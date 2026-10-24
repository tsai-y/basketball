from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from development.partial_finetune_v1.data import RGBTrainingDataset, check_split_metadata
from development.partial_finetune_v1.engine import evaluate_loader, metrics_from_logits, train_one_epoch
from development.partial_finetune_v1 import train_rgb


class SyntheticDataset(Dataset):
    """Small vectors, not real videos; optimizer tests must never touch project data."""

    def __init__(self, count=7, split="train"):
        self.x = torch.randn(count, 3)
        self.y = torch.arange(count) % 2
        self.rows = [{"video_id": f"{split}/{i}", "binary_label": str(int(self.y[i])),
                      "original_class": "ft" + str(int(self.y[i])), "split": split,
                      "excluded": "0", "source_group": split + "_source"}
                     for i in range(count)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"tensor": self.x[index], "binary_label": int(self.y[index]),
                "video_id": self.rows[index]["video_id"]}


class SyntheticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.head = nn.Linear(3, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)

    def optimizer_groups(self):
        return [{"params": self.backbone.parameters(), "lr": 1e-5},
                {"params": self.head.parameters(), "lr": 1e-4}]


class RGBTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(20261024)

    def test_accumulation_matches_true_large_batches_including_final_25(self):
        # 89 = 64 + 25. Micro-batch=2 gives a final one-video micro-batch.
        data = SyntheticDataset(89)
        actual = SyntheticModel()
        reference = deepcopy(actual)
        optimizer = torch.optim.SGD(actual.parameters(), lr=.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        for batch in DataLoader(data, batch_size=64):
            reference_optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                reference(batch["tensor"]), batch["binary_label"].float())
            loss.backward()
            nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            reference_optimizer.step()
        info = train_one_epoch(actual, DataLoader(data, batch_size=2), optimizer,
                               device=torch.device("cpu"), progress=None)
        self.assertEqual(info["optimizer_steps"], 2)
        self.assertEqual(info["videos"], 89)
        for a, b in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    def test_micro_batch_one_keeps_the_same_effective_batch(self):
        data = SyntheticDataset(7)
        a = SyntheticModel()
        b = deepcopy(a)
        for model, micro in ((a, 1), (b, 2)):
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            info = train_one_epoch(model, DataLoader(data, batch_size=micro), optimizer,
                                   device="cpu", effective_batch_size=4, progress=None)
            self.assertEqual(info["optimizer_steps"], 2)
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-7)

    def test_real_train_size_means_113_updates_and_final_25(self):
        total = 7193
        groups = [min(64, total - start) for start in range(0, total, 64)]
        self.assertEqual(len(groups), 113)
        self.assertEqual(groups[-1], 25)
        self.assertEqual(sum(groups), total)

    def test_validation_matches_global_metrics_and_never_changes_model(self):
        data = SyntheticDataset(5, "validation")
        model = SyntheticModel()
        before = {n: v.clone() for n, v in model.state_dict().items()}
        expected_logits = model(data.x).detach()
        expected = metrics_from_logits(expected_logits, data.y.float())
        ids = [r["video_id"] for r in data.rows]
        actual, logits, labels, actual_ids = evaluate_loader(
            model, DataLoader(data, batch_size=2), device="cpu", expected_ids=ids, progress=None)
        self.assertEqual(actual["confusion_matrix"], expected["confusion_matrix"])
        self.assertAlmostEqual(actual["loss"], expected["loss"], places=6)
        torch.testing.assert_close(logits, expected_logits)
        self.assertEqual(actual_ids, ids)
        self.assertTrue(torch.equal(labels, data.y.float()))
        self.assertFalse(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_validation_order_and_nonfinite_outputs_rejected(self):
        data = SyntheticDataset(3, "validation")
        model = SyntheticModel()
        with self.assertRaisesRegex(RuntimeError, "order/IDs"):
            evaluate_loader(model, DataLoader(data, batch_size=2), device="cpu",
                            expected_ids=["wrong"], progress=None)
        with torch.no_grad():
            model.head.weight.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "validation logits"):
            evaluate_loader(model, DataLoader(data, batch_size=2), device="cpu", progress=None)

    def test_nonfinite_inputs_fail_before_any_optimizer_update(self):
        data = SyntheticDataset(3)
        data.x[0, 0] = float("nan")
        model = SyntheticModel()
        optimizer = torch.optim.AdamW(model.parameters())
        with patch.object(optimizer, "step", side_effect=AssertionError("Update")):
            with self.assertRaisesRegex(RuntimeError, "input tensor"):
                train_one_epoch(model, DataLoader(data, batch_size=2), optimizer,
                                device="cpu", progress=None)

    def test_duplicate_training_ids_rejected(self):
        data = SyntheticDataset(3)
        data.rows[1]["video_id"] = data.rows[0]["video_id"]
        model = SyntheticModel()
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(RuntimeError, "Duplicate training IDs"):
            train_one_epoch(model, DataLoader(data, batch_size=2), optimizer, device="cpu", progress=None)

    def test_dropping_videos_and_invalid_accumulation_rejected(self):
        data = SyntheticDataset(3)
        model = SyntheticModel()
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(RuntimeError, "drop_last"):
            train_one_epoch(model, DataLoader(data, batch_size=2, drop_last=True),
                            optimizer, device="cpu", progress=None)
        with self.assertRaisesRegex(RuntimeError, "multiple"):
            train_one_epoch(model, DataLoader(data, batch_size=2), optimizer,
                            device="cpu", effective_batch_size=3, progress=None)

    def test_explicit_mode_and_valid_arguments_required_before_runtime(self):
        cases = ([], ["--train", "--dry-run"], ["--dry-run", "--epochs", "0"],
                 ["--dry-run", "--micro-batch", "64"], ["--dry-run", "--seed", "-1"])
        for args in cases:
            with self.subTest(args=args), redirect_stderr(io.StringIO()), \
                 patch.object(train_rgb, "configure_runtime", side_effect=AssertionError("Runtime")):
                with self.assertRaises(SystemExit):
                    train_rgb.main(args)

    def test_dry_run_cannot_create_optimizer_run_directory_or_start_training(self):
        train, val = SyntheticDataset(4), SyntheticDataset(4, "validation")
        model = SyntheticModel()
        with patch.object(train_rgb, "configure_runtime", return_value=torch.device("cpu")), \
             patch.object(train_rgb, "load_gate_config", return_value=object()), \
             patch.object(train_rgb, "RGBTrainingDataset", side_effect=[train, val]), \
             patch.object(train_rgb, "check_split_metadata", return_value={}), \
             patch.object(train_rgb, "load_initialized_model", return_value=(model, {})), \
             patch.object(train_rgb, "make_config", return_value={"accumulation_steps": 32}), \
             patch.object(torch.optim, "AdamW", side_effect=AssertionError("Optimizer")), \
             patch.object(train_rgb, "mkdtemp", side_effect=AssertionError("Write")), \
             patch.object(train_rgb, "run_training", side_effect=AssertionError("Train")), \
             redirect_stdout(io.StringIO()) as output:
            train_rgb.main(["--dry-run"])
        self.assertIn('"optimizer_steps": 0', output.getvalue())
        self.assertIn('"all_model_state_unchanged": true', output.getvalue())
        self.assertIn('"output_files_written": false', output.getvalue())

    def test_training_function_itself_requires_explicit_train(self):
        args = train_rgb.parse_args(["--dry-run"])
        with patch.object(train_rgb, "mkdtemp", side_effect=AssertionError("Write")):
            with self.assertRaisesRegex(RuntimeError, "Explicit --train"):
                train_rgb.run_training(None, None, None, args, None, {})

    def test_synthetic_only_training_writes_best_last_history_and_predictions(self):
        train, val, model = SyntheticDataset(5), SyntheticDataset(3, "validation"), SyntheticModel()
        args = train_rgb.parse_args(["--train", "--epochs", "2"])
        config = {"experiment": "synthetic_unit_test_only", "seed": args.seed}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(train_rgb, "FOLDER", Path(temporary)), redirect_stdout(io.StringIO()):
            output = train_rgb.run_training(model, train, val, args, torch.device("cpu"), config)
            best = torch.load(output / "best.pt", weights_only=True)
            last = torch.load(output / "last.pt", weights_only=True)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(last["epoch"], 2)
            self.assertEqual(best["epoch"], summary["best_epoch"])
            self.assertEqual(best["config"]["experiment"], "synthetic_unit_test_only")
            with (output / "history.csv").open(newline="") as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual([int(r["epoch"]) for r in history], [0, 1, 2])
            minimum = min(history, key=lambda r: float(r["val_loss"]))
            self.assertEqual(int(minimum["epoch"]), summary["best_epoch"])
            with (output / "validation_predictions.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["video_id"] for r in rows], [r["video_id"] for r in val.rows])

    def test_epoch_zero_is_retained_if_all_finetuning_epochs_worsen(self):
        train, val, model = SyntheticDataset(2), SyntheticDataset(2, "validation"), SyntheticModel()
        args = train_rgb.parse_args(["--train", "--epochs", "50"])
        labels = val.y.float()
        ids = [r["video_id"] for r in val.rows]
        initial, worse = torch.tensor([-2., 2.]), torch.zeros(2)
        calls = 0

        def fake_evaluate(*args, **kwargs):
            nonlocal calls
            logits = initial if calls == 0 else worse
            calls += 1
            return metrics_from_logits(logits, labels), logits, labels, ids

        config = {"experiment": "synthetic_no_updates_early_stop_test", "seed": args.seed}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(train_rgb, "FOLDER", Path(temporary)), \
             patch.object(train_rgb, "evaluate_loader", side_effect=fake_evaluate), \
             patch.object(train_rgb, "train_one_epoch", return_value={"loss": .5, "optimizer_steps": 1}), \
             redirect_stdout(io.StringIO()):
            output = train_rgb.run_training(model, train, val, args, torch.device("cpu"), config)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["epochs_completed"], 10)
            self.assertEqual(summary["stop_reason"], "early_stopping")
            self.assertEqual(summary["best_epoch"], 0)
            self.assertTrue(summary["selected_initial_weights"])
            self.assertEqual(calls, 11)

    def test_metrics_reject_bad_labels(self):
        with self.assertRaisesRegex(RuntimeError, "metric labels"):
            metrics_from_logits(torch.zeros(2), torch.tensor([0., 2.]))


class SplitMetadataTests(unittest.TestCase):
    def make_splits(self):
        splits = []
        for name, counts in (("train", (3421, 3772)), ("validation", (740, 807))):
            rows = [{"video_id": f"{name}/{label}/{i}", "binary_label": str(label), "split": name,
                     "excluded": "0", "source_group": name + "_source"}
                    for label, count in enumerate(counts) for i in range(count)]
            dataset = SimpleNamespace(rows=rows, split=name)
            # SimpleNamespace has no __len__; use a tiny read-only metadata container.
            class MetadataDataset:
                def __init__(self, rows, split):
                    self.rows, self.split = rows, split
                def __len__(self):
                    return len(self.rows)
            splits.append(MetadataDataset(dataset.rows, dataset.split))
        return splits

    def test_full_counts_and_source_disjointness(self):
        info = check_split_metadata(*self.make_splits())
        self.assertEqual(info["train"]["videos"], 7193)
        self.assertEqual(info["validation"]["videos"], 1547)
        self.assertTrue(info["source_groups_disjoint"])

    def test_wrong_counts_labels_excluded_flags_and_shared_sources_rejected(self):
        for change in ("count", "label", "excluded", "source"):
            train, val = self.make_splits()
            if change == "count":
                train.rows.pop()
            elif change == "label":
                train.rows[0]["binary_label"] = "1"
            elif change == "excluded":
                val.rows[0]["excluded"] = "1"
            else:
                val.rows[0]["source_group"] = train.rows[0]["source_group"]
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                check_split_metadata(train, val)

    def test_test_split_rejected_before_underlying_dataset_access(self):
        with patch("development.partial_finetune_v1.data.BasketballThreeWindowDataset",
                   side_effect=AssertionError("Accessed dataset")):
            with self.assertRaises(ValueError):
                RGBTrainingDataset(None, split="test")


if __name__ == "__main__":
    unittest.main()
