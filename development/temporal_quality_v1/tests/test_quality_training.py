"""Tiny synthetic training tests; production-data checks never step an optimizer."""

import copy
from contextlib import redirect_stderr, redirect_stdout
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
from torch.utils.data import DataLoader, TensorDataset

from development.temporal_baseline_v1 import train_temporal
from development.temporal_quality_v1 import train_quality as train
from development.temporal_quality_v1 import weights as quality


def quality_row(video_id, label, counts):
    return {"video_id": video_id, "label": label, "shot_type": "ft",
            "original_class": "ft1" if label == "Make" else "ft0",
            "source_group": "source", "fps_rational": "24/1",
            "candidate_overlap_frames": counts, "window_exposure_ratio": sum(counts) / 192}


class QualityWeightAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rows = [quality_row("a", "Miss", [0, 0, 0]),
                     quality_row("b", "Make", [64, 64, 64])]

    def test_order_independent_id_join_and_global_mean(self):
        weights, info, _ = quality.aligned_weights(self.rows, ["b", "a"], torch.tensor([1., 0.]))
        torch.testing.assert_close(weights, torch.tensor([.75 / .875, 1 / .875]))
        self.assertAlmostEqual(float(weights.double().mean()), 1, places=6)
        self.assertEqual(info["alpha"], .25)
        self.assertFalse(info["validation_weighting"])

    def test_alpha05_uses_different_weights_but_same_global_normalization(self):
        w025, i025, _ = quality.aligned_weights(self.rows, ["a", "b"], torch.tensor([0., 1.]))
        w05, i05, report = quality.aligned_weights(self.rows, ["a", "b"], torch.tensor([0., 1.]), alpha=.5)
        torch.testing.assert_close(w05, torch.tensor([1 / .75, .5 / .75]))
        self.assertEqual(i05["alpha"], report["alpha"])
        self.assertEqual(i05["alpha"], .5)
        self.assertAlmostEqual(float(w05.mean()), 1)
        self.assertNotEqual(i025["aligned_weights_sha256"], i05["aligned_weights_sha256"])
        self.assertFalse(torch.equal(w025, w05))

    def test_invalid_alpha_rejected_before_io(self):
        with patch.object(quality.audit, "run_audit", side_effect=AssertionError("IO")):
            with self.assertRaises(ValueError):
                quality.load_train_weights([], torch.tensor([]), cache_root=Path("/not/read"), alpha=.75)

    def test_other_split_rejected_before_io(self):
        for split in ("validation", "test"):
            with self.subTest(split=split), patch.object(quality.audit, "run_audit", side_effect=AssertionError("IO")):
                with self.assertRaises(ValueError):
                    quality.load_train_weights([], torch.tensor([]), cache_root=Path("/not/read"), split=split)

    def test_duplicate_missing_or_wrong_label_fail_closed(self):
        cases = [(["a", "a"], [0., 0.]), (["a", "other"], [0., 1.]),
                 (["a", "b"], [1., 0.]), (["a"], [0.])]
        for ids, labels in cases:
            with self.subTest(ids=ids, labels=labels), self.assertRaises(ValueError):
                quality.aligned_weights(self.rows, ids, torch.tensor(labels))


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x.mean((1, 2))).squeeze(-1)


class BaselineReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = {}
        for seed in (20261024, 20261025, 20261026):
            path = Path(self.temp.name) / str(seed)
            path.mkdir()
            config = {
                "experiment": "frozen_mixnet_i3d_window_transformer_v1", "seed": seed,
                "epochs": 50, "cpu_threads": 4, "batch_size": 64, "eval_batch_size": 64,
                "optimizer": "AdamW", "lr": .0001, "weight_decay": .0001,
                "max_grad_norm": 1., "selection": "minimum_validation_loss", "threshold": .5,
                "cache_run_id": train.CACHE_RUN_ID, "quality_weighting_used": False,
                "backbone_trainable": False, "device": "cpu", "precision": "float32",
                "train_count": 7193, "validation_count": 1547, "trainable_parameters": 1450497,
                "deterministic_algorithms": True, "torch_version": str(torch.__version__),
                "source_sha256": {},
                "early_stopping": {"monitor": "validation_loss", "patience": 10, "min_delta": .0001}}
            summary = {"experiment": config["experiment"], "seed": seed,
                       "cache_run_id": train.CACHE_RUN_ID, "validation": {"loss": .5}}
            (path / "config.json").write_text(json.dumps(config))
            (path / "summary.json").write_text(json.dumps(summary))
            self.paths[seed] = path

    def test_each_seed_selects_its_own_predeclared_baseline(self):
        with patch.object(train, "BASELINES", self.paths):
            for seed, path in self.paths.items():
                config, reference = train.load_baseline_reference(seed)
                self.assertEqual(config["seed"], seed)
                self.assertEqual(reference["paired_seed"], seed)
                self.assertEqual(reference["path"], str(path))
                self.assertFalse(reference["head_checkpoint_loaded"])

    def test_wrong_seed_reference_is_rejected_not_silently_reused(self):
        with patch.object(train, "BASELINES", {20261025: self.paths[20261024]}):
            with self.assertRaisesRegex(RuntimeError, "Baseline setting changed: seed"):
                train.load_baseline_reference(20261025)

    def test_unlisted_seed_rejected_before_file_access(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("Read")):
            with self.assertRaisesRegex(RuntimeError, "No predeclared"):
                train.load_baseline_reference(999)


class QualityTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        cls.old_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        torch.use_deterministic_algorithms(cls.old_deterministic)

    def setUp(self):
        torch.manual_seed(20261024)
        self.model = TinyHead()
        self.x = torch.randn(5, 3, 8, 4)
        self.y = torch.tensor([0., 1., 0., 1., 1.])
        self.w = torch.tensor([1.02, .97, 1.01, 1., 1.])

    def test_weighted_bce_and_gradient_match_manual_reference(self):
        logits = torch.tensor([-.2, .7], requires_grad=True)
        y, w = torch.tensor([0., 1.]), torch.tensor([.95, 1.05])
        loss, ordinary = train.weighted_bce(logits, y, w)
        expected = (nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        torch.testing.assert_close(ordinary, nn.functional.binary_cross_entropy_with_logits(logits, y))
        expected_gradient = w * (logits.detach().sigmoid() - y) / 2
        loss.backward()
        torch.testing.assert_close(logits.grad, expected_gradient)

    def test_no_per_batch_weight_renormalization(self):
        logit = torch.tensor([0.], requires_grad=True)
        loss, _ = train.weighted_bce(logit, torch.tensor([0.]), torch.tensor([.9]))
        self.assertAlmostEqual(float(loss.detach()), .9 * float(torch.log(torch.tensor(2.))), places=6)

    def test_invalid_weight_or_nonfinite_loss_rejected(self):
        for w in (torch.tensor([0., 1.]), torch.tensor([float("nan"), 1.]),
                  torch.ones(2, requires_grad=True)):
            with self.subTest(w=w), self.assertRaises(RuntimeError):
                train.weighted_bce(torch.zeros(2), torch.tensor([0., 1.]), w)
        with self.assertRaises(RuntimeError):
            train.weighted_bce(torch.tensor([float("nan")]), torch.zeros(1), torch.ones(1))

    def test_identical_shuffle_to_baseline_and_complete_coverage(self):
        loader = train.make_loader(self.x, self.y, self.w, 123, batch_size=2)
        reference = DataLoader(TensorDataset(torch.arange(5)), batch_size=2, shuffle=True,
                               generator=torch.Generator().manual_seed(123), num_workers=0)
        actual = torch.cat([batch[3] for batch in loader])
        expected = torch.cat([batch[0] for batch in reference])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(sorted(actual.tolist()), list(range(5)))

    def test_all_one_weights_reproduce_original_training_helper(self):
        reference = copy.deepcopy(self.model)
        old_loader = DataLoader(TensorDataset(self.x, self.y), batch_size=2, shuffle=True,
                                generator=torch.Generator().manual_seed(123), num_workers=0)
        new_loader = train.make_loader(self.x, self.y, torch.ones(5), 123, batch_size=2)
        old_opt = torch.optim.AdamW(reference.parameters(), lr=.001)
        new_opt = torch.optim.AdamW(self.model.parameters(), lr=.001)
        old_loss = train_temporal.train_one_epoch(reference, old_loader, old_opt)
        result = train.train_one_epoch(self.model, new_loader, new_opt)
        self.assertAlmostEqual(old_loss, result["weighted_loss"], places=7)
        self.assertEqual(result["videos_seen"], 5)
        self.assertEqual(result["optimizer_steps"], 3)  # Includes the last one-video batch.
        for actual, expected in zip(self.model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_all_one_real_transformer_with_dropout_matches_baseline(self):
        # Actual architecture, but RANDOM synthetic tensors, never cached videos.
        x = torch.randn(3, 3, 8, 1536)
        y = torch.tensor([0., 1., 0.])
        model = train.WindowTemporalTransformer()
        reference = copy.deepcopy(model)
        old_loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=True,
                                generator=torch.Generator().manual_seed(123), num_workers=0)
        new_loader = train.make_loader(x, y, torch.ones(3), 123, batch_size=2)
        old_opt = torch.optim.AdamW(reference.parameters(), lr=.0001)
        new_opt = torch.optim.AdamW(model.parameters(), lr=.0001)
        torch.manual_seed(77)
        train_temporal.train_one_epoch(reference, old_loader, old_opt)
        torch.manual_seed(77)
        train.train_one_epoch(model, new_loader, new_opt)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_drop_last_rejected(self):
        loader = DataLoader(TensorDataset(self.x, self.y, self.w, torch.arange(5)),
                            batch_size=2, drop_last=True)
        with self.assertRaisesRegex(RuntimeError, "Full training coverage"):
            train.train_one_epoch(self.model, loader, torch.optim.SGD(self.model.parameters(), lr=.01))

    def test_duplicate_indices_rejected(self):
        loader = DataLoader(TensorDataset(self.x, self.y, self.w, torch.tensor([0, 0, 2, 3, 4])), batch_size=5)
        before = train.state_sha(self.model)
        with self.assertRaisesRegex(RuntimeError, "Duplicate/invalid"):
            train.train_one_epoch(self.model, loader, torch.optim.SGD(self.model.parameters(), lr=.01))
        self.assertEqual(train.state_sha(self.model), before)

    def test_validation_is_ordinary_baseline_evaluation(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.x)
            expected = nn.functional.binary_cross_entropy_with_logits(logits, self.y)
        metrics, actual = train.evaluate(self.model, self.x, self.y, batch_size=2)
        self.assertAlmostEqual(metrics["loss"], float(expected), places=6)
        torch.testing.assert_close(actual, logits)
        self.assertIs(train.evaluate, train_temporal.evaluate)

    def prepared(self, alpha=.25):
        return {"model": self.model, "train_x": self.x, "train_y": self.y,
                "train_ids": [f"train{i}" for i in range(5)], "weights": self.w,
                "val_x": self.x[:4], "val_y": self.y[:4],
                "val_ids": [f"val{i}" for i in range(4)],
                "val_groups": ["2p0", "2p1", "ft0", "ft1"],
                "weight_audit": {}, "config": {
                    "experiment": "synthetic_only", "alpha": alpha, "batch_size": 2, "eval_batch_size": 2,
                    "lr": .001, "weight_decay": .0001, "max_grad_norm": 1.,
                    "quality_weighting": {"alpha": alpha}, "early_stopping": {"patience": 10, "min_delta": .0001},
                    "baseline_reference": {"summary": {"validation": {
                        "loss": 1., "accuracy": .5, "macro_f1": .5}}}}}

    def test_dry_run_never_creates_optimizer_directory_or_starts_training(self):
        prepared = self.prepared()
        before = train.state_sha(self.model)
        with patch.object(train, "prepare", return_value=prepared), \
             patch.object(torch.optim, "AdamW", side_effect=AssertionError("Optimizer created")), \
             patch.object(train, "mkdtemp", side_effect=AssertionError("Output directory")), \
             patch.object(train, "save_checkpoint", side_effect=AssertionError("Checkpoint written")), \
             patch.object(train, "run_training", side_effect=AssertionError("Training started")), \
             redirect_stdout(io.StringIO()) as output:
            train.main(["--dry-run", "--threads", "1"])
        result = json.loads(output.getvalue())
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertFalse(result["output_files_written"])
        self.assertEqual(train.state_sha(self.model), before)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_explicit_mode_and_fixed_experiment_arguments(self):
        for argv in ([], ["--train", "--dry-run"], ["--train", "--seed", "20261027"],
                     ["--train", "--alpha", ".75"], ["--train", "--threads", "1"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(argv)

    def test_three_predeclared_seeds_are_accepted(self):
        for seed in (20261024, 20261025, 20261026):
            self.assertEqual(train.parse_args(["--dry-run", "--seed", str(seed)]).seed, seed)

    def test_alpha_arguments_and_output_tags_are_unambiguous(self):
        self.assertEqual(train.parse_args(["--dry-run"]).alpha, .25)
        self.assertEqual(train.parse_args(["--dry-run", "--alpha", ".5"]).alpha, .5)
        self.assertEqual(train.alpha_tag(.25), "025")
        self.assertEqual(train.alpha_tag(.5), "050")

    def test_programmatic_training_requires_explicit_flag(self):
        with self.assertRaisesRegex(RuntimeError, "explicit --train"):
            train.run_training(None, SimpleNamespace(train=False, dry_run=True))

    def test_synthetic_checkpoint_history_and_summary_lifecycle(self):
        # Only a four-channel toy model and five random tensors are optimized.
        # This test does not open the Basketball-51 cache or start a real run.
        prepared = self.prepared()
        args = SimpleNamespace(train=True, dry_run=False, seed=20261024, epochs=2, alpha=.25)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(train, "FOLDER", Path(folder)), redirect_stdout(io.StringIO()):
                output = train.run_training(prepared, args)
            best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(last["epoch"], 2)
            self.assertEqual(best["epoch"], summary["best_epoch"])
            self.assertEqual(summary["alpha"], .25)
            self.assertTrue(output.name.startswith("quality_alpha025_seed"))
            self.assertFalse(summary["validation_weighting_used"])
            self.assertFalse(summary["test_used"])
            with (output / "history.csv").open(newline="") as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual(len(history), 2)
            self.assertTrue(all(row["train_videos"] == "5" and row["optimizer_steps"] == "3" for row in history))
            self.assertTrue((output / "train_weights.pt").is_file())
            self.assertTrue((output / "validation_predictions.csv").is_file())

    def test_early_stopping_and_minimum_loss_checkpoint_are_separate(self):
        prepared = self.prepared()
        prepared["config"]["early_stopping"]["patience"] = 1
        args = SimpleNamespace(train=True, dry_run=False, seed=20261024, epochs=5, alpha=.25)
        logits = torch.tensor([-1., 1., -1., 1.])
        metrics = {"loss": .6, "accuracy": 1., "macro_f1": 1., "confusion_matrix": [[2, 0], [0, 2]],
                   "per_class": {}}
        tiny_improvement = {**metrics, "loss": .59995}  # Smaller than min_delta.
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(train, "FOLDER", Path(folder)), \
                 patch.object(train, "evaluate", side_effect=[(metrics, logits),
                              (tiny_improvement, logits), (tiny_improvement, logits)]), \
                 redirect_stdout(io.StringIO()):
                output = train.run_training(prepared, args)
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["epochs_completed"], 2)
            self.assertEqual(result["stop_reason"], "early_stopping")
            self.assertEqual(result["best_epoch"], 2)  # Absolute lowest loss is still saved.

    def test_alpha05_synthetic_outputs_do_not_use_alpha025_names_or_metadata(self):
        prepared = self.prepared(alpha=.5)
        args = SimpleNamespace(train=True, dry_run=False, seed=20261024, epochs=1, alpha=.5)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(train, "FOLDER", Path(folder)), redirect_stdout(io.StringIO()):
                output = train.run_training(prepared, args)
            summary = json.loads((output / "summary.json").read_text())
            config = json.loads((output / "config.json").read_text())
            self.assertTrue(output.name.startswith("quality_alpha050_seed20261024_"))
            self.assertEqual(summary["alpha"], .5)
            self.assertEqual(config["quality_weighting"]["alpha"], .5)

    def test_alpha_mismatch_rejected_before_output_or_optimizer(self):
        args = SimpleNamespace(train=True, dry_run=False, seed=20261024, epochs=1, alpha=.5)
        with patch.object(train, "mkdtemp", side_effect=AssertionError("Output")), \
             patch.object(torch.optim, "AdamW", side_effect=AssertionError("Optimizer")):
            with self.assertRaisesRegex(RuntimeError, "alpha mismatch"):
                train.run_training(self.prepared(alpha=.25), args)


if __name__ == "__main__":
    unittest.main()
