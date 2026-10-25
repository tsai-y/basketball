import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from development.aca_net_v1 import train
from development.aca_net_v1.readiness import ToyDataset, synthetic_resume_probe


class TrainingTests(unittest.TestCase):
    def test_config_blocks_unresolved_aca_and_unsupported_settings(self):
        for kwargs in ({"variant": "aca"}, {"micro_batch": 3}, {"epochs": 0},
                       {"lr": .1}, {"workers": -1}, {"seed": True}):
            with self.assertRaises(ValueError):
                train.TrainConfig(**kwargs)

    def test_fixed_epoch_order_coverage_and_rng_isolation(self):
        torch.manual_seed(6)
        before = torch.get_rng_state().clone()
        first = train.epoch_order(7193, 123, 0)
        self.assertEqual(sorted(first), list(range(7193)))
        self.assertEqual(first, train.epoch_order(7193, 123, 0))
        self.assertNotEqual(first, train.epoch_order(7193, 123, 1))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_loader_resume_order_and_no_model_rng_consumption(self):
        dataset = ToyDataset(19)
        config = train.TrainConfig(workers=0)
        indices = train.epoch_order(19, config.seed, 0)[16:]
        before = torch.get_rng_state().clone()
        batches = list(train.make_loader(dataset, indices, config, 0))
        self.assertEqual([b["video_id"][0] for b in batches], [f"synthetic-{i}" for i in indices])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_metrics_binary_and_shot_type(self):
        metric = train.Metrics()
        metric.add(torch.tensor([[3., 0.], [0., 3.], [0., 3.], [0., 3.]]),
                   torch.tensor([0, 0, 1, 1]), 2., ["2p0", "2p0", "ft1", "ft1"])
        result = metric.result()
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["loss"], .5)
        self.assertEqual(result["miss_recall"], .5)
        self.assertEqual(result["make_recall"], 1.)
        self.assertAlmostEqual(result["macro_f1"], (.6666666666666666 + .8) / 2)
        self.assertEqual(result["by_shot_type"]["ft"]["count"], 2)

    def test_synthetic_production_loop_resume_is_bitwise_equal(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = synthetic_resume_probe(Path(directory))
        self.assertTrue(result["bitwise_equal"])
        self.assertEqual(result["last_effective_batch"], 3)

    def test_checkpoint_rejects_engineering_and_mismatched_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            model = nn.Linear(3, 2)
            optimizer = torch.optim.Adam(model.parameters())
            plan = {"config": vars(train.TrainConfig()), "train_count": 19}
            torch.save({"schema": train.SCHEMA, "engineering_only": True}, path)
            with self.assertRaises(ValueError):
                train.load_checkpoint(path, model, optimizer, plan)
            train.save_checkpoint(path, model, optimizer, train.initial_progress(), plan)
            with self.assertRaises(ValueError):
                train.load_checkpoint(path, model, optimizer, {**plan, "train_count": 20})

    def test_readiness_rejects_stale_or_trained_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            plan = {"a": 1}
            good = {"status": "passed", "optimizer_steps": 0, "formal_training_started": False,
                    "plan_sha256": train.canonical_hash(plan), "cuda_resume_bitwise_equal": True}
            train.atomic_json(path, good)
            train.verify_preflight(path, plan)
            for bad in ({**good, "optimizer_steps": 1}, {**good, "plan_sha256": "wrong"},
                        {**good, "cuda_resume_bitwise_equal": False}):
                train.atomic_json(path, bad)
                with self.assertRaises(ValueError):
                    train.verify_preflight(path, plan)

    def test_no_cuda_fails_before_training_side_effects(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(train, "build_plan", side_effect=AssertionError("Must not open data")):
            with self.assertRaises(ValueError):
                train.train_run(Path("/nonexistent"), train.TrainConfig(), Path("/nonexistent"))

    def test_default_cli_only_plans_and_never_trains(self):
        with patch.object(sys, "argv", ["train", "--source-root", "/nonexistent"]), \
             patch.object(train, "build_plan", return_value={"formal_training_started": False}) as planner, \
             patch.object(train, "train_run", side_effect=AssertionError("Must not train")), \
             contextlib.redirect_stdout(io.StringIO()):
            train.main()
        planner.assert_called_once()

    def test_validation_loss_ties_keep_first_best_and_early_stop(self):
        config = train.TrainConfig(epochs=3, patience=1, workers=0)
        plan = {"config": vars(config), "train_count": 19, "validation_count": 5}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            model = nn.Linear(3, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            with patch.object(train, "validation_pass", return_value={"count": 5, "loss": .7}):
                result = train.fit(model, optimizer, ToyDataset(19), ToyDataset(5), config, plan,
                                   output, train.initial_progress(), device="cpu")
            self.assertEqual(result["epochs"], 2)
            self.assertEqual(result["stop_reason"], "early_stop")
            best = torch.load(output / "best.pt", weights_only=True)
            last = torch.load(output / "last.pt", weights_only=True)
            self.assertEqual(best["progress"]["epoch"], 1)
            self.assertEqual(last["progress"]["epoch"], 2)
            self.assertFalse(last["progress"]["history"][-1]["selected_as_best"])

    def test_validation_interruption_does_not_repeat_training_updates(self):
        config = train.TrainConfig(epochs=1, patience=1, workers=0)
        plan = {"config": vars(config), "train_count": 19, "validation_count": 5}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            model = nn.Linear(3, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            with patch.object(train, "validation_pass", return_value=None):
                result = train.fit(model, optimizer, ToyDataset(19), ToyDataset(5), config, plan,
                                   output, train.initial_progress(), device="cpu")
            self.assertEqual(result["status"], "stopped_validation_will_repeat")
            progress = train.load_checkpoint(output / "last.pt", model, optimizer, plan)
            self.assertEqual(progress["cursor"], 19)
            with patch.object(optimizer, "step", side_effect=AssertionError("Must not retrain")):
                result = train.fit(model, optimizer, ToyDataset(19), ToyDataset(5), config, plan,
                                   output, progress, device="cpu")
            self.assertEqual(result["optimizer_steps"], 2)
            self.assertEqual(result["epochs"], 1)

    def test_production_loop_partial_batch_matches_mean_loss_reference(self):
        config = train.TrainConfig(epochs=1, patience=1, workers=0)
        plan = {"config": vars(config), "train_count": 19, "validation_count": 5}
        train.configure_runtime(7, cuda=False)
        model = nn.Linear(3, 2).double()
        reference = nn.Linear(3, 2).double()
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=config.lr)

        class DoubleToy(ToyDataset):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                item["tensor"] = item["tensor"].double()
                return item

        dataset = DoubleToy(19)
        order = train.epoch_order(19, config.seed, 0)
        for start in (0, 16):
            items = [dataset[i] for i in order[start:start+16]]
            x = torch.stack([item["tensor"] for item in items])
            y = torch.tensor([item["binary_label"] for item in items])
            reference_optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(reference(x), y).backward()
            nn.utils.clip_grad_norm_(reference.parameters(), config.clip_norm)
            reference_optimizer.step()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            train.fit(model, optimizer, dataset, DoubleToy(5), config, plan, Path(directory),
                      train.initial_progress(), device="cpu")
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
