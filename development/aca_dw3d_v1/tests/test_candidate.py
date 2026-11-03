import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from development.aca_dw3d_v1 import engine, experiment as exp, preflight, train
from development.aca_net_v1 import train as baseline
from development.aca_net_v1.baseline_preflight import freeze_batchnorm
from development.aca_net_v1.modules import ACAModule
from development.aca_dw3d_v1.model import DW3DConfig as ReconstructionConfig, convert_lsta_gating
from development.aca_net_v1.readiness import ToyDataset


class TinyVideoData(ToyDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["tensor"] = (torch.arange(144).float().reshape(3, 3, 4, 4) * .01 + index * .03).sin()
        return item


class TinyVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv3d(3, 4, 1)
        self.aca = ACAModule(4, ReconstructionConfig(frames=3))
        convert_lsta_gating(self.aca.lsta)
        self.dropout = nn.Dropout(.2)
        self.fc = nn.Linear(4, 2)
        freeze_batchnorm(self)

    def forward(self, x):
        return self.fc(self.dropout(self.aca(self.stem(x)).mean(dim=(2, 3, 4))))


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_candidate_config_and_no_silent_architecture_substitution(self):
        config = exp.config_from(exp.FOLDER / "config.json")
        self.assertEqual(config.variant, "aca_dw3d_candidate")
        self.assertEqual(ReconstructionConfig().gate_groups, "depthwise_3x3")
        for kwargs in ({"variant": "baseline"}, {"variant": "aca"}, {"micro_batch": 2},
                       {"lr": .001}, {"seed": True}, {"workers": -1}):
            with self.assertRaises(ValueError):
                exp.CandidateConfig(**kwargs)

    def test_environment_uses_only_NVML_metadata_without_CUDA_context(self):
        with patch.object(exp.subprocess, "run", return_value=SimpleNamespace(stdout="GPU, GPU-uuid, 1.0, 16303 MiB\n")), \
             patch.object(torch.cuda, "init", side_effect=AssertionError("Must not initialize CUDA")):
            identity = exp.environment()
        self.assertEqual(identity["gpu"][1], "GPU-uuid")

    def test_second_compute_process_blocks_GPU_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            baseline.atomic_json(path / "summary.json", {"status": "completed"})
            with patch.object(exp, "REFERENCE_RUN", path), \
                 patch.object(exp.subprocess, "run", return_value=SimpleNamespace(stdout="999999\n")), \
                 patch.object(Path, "resolve", return_value=Path("/usr/bin/python3.10")):
                with self.assertRaisesRegex(ValueError, "Other GPU compute"):
                    exp.assert_launch_prerequisites(True)

    def test_default_cli_never_launches_or_decodes(self):
        with patch.object(sys, "argv", ["candidate", "--source-root", "/not-used"]), \
             patch.object(train, "build_plan", return_value={"formal_training_started": False}), \
             patch.object(train, "run_training", side_effect=AssertionError("Must not train")), \
             patch.object(train, "prepare", side_effect=AssertionError("Must not write")), \
             contextlib.redirect_stdout(io.StringIO()):
            train.main()

    def test_launch_acknowledgement_required_before_any_gpu_query(self):
        with patch.object(exp.subprocess, "run", side_effect=AssertionError("Must not query GPU")):
            with self.assertRaises(ValueError):
                exp.assert_launch_prerequisites(False)

    def test_unfinished_baseline_rejected_before_gpu_query_or_context(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(exp, "REFERENCE_RUN", Path(directory)), \
             patch.object(exp.subprocess, "run", side_effect=AssertionError("Must not query")), \
             patch.object(torch.cuda, "init", side_effect=AssertionError("Must not initialize CUDA")):
            with self.assertRaisesRegex(ValueError, "Baseline is not complete"):
                exp.assert_launch_prerequisites(True)

    def test_failed_baseline_not_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            baseline.atomic_json(path / "summary.json", {"status": "failed"})
            with patch.object(exp, "REFERENCE_RUN", path), \
                 patch.object(exp.subprocess, "run", side_effect=AssertionError("Must not query")):
                with self.assertRaises(ValueError):
                    exp.assert_launch_prerequisites(True)

    def test_cpu_baseline_and_stale_reports_do_not_unlock_training(self):
        plan = {"identity": "candidate"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            good = {"schema": exp.PREFLIGHT_SCHEMA, "status": "passed", "plan_sha256": baseline.canonical_hash(plan),
                    "engineering_only": True, "formal_training_started": False,
                    "checkpoint_may_initialize_formal_training": False,
                    "checkpoint_roundtrip_logits_bitwise_equal": True, "synthetic_cuda_resume_bitwise_equal": True}
            baseline.atomic_json(path, good)
            exp.verify_gpu_report(path, plan)
            for bad in ({**good, "status": "cpu_preparation_passed_gpu_preflight_pending"},
                        {**good, "schema": baseline.SCHEMA}, {**good, "plan_sha256": "stale"},
                        {**good, "checkpoint_may_initialize_formal_training": True}):
                baseline.atomic_json(path, bad)
                with self.assertRaises(ValueError):
                    exp.verify_gpu_report(path, plan)

    def test_engine_rejects_baseline_engineering_and_synthetic_checkpoints(self):
        config = exp.CandidateConfig(workers=0)
        plan = {"config": vars(config), "train_count": 19, "validation_count": 5}
        model = nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            for schema, engineering, kind in ((baseline.SCHEMA, False, "formal_aca_dw3d_candidate"),
                                             ("aca_dense_candidate_resume_v1", False, "formal_aca_candidate"),
                                             ("aca_dw3d_candidate_engineering_v1", True, "engineering"),
                                             (exp.SCHEMA, True, "synthetic_toy")):
                torch.save({"schema": schema, "engineering_only": engineering, "kind": kind}, path)
                with self.assertRaises(ValueError):
                    engine.load_checkpoint(path, model, optimizer, plan)
            engine.save_checkpoint(path, model, optimizer, baseline.initial_progress(), plan)
            engine.load_checkpoint(path, model, optimizer, plan)
            with self.assertRaises(ValueError):
                engine.load_checkpoint(path, model, optimizer, {**plan, "train_count": 20})

    def test_corrupt_optimizer_cursor_rejected(self):
        config = exp.CandidateConfig(workers=0)
        plan = {"config": vars(config), "train_count": 19, "validation_count": 5}
        model = nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters())
        progress = baseline.initial_progress()
        progress["cursor"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            engine.save_checkpoint(path, model, optimizer, progress, plan)
            with self.assertRaises(ValueError):
                engine.load_checkpoint(path, model, optimizer, plan)

    def test_candidate_loop_toy_resume_is_bitwise_equal(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = preflight.synthetic_resume_probe(Path(directory))
        self.assertTrue(result["bitwise_equal"])

    def test_small_aca_model_resume_with_dropout_and_frozen_BN(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = preflight.synthetic_resume_probe(Path(directory),
                model_factory=lambda device: TinyVideoModel().to(device), dataset_factory=TinyVideoData)
        self.assertTrue(result["bitwise_equal"])

    def test_tail_batch_gradient_matches_full_mean_reference(self):
        config = exp.CandidateConfig(epochs=1, patience=1, workers=0)
        plan = {"config": vars(config), "train_count": 19, "validation_count": 5,
                "engineering_only_synthetic_toy": True}

        class DoubleToy(ToyDataset):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                item["tensor"] = item["tensor"].double()
                return item

        model = nn.Linear(3, 2).double()
        reference = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        ref_optimizer = torch.optim.Adam(reference.parameters(), lr=config.lr)
        data = DoubleToy(19)
        order = baseline.epoch_order(19, config.seed, 0)
        for start in (0, 16):
            items = [data[i] for i in order[start:start+16]]
            x = torch.stack([item["tensor"] for item in items])
            y = torch.tensor([item["binary_label"] for item in items])
            ref_optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(reference(x), y).backward()
            nn.utils.clip_grad_norm_(reference.parameters(), config.clip_norm)
            ref_optimizer.step()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            engine.fit(model, optimizer, data, DoubleToy(5), config, plan, Path(directory),
                       baseline.initial_progress(), device="cpu")
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_early_stop_loss_tie_keeps_first_checkpoint(self):
        config = exp.CandidateConfig(epochs=3, patience=1, workers=0)
        plan = {"config": vars(config), "train_count": 19, "validation_count": 5,
                "engineering_only_synthetic_toy": True}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory)
            model = nn.Linear(3, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            with patch.object(engine, "validation_pass", return_value={"count": 5, "loss": .7}):
                result = engine.fit(model, optimizer, ToyDataset(19), ToyDataset(5), config, plan,
                                    path, baseline.initial_progress(), device="cpu")
            self.assertEqual(result["epochs"], 2)
            self.assertEqual(result["stop_reason"], "early_stop")
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(torch.load(path / "best.pt", weights_only=True)["progress"]["epoch"], 1)

    def test_validation_interruption_never_repeats_train_updates(self):
        config = exp.CandidateConfig(epochs=1, patience=1, workers=0)
        plan = {"config": vars(config), "train_count": 19, "validation_count": 5,
                "engineering_only_synthetic_toy": True}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory)
            model = nn.Linear(3, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            with patch.object(engine, "validation_pass", return_value=None):
                engine.fit(model, optimizer, ToyDataset(19), ToyDataset(5), config, plan,
                           path, baseline.initial_progress(), device="cpu")
            progress = engine.load_checkpoint(path / "last.pt", model, optimizer, plan)
            self.assertEqual(progress["cursor"], 19)
            with patch.object(optimizer, "step", side_effect=AssertionError("Must not repeat train")):
                result = engine.fit(model, optimizer, ToyDataset(19), ToyDataset(5), config, plan, path, progress, device="cpu")
            self.assertEqual(result["optimizer_steps"], 2)
            (path / "summary.json").unlink()  # Synthetic fixture only; exercise final-mirror recovery.
            engine.fit(model, optimizer, ToyDataset(19), ToyDataset(5), config, plan, path, progress, device="cpu")
            self.assertTrue((path / "summary.json").exists())

    def test_no_CUDA_fallback_or_initialization_when_unavailable(self):
        with patch.object(train, "assert_launch_prerequisites"), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(train, "initialized_candidate", side_effect=AssertionError("Must not create model")):
            with self.assertRaises(ValueError):
                train.run_training(Path("/not-used"), exp.CandidateConfig(), Path("/not-used"), accepted=True)


if __name__ == "__main__":
    unittest.main()
