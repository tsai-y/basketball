import contextlib
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from development.aca_extension_v1 import train as ext
from development.aca_candidate_v1 import engine, experiment as exp
from development.aca_net_v1 import train as base
from development.aca_net_v1.readiness import nested_equal, ToyDataset, toy_model


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_only_budget_changes(self):
        old = asdict(exp.CandidateConfig(epochs=5))
        new = asdict(exp.CandidateConfig(epochs=15))
        ext.validate_extension(old, new)
        for bad in ({**new, "epochs": 50}, {**new, "patience": 10},
                    {**new, "lr": 1e-5}, {**new, "workers": 0}, {**new, "seed": 1}):
            with self.assertRaises(ValueError):
                ext.validate_extension(old, bad)

    def test_reopen_preserves_all_other_state_and_parent_immutable(self):
        old = asdict(exp.CandidateConfig(epochs=5))
        new = asdict(exp.CandidateConfig(epochs=15))
        progress = {"finished": True, "epoch": 5, "cursor": 0, "bad_epochs": 3,
                    "best_loss": .4, "history": [{"epoch": 2}], "optimizer_steps": 2250}
        before = copy.deepcopy(progress)
        actual = ext.reopen_progress(progress, old, new)
        self.assertEqual(progress, before)
        self.assertEqual(actual, {**before, "finished": False})
        actual["history"].append({"epoch": 3})
        self.assertEqual(progress, before)

    def test_early_stopped_partial_or_unfinished_parent_rejected(self):
        old = asdict(exp.CandidateConfig(epochs=5))
        new = asdict(exp.CandidateConfig(epochs=15))
        good = {"finished": True, "epoch": 5, "cursor": 0, "bad_epochs": 0}
        for bad in ({**good, "bad_epochs": 5}, {**good, "cursor": 16},
                    {**good, "finished": False}, {**good, "epoch": 4}):
            with self.assertRaises(ValueError):
                ext.reopen_progress(bad, old, new)

    def test_cpu_completed_extension_matches_continuous_with_dropout(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = ext.synthetic_extension_probe(Path(directory))
        self.assertTrue(result["bitwise_equal"])

    def test_import_preserves_parent_and_best_Adam_RNG_history(self):
        old_cfg = exp.CandidateConfig(epochs=5, workers=0)
        new_cfg = exp.CandidateConfig(epochs=15, workers=0)
        old = {"config": asdict(old_cfg), "train_count": 19, "validation_count": 5,
               "engineering_only_synthetic_toy": True}
        new = {**old, "config": asdict(new_cfg)}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            parent, output = root / "parent", root / "new"
            parent.mkdir()
            output.mkdir()
            base.configure_runtime(old_cfg.seed, cuda=False)
            model = toy_model("cpu")
            optimizer = torch.optim.Adam(model.parameters(), lr=old_cfg.lr)
            # Best is earlier than last; extension must carry that best forward.
            values = iter([.4, .3, .35, .36, .37])
            with patch.object(engine, "validation_pass", side_effect=lambda *a: {"count": 5, "loss": next(values)}):
                engine.fit(model, optimizer, ToyDataset(19), ToyDataset(5), old_cfg, old,
                           parent, base.initial_progress(), device="cpu")
            digests = {p.name: base.file_sha256(p) for p in parent.iterdir() if p.is_file()}
            original = torch.load(parent / "last.pt", weights_only=True)
            model = toy_model("cpu")
            optimizer = torch.optim.Adam(model.parameters(), lr=old_cfg.lr)
            with patch.object(ext, "PARENT", parent):
                progress = ext.import_parent_checkpoints(output, model, optimizer, old, new)
            actual = torch.load(output / "last.pt", weights_only=True)
            inherited_best = torch.load(output / "best.pt", weights_only=True)
            prior_best = torch.load(parent / "best.pt", weights_only=True)
            self.assertTrue(nested_equal(original["model"], actual["model"]))
            self.assertTrue(nested_equal(original["optimizer"], actual["optimizer"]))
            self.assertTrue(nested_equal(original["rng"], actual["rng"]))
            self.assertEqual(progress, {**original["progress"], "finished": False})
            self.assertEqual(progress["bad_epochs"], 3)
            self.assertEqual(inherited_best["progress"]["epoch"], 2)
            self.assertTrue(nested_equal(inherited_best["model"], prior_best["model"]))
            self.assertEqual(actual["plan_sha256"], base.canonical_hash(new))
            self.assertEqual(digests, {p.name: base.file_sha256(p) for p in parent.iterdir() if p.is_file()})

    def test_checkpoint_compatibility_not_relaxed(self):
        model = toy_model("cpu")
        opt = torch.optim.Adam(model.parameters())
        old = {"config": asdict(exp.CandidateConfig(epochs=5)), "train_count": 19, "validation_count": 5}
        new = {**old, "config": asdict(exp.CandidateConfig(epochs=15))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            engine.save_checkpoint(path, model, opt, base.initial_progress(), old)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                engine.load_checkpoint(path, model, opt, new)

    def test_wrong_or_stale_preflight_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            base.atomic_json(path, {"schema": exp.PREFLIGHT_SCHEMA, "status": "passed"})
            with self.assertRaises(ValueError):
                ext.verify_report(path, {})

    def test_explicit_acceptance_before_cuda_work(self):
        with patch.object(exp, "assert_launch_prerequisites", side_effect=ValueError("accept")), \
             patch.object(base, "configure_runtime", side_effect=AssertionError("must not init")):
            with self.assertRaises(ValueError):
                ext.run(Path("/unused"), Path("/unused"), False)


if __name__ == "__main__":
    unittest.main()
