import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from development.aca_dw3d_v1 import after_baseline as auto, experiment as exp
from development.aca_net_v1.train import atomic_json, canonical_hash


class AutomaticLaunchTests(unittest.TestCase):
    def test_default_status_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "missing"
            self.assertEqual(auto.status(control), {"status": "not_launched"})
            self.assertFalse(control.exists())

    def test_unready_baseline_never_claims_or_spawns(self):
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "control"
            with patch.object(exp, "assert_launch_prerequisites", side_effect=ValueError("not complete")), \
                 patch.object(auto.subprocess, "Popen", side_effect=AssertionError("Must not spawn")):
                with self.assertRaises(ValueError):
                    auto.launch(Path(directory), Path(directory), accepted=True, control=control)
            self.assertFalse(control.exists())

    def test_explicit_acceptance_is_forwarded(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(exp, "assert_launch_prerequisites", side_effect=ValueError("accept required")) as gate:
            with self.assertRaises(ValueError):
                auto.launch(Path(directory), Path(directory), accepted=False, control=Path(directory) / "new")
            gate.assert_called_once_with(False)

    def test_only_one_detached_process_and_persistent_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            with patch.object(exp, "assert_launch_prerequisites"), patch.object(auto, "verify_preparation"), \
                 patch.object(auto.subprocess, "Popen", return_value=SimpleNamespace(pid=123)) as spawn:
                first = auto.launch(root, root, accepted=True, control=control)
                second = auto.launch(root, root, accepted=True, control=control)
                self.assertEqual(first["status"], "dispatched")
                self.assertEqual(second["status"], "claimed")
                spawn.assert_called_once()
                self.assertTrue(spawn.call_args.kwargs["start_new_session"])
                self.assertIn("development.aca_dw3d_v1.after_baseline", spawn.call_args.args[0])
                self.assertIn("--worker", spawn.call_args.args[0])
                self.assertFalse(json.loads((control / "claim.json").read_text())["automatic_retries"])

    def test_failed_spawn_does_not_enable_automatic_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / "control"
            with patch.object(exp, "assert_launch_prerequisites"), patch.object(auto, "verify_preparation"), \
                 patch.object(auto.subprocess, "Popen", side_effect=OSError("failure")) as spawn:
                with self.assertRaises(OSError):
                    auto.launch(root, root, accepted=True, control=control)
                self.assertEqual(auto.launch(root, root, accepted=True, control=control)["status"], "failed")
                spawn.assert_called_once()

    def test_stale_cpu_preparation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"source": "old"}
            atomic_json(root / "plan.json", plan)
            atomic_json(root / "report.json", {
                "status": "cpu_preparation_passed_gpu_preflight_pending", "parameters": exp.PARAMETERS,
                "cuda_context_initialized": False, "test_videos_opened": 0, "plan_sha256": canonical_hash(plan)})
            with patch.object(exp, "build_plan", return_value={"source": "new"}):
                with self.assertRaisesRegex(ValueError, "changed"):
                    auto.verify_preparation(root, root, exp.CandidateConfig())

    def test_dead_or_recycled_pid_is_not_reported_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "state.json", {"status": "training", "pid": 123, "process_start": "old"})
            with patch.object(auto, "process_start", return_value="different"):
                self.assertFalse(auto.status(root)["process_alive"])

    def test_worker_preflight_failure_never_trains(self):
        from development.aca_dw3d_v1 import preflight, train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "claim.json", {"source_root": str(root), "preparation": str(root)})
            with patch.object(exp, "assert_launch_prerequisites"), patch.object(auto, "verify_preparation"), \
                 patch.object(preflight, "run", side_effect=ValueError("bad GPU preflight")), \
                 patch.object(train, "run_training", side_effect=AssertionError("Must not train")):
                with self.assertRaisesRegex(ValueError, "bad GPU"):
                    auto.worker(root, root, root)
            self.assertEqual(auto.status(root)["status"], "failed")

    def test_worker_success_and_duplicate_worker_refused(self):
        from development.aca_dw3d_v1 import preflight, train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "claim.json", {"source_root": str(root), "preparation": str(root)})

            def fake_training(*args, **kwargs):
                kwargs["on_created"](root / "formal_run")
                self.assertEqual(auto.status(root)["status"], "training")
                return {"status": "completed", "output": str(root / "formal_run")}

            with patch.object(exp, "assert_launch_prerequisites"), patch.object(auto, "verify_preparation"), \
                 patch.object(preflight, "run", return_value={"output": str(root / "preflight")}) as check, \
                 patch.object(train, "run_training", side_effect=fake_training) as training:
                auto.worker(root, root, root)
                self.assertEqual(auto.status(root)["status"], "completed")
                with self.assertRaises(FileExistsError):
                    auto.worker(root, root, root)
                check.assert_called_once()
                training.assert_called_once()


if __name__ == "__main__":
    unittest.main()
