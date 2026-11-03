import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from .analyze import (align, compare, match_metrics, summarize, tensor_digest,
                      validate_payload, write_new)


def rows(predictions, probabilities, labels=(0, 1, 0, 1)):
    return [{"video_id": str(i), "split": "validation", "label": y,
             "original_class": f"ft{y}", "shot_type": "ft", "source_group": str(i),
             "prediction": pred, "p_make": p, "ce_loss": 0.5}
            for i, (y, pred, p) in enumerate(zip(labels, predictions, probabilities))]


def expected(data):
    return [{"video_id": r["video_id"], "binary_label": str(r["label"]),
             "original_class": r["original_class"], "source_group": r["source_group"]} for r in data]


class AnalysisTests(unittest.TestCase):
    def test_alignment_by_id_not_position(self):
        a = rows([0, 1, 0, 1], [.1, .9, .2, .8])
        pairs = align(a, list(reversed(a)), expected(a))
        self.assertTrue(all(x["video_id"] == y["video_id"] for x, y in pairs))

    def test_missing_duplicate_and_foreign_ids_refused(self):
        a = rows([0, 1, 0, 1], [.1, .9, .2, .8])
        for b in (a[:-1], a[:3]+a[:1], [{**r, "video_id": "test"} for r in a]):
            with self.assertRaises(ValueError):
                align(a, b, expected(a))

    def test_test_split_and_wrong_labels_refused(self):
        a = rows([0, 1, 0, 1], [.1, .9, .2, .8])
        for b in ([{**r, "split": "test"} for r in a], [{**r, "label": 1-r["label"]} for r in a]):
            with self.assertRaises(ValueError):
                align(a, b, expected(a))

    def test_complementarity_and_fixed_fusion(self):
        a = rows([0, 1, 1, 0], [.1, .9, .6, .2])
        b = rows([0, 0, 0, 0], [.2, .2, .1, .1])
        report, fused = compare(a, b, expected(a))
        self.assertEqual(report["complementarity"], {"both_correct": 1, "dense_only_correct": 1,
                                                   "dw3d_only_correct": 1, "both_wrong": 1})
        self.assertEqual([r["prediction"] for r in fused], [0, 1, 0, 0])
        self.assertEqual(report["equal_weight_fusion"]["accuracy"], .75)
        self.assertEqual(report["changes_vs_single"]["dense"], {"rescued": 1, "harmed": 0})
        self.assertFalse(report["label_oracle_ceiling"]["deployable"])

    def test_probability_tie_keeps_class_zero_like_argmax(self):
        a = rows([0, 1, 0, 1], [.2, .2, .2, .2])
        b = rows([1, 1, 1, 1], [.8, .8, .8, .8])
        _, fused = compare(a, b, expected(a))
        self.assertTrue(all(r["prediction"] == 0 for r in fused))

    def test_nonfinite_metrics_refused(self):
        a = rows([0, 1, 0, 1], [.1, .9, .2, float("nan")])
        with self.assertRaises(ValueError):
            summarize(a)

    def test_stable_fusion_ce_for_saturated_probabilities(self):
        a = rows([1, 1, 1, 1], [1., 1., 1., 1.])
        for row in a:
            row["log_probabilities"] = [-1000., 0.]
        _, fused = compare(a, a, expected(a))
        self.assertAlmostEqual(fused[0]["ce_loss"], 1000.)

    def test_inference_path_with_cpu_toy_no_optimizer_or_video(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from . import analyze
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)).eval().requires_grad_(False)
        sources = expected(rows([0, 1, 0, 1], [.1, .9, .2, .8]))
        batches, reference = [], []
        for row in sources:
            x, y = torch.ones(1, 2), torch.tensor([int(row["binary_label"])])
            with torch.inference_mode():
                logits = model(x)
            reference.append({"label": y.item(), "prediction": logits.argmax(-1).item(),
                              "shot_type": "ft", "p_make": logits.double().softmax(-1)[0,1].item(),
                              "ce_loss": torch.nn.functional.cross_entropy(logits, y).item()})
            batches.append({"tensor": x, "binary_label": y, "video_id": [row["video_id"]], "split": ["validation"]})
        dataset = MagicMock()
        dataset.__len__.return_value = 4
        dataset.scope.rows = sources
        spec = {"plan": {"config": {}}, "summary": {"best_epoch": 1, "best_validation": summarize(reference)}}
        before = tensor_digest(model.state_dict())
        with tempfile.TemporaryDirectory() as tmp, patch.object(analyze, "load_model", return_value=(model, before)), \
                patch.object(analyze, "make_loader", return_value=batches):
            result = analyze.infer("toy", spec, dataset, Path(tmp), device="cpu")
            self.assertEqual(len(result), 4)
            self.assertTrue((Path(tmp) / "toy_verification.json").is_file())
            self.assertEqual(before, tensor_digest(model.state_dict()))

    def test_metric_reproduction_gate(self):
        a = summarize(rows([0, 1, 0, 1], [.1, .9, .2, .8]))
        match_metrics(a, copy.deepcopy(a))
        with self.assertRaises(ValueError):
            match_metrics({**a, "loss": a["loss"] + .001}, a)

    def test_digest_catches_bn_buffer_or_parameter_change(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)).eval()
        before = tensor_digest(model.state_dict())
        with torch.inference_mode():
            model(torch.ones(1, 2))
        self.assertEqual(before, tensor_digest(model.state_dict()))
        model[1].running_mean.add_(1)
        self.assertNotEqual(before, tensor_digest(model.state_dict()))

    def test_engineering_and_incorrect_plan_checkpoint_refused(self):
        spec = {"plan": {}, "schema": "s", "kind": "formal", "summary": {}}
        for payload in ({"schema": "s", "kind": "formal", "engineering_only": True},
                        {"schema": "s", "kind": "formal", "engineering_only": False, "plan_sha256": "bad"}):
            with self.assertRaises(ValueError):
                validate_payload(payload, spec)

    def test_existing_output_not_overwritten(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            write_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                write_new(path, {})

    def test_default_mode_no_inference(self):
        from . import analyze
        with patch.object(analyze, "VideoScope") as scope, patch.object(analyze, "metadata") as meta, \
                patch.object(analyze, "configure_runtime") as gpu:
            scope.return_value.rows = []
            scope.return_value.hashes = {}
            meta.return_value = {"checkpoint": "best.pt", "plan": {
                "input_hashes": {}, "validation_ordered_ids_sha256": analyze.canonical_hash([]),
                "source_root": "/tmp"}}
            analyze.run(Path("/tmp"), False)
            gpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
