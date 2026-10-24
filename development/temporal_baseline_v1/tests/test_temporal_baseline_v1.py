import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from development.temporal_baseline_v1 import cached_features
from development.temporal_baseline_v1.cached_features import load_sequences
from development.temporal_baseline_v1.model import WindowTemporalTransformer


class CachedSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="temporal_cache_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.manifest = self.root / "data/processed/dataset_manifest.csv"
        self.manifest.parent.mkdir(parents=True)
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("video_id", "split", "excluded", "binary_label")
            )
            writer.writeheader()
            writer.writerow({
                "video_id": "2p0/train.mp4", "split": "train",
                "excluded": "0", "binary_label": "0",
            })
            writer.writerow({
                "video_id": "2p1/validation.mp4", "split": "validation",
                "excluded": "0", "binary_label": "1",
            })
        self.protocol = {
            "schema_version": "three_window_feature_cache_v1",
            "quality_weighting_used": False,
            "quality_aware_selection_used": False,
            "input_identities": {
                "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest()
            },
        }
        self.run_id = hashlib.sha256(
            json.dumps(self.protocol, sort_keys=True).encode()
        ).hexdigest()
        self.cache.mkdir()
        torch.save(self.protocol, self.cache / "run.pt")
        self.features = torch.arange(3 * 8 * 1536, dtype=torch.float32).reshape(3, 8, 1536)
        self.paths = {}
        for split, video_id, label in (
            ("train", "2p0/train.mp4", 0),
            ("validation", "2p1/validation.mp4", 1),
        ):
            folder = self.cache / split
            folder.mkdir()
            torch.save({
                "run_id": self.run_id, "split": split,
                "video_ids": [video_id], "count": 1,
            }, folder / "index.pt")
            path = folder / (hashlib.sha256(video_id.encode()).hexdigest() + ".pt")
            self.paths[split] = path
            torch.save({
                "run_id": self.run_id, "video_id": video_id, "split": split,
                "binary_label": label, "quality_aware_selection_used": False,
                "features": self.features,
                "features_sha256": hashlib.sha256(self.features.numpy().tobytes()).hexdigest(),
            }, path)
        self.root_patch = patch.object(cached_features, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def change_record(self, **changes):
        path = self.paths["train"]
        record = torch.load(path, map_location="cpu", weights_only=True)
        record.update(changes)
        torch.save(record, path)

    def test_preserves_windows_time_and_labels(self):
        for split, label in (("train", 0), ("validation", 1)):
            with self.subTest(split=split):
                x, y, ids, run_id = load_sequences(self.cache, split=split, limit=1)
                self.assertEqual(tuple(x.shape), (1, 3, 8, 1536))
                torch.testing.assert_close(x[0], self.features, rtol=0, atol=0)
                self.assertEqual(y.dtype, torch.float32)
                self.assertEqual(y.tolist(), [float(label)])
                self.assertEqual(len(ids), 1)
                self.assertEqual(run_id, self.run_id)

    def test_test_split_rejected_before_file_access(self):
        with patch.object(torch, "load", side_effect=AssertionError("File access")):
            with self.assertRaises(ValueError):
                load_sequences(self.root / "missing", split="test")

    def test_invalid_limits_rejected_before_file_access(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit):
                with patch.object(torch, "load", side_effect=AssertionError("File access")):
                    with self.assertRaises(ValueError):
                        load_sequences(self.cache, split="train", limit=limit)

    def test_record_identity_and_label_mismatches_rejected(self):
        for changes in (
            {"run_id": "wrong"}, {"video_id": "wrong.mp4"},
            {"split": "validation"}, {"binary_label": 1},
            {"quality_aware_selection_used": True},
        ):
            with self.subTest(changes=changes):
                path = self.paths["train"]
                original = torch.load(path, map_location="cpu", weights_only=True)
                try:
                    self.change_record(**changes)
                    with self.assertRaises(RuntimeError):
                        load_sequences(self.cache, split="train")
                finally:
                    torch.save(original, path)

    def test_bad_shape_dtype_and_nonfinite_features_rejected(self):
        for features in (
            torch.zeros(24, 1536), self.features.double(),
            torch.full_like(self.features, float("nan")),
        ):
            with self.subTest(shape=features.shape, dtype=features.dtype):
                self.change_record(features=features)
                with self.assertRaises(RuntimeError):
                    load_sequences(self.cache, split="train")

    def test_corrupt_feature_hash_rejected(self):
        changed = self.features.clone()
        changed[0, 0, 0] += 1
        self.change_record(features=changed)
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            load_sequences(self.cache, split="train")

    def test_manifest_change_rejected(self):
        with self.manifest.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(RuntimeError, "Manifest identity"):
            load_sequences(self.cache, split="train")

    def test_duplicate_index_ids_rejected(self):
        path = self.cache / "train/index.pt"
        index = torch.load(path, map_location="cpu", weights_only=True)
        index["video_ids"] *= 2
        index["count"] = 2
        torch.save(index, path)
        with self.assertRaisesRegex(RuntimeError, "Duplicate index"):
            load_sequences(self.cache, split="train")


class TemporalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(20261024)
        self.model = WindowTemporalTransformer()
        self.features = torch.randn(2, 3, 8, 1536)

    def test_forward_and_all_parameter_gradients(self):
        logits = self.model(self.features)
        self.assertEqual(tuple(logits.shape), (2,))
        loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0]))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_invalid_shapes_rejected(self):
        for shape in ((2, 24, 1536), (2, 3, 7, 1536), (2, 2, 8, 1536)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    self.model(torch.zeros(shape))

    def test_windows_are_independent_and_equally_pooled(self):
        self.model.eval()
        with torch.no_grad():
            original = self.model(self.features)
            reordered = self.model(self.features[:, [2, 0, 1]])
        torch.testing.assert_close(original, reordered, rtol=1e-5, atol=1e-6)

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 1450497)


if __name__ == "__main__":
    unittest.main()
