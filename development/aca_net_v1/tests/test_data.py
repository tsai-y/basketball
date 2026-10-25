from pathlib import Path
import csv
import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from development.aca_net_v1 import data
from development.aca_net_v1.audit_architecture import run as audit


class FakeCapture:
    def __init__(self, *args):
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return 73 if prop == cv2.CAP_PROP_FRAME_COUNT else 25

    def read(self):
        if self.index == 73:
            return False, None
        x = np.empty((240, 320, 3), np.uint8)
        x[:] = (self.index, 10, 200)
        self.index += 1
        return True, x

    def release(self):
        self.released = True


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scope = data.VideoScope(source_root=self.root, split="train")
        self.video_id = self.scope.rows[0]["video_id"]

    def create_fake_source(self):
        row, path = self.scope.authorize(self.video_id)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"unit-test-only placeholder, VideoCapture is mocked")
        return path

    def test_indices_cover_endpoints_without_duplicates(self):
        for n in (50, 51, 73, 146, 180, 1000):
            values = data.uniform_indices(n)
            self.assertEqual((len(values), len(set(values)), values[0], values[-1]), (50, 50, 0, n - 1))
            self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
        for bad in (0, 49, 50.0, True):
            with self.assertRaises(ValueError):
                data.uniform_indices(bad)

    def test_normalization_rgb_channel_order_and_roundtrip(self):
        rgb = np.empty((50, 120, 160, 3), np.uint8)
        rgb[:] = (255, 128, 0)
        x = data.normalize_rgb(rgb)
        expected = (torch.tensor([1., 128 / 255, 0.]) - torch.tensor(data.MEAN)) / torch.tensor(data.STD)
        torch.testing.assert_close(x[:, 0, 0, 0], expected)
        self.assertEqual(tuple(x.shape), (3, 50, 120, 160))
        self.assertEqual(x.dtype, torch.float32)
        np.testing.assert_array_equal(data.preview_rgb(x), rgb)

    def test_forbidden_split_rejected_before_any_metadata_read(self):
        with patch.object(data, "file_sha256", side_effect=AssertionError("Must not read")):
            with self.assertRaises(ValueError):
                data.VideoScope(source_root=self.root, split="test")

    def test_test_excluded_and_cross_split_ids_rejected_before_open(self):
        validation = data.VideoScope(source_root=self.root, split="validation")
        with (data.ROOT / "data/processed/dataset_manifest.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        denied = [next(r["video_id"] for r in rows if r["split"] == s)
                  for s in ("test", "excluded")]
        with patch.object(data.gate1a, "open_guarded", side_effect=AssertionError("Must not open")):
            for video_id in [*denied, validation.rows[0]["video_id"], "2p0/unknown.mp4", "../outside.mp4"]:
                with self.assertRaises(ValueError):
                    data.decode_uniform50(self.scope, video_id)

    def test_frozen_identity_mismatch_rejected(self):
        with patch.object(data, "file_sha256", return_value="changed"):
            with self.assertRaises(ValueError):
                data.VideoScope(source_root=self.root, split="train")

    def test_one_capture_full_pass_correct_indices_and_bgr_conversion(self):
        self.create_fake_source()
        cap = FakeCapture()
        with patch.object(data.cv2, "VideoCapture", return_value=cap) as factory:
            rgb, meta, _ = data.decode_uniform50(self.scope, self.video_id)
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(cap.released)
        self.assertEqual(meta["decoded_frames"], 73)
        self.assertEqual(meta["video_open_count"], 1)
        np.testing.assert_array_equal(rgb[:, 0, 0, 2], data.uniform_indices(73))
        self.assertTrue(np.all(rgb[..., 0] == 200))
        self.assertTrue(np.all(rgb[..., 1] == 10))

    def test_truncation_rejected_and_capture_released(self):
        self.create_fake_source()
        cap = FakeCapture()
        read = cap.read
        cap.read = lambda: (False, None) if cap.index == 20 else read()
        with patch.object(data.cv2, "VideoCapture", return_value=cap):
            with self.assertRaises(ValueError):
                data.decode_uniform50(self.scope, self.video_id)
        self.assertTrue(cap.released)

    def test_symlink_alias_rejected_before_decoder(self):
        row, path = self.scope.authorize(self.video_id)
        path.parent.mkdir(parents=True)
        target = self.root / "foreign.bin"
        target.write_bytes(b"foreign")
        path.symlink_to(target)
        with patch.object(data.cv2, "VideoCapture", side_effect=AssertionError("Must not decode")):
            with self.assertRaises(ValueError):
                data.decode_uniform50(self.scope, self.video_id)

    def test_sampling_is_deterministic_and_source_groups_distinct(self):
        for split in ("train", "validation"):
            scope = data.VideoScope(source_root=self.root, split=split)
            ids = data.select_smoke_ids(scope)
            self.assertEqual(ids, data.select_smoke_ids(scope))
            rows = [scope.authorize(i)[0] for i in ids]
            self.assertEqual(len({r["source_group"] for r in rows}), 4)
            self.assertEqual([r["binary_label"] for r in rows], ["0", "1", "0", "1"])

    def test_guarded_hash_rejects_symlink_and_hardlink(self):
        path = self.create_fake_source()
        os.link(path, self.root / "alias.bin")
        with self.assertRaises(ValueError):
            data.guarded_source_sha256(self.scope, self.video_id)

    def test_extra_frames_rejected_and_capture_released(self):
        self.create_fake_source()
        cap = FakeCapture()
        cap.get = lambda prop: 72 if prop == cv2.CAP_PROP_FRAME_COUNT else 25
        with patch.object(data.cv2, "VideoCapture", return_value=cap):
            with self.assertRaises(ValueError):
                data.decode_uniform50(self.scope, self.video_id)
        self.assertTrue(cap.released)

    def test_dataset_tensor_and_label_match_manifest(self):
        self.create_fake_source()
        dataset = data.Basketball50Dataset(source_root=self.root, split="train", video_ids=[self.video_id])
        with patch.object(data.cv2, "VideoCapture", return_value=FakeCapture()):
            sample = dataset[0]
        self.assertEqual(sample["video_id"], self.video_id)
        self.assertEqual(sample["binary_label"], int(self.scope.rows[0]["binary_label"]))
        self.assertEqual(tuple(sample["tensor"].shape), (3, 50, 120, 160))
        self.assertTrue(torch.isfinite(sample["tensor"]).all())

    def test_parameter_formula_and_category_totals(self):
        report = audit()
        dense = report["variants_8_classes"]["aca"]
        self.assertEqual(dense["total"], 93_176_164)
        self.assertEqual(sum(dense["categories"].values()), dense["total"])
        self.assertEqual(report["gated_conv_formula_total"], 69_160_960)
        self.assertEqual(report["variants_8_classes"]["lsta"]["categories"]["lsta_gated_3d_convolutions"],
                         69_160_960)
        self.assertEqual(sum(dense["gated_conv_by_stage"].values()), 69_160_960)
        self.assertEqual(report["aca_2_class_parameters"], 93_176_164 - 6 * 2049)
        self.assertEqual(report["binary_logits_shape"], [1, 2])
        self.assertEqual(report["videos_opened"], 0)


if __name__ == "__main__":
    unittest.main()
