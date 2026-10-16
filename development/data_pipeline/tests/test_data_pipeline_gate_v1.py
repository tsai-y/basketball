from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from development.data_pipeline.pipeline import (
    AccessLedger,
    BasketballClipDataset,
    DEFAULT_CONFIG,
    GateError,
    TrainQualityIndex,
    load_gate_config,
    select_validation_clip,
    validate_decoded_clip,
)
from development.data_quality.samplers.quality_aware_clip_sampler import select_clips
from development.data_quality.transforms.video_clip_transforms import (
    ClipTransform,
    derived_seed,
    normalize_tensor,
)


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
RUNNER = ROOT / "development/data_pipeline/scripts/run_data_pipeline_gate_v1.py"


class FakeDecoder:
    def __init__(self, *, frame_count=101, dtype=np.uint8, returned_frames=64):
        self.frame_count = frame_count
        self.dtype = dtype
        self.returned_frames = returned_frames
        self.calls = 0

    def __call__(self, row, guard, *, selection, expected_frame_count, expected_fps_rational):
        self.calls += 1
        frame_count = expected_frame_count or self.frame_count
        if selection is None:
            selection = select_validation_clip(frame_count)
        start, end = selection["start_frame"], selection["end_frame_exclusive"]
        base = np.arange(self.returned_frames * 8 * 8 * 3, dtype=np.uint32)
        clip = (base % 256).astype(self.dtype).reshape(self.returned_frames, 8, 8, 3)
        indices = list(range(start, start + self.returned_frames))
        metadata = {
            "decoded_frame_indices": indices,
            "retained_frames": self.returned_frames,
            "decoder_read_frames": end,
            "prefix_frames": start,
            "timestamps_match_indices": True,
            "source_unchanged": True,
        }
        fps = float(expected_fps_rational.split("/")[0]) / float(expected_fps_rational.split("/")[1]) \
            if expected_fps_rational else 29.97
        return clip, metadata, selection, frame_count, fps


def runs_snapshot():
    return {str(path.relative_to(RUNS)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in RUNS.rglob("*") if path.is_file()}


class DataPipelineGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_gate_config(DEFAULT_CONFIG)
        cls.quality = TrainQualityIndex(cls.config)
        cls.fallback_ids = sorted(video_id for video_id, row in cls.quality.summaries.items()
                                  if row["has_clean_64"] == "false")
        with cls.config.paths["manifest"].open(newline="", encoding="utf-8") as stream:
            cls.manifest = list(csv.DictReader(stream))
        cls.validation_id = next(row["video_id"] for row in cls.manifest if row["split"] == "validation")
        cls.test_id = next(row["video_id"] for row in cls.manifest if row["split"] == "test")

    def test_train_sampler_is_reproducible_for_same_inputs(self):
        video_id = self.fallback_ids[0]
        row = self.quality.summaries[video_id]
        kwargs = dict(video_id=video_id, frame_count=int(row["frame_count"]),
                      candidate_segments=self.quality.segments.get(video_id, []), clip_length=64,
                      seed=20261009, epoch=3, num_clips=1)
        first = select_clips(**kwargs)[0]
        second = select_clips(**kwargs)[0]
        self.assertEqual(first, second)
        self.assertEqual(list(range(first.start_frame, first.end_frame_exclusive)),
                         list(range(second.start_frame, second.end_frame_exclusive)))

    def test_reference_tensor_is_bitwise_identical_and_identity_is_explicit(self):
        transform_cfg = {key: value for key, value in self.config.raw["transform"].items()
                         if key != "policy"}
        clip = np.arange(64 * 12 * 16 * 3, dtype=np.uint32)
        clip = (clip % 256).astype(np.uint8).reshape(64, 12, 16, 3)
        seed = derived_seed(self.config.raw["seed"], "synthetic", 0, 0, "reference")
        transform = ClipTransform(transform_cfg, "reference")
        first, _ = transform(clip, seed=seed, split="train")
        second, _ = transform(clip.copy(), seed=seed, split="train")
        self.assertTrue(np.array_equal(first, second))
        normalized, metadata = normalize_tensor(first, mode="identity")
        self.assertTrue(np.array_equal(first, normalized))
        self.assertEqual(metadata["mode"], "identity")
        self.assertEqual(normalized.shape, (3, 64, 192, 256))
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(normalized.flags.c_contiguous)
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)

    def test_all_five_fallback_videos_return_samples_and_metadata(self):
        self.assertEqual(len(self.fallback_ids), 5)
        dataset = BasketballClipDataset(
            self.config, split="train", video_ids=self.fallback_ids, decoder=FakeDecoder()
        )
        for sample in dataset:
            self.assertTrue(sample["fallback"])
            self.assertEqual(sample["fallback_reason"], "no_zero_candidate_overlap_window")
            self.assertGreater(sample["candidate_overlap"], 0)
            self.assertEqual(sample["sampler_version"], "quality_aware_sampler_v1")
            self.assertTrue(sample["sidecar_identity"]["used"])

    def test_validation_center_rule_and_no_train_sidecar(self):
        self.assertEqual(select_validation_clip(65)["start_frame"], 0)
        self.assertEqual(select_validation_clip(66)["start_frame"], 1)
        dataset = BasketballClipDataset(
            self.config, split="validation", video_ids=[self.validation_id], decoder=FakeDecoder(frame_count=101)
        )
        self.assertIsNone(dataset.quality_index)
        sample = dataset[0]
        self.assertEqual((sample["start_frame"], sample["end_frame_exclusive"]), (18, 82))
        self.assertFalse(sample["sidecar_identity"]["used"])
        self.assertIsNone(sample["sidecar_identity"]["sha256"])
        self.assertIsNone(sample["candidate_overlap"])

    def test_test_split_and_test_id_are_rejected_before_open(self):
        ledger = AccessLedger()
        with self.assertRaises(GateError):
            BasketballClipDataset(self.config, split="test", ledger=ledger)
        self.assertEqual(ledger.events, [])
        decoder = FakeDecoder()
        with self.assertRaises(GateError):
            BasketballClipDataset(self.config, split="validation", video_ids=[self.test_id],
                                  decoder=decoder, ledger=ledger)
        self.assertEqual(decoder.calls, 0)
        self.assertEqual(ledger.events, [])

    def test_bad_decoded_dtype_count_indices_and_short_video_fail_closed(self):
        selection = select_validation_clip(100)
        valid = np.zeros((64, 8, 8, 3), dtype=np.uint8)
        with self.assertRaises(GateError):
            validate_decoded_clip(valid.astype(np.float32), range(18, 82), selection)
        with self.assertRaises(GateError):
            validate_decoded_clip(valid[:63], range(18, 81), selection)
        with self.assertRaises(GateError):
            validate_decoded_clip(valid, range(19, 83), selection)
        with self.assertRaises(GateError):
            select_validation_clip(63)

        class WrongFrameCount(FakeDecoder):
            def __call__(self, *args, **kwargs):
                clip, metadata, selected, frame_count, fps = super().__call__(*args, **kwargs)
                return clip, metadata, selected, frame_count + 1, fps

        dataset = BasketballClipDataset(
            self.config, split="train", video_ids=[self.fallback_ids[0]], decoder=WrongFrameCount()
        )
        with self.assertRaises(GateError):
            dataset[0]

    def test_sidecar_identity_mismatch_fails_closed(self):
        with self.assertRaises(GateError):
            TrainQualityIndex(self.config, expected_run_identity="0" * 64)

    def test_help_and_dry_run_do_not_write_historical_runs(self):
        before = runs_snapshot()
        help_result = subprocess.run(
            [sys.executable, "-B", str(RUNNER), "--help"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        dry_result = subprocess.run(
            [sys.executable, "-B", str(RUNNER), "--dry-run"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(dry_result.returncode, 0, dry_result.stderr)
        payload = json.loads(dry_result.stdout)
        self.assertEqual(payload["test_video_open_count"], 0)
        self.assertEqual(payload["test_video_decode_count"], 0)
        self.assertEqual(before, runs_snapshot())

        forbidden = subprocess.run(
            [sys.executable, "-B", str(RUNNER), "--dry-run", "--output", str(RUNS / "gate.json")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(forbidden.returncode, 2)
        self.assertFalse((RUNS / "gate.json").exists())
        self.assertEqual(before, runs_snapshot())


if __name__ == "__main__":
    unittest.main()
