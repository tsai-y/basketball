"""Metadata-only tests for quality-aware continuous clip sampling v1."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


DQ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DQ))
sys.path.insert(0, str(DQ / "scripts"))
from samplers.quality_aware_clip_sampler import (  # noqa: E402
    CandidateSegment, SamplerError, candidate_overlap, select_clips, window_costs,
)
import audit_quality_aware_sampler_v1 as audit  # noqa: E402


class QualityAwareSamplerV1(unittest.TestCase):
    def select(self, frame_count=20, segments=(), clip_length=5, **kwargs):
        return select_clips(video_id="train/video.mp4", frame_count=frame_count,
                            candidate_segments=segments, clip_length=clip_length,
                            seed=20261009, epoch=0, **kwargs)

    def test_no_candidate(self):
        result = self.select()[0]
        self.assertEqual(result.candidate_overlap, 0)
        self.assertFalse(result.fallback)

    def test_candidate_at_beginning(self):
        result = self.select(segments=[(0, 4)])[0]
        self.assertGreaterEqual(result.start_frame, 4)
        self.assertEqual(result.candidate_overlap, 0)

    def test_candidate_in_middle(self):
        result = self.select(frame_count=30, segments=[(12, 18)])[0]
        self.assertEqual(candidate_overlap(result.start_frame, result.end_frame_exclusive,
                                           [CandidateSegment(12, 18)]), 0)

    def test_candidate_at_end(self):
        result = self.select(segments=[(16, 20)])[0]
        self.assertLessEqual(result.end_frame_exclusive, 16)

    def test_exactly_one_clean_window(self):
        result = self.select(frame_count=10, segments=[(0, 3), (7, 10)], clip_length=4)[0]
        self.assertEqual((result.start_frame, result.end_frame_exclusive), (3, 7))

    def test_no_clean_window_is_explicit_fallback(self):
        result = self.select(frame_count=8, segments=[(0, 8)], clip_length=4)[0]
        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, "no_zero_candidate_overlap_window")
        self.assertEqual(result.candidate_overlap, 4)

    def test_minimum_overlap_fallback(self):
        segments = [(0, 2), (4, 8)]
        result = self.select(frame_count=8, segments=segments, clip_length=5)[0]
        costs = window_costs(8, 5, [CandidateSegment(*value) for value in segments])
        self.assertEqual(result.candidate_overlap, min(costs))

    def test_multiple_equal_cost_windows(self):
        first = self.select(frame_count=20, segments=[], clip_length=5)[0]
        second = self.select(frame_count=20, segments=[], clip_length=5)[0]
        self.assertEqual(first, second)

    def test_same_seed_epoch_reproducible(self):
        a = self.select(frame_count=100, clip_length=16, num_clips=5)
        b = self.select(frame_count=100, clip_length=16, num_clips=5)
        self.assertEqual(a, b)

    def test_different_epoch_can_change(self):
        a = select_clips(video_id="v", frame_count=100, candidate_segments=[], clip_length=16,
                         seed=7, epoch=0)[0]
        b = select_clips(video_id="v", frame_count=100, candidate_segments=[], clip_length=16,
                         seed=7, epoch=1)[0]
        self.assertNotEqual(a.start_frame, b.start_frame)

    def test_multiple_clips_prefer_unique_starts(self):
        values = self.select(frame_count=20, clip_length=5, num_clips=16)
        self.assertEqual(len({value.start_frame for value in values}), 16)
        repeated = self.select(frame_count=20, clip_length=5, num_clips=17)
        self.assertEqual(len({value.start_frame for value in repeated}), 16)

    def test_clip_length_equals_frame_count(self):
        result = self.select(frame_count=10, clip_length=10)[0]
        self.assertEqual((result.start_frame, result.end_frame_exclusive), (0, 10))

    def test_clip_length_greater_than_frame_count_fails(self):
        with self.assertRaisesRegex(SamplerError, "exceeds"):
            self.select(frame_count=10, clip_length=11)

    def test_half_open_off_by_one(self):
        values = [CandidateSegment(5, 6)]
        self.assertEqual(candidate_overlap(0, 5, values), 0)
        self.assertEqual(candidate_overlap(6, 10, values), 0)
        self.assertEqual(candidate_overlap(5, 6, values), 1)

    def test_overlapping_or_unsorted_segments_rejected(self):
        for segments in ([(1, 5), (4, 7)], [(5, 7), (1, 2)]):
            with self.subTest(segments=segments), self.assertRaisesRegex(SamplerError, "sorted and non-overlapping"):
                self.select(segments=segments)

    def _manifest_with(self, row: dict[str, str], directory: Path) -> Path:
        path = directory / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader(); writer.writerow(row)
        return path

    def test_nontrain_id_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            manifest = self._manifest_with({"video_id": "unknown.mp4", "split": "train"}, Path(name))
            with self.assertRaisesRegex(audit.AuditError, "Non-train/unknown"):
                audit.load_sidecars(manifest, audit.DEFAULT_SEGMENTS, audit.DEFAULT_SUMMARY)

    def test_held_out_id_rejected_without_video_access(self):
        held_out = next(row for row in audit.read_csv(audit.DEFAULT_MANIFEST) if row["split"] == "validation")
        with tempfile.TemporaryDirectory() as name:
            manifest = self._manifest_with({"video_id": held_out["video_id"], "split": "validation"}, Path(name))
            with patch("builtins.open", side_effect=AssertionError("No generic/video opener")):
                # pathlib.Path.open remains the only metadata read path; builtins.open is never a video route here.
                with self.assertRaisesRegex(audit.AuditError, "Non-train/unknown"):
                    audit.load_sidecars(manifest, audit.DEFAULT_SEGMENTS, audit.DEFAULT_SUMMARY)

    def test_sidecar_hash_and_run_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for source in (audit.DEFAULT_SEGMENTS, audit.DEFAULT_SUMMARY,
                           audit.DEFAULT_SEGMENTS.parent / audit.METADATA_NAME):
                shutil.copyfile(source, root / source.name)
            summary = root / audit.DEFAULT_SUMMARY.name
            rows = audit.read_csv(summary)
            rows[0]["run_identity"] = "bad"
            summary.write_text(audit.csv_text(rows, list(rows[0])), encoding="utf-8")
            metadata_path = root / audit.METADATA_NAME
            metadata = json.loads(metadata_path.read_text())
            metadata["csv_files"][summary.name]["sha256"] = hashlib.sha256(summary.read_bytes()).hexdigest()
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(audit.AuditError, "Run identity mismatch"):
                audit.load_sidecars(audit.DEFAULT_MANIFEST, root / audit.DEFAULT_SEGMENTS.name, summary)
            with summary.open("a", encoding="utf-8") as stream:
                stream.write("tamper")
            with self.assertRaisesRegex(audit.AuditError, "Video-summary CSV hash mismatch"):
                audit.load_sidecars(audit.DEFAULT_MANIFEST, root / audit.DEFAULT_SEGMENTS.name, summary)

    def test_every_output_preserves_original_contiguous_order(self):
        for result in self.select(frame_count=50, segments=[(10, 15)], clip_length=8, num_clips=10):
            self.assertEqual(result.end_frame_exclusive - result.start_frame, 8)
            self.assertGreaterEqual(result.start_frame, 0)
            self.assertLessEqual(result.end_frame_exclusive, 50)

    def test_candidate_overlap_independent_recalculation(self):
        segments = [CandidateSegment(0, 3), CandidateSegment(8, 11), CandidateSegment(18, 20)]
        for result in self.select(frame_count=20, segments=segments, clip_length=7, num_clips=8):
            manual = sum(1 for frame in range(result.start_frame, result.end_frame_exclusive)
                         if any(item.start_frame <= frame < item.end_frame_exclusive for item in segments))
            self.assertEqual(result.candidate_overlap, manual)

    def test_fixed_smoke_is_balanced_diverse_and_reproducible(self):
        summaries, _, _ = audit.load_sidecars(audit.DEFAULT_MANIFEST,
                                               audit.DEFAULT_SEGMENTS, audit.DEFAULT_SUMMARY)
        a = audit.choose_smoke(summaries.values(), 20261009)
        b = audit.choose_smoke(reversed(list(summaries.values())), 20261009)
        self.assertEqual(a, b)
        self.assertEqual(sum(row["label"] == "Make" for row in a), 10)
        self.assertEqual(len({row["source_group"] for row in a}), 20)
        self.assertEqual(sum(row["has_clean_64"] == "false" for row in a), 5)
        self.assertTrue(any(int(row["candidate_frame_count"]) == 0 for row in a))
        self.assertTrue(any(int(row["candidate_frame_count"]) > 0 and row["has_clean_64"] == "true" for row in a))


if __name__ == "__main__":
    unittest.main()
