"""Synthetic tests: no video or production feature-cache reads."""

import unittest

import torch

from development.data_quality.scripts import audit_three_window_quality_v1 as audit


class ThreeWindowQualityAuditTests(unittest.TestCase):
    def setUp(self):
        self.windows = audit.select_covering_windows(146)

    def test_half_open_boundaries_and_empty(self):
        self.assertEqual(audit.independent_counts(146, self.windows, []), ((0, 0, 0), 0))
        self.assertEqual(audit.independent_counts(146, self.windows, [(64, 65)]), ((0, 1, 0), 1))
        self.assertEqual(audit.independent_counts(146, self.windows, [(90, 98)]), ((0, 8, 8), 8))

    def test_same_candidate_in_two_windows_is_not_two_unique_frames(self):
        counts, unique = audit.independent_counts(146, self.windows, [(90, 98)])
        self.assertEqual(sum(counts), 16)
        self.assertEqual(unique, 8)
        self.assertNotEqual(sum(counts) / 192, unique / 146)

    def test_overlapping_unsorted_out_of_bounds_or_empty_segments_rejected(self):
        for segments in ([(1, 4), (3, 6)], [(8, 10), (1, 3)],
                         [(-1, 2)], [(145, 147)], [(4, 4)]):
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                audit.independent_counts(146, self.windows, segments)

    def test_adjacent_intervals_are_valid(self):
        self.assertEqual(audit.independent_counts(146, self.windows, [(0, 8), (8, 16)]),
                         ((16, 0, 0), 16))

    def test_gap_in_window_coverage_rejected(self):
        with self.assertRaises(ValueError):
            audit.independent_counts(200, ((0, 64), (70, 134), (136, 200)), [])

    def test_histogram_boundaries(self):
        result = audit.histogram([0, .01, .05, .1, .25, 1])
        self.assertEqual(list(result.values()), [1] * 6)

    def test_cache_metadata_fail_closed(self):
        row = {"video_id": "2p0/a.mp4", "binary_label": "0", "split": "train", "excluded": "0"}
        record = {"video_id": row["video_id"], "run_id": "run", "binary_label": 0,
                  "split": "train", "frame_count": 146, "windows": self.windows,
                  "quality_aware_selection_used": False, "quality_metadata_available": True,
                  "candidate_overlap_frames": torch.tensor([0, 8, 8], dtype=torch.int64)}

        def check(value, manifest_row=row):
            audit.validate_record(value, manifest_row, "run", 146, self.windows, (0, 8, 8))

        check(record)
        for key, value in (("split", "test"), ("run_id", "other"),
                           ("quality_metadata_available", False),
                           ("quality_aware_selection_used", True), ("frame_count", 145),
                           ("candidate_overlap_frames", torch.tensor([0, 7, 8])),
                           ("candidate_overlap_frames", torch.tensor([0., 8., 8.]))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check({**record, key: value})
        with self.assertRaises(ValueError):
            check(record, {**row, "split": "test"})


if __name__ == "__main__":
    unittest.main()
