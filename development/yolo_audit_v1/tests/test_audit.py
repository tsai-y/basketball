"""Metadata/geometry tests; never load YOLO weights or original videos."""

from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from development.yolo_audit_v1 import audit


class YoloAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.summaries = [], {}
        for group in audit.CLASSES:
            for number in range(12):
                video_id = f"{group}/{number}.mp4"
                self.rows.append({"video_id": video_id, "video_path": video_id,
                                  "original_class": group, "binary_label": group[-1],
                                  "source_group": f"source{number}", "split": "train", "excluded": "0"})
                self.summaries[video_id] = {"candidate_frame_count": 5 if number % 2 else 0,
                                           "fps_rational": "24/1", "frame_count": 146}

    def test_selection_is_reproducible_with_fixed_quotas(self):
        a = audit.select_samples(self.rows, self.summaries)
        b = audit.select_samples(list(reversed(self.rows)), self.summaries)
        self.assertEqual(a, b)
        self.assertEqual(len({r["video_id"] for r in a}), 20)
        self.assertEqual(sum(r["binary_label"] == "1" for r in a), 10)
        self.assertEqual(sum(r["candidate_present"] for r in a), 10)
        self.assertEqual([r["sample_id"] for r in a], [f"Y{i:03d}" for i in range(1, 21)])

    def test_nontrain_or_duplicate_rows_rejected(self):
        for rows in ([{**self.rows[0], "split": "test"}], self.rows + [self.rows[0]]):
            with self.subTest(rows=rows[:1]), self.assertRaises(ValueError):
                audit.select_samples(rows, self.summaries)

    def test_missing_stratum_fails_instead_of_changing_sampling(self):
        with self.assertRaisesRegex(ValueError, "Insufficient train stratum"):
            audit.select_samples([r for r in self.rows if r["original_class"] != "ft0"], self.summaries)

    def test_frame_indices_are_fixed_and_unique(self):
        self.assertEqual(audit.sample_frame_indices(146), (14, 43, 73, 102, 131))
        self.assertEqual(len(set(audit.sample_frame_indices(180))), 5)

    def test_roi_union_padding_and_no_detection_fallback(self):
        roi, fallback = audit.person_union_roi([[100, 100, 200, 200]], 320, 240)
        self.assertEqual(roi, [90, 90, 210, 210])
        self.assertFalse(fallback)
        self.assertEqual(audit.person_union_roi([], 320, 240), ([0, 0, 320, 240], True))
        self.assertEqual(audit.person_union_roi([[0, 0, 320, 240]], 320, 240)[0], [0, 0, 320, 240])

    def test_invalid_roi_boxes_rejected(self):
        for boxes in ([[0, 0, 0, 4]], [[0, 0, float("nan"), 4]]):
            with self.assertRaises(ValueError):
                audit.person_union_roi(boxes, 320, 240)

    def test_unconfirmed_detections_are_retained_not_silently_dropped(self):
        boxes = np.array([[1, 2, 10, 20], [20, 20, 40, 40]])
        tracks = np.array([[1, 2, 10, 20, 7, .8, 0, 0]])
        result = audit.attach_track_ids(boxes, [.8, .7], tracks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["track_id"], 7)
        self.assertIsNone(result[1]["track_id"])

    def test_installed_bytetrack_api_and_detection_index_schema(self):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        config = SimpleNamespace(track_buffer=30, track_high_thresh=.25, track_low_thresh=.1,
                                 new_track_thresh=.25, match_thresh=.8, fuse_score=True)
        box = Boxes(np.array([[10, 10, 40, 60, .8, 0]], dtype=np.float32), orig_shape=(240, 320))
        for _ in range(2):
            tracker = BYTETracker(args=config)  # A new tracker resets IDs per video.
            result = tracker.update(box.numpy(), np.zeros((240, 320, 3), dtype=np.uint8))
            self.assertEqual(result.shape, (1, 8))
            linked = audit.attach_track_ids(box.xyxy, box.conf, result)
            self.assertEqual(linked[0]["track_id"], 1)

    def test_invalid_track_mapping_rejected(self):
        with self.assertRaises(ValueError):
            audit.attach_track_ids([[0, 0, 10, 10]], [.8], [[0, 0, 10, 10, 7, .8, 0, 3]])

    def test_visualization_does_not_modify_source_pixels(self):
        original = np.full((240, 320, 3), 40, dtype=np.uint8)
        before = original.copy()
        pair = audit.draw_pair(original, [], [0, 0, 320, 240], sample_id="Y001", frame_index=0, fps=24.)
        self.assertEqual(pair.shape, (264, 640, 3))
        np.testing.assert_array_equal(original, before)
        np.testing.assert_array_equal(pair[24:, :320], original)

    def test_process_rejects_test_before_guard_or_decoder(self):
        sample = {"split": "test"}
        with patch.object(audit.cv2, "VideoCapture", side_effect=AssertionError("Video opened")):
            with self.assertRaisesRegex(ValueError, "Non-train"):
                audit.process_video(sample, None, None, None, None, "cpu")

    def test_cli_requires_explicit_mode(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            audit.main([])


if __name__ == "__main__":
    unittest.main()
