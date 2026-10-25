"""ROI geometry and visualization tests; no model or dataset videos."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import numpy as np

from development.yolo_audit_v1 import roi_probe as probe


class ContextRoiProbeTests(unittest.TestCase):
    def test_legacy_rule_matches_audit(self):
        self.assertEqual(probe.legacy_roi([[100, 100, 200, 200]]), [90, 90, 210, 210])
        self.assertEqual(probe.legacy_roi([]), [0, 0, 320, 240])

    def test_no_detections_keep_full_frame_not_delete_frame(self):
        result = probe.context_roi([])
        self.assertEqual(result['roi_xyxy'], [0, 0, 320, 240])
        self.assertEqual(result['fallback_reasons'], ['no_person_boxes'])
        self.assertTrue(result['fallback_used'])

    def test_one_person_cannot_become_sole_model_view(self):
        result = probe.context_roi([[176, 140, 214, 190]])
        self.assertEqual(result['roi_xyxy'], [0, 0, 320, 240])
        self.assertIn('few_person_boxes', result['fallback_reasons'])

    def test_bottom_spectator_cluster_uses_full_frame(self):
        boxes = [[60, 180, 70, 210], [100, 180, 112, 210],
                 [130, 180, 140, 210], [145, 180, 155, 210]]
        result = probe.context_roi(boxes)
        self.assertTrue(result['full_frame_used'])
        self.assertIn('bottom_cluster', result['fallback_reasons'])

    def test_ordinary_context_keeps_width_top_and_ninety_percent_height(self):
        boxes = [[25, 60, 50, 180], [130, 80, 155, 180], [250, 70, 280, 180]]
        result = probe.context_roi(boxes)
        self.assertFalse(result['fallback_used'])
        self.assertEqual(result['roi_xyxy'], [0, 0, 320, 216])
        self.assertEqual(result['kept_area_fraction'], .9)

    def test_expansion_is_not_misreported_as_rule_fallback(self):
        boxes = [[0, 0, 70, 240], [120, 0, 180, 240], [250, 0, 320, 240]]
        result = probe.context_roi(boxes)
        self.assertTrue(result['full_frame_used'])
        self.assertFalse(result['fallback_used'])

    def test_new_rectangle_contains_old_and_has_bounded_area(self):
        rng = np.random.default_rng(20261024)
        for _ in range(100):
            boxes = []
            for _ in range(int(rng.integers(0, 12))):
                x1, x2 = sorted(rng.choice(321, size=2, replace=False))
                y1, y2 = sorted(rng.choice(241, size=2, replace=False))
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
            old = probe.legacy_roi(boxes)
            result = probe.context_roi(boxes)
            new = result['roi_xyxy']
            self.assertTrue(new[0] <= old[0] < old[2] <= new[2])
            self.assertTrue(new[1] <= old[1] < old[3] <= new[3])
            self.assertTrue(.9 <= result['kept_area_fraction'] <= 1)

    def test_invalid_boxes_and_policy_fail_closed(self):
        for boxes in ([[0, 0, 0, 2]], [[-1, 0, 10, 20]], [[0, 0, 321, 240]],
                      [[0, 0, float('nan'), 10]], [[0, 0, 1]]):
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                probe.context_roi(boxes)
        with self.assertRaises(ValueError):
            probe.context_roi([], policy=replace(probe.ContextPolicy(), min_context_height_fraction=1.1))

    def test_compare_preserves_source_and_does_not_resize_kept_pixels(self):
        bgr = np.full((240, 320, 3), 70, dtype=np.uint8)
        before = bgr.copy()
        old = [90, 90, 210, 210]
        candidate = probe.context_roi([[100, 100, 200, 200]])
        image = probe.draw_comparison(bgr, old, candidate, sid='Y018', index=92, fps=24, count=1)
        self.assertEqual(image.shape, (312, 960, 3))
        np.testing.assert_array_equal(bgr, before)
        np.testing.assert_array_equal(image[48:288, :320], before)
        self.assertTrue((image[48 + 20, 320 + 20] == 0).all())
        np.testing.assert_array_equal(image[48 + 120, 320 + 120], before[120, 120])

    def test_non_train_manifest_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Only cached train'):
            probe.select_cached_samples({'split': 'test', 'samples': []})

    def test_dry_run_cannot_open_video_or_create_output(self):
        fake = [{'frame_count': 146}] * 4
        with patch.object(probe, 'load_inputs', return_value=(None, fake, {}, {})), \
                patch.object(probe.cv2, 'VideoCapture', side_effect=AssertionError('Video opened')), \
                patch.object(probe, 'mkdtemp', side_effect=AssertionError('Output created')), \
                redirect_stdout(io.StringIO()) as stream:
            probe.main(['--audit-run', 'ignored', '--dry-run'])
        self.assertIn('"video_opens": 0', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
