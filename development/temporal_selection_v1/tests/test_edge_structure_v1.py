from dataclasses import replace
import unittest

import cv2
import numpy as np

from development.temporal_selection_v1.edge_structure_v1 import (
    EdgePolicy, analyze_edges, edge_change, gray_frame, pair_metrics,
)
from development.temporal_selection_v1.selector_v5 import analyze_v5


class EdgeStructureTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)

    def texture(self, seed=5):
        rng = np.random.default_rng(seed)
        image = np.zeros((240, 320, 3), np.uint8) + 50
        for _ in range(140):
            x, y = rng.integers([5, 28], [300, 190])
            color = tuple(int(v) for v in rng.integers(70, 255, 3))
            cv2.rectangle(image, (int(x), int(y)), (int(x+9), int(y+8)), color, -1)
        return image

    def test_translation_is_explained_by_motion_not_scene_cut(self):
        a = self.texture()
        b = cv2.warpAffine(a, np.float32([[1, 0, 9], [0, 1, 4]]), (320, 240))
        m = pair_metrics(gray_frame(a), gray_frame(b))
        self.assertTrue(m['motion']['accepted'], m['motion'])
        self.assertTrue(m['compensation_used'])
        self.assertGreater(m['raw']['ecr'], .2)
        self.assertLess(m['effective_ecr'], .15)
        self.assertAlmostEqual(m['motion']['matrix'][0][2], 9*.8, delta=1.)

    def test_unrelated_same_palette_scene_still_has_structural_change(self):
        a, b = self.texture(), self.texture(51)
        m = pair_metrics(gray_frame(a), gray_frame(b))
        self.assertGreater(m['effective_ecr'], .35)
        self.assertFalse(m['motion']['accepted'])

    def test_uniform_frames_are_unknown_not_zero_change(self):
        a = np.zeros((240, 320, 3), np.uint8)
        m = pair_metrics(gray_frame(a), gray_frame(a+100))
        self.assertFalse(m['score_valid'])
        self.assertIsNone(m['effective_ecr'])
        self.assertFalse(m['compensation_used'])

    def test_invalid_warp_borders_are_excluded(self):
        a = np.zeros((80, 120), bool)
        a[10:70:5, 10:90] = True
        b = a.copy()
        b[:, 100:] = True
        valid = np.ones(a.shape, np.uint8)
        valid[:, 95:] = 0
        m = edge_change(a, b, valid)
        self.assertTrue(m['valid'])
        self.assertEqual(m['ecr'], 0.)

    def test_source_pixels_baseline_and_no_training_contract_unchanged(self):
        a = self.texture()
        frames = [a.copy() for _ in range(8)]
        snapshot = [f.copy() for f in frames]
        baseline = analyze_v5(frames)
        import copy
        before = copy.deepcopy(baseline)
        r = analyze_edges(frames, baseline)
        self.assertEqual(before, baseline)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(frames, snapshot)))
        self.assertEqual(r['hybrid']['hard_cut_candidates'], [])
        self.assertFalse(r['hybrid']['training_use_approved'])
        self.assertFalse(r['hybrid']['target_event_selected'])
        self.assertFalse(r['laplacian_used_for_selection'])

    def test_exposure_confirmation_cannot_create_new_boundary(self):
        a = self.texture()
        bright = np.clip(a.astype(int)+90, 0, 255).astype(np.uint8)
        frames = [a]*8 + [bright] + [a]*8
        baseline = analyze_v5(frames)
        self.assertTrue(baseline['exposure_flash_candidates'])
        r = analyze_edges(frames, baseline)
        self.assertNotIn(8, r['hybrid']['hard_cut_candidates'])
        self.assertNotIn(9, r['hybrid']['hard_cut_candidates'])

    def test_frozen_policy_and_native_input_validation(self):
        with self.assertRaises(ValueError):
            gray_frame(self.texture(), replace(EdgePolicy(), abrupt_ecr=.5))
        with self.assertRaises(ValueError):
            gray_frame(np.zeros((120, 160, 3), np.uint8))
        with self.assertRaises(ValueError):
            analyze_edges([], {'per_frame': []})


if __name__ == '__main__':
    unittest.main()
