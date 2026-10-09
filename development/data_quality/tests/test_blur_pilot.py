"""Safety and determinism checks; no dataset video is opened."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_blur_pilot.py'
spec = importlib.util.spec_from_file_location('pilot', SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotSafety(unittest.TestCase):
    def setUp(self):
        self.cfg = pilot.load_config(pilot.DEFAULT_CONFIG)
        self.guard = pilot.TrainGuard()
        self.rows = pilot.read_csv(pilot.gate.MANIFEST)
        self.selected = pilot.choose(self.guard, self.cfg, self.cfg['seed'])

    def test_selection_reproducible_and_balanced(self):
        self.assertEqual(self.selected, pilot.choose(self.guard, self.cfg, self.cfg['seed']))
        self.assertEqual(len(self.selected), 30)
        self.assertEqual(len({r['source_group'] for r in self.selected}), 30)
        self.assertEqual(sum(r['binary_label']=='1' for r in self.selected), 15)
        self.guard.allowed = dict(reversed(list(self.guard.allowed.items())))
        self.assertEqual(self.selected, pilot.choose(self.guard, self.cfg, self.cfg['seed']))

    def test_all_selected_ids_and_paths_authorized_without_io(self):
        with patch.object(pilot.gate, 'open_guarded', side_effect=AssertionError('Unexpected video open')):
            for r in self.selected:
                self.guard.authorize(r['video_id'], r['video_path'])
        self.assertEqual(self.guard.events, [])

    def test_all_heldout_ids_paths_and_exclusions_rejected_before_open(self):
        train = self.selected[0]
        with patch.object(pilot.gate, 'open_guarded', side_effect=AssertionError('Unexpected video open')):
            for row in self.rows:
                if row['split'] == 'train':
                    continue
                for vid, path in [(row['video_id'], row['video_path']),
                                  (train['video_id'], row['video_path'])]:
                    with self.assertRaises(ValueError):
                        self.guard.open(vid, path, 'decode')
        self.assertEqual(self.guard.events, [])

    def test_unknown_absolute_traversal_and_tampered_split_rejected(self):
        r = self.selected[0]
        for vid, path in [('unknown', None), (r['video_id'], '/tmp/example.mp4'),
                          (r['video_id'], '../example.mp4')]:
            with self.assertRaises(ValueError):
                self.guard.authorize(vid, path)
        self.guard.allowed[r['video_id']] = dict(r, split='validation')
        with self.assertRaises(ValueError):
            self.guard.authorize(r['video_id'])

    def test_segments_edges_and_metric_zero(self):
        self.assertEqual(pilot.runs([True, True, False, True]), [(0,1),(3,3)])
        self.assertEqual(pilot.runs([]), [])
        self.assertEqual(pilot.score_frame(np.zeros((10,10,3),dtype=np.uint8), self.cfg), (0.,0.))
        frames = [dict(laplacian_variance=0., tenengrad=0.) for _ in range(8)]
        flags, _ = pilot.flags_for(frames, 25., dict(laplacian_variance=1.,tenengrad=1.), self.cfg)
        self.assertEqual(int(flags['smoothed_both_low'].sum()), 8)
        flags, _ = pilot.flags_for(frames[:1], 25., dict(laplacian_variance=1.,tenengrad=1.), self.cfg)
        self.assertEqual(int(flags['smoothed_both_low'].sum()), 0)

    def test_short_decode_has_failure_status_not_low_quality(self):
        class EmptyCapture:
            def isOpened(self): return True
            def read(self): return False, None
            def release(self): pass
        row = self.selected[0]
        with patch.object(self.guard, 'open', return_value=123), \
             patch.object(pilot.cv2, 'VideoCapture', return_value=EmptyCapture()), \
             patch.object(pilot.os, 'close'):
            frames, status = pilot.decode(row, dict(frame_count=146, fps=23.976,
                fps_rational='2997/125',r_frame_rate='2997/125'), self.guard, self.cfg, {})
        self.assertEqual(frames, [])
        self.assertEqual(status['decode_status'], 'frame_count_mismatch')


if __name__ == '__main__':
    unittest.main(verbosity=2)
