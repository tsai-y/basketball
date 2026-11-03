from dataclasses import replace
import unittest
import numpy as np

from development.temporal_selection_v1.selector import descriptor
from development.temporal_selection_v1.selector_v2 import exposure_events
from development.temporal_selection_v1.selector_v3 import ScenePolicyV3, exposure_events_v3, analyze_v3
from development.temporal_selection_v1.selector_v2 import correlation


class ExposureTailTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(9)
        # Spatial color patches survive descriptor downsampling; random independent
        # RGB pixels can average to gray and hide a real histogram change.
        texture = rng.integers(0,30,(12,16,1),dtype=np.uint8).repeat(20,axis=0).repeat(20,axis=1)
        a = texture + np.array([20,65,110],dtype=np.uint8)
        return a, (a.astype(int)+100).astype(np.uint8), (a.astype(int)+20).astype(np.uint8)

    def events(self, frames):
        return exposure_events_v3([descriptor(f) for f in frames])[0]

    def test_asymmetric_tail_endpoint_not_peak_endpoint(self):
        a, peak, tail = self.fixture(); frames = [a]*10+[peak,tail]+[a]*10
        self.assertEqual(exposure_events([descriptor(f) for f in frames])[0], [])
        events = self.events(frames)
        self.assertEqual([(e['start_frame'],e['end_frame_exclusive']) for e in events], [(10,12)])
        self.assertEqual(events[0]['peak_end_frame_exclusive'],11)
        r = analyze_v3(frames)
        self.assertEqual(r['hard_cut_candidates'],[]); self.assertEqual(r['gradual_change_regions'],[])
        self.assertEqual(r['contiguous_scene_candidates'][0]['length'],len(frames))
        self.assertFalse(events[0]['pixels_removed']); self.assertFalse(r['automatic_training_use_enabled'])

    def test_real_persistent_bright_scene_is_not_flash(self):
        a, peak, _ = self.fixture()
        self.assertEqual(self.events([a]*10+[peak]*12),[])
        self.assertIn(10,analyze_v3([a]*10+[peak]*12)['hard_cut_candidates'])

    def test_brief_unrelated_bright_scene_then_return_not_flash(self):
        a, peak, _ = self.fixture(); other = np.flip(peak, axis=1).copy()
        self.assertEqual(self.events([a]*10+[other]*2+[a]*10),[])
        self.assertTrue(analyze_v3([a]*10+[other]*2+[a]*10)['hard_cut_candidates'])

    def test_flash_followed_by_true_cut_does_not_get_suppressed(self):
        a, peak, _ = self.fixture(); other = np.flip(a, axis=1).copy()
        self.assertEqual(self.events([a]*10+[peak]+[other]*12),[])

    def test_recovery_must_persist_not_one_return_frame(self):
        a, peak, _ = self.fixture(); other = np.flip(a, axis=1).copy()
        self.assertEqual(self.events([a]*10+[peak,a]+[other]*10),[])

    def test_hold_follows_returned_scene_not_stale_preflash_anchor(self):
        a, peak, _ = self.fixture()
        # Feature-level controlled pan: first return still matches pre-flash;
        # next frame drifts farther from the old image but matches its neighbor.
        g=descriptor(a)['gray'];rng=np.random.default_rng(77)
        noise=rng.normal(size=g.shape);noise-=noise.mean();noise*=float(g.std())/float(noise.std())
        first=g+.70*noise;second=g+1.05*noise
        self.assertGreaterEqual(correlation(g,first),.75)
        self.assertLess(correlation(g,second),.75)
        self.assertGreater(correlation(first,second),.75)
        baseline=descriptor(a)
        before=[dict(baseline) for _ in range(10)]
        bright={**baseline,'gray':g+.20}
        returned={**baseline,'gray':first.astype(np.float32)}
        moved={**baseline,'gray':second.astype(np.float32)}
        features=before+[bright,returned,moved]+[dict(moved) for _ in range(6)]
        events,_=exposure_events_v3(features)
        self.assertEqual([(e['start_frame'],e['end_frame_exclusive']) for e in events],[(10,11)])

    def test_search_bounded_and_end_without_hold_not_confirmed(self):
        a, peak, tail = self.fixture()
        self.assertEqual(self.events([a]*10+[peak]+[tail]*4+[a]*10),[])
        self.assertEqual(self.events([a]*10+[peak,tail,a]),[])

    def test_original_pixels_unchanged_and_no_label_input(self):
        a, peak, tail = self.fixture(); frames=[a]*10+[peak,tail]+[a]*10
        copies=[f.copy() for f in frames]; analyze_v3(frames)
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(frames,copies)))
        with self.assertRaises(TypeError):analyze_v3(frames,binary_label=1)
        with self.assertRaises(ValueError):analyze_v3(frames,replace(ScenePolicyV3(),recovery_hold_frames=0))


if __name__=='__main__':unittest.main()
