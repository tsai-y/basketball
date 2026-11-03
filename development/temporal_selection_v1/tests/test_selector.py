from dataclasses import replace
import unittest
import cv2
import numpy as np

from development.temporal_selection_v1.selector import (
    Policy,analyze,descriptor,cut_candidates,propose_intervals,runs,
)


class SelectorTests(unittest.TestCase):
    def test_half_open_runs_and_unchanged_input(self):
        flags=[False,True,True,False,True]
        self.assertEqual(runs(flags),[(1,3),(4,5)]);self.assertEqual(flags,[False,True,True,False,True])

    def test_one_long_interval_is_only_unverified_proposal(self):
        r=propose_intervals([True]*100,[])
        self.assertEqual(r['proposed_contiguous_interval']['end_frame_exclusive'],100)
        self.assertFalse(r['event_semantics_verified']);self.assertFalse(r['automatic_training_use_enabled'])
        self.assertIn('original_three_window',r['effective_training_policy'])

    def test_short_or_no_court_falls_back_without_padding(self):
        for flags in ([True]*63,[False]*100):
            r=propose_intervals(flags,[])
            self.assertIsNone(r['proposed_contiguous_interval']);self.assertFalse(r['frames_deleted'])

    def test_cut_not_bridged_and_later_free_throw_not_silently_selected(self):
        # Without target-event information, two court scenes cannot be resolved by length alone.
        r=propose_intervals([True]*20+[False]*30+[True]*90,[20,50])
        self.assertEqual([(x['start_frame'],x['end_frame_exclusive']) for x in r['court_interval_candidates']],[(0,20),(50,140)])
        self.assertIsNone(r['proposed_contiguous_interval'])
        self.assertIn('multiple_court_intervals',r['candidate_fallback_reasons'][0])
        r=propose_intervals([True]*100,[50]);self.assertIsNone(r['proposed_contiguous_interval'])
        self.assertFalse(r['frames_concatenated_across_cuts'])

    def test_invalid_boundaries_and_policy_rejected(self):
        for cuts in ([0],[100],[30,20],[20,20]):
            with self.assertRaises(ValueError):propose_intervals([True]*100,cuts)
        with self.assertRaises(ValueError):propose_intervals([True]*100,[],replace(Policy(),floor_area_min=float('nan')))

    def test_native_input_and_finite_descriptors(self):
        raw=np.full((240,320,3),(100,180,220),np.uint8);d=descriptor(raw)
        self.assertTrue(np.isfinite(d['hist']).all());self.assertGreater(d['floor_area'],.9)
        for bad in (raw.astype(float),raw[:100],raw[:,:,0]):
            with self.assertRaises(ValueError):descriptor(bad)

    def test_obvious_cut_and_flash_recovery(self):
        a=np.full((240,320,3),(100,180,220),np.uint8)
        b=np.full_like(a,(200,10,10));flash=np.full_like(a,255)
        cuts,flashes,_=cut_candidates([descriptor(x) for x in [a,a,b,b]])
        self.assertEqual(cuts,[2]);self.assertEqual(flashes,[])
        cuts,flashes,_=cut_candidates([descriptor(x) for x in [a,a,flash,a,a]])
        self.assertEqual(cuts,[]);self.assertEqual(flashes,[2])

    def test_static_court_no_fake_motion_event(self):
        raw=np.full((240,320,3),(100,180,220),np.uint8)
        r=analyze([raw]*70)
        self.assertEqual(r['cut_frame_candidates'],[])
        self.assertFalse(r['event_semantics_verified'])


if __name__=='__main__':unittest.main()
