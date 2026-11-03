from dataclasses import replace
import unittest
import numpy as np

from development.temporal_selection_v1.selector import descriptor
from development.temporal_selection_v1.selector_v2 import (
    ScenePolicy,exposure_events,clean_segments,merge_regions,analyze_v2,distinguish_hard_regions,
)


class SceneV2Tests(unittest.TestCase):
    def images(self):
        rng=np.random.default_rng(4)
        a=rng.integers(30,110,(240,320,3),dtype=np.uint8)
        return a,np.clip(a.astype(int)+100,0,255).astype(np.uint8)

    def test_multi_frame_exposure_recovers_without_deleting_pixels(self):
        a,bright=self.images();frames=[a]*6+[bright]*2+[a]*6
        events,_=exposure_events([descriptor(x) for x in frames])
        self.assertEqual([(e['start_frame'],e['end_frame_exclusive']) for e in events],[(6,8)])
        self.assertFalse(events[0]['pixels_removed'])

    def test_permanent_new_bright_scene_is_not_recovery(self):
        a,bright=self.images();events,_=exposure_events([descriptor(x) for x in [a]*8+[bright]*8])
        self.assertEqual(events,[])

    def test_regions_are_half_open_clipped_and_merged(self):
        self.assertEqual(merge_regions([(2,5),(7,9)],10,gap=2,pad=0),[(2,9)])
        self.assertEqual(merge_regions([(0,2),(8,10)],10,pad=2),[(0,4),(6,10)])
        with self.assertRaises(ValueError):merge_regions([(8,11)],10)

    def test_no_stitching_and_no_target_from_unique_long_length(self):
        r=clean_segments(146,[],[(8,37)])
        self.assertEqual([(s['start_frame'],s['end_frame_exclusive']) for s in r['contiguous_scene_candidates']],[(0,8),(37,146)])
        self.assertEqual(r['unique_length_eligible_candidate']['start_frame'],37)
        self.assertFalse(r['target_event_selected']);self.assertFalse(r['automatic_training_use_enabled'])
        self.assertTrue(r['short_other_scenes_present'])

    def test_hard_cut_and_multiple_long_scenes_not_ranked_by_length(self):
        r=clean_segments(180,[80],[])
        self.assertEqual(len(r['contiguous_scene_candidates']),2)
        self.assertIsNone(r['unique_length_eligible_candidate'])

    def test_no_floor_exclusion_and_static_no_transition(self):
        image=np.full((240,320,3),(200,10,10),np.uint8)
        r=analyze_v2([image]*70)
        self.assertEqual(r['hard_cut_candidates'],[]);self.assertEqual(r['gradual_change_regions'],[])
        self.assertFalse(r['floor_appearance_used_as_exclusion_rule'])
        self.assertEqual(r['contiguous_scene_candidates'][0]['length'],70)
        self.assertFalse(r['target_event_selected'])

    def test_no_label_input_or_invalid_policy(self):
        with self.assertRaises(TypeError):analyze_v2([],binary_label=1)
        with self.assertRaises(ValueError):clean_segments(100,[0],[])
        with self.assertRaises(ValueError):clean_segments(100,[],[],replace(ScenePolicy(),luma_peak=float('nan')))

    def test_flash_temporal_wings_are_not_gradual_regions(self):
        a,bright=self.images();r=analyze_v2([a]*12+[bright]*2+[a]*12)
        self.assertEqual(len(r['exposure_flash_candidates']),1)
        self.assertEqual(r['gradual_change_regions'],[]);self.assertEqual(r['hard_cut_candidates'],[])
        self.assertEqual(r['contiguous_scene_candidates'][0]['length'],26)

    def test_localized_hard_cut_not_dilated_into_gradual_scene(self):
        p=ScenePolicy()
        self.assertEqual(distinguish_hard_regions([(37,51)],[44],p),[])
        self.assertEqual(distinguish_hard_regions([(2,44)],[36],p),[(2,44)])


if __name__=='__main__':unittest.main()
