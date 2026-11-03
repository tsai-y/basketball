from dataclasses import replace,asdict
import unittest

import numpy as np

from development.temporal_selection_v1.selector import descriptor
from development.temporal_selection_v1.selector_v3 import ScenePolicyV3,exposure_events_v3
from development.temporal_selection_v1.selector_v4 import BoundaryPolicy
from development.temporal_selection_v1.selector_v5 import (
    ExposurePolicyV5,analyze_v5,exposure_events_v5,largest_component,local_seeds,
)


class LocalExposureTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(9)
        # Enough spatial contrast that a partial pulse preserves structure;
        # near-uniform weak texture correctly fails the unchanged correlation gate.
        texture=rng.integers(0,140,(12,16,1),dtype=np.uint8).repeat(20,0).repeat(20,1)
        a=texture+np.array([20,65,110],dtype=np.uint8)
        partial=a.copy();partial[144:204]=np.clip(partial[144:204].astype(int)+60,0,255).astype(np.uint8)
        return a,partial

    def coords(self,events):return [(e['start_frame'],e['end_frame_exclusive']) for e in events]

    def events(self,frames):return exposure_events_v5([descriptor(f) for f in frames])[0]

    def test_partial_pulse_below_global_threshold_recovers_without_fake_regions(self):
        a,p=self.fixture();frames=[a]*10+[p]+[a]*10;features=[descriptor(f) for f in frames]
        self.assertLess(float(features[10]['gray'].mean()-features[9]['gray'].mean()),.1)
        self.assertEqual(exposure_events_v3(features)[0],[])
        self.assertEqual(self.coords(self.events(frames)),[(10,11)])
        r=analyze_v5(frames)
        self.assertEqual(r['hard_cut_candidates'],[]);self.assertEqual(r['gradual_change_regions'],[])
        self.assertFalse(r['training_use_approved']);self.assertFalse(r['automatic_training_use_enabled'])

    def test_connectivity_not_diagonal_or_single_bright_tile(self):
        self.assertEqual(largest_component(np.eye(3,dtype=bool)),1)
        self.assertEqual(largest_component(np.array([[True,True],[False,True]])),3)
        a,_=self.fixture();base=descriptor(a);frames=[dict(base) for _ in range(21)]
        pulse=base['gray'].copy();pulse[:18,:24]+=.2
        frames[10]={**base,'gray':pulse}
        self.assertFalse(local_seeds(frames,ExposurePolicyV5())[0][10])

    def test_global_rule_preserved_and_not_counted_twice(self):
        a,_=self.fixture();p=np.clip(a.astype(int)+100,0,255).astype(np.uint8);frames=[a]*10+[p]+[a]*10
        features=[descriptor(f) for f in frames];old,_=exposure_events_v3(features)
        events=exposure_events_v5(features)[0]
        self.assertEqual(self.coords(events),self.coords(old));self.assertEqual(len(events),1)
        self.assertEqual(events[0]['detection_basis'],'unchanged_v3_global_luma_rule')

    def test_persistent_bright_new_scene_is_not_recovered_exposure(self):
        a,p=self.fixture();self.assertEqual(self.events([a]*10+[p]*12),[])

    def test_brief_unrelated_bright_scene_and_cut_after_pulse_not_suppressed(self):
        a,p=self.fixture();other=np.flip(p,axis=1).copy()
        self.assertEqual(self.events([a]*10+[other]*2+[a]*10),[])
        self.assertEqual(self.events([a]*10+[p]+[np.flip(a,axis=1).copy()]*10),[])

    def test_real_cut_near_confirmed_flash_is_retained(self):
        a,p=self.fixture();other=np.flip(a,axis=1).copy()
        # A real color/structure step after the two-frame hold cannot be swallowed.
        other=np.roll(other,1,axis=2).copy()
        r=analyze_v5([a]*10+[p]+[a]*3+[other]*12)
        self.assertEqual(self.coords(r['exposure_flash_candidates']),[(10,11)])
        self.assertIn(14,r['hard_cut_candidates'])

    def test_no_hold_no_confirmation_and_existing_length_limit(self):
        a,p=self.fixture()
        self.assertEqual(self.events([a]*10+[p,a]),[])
        self.assertEqual(self.events([a]*10+[p]*5+[a]*10),[])

    def test_no_input_mutation_labels_or_old_threshold_changes(self):
        a,p=self.fixture();frames=[a]*10+[p]+[a]*10;copies=[f.copy() for f in frames]
        analyze_v5(frames)
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(frames,copies)))
        old=asdict(BoundaryPolicy());new=asdict(ExposurePolicyV5())
        self.assertTrue(all(new[k]==v for k,v in old.items() if k!='version'))
        with self.assertRaises(TypeError):analyze_v5(frames,binary_label=1)
        with self.assertRaises(ValueError):analyze_v5(frames,replace(ExposurePolicyV5(),local_grid_rows=5))
        with self.assertRaises(ValueError):analyze_v5(frames,replace(ExposurePolicyV5(),local_min_connected_tiles=1))


if __name__=='__main__':unittest.main()
