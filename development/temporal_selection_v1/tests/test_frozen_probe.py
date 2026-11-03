import unittest
from development.temporal_selection_v1.frozen_probe_v2 import choose_samples, nested_video_ids, boundary_indices


class FrozenProbeTests(unittest.TestCase):
    def fixture(self):
        return {f'{g}/{i}.mp4':{'video_id':f'{g}/{i}.mp4','source_group':g,'split':'train','excluded':'0'}
                for g in ('old','fresh1','fresh2') for i in range(7)}

    def test_deterministic_disjoint_source_not_only_id(self):
        rows=self.fixture();known={'old/0.mp4'}
        first,groups=choose_samples(rows,known)
        again,_=choose_samples(dict(reversed(list(rows.items()))),known)
        self.assertEqual(first,again);self.assertEqual(groups,['old'])
        self.assertEqual(len(first),12)
        self.assertTrue(all(s['source_group']!='old' for s in first))

    def test_insufficient_sources_no_silent_relaxation(self):
        with self.assertRaises(ValueError):choose_samples(self.fixture(),{'old/0.mp4','fresh1/0.mp4'})

    def test_nontrain_excluded_or_canonical_key_mismatch_rejected(self):
        for field,value in (('split','validation'),('excluded','1'),('video_id','wrong')):
            rows=self.fixture();rows['old/0.mp4'][field]=value
            with self.assertRaises(ValueError):choose_samples(rows,set())

    def test_nested_inventory_ignores_reservations_without_actual_ids(self):
        self.assertEqual(set(nested_video_ids({'source_assignments':{'all_groups':'reserved'},
            'records':[{'video_id':'a'},{'nested':{'video_id':'b'}}]})),{'a','b'})

    def test_boundaries_clipped_and_sorted(self):
        r={'hard_cut_candidates':[1], 'exposure_flash_candidates':[{'start_frame':8,'end_frame_exclusive':10}],
           'gradual_change_regions':[]}
        self.assertEqual(boundary_indices(10,r),[0,1,2,3,6,7,8,9])


if __name__=='__main__':unittest.main()
