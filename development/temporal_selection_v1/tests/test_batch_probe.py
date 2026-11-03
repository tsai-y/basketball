import copy
import unittest

from development.temporal_selection_v1.batch_probe_v1 import choose_samples, plan_input, geometry


class BatchProbeTests(unittest.TestCase):
    def rows(self):
        return {f'{g}/{i}':{'video_id':f'{g}/{i}','source_group':g,'split':'train','excluded':'0',
                           'binary_label':str(i%2),'original_class':'ft0' if i%2==0 else '2p1'}
                for g in ('a','b','c','d','e') for i in range(25)}

    def analysis(self,*spans):
        return {'all_segments_including_short':[{'start_frame':a,'end_frame_exclusive':b,'length':b-a,
                                                 'inside_proposed_change_span':changing}
                                                for a,b,changing in spans]}

    def test_balanced100_all_sources_deterministic_and_excludes_known(self):
        rows=self.rows();known={'a/0','b/1','c/2'}
        first=choose_samples(rows,known)
        self.assertEqual(first,choose_samples(dict(reversed(list(rows.items()))),known))
        self.assertEqual(len({r['video_id'] for r in first}),100)
        self.assertFalse({r['video_id'] for r in first}&known)
        self.assertTrue(all(sum(r['source_group']==g for r in first)==20 for g in ('a','b','c','d','e')))

    def test_labels_do_not_change_sampling(self):
        a=self.rows();b=copy.deepcopy(a)
        for r in b.values():r['binary_label']='9';r['original_class']='irrelevant'
        self.assertEqual([r['video_id'] for r in choose_samples(a,set())],
                         [r['video_id'] for r in choose_samples(b,set())])

    def test_no_silent_relaxation_or_foreign_rows(self):
        rows=self.rows()
        with self.assertRaises(ValueError):choose_samples(rows,set(rows))
        with self.assertRaises(ValueError):choose_samples(rows,set(),count=4)
        for field,value in (('split','test'),('excluded','1'),('video_id','wrong')):
            bad=copy.deepcopy(rows);bad['a/0'][field]=value
            with self.assertRaises(ValueError):choose_samples(bad,set())

    def test_full_source_identity_no_cropping(self):
        p=plan_input(146,self.analysis((0,146,False)))
        self.assertEqual(p['mode'],'identity_original');self.assertEqual(p['effective_windows'],p['original_windows'])
        self.assertFalse(p['training_use_approved'])

    def test_single_crop_geometry_is_not_event_or_approval(self):
        p=plan_input(146,self.analysis((0,37,True),(37,142,False),(142,146,True)))
        self.assertEqual(p['mode'],'fallback_original')
        self.assertEqual(p['single_crop_proposal']['geometry']['windows'],((37,101),(57,121),(78,142)))
        self.assertEqual(p['potential_removed_frames_geometry_only'],41)
        self.assertEqual(p['actual_removed_frames'],0)
        self.assertFalse(p['single_crop_proposal']['event_identity_verified'])

    def test_tiny_other_scene_is_not_ignored(self):
        p=plan_input(146,self.analysis((0,3,False),(3,20,True),(20,146,False)))
        self.assertEqual(p['reason'],'multiple_scene_proposals_target_unconfirmed')
        self.assertEqual(len(p['all_clean_scene_candidates']),2)
        self.assertIsNone(p['single_crop_proposal'])

    def test_short_uncoverable_and_absent_fall_back(self):
        self.assertEqual(plan_input(146,self.analysis((0,140,True),(140,146,False)))['reason'],'shorter_than_66')
        self.assertEqual(plan_input(146,self.analysis((0,146,True)))['reason'],'no_clean_scene_proposal')
        self.assertEqual(plan_input(193,self.analysis((0,193,False)))['reason'],'original_policy_rejects_source_no_silent_padding')
        self.assertFalse(geometry(0,193)['eligible'])

    def test_partition_gap_overlap_out_of_source_rejected(self):
        for spans in (((0,20,False),(21,146,False)),((0,30,False),(20,146,False)),((0,147,False),)):
            with self.assertRaises(ValueError):plan_input(146,self.analysis(*spans))


if __name__=='__main__':unittest.main()
