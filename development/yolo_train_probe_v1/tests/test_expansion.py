import unittest
from copy import deepcopy

import numpy as np

from development.yolo_train_probe_v1.prepare_expansion_v1 import (
    assign_sources, choose_frames, validate_native_pair,
)
from development.yolo_audit_v1.target_label_pilot import CLASS_NAMES, PROTOCOL, validate_review


class ExpansionTests(unittest.TestCase):
    def fixture(self):
        return [{'sample_id':f'U{i+1:03d}','video_id':f'{i}.mp4','source_group':f'g{i}',
                 'split':'train','excluded':'0','original_class':shot+str(j%2),'frame_count':146}
                for i,(shot,j) in enumerate((s,j) for s in ('2p','3p','ft','mp') for j in range(4))]

    def test_partition_is_deterministic_disjoint_and_stratified(self):
        rows=self.fixture();a=assign_sources(rows,['old1','old2'])
        self.assertEqual(a,assign_sources(list(reversed(rows)),['old1','old2']))
        held=[r for r in rows if a[r['source_group']]=='detector_dev_validation']
        self.assertEqual(len(held),4)
        self.assertEqual({r['original_class'][:-1] for r in held},{'2p','3p','ft','mp'})
        self.assertEqual(a['old1'],'detector_train')
        self.assertEqual(sum(a[r['source_group']]=='detector_train' for r in rows),12)

    def test_nontrain_duplicate_and_old_source_fail(self):
        for field,value in [('split','test'),('split','validation'),('excluded','1')]:
            rows=self.fixture();rows[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):assign_sources(rows,[])
        rows=self.fixture();rows[1]['source_group']=rows[0]['source_group']
        with self.assertRaises(ValueError):assign_sources(rows,[])
        with self.assertRaises(ValueError):assign_sources(self.fixture(),['g0'])

    def test_heldout_positions_not_adapted_to_known_failure(self):
        row=self.fixture()[0]
        self.assertEqual(choose_frames(row,{'g0':'detector_dev_validation'}),(43,102))
        self.assertEqual(choose_frames(row,{'g0':'detector_train'}),(73,131))

    def test_native_pixels_must_match_lossless_pair(self):
        raw=np.zeros((240,320,3),np.uint8);pair=np.zeros((264,640,3),np.uint8)
        validate_native_pair(raw,pair)
        pair[24,0,0]=1
        with self.assertRaises(ValueError):validate_native_pair(raw,pair)
        with self.assertRaises(ValueError):validate_native_pair(raw[:200],pair)

    def test_AI_proposal_does_not_make_ready_and_transition_blocks_readiness(self):
        row={'image_id':'E001','image_sha256':'x','status':'pending','scene':'unreviewed',
             'annotations':[],'proposals':[],'proposal_states':{},
             'class_visibility':{n:'unreviewed' for n in CLASS_NAMES},'annotation_complete':False,'notes':''}
        draft={'schema':'yolo_target_label_review_v1','pilot_id':'p','protocol':PROTOCOL,
               'classes':CLASS_NAMES,'images':[row]}
        self.assertEqual(validate_review(draft,draft)['training_ready_images'],0)
        review=deepcopy(draft);r=review['images'][0]
        r.update(status='reviewed',scene='transition',annotation_complete=True,
                 class_visibility={n:'absent' for n in CLASS_NAMES})
        with self.assertRaises(ValueError):validate_review(draft,review)


if __name__=='__main__':unittest.main()
