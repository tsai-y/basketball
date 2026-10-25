import unittest
from unittest.mock import MagicMock
import numpy as np
from development.yolo_train_probe_v1.unseen_video_audit import (
    select_samples,detections_from_arrays,draw_pair,summarize,process_video,
)


class UnseenAuditTests(unittest.TestCase):
    def fixture(self):
        return {'split':'train','samples':[{'sample_id':f'Y{i:03d}','video_id':f'{i}.mp4',
            'source_group':f'g{i}','split':'train','excluded':'0'} for i in range(14)]}

    def test_selection_excludes_entire_training_sources_not_just_image_ids(self):
        a=self.fixture();t={'source_groups':['g0','g1'],'records':[{'video_id':'other0.mp4'}]}
        selected=select_samples(a,t)
        self.assertEqual(len(selected),12)
        self.assertFalse({'g0','g1'} & {r['source_group'] for r in selected})
        self.assertEqual(selected,select_samples(a,t))

    def test_nontrain_rejected_before_path_access(self):
        guard=MagicMock();model=MagicMock()
        for row in ({'split':'test'},{'split':'validation'}, {'split':'train','source_group':'trained'}):
            with self.subTest(row=row),self.assertRaises(ValueError):process_video(row,guard,model,None,{'trained'})
        guard.identity.assert_not_called();guard.open.assert_not_called();model.predict.assert_not_called()

    def test_bad_audit_and_insufficient_unseen_samples_rejected(self):
        a=self.fixture();a['samples'][0]['split']='test'
        with self.assertRaises(ValueError):select_samples(a,{'source_groups':[],'records':[]})
        with self.assertRaises(ValueError):select_samples(self.fixture(),{'source_groups':[f'g{i}' for i in range(10)],'records':[]})

    def test_detection_classes_finite_and_native_bounds(self):
        boxes=[[0,0,10,10],[11,12,15,16],[20,30,40,35]]
        d=detections_from_arrays(boxes,[0.,1.,2.],[.3,.5,.9])
        self.assertEqual([x['detection_id'] for x in d],['P1','B1','R1'])
        self.assertTrue(all(x['track_id'] is None for x in d))
        for cls,box,score in [(3,[0,0,2,2],.5),(0,[0,0,321,2],.5),(1,[0,0,2,2],float('nan'))]:
            with self.subTest(cls=cls),self.assertRaises(ValueError):detections_from_arrays([box],[cls],[score])

    def test_pair_preserves_source_pixels_and_presence_is_not_recall(self):
        raw=np.full((240,320,3),100,dtype=np.uint8);copy=raw.copy()
        d=detections_from_arrays([[10,10,30,50]],[0],[.6])
        pair=draw_pair(raw,d,'U001',10,30)
        self.assertEqual(pair.shape,(264,640,3))
        np.testing.assert_array_equal(raw,copy);np.testing.assert_array_equal(pair[24:,:320],copy)
        s=summarize([{'detections':d},{'detections':[]}])
        self.assertEqual(s['active_player']['frames_with_detection'],1)
        self.assertEqual(s['ball']['frames_without_detection'],2)
        self.assertNotIn('recall',s['ball'])


if __name__=='__main__':unittest.main()
