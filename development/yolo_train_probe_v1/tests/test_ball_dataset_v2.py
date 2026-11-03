import unittest
from copy import deepcopy

from development.yolo_train_probe_v1.ball_dataset_v2 import (
    human_rows, validate_partitions, remaining_unlabeled,
)
from development.yolo_train_probe_v1.ball_source_review_v2 import PROTOCOL


class BallDatasetV2Tests(unittest.TestCase):
    def human_fixture(self):
        rows=[]
        for iid in ('B001','B006','B012','B029'):
            rows.append({'image_id':iid,'source_split':'train',
                'detector_partition':'development' if iid=='B029' else 'train',
                'ball_boxes_proposed':[[1,2,7,9]],'image_sha256':iid+'native',
                'shown_panel_sha256':iid+'panel','status':'pending_human_ball_only_full_image_review',
                'human_confirmed':False,'visible_ball_completeness_confirmed':False,'training_ready':False})
        draft={'protocol':PROTOCOL,'images':rows}
        progress=deepcopy(draft)
        progress.update(batch_annotation_complete=True,pending_ids=[],next_human_question=None,
            annotation_complete_images=4,training_ready_images=3,
            detector_train_confirmed_ids=['B001','B006','B012'],detector_development_confirmed_ids=['B029'])
        for r in progress['images']:
            r.update(status='human_confirmed_ball_only_full_image',human_confirmed=True,
                visible_ball_completeness_confirmed=True,annotation_complete=True,
                training_ready=r['detector_partition']=='train',annotations=[{
                    'id':'ball1','class_id':0,'xyxy':[1,2,7,9],
                    'origin':'AI_box_accepted_in_direct_human_ball_only_review'}])
        return progress,draft

    def test_scoped_four_acceptances_do_not_mutate_draft(self):
        p,d=self.human_fixture();before=deepcopy(d)
        accepted=human_rows(p,d)
        self.assertEqual(len(accepted),4);self.assertEqual(d,before)
        self.assertFalse(accepted[-1]['training_ready'])

    def test_pending_or_unconfirmed_batch_rejected(self):
        for field,value in [('human_confirmed',False),('annotation_complete',False),
                            ('visible_ball_completeness_confirmed',False),('status','pending')]:
            with self.subTest(field=field):
                p,d=self.human_fixture();p['images'][0][field]=value
                with self.assertRaises(ValueError):human_rows(p,d)
        p,d=self.human_fixture();p['pending_ids']=['B006']
        with self.assertRaises(ValueError):human_rows(p,d)

    def test_stale_geometry_pixels_or_class_rejected(self):
        for field,value in [('ball_boxes_proposed',[[2,2,8,9]]),('image_sha256','changed'),
                            ('shown_panel_sha256','stale')]:
            with self.subTest(field=field):
                p,d=self.human_fixture();p['images'][0][field]=value
                with self.assertRaises(ValueError):human_rows(p,d)
        p,d=self.human_fixture();p['images'][0]['annotations'][0]['class_id']=1
        with self.assertRaises(ValueError):human_rows(p,d)

    def test_development_not_promoted_to_training(self):
        p,d=self.human_fixture();p['images'][-1]['training_ready']=True
        with self.assertRaises(ValueError):human_rows(p,d)
        p,d=self.human_fixture();p['images'][-1]['detector_partition']='train'
        with self.assertRaises(ValueError):human_rows(p,d)

    def split_fixture(self):
        records=[{'image_id':str(i),'source_split':'train','partition':s,
            'source_group':'source'+str(i),'video_id':'video'+str(i),'image_sha256':'sha'+str(i),
            'annotations':[{'class_id':0,'xyxy':[1,2,7,9]}]} for i,s in enumerate(('train','val'))]
        assignments={'source0':'detector_train','source1':'detector_dev_validation'}
        return records,assignments

    def test_positive_ball_source_separation(self):
        r,a=self.split_fixture();validate_partitions(r,a,{'train':1,'val':1})
        for field in ('source_group','video_id','image_sha256'):
            with self.subTest(field=field):
                r,a=self.split_fixture();r[1][field]=r[0][field]
                with self.assertRaises(ValueError):validate_partitions(r,a)

    def test_nontrain_original_empty_or_other_class_labels_rejected(self):
        for field,value in [('source_split','validation'),('source_split','test'),
                            ('annotations',[]),('annotations',[{'class_id':1,'xyxy':[1,2,7,9]}])]:
            with self.subTest(field=field,value=value):
                r,a=self.split_fixture();r[0][field]=value
                with self.assertRaises(ValueError):validate_partitions(r,a)
        r,a=self.split_fixture();r.append(deepcopy(r[0]))
        with self.assertRaises(ValueError):validate_partitions(r,a)

    def test_diagnostics_remove_all_actual_train_and_development_sources(self):
        rows=[{'item_id':str(i),'source_group':g,'ground_truth':gt}
              for i,(g,gt) in enumerate((('newtrain',None),('newdev',None),('other',None),('olddev',[])))]
        before=deepcopy(rows)
        self.assertEqual([r['item_id'] for r in remaining_unlabeled(rows,{'newtrain','newdev'})],['2'])
        self.assertEqual(rows,before)
        with self.assertRaises(ValueError):remaining_unlabeled(rows+[deepcopy(rows[0])],set())


if __name__=='__main__':unittest.main()
