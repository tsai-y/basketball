import copy
import unittest

from development.yolo_audit_v1.tests.test_target_label_pilot import fixture, completed
from development.yolo_train_probe_v1.probe import select_human_complete, xyxy_to_yolo
from development.yolo_train_probe_v1.check_predictions import box_iou, matched_counts


class ProbeTests(unittest.TestCase):
    def test_normalized_label_and_round_trip(self):
        line=xyxy_to_yolo(1,[160,5,168,13])
        cls,cx,cy,w,h=map(float,line.split())
        self.assertEqual(cls,1)
        actual=[(cx-w/2)*320,(cy-h/2)*240,(cx+w/2)*320,(cy+h/2)*240]
        for a,b in zip(actual,[160,5,168,13]):self.assertAlmostEqual(a,b,places=5)

    def test_invalid_geometry_and_class_rejected(self):
        for cls,box in [(3,[0,0,5,5]),(True,[0,0,5,5]),(0,[-1,0,5,5]),(1,[0,0,0,5])]:
            with self.subTest(cls=cls,box=box), self.assertRaises(ValueError):xyxy_to_yolo(cls,box)

    def base(self):
        d=fixture()
        manifest={'protocol':d['protocol'],'classes':d['classes'],'pilot_id':d['pilot_id'],
            'source_videos':[{'sample_id':'Y001','split':'train','excluded':'0','source_group':'g'}],
            'images':[{'image_id':'L001','audit_sample_id':'Y001','image_sha256':'pixels'}]}
        return manifest,d

    def test_AI_pending_rows_never_become_labels(self):
        m,d=self.base()
        with self.assertRaisesRegex(ValueError,'at least two'):select_human_complete(m,d,copy.deepcopy(d))

    def test_non_train_source_rejected_before_images_open(self):
        for split in ('validation','test'):
            m,d=self.base();m['source_videos'][0]['split']=split
            with self.subTest(split=split), self.assertRaisesRegex(ValueError,'original train'):
                select_human_complete(m,d,copy.deepcopy(d))

    def test_reviewed_without_human_whole_image_event_rejected(self):
        m,d=self.base();r=completed(d)
        with self.assertRaisesRegex(ValueError,'explicit whole-image human'):select_human_complete(m,d,r)

    def test_uncertain_ball_cannot_be_exported(self):
        m,d=self.base();r=completed(d);r['images'][0]['class_visibility']['ball']='uncertain'
        with self.assertRaisesRegex(ValueError,'uncertain'):select_human_complete(m,d,r)

    def test_matching_needs_same_class_and_is_one_to_one(self):
        gt=[{'class_id':1,'xyxy':[0,0,10,10]}]
        pred=[{'class_id':0,'xyxy':[0,0,10,10],'confidence':.99},
              {'class_id':1,'xyxy':[0,0,10,10],'confidence':.8},
              {'class_id':1,'xyxy':[0,0,10,10],'confidence':.7}]
        matches=matched_counts(gt,pred)
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0]['prediction_index'],1)
        self.assertEqual(box_iou([0,0,10,10],[20,20,30,30]),0)


if __name__=='__main__':unittest.main()
