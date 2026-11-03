import unittest
from copy import deepcopy
import numpy as np
from development.yolo_train_probe_v1.ball_negative_crops_v1 import source_balls,choose_crop,crop_pixels,intersection_area


class NegativeCropTests(unittest.TestCase):
    def record(self,box=(209,88,220,101)):
        return {'source_split':'train','partition':'train',
                'original_human_accepted_annotations':[{'id':'B1','class_id':1,'xyxy':list(box),'human_confirmed':True}],
                'human_full_image_confirmation':{'whole_image_confirmed':True,'user_reply':'可接受',
                    'source':'direct_human_chat_reply','annotation_complete':True,
                    'accepted_counts':{'ball':1},'class_visibility':{'ball':'present'}}}

    def test_deterministic_safe_and_nonmutating(self):
        r=self.record();before=deepcopy(r);p=choose_crop(r)
        self.assertEqual(p,choose_crop(r));self.assertEqual(r,before)
        self.assertEqual(p['crop_xyxy'],[0,0,160,120]);self.assertEqual(p['intersection_with_each_expanded_ball'],[0])
        self.assertEqual(p['native_margin_px'],12)

    def test_dev_and_classification_nontrain_rejected(self):
        for field,value in [('partition','val'),('source_split','validation'),('source_split','test')]:
            r=self.record();r[field]=value
            with self.assertRaises(ValueError):choose_crop(r)

    def test_partial_or_uncertain_cannot_make_negative(self):
        for field,value in [('whole_image_confirmed',False),('user_reply',''),('class_visibility',{'ball':'uncertain'})]:
            r=self.record();r['human_full_image_confirmation'][field]=value
            with self.assertRaises(ValueError):source_balls(r)

    def test_count_mismatch_and_AI_only_rejected(self):
        r=self.record();r['human_full_image_confirmation']['accepted_counts']['ball']=2
        with self.assertRaises(ValueError):source_balls(r)

    def test_legacy_complete_event_not_AI_origin_alone(self):
        r=self.record();a=r['original_human_accepted_annotations'][0];a.pop('human_confirmed')
        a['origin']='AI_proposed_box_explicitly_confirmed_by_human'
        r['human_full_image_confirmation']={'source':'human_chat_reply','user_quote':'整張可接受',
            'whole_image_confirmed':True,'annotation_complete_confirmed':True,'accepted_counts':{'ball':1},
            'confirmation_scope':'full_image_three_classes_boxes_and_completeness'}
        self.assertEqual(len(source_balls(r)),1)
        r['human_full_image_confirmation']['annotation_complete_confirmed']=False
        with self.assertRaises(ValueError):source_balls(r)
        r=self.record();r['original_human_accepted_annotations'][0]['human_confirmed']=False
        with self.assertRaises(ValueError):source_balls(r)

    def test_all_balls_excluded_not_just_first(self):
        r=self.record((10,10,20,20));r['original_human_accepted_annotations'].append(
            {'id':'B2','class_id':1,'xyxy':[300,10,310,20],'human_confirmed':True})
        r['human_full_image_confirmation']['accepted_counts']['ball']=2
        p=choose_crop(r);self.assertEqual(len(p['intersection_with_each_expanded_ball']),2)
        self.assertTrue(all(v==0 for v in p['intersection_with_each_expanded_ball']))

    def test_no_safe_crop_defers_without_reducing_margin(self):
        with self.assertRaisesRegex(ValueError,'No safe'):choose_crop(self.record((0,0,320,240)))

    def test_native_pixels_no_resize_or_alias(self):
        a=np.arange(240*320*3,dtype=np.uint8).reshape(240,320,3);b=crop_pixels(a,[0,0,160,120])
        self.assertEqual(b.shape,(120,160,3));self.assertTrue(np.array_equal(a[:120,:160],b))
        b[:]=0;self.assertTrue(a[:120,:160].any())

    def test_invalid_geometry_and_wrong_source_fail(self):
        for box in [(-1,0,20,20),(2,2,1,3),(float('nan'),1,4,5)]:
            with self.assertRaises(ValueError):choose_crop(self.record(box))
        with self.assertRaises(ValueError):crop_pixels(np.zeros((120,160,3),np.uint8),[0,0,160,120])
        self.assertEqual(intersection_area([0,0,2,2],[2,0,3,2]),0)


if __name__=='__main__':unittest.main()
