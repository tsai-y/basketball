import unittest
from copy import deepcopy
from development.yolo_train_probe_v1.ball_source_review_v2 import select_sources,check_proposal,validate_pending,corrected_location_proposal,accept_other_ball_images,accept_current_ball_image,PROTOCOL


class BallSourceReviewTests(unittest.TestCase):
    def fixture(self):
        rows=[];assignment={}
        for i,cls in enumerate(('2p0','2p1','3p0','3p1','ft0','mp1')):
            g=f'new{i}';rows.append({'video_id':f'v{i}.mp4','source_group':g,'split':'train','excluded':'0','original_class':cls})
            assignment[g]='detector_train' if i<4 else 'detector_dev_validation'
        return rows,assignment
    def test_four_training_two_reserved_sources(self):
        rows,a=self.fixture();s=select_sources(rows,a,set())
        self.assertEqual(len(s),6);self.assertEqual({r['original_class'] for r in s[:4]},{'2p0','2p1','3p0','3p1'})
        self.assertTrue(all(a[r['source_group']]=='detector_dev_validation' for r in s[4:]))
        self.assertEqual(s,select_sources(list(reversed(rows)),a,set()))
    def test_used_source_not_reintroduced(self):
        rows,a=self.fixture()
        with self.assertRaises(ValueError):select_sources(rows,a,{'new0'})
    def test_original_validation_and_duplicate_fail(self):
        rows,a=self.fixture();rows[0]['split']='validation'
        with self.assertRaises(ValueError):select_sources(rows,a,set())
        rows,a=self.fixture();rows.append(deepcopy(rows[0]))
        with self.assertRaises(ValueError):select_sources(rows,a,set())
    def test_reserved_source_cannot_fill_train_slot(self):
        rows,a=self.fixture();a['new0']='detector_dev_validation'
        with self.assertRaises(ValueError):select_sources(rows,a,set())
    def test_native_box_validation_and_empty_not_negative(self):
        check_proposal([[1,2,7,9]])
        for boxes in ([],[[0,0,321,20]],[[4,4,2,8]],[[1.,2,8,9]],[[1,2,8,9],[1,2,8,9]]):
            with self.assertRaises(ValueError):check_proposal(boxes)

    def test_AI_draft_cannot_become_ready_or_human_confirmed(self):
        r={'ball_boxes_proposed':[[1,2,7,9]],'status':'pending_human_ball_only_full_image_review',
           'human_confirmed':False,'visible_ball_completeness_confirmed':False,'training_ready':False}
        validate_pending(r)
        for field in ('human_confirmed','visible_ball_completeness_confirmed','training_ready'):
            bad=deepcopy(r);bad[field]=True
            with self.assertRaises(ValueError):validate_pending(bad)

    def test_location_correction_does_not_accept_box_or_other_images(self):
        ai={'origin':'AI_proposals_not_ground_truth','selected':[
            {'image_id':'B001','ball_boxes':[[1,2,7,9]]},
            {'image_id':'B006','ball_boxes':[[80,114,89,124]]}]}
        before=deepcopy(ai)
        updated,event=corrected_location_proposal(ai,'B006',[[269,113,277,121]],'B006球再這')
        self.assertEqual(ai,before);self.assertEqual(updated['selected'][0],ai['selected'][0])
        self.assertEqual(updated['selected'][1]['ball_boxes'],[[269,113,277,121]])
        self.assertFalse(event['ball_box_boundary_confirmed']);self.assertFalse(event['whole_image_confirmed'])
        self.assertFalse(event['batch_confirmed']);self.assertEqual(event['training_ready_images'],0)

    def test_unknown_correction_and_missing_reply_rejected(self):
        ai={'origin':'AI_proposals_not_ground_truth','selected':[{'image_id':'B006','ball_boxes':[[1,2,7,9]]}]}
        for iid,reply in [('B999','球在這'),('B006','')]:
            with self.assertRaises(ValueError):corrected_location_proposal(ai,iid,[[2,3,8,10]],reply)

    def test_other_images_reply_excludes_corrected_B006(self):
        rows=[{'image_id':iid,'ball_boxes_proposed':[[1,2,7,9]],
               'status':'pending_human_ball_only_full_image_review','human_confirmed':False,
               'visible_ball_completeness_confirmed':False,'training_ready':False,
               'detector_partition':'development' if iid=='B029' else 'train'}
              for iid in ('B001','B006','B012','B029')]
        d={'protocol':PROTOCOL,'images':rows,'human_confirmation_events':[]};before=deepcopy(d)
        p=accept_other_ball_images(d,'B006','其餘都正常')
        self.assertEqual(d,before);self.assertEqual(p['annotation_complete_images'],3)
        self.assertEqual(p['detector_train_confirmed_ids'],['B001','B012'])
        self.assertEqual(p['detector_development_confirmed_ids'],['B029'])
        self.assertEqual(p['pending_ids'],['B006']);validate_pending(p['images'][1])
        self.assertFalse(p['human_confirmation_events'][-1]['batch_fully_confirmed'])
        with self.assertRaises(ValueError):accept_other_ball_images(d,'B999','其餘都正常')
        with self.assertRaises(ValueError):accept_other_ball_images(d,'B006','正常')

    def pending_corrected(self):
        r={'image_id':'B006','ball_boxes_proposed':[[269,113,277,121]],
           'status':'pending_human_ball_only_full_image_review','human_confirmed':False,
           'visible_ball_completeness_confirmed':False,'training_ready':False,'detector_partition':'train',
           'image_sha256':'native','shown_panel':'panels/B006.png','shown_panel_sha256':'corrected-v2'}
        q={'image_id':'B006','response':None,'shown_panel':r['shown_panel'],'shown_panel_sha256':r['shown_panel_sha256'],
           'scope':'B006 corrected box boundaries and full-image visible-ball completeness only'}
        return {'protocol':PROTOCOL,'images':[r],'pending_ids':['B006'],'next_human_question':q,'human_confirmation_events':[]}

    def test_current_reply_accepts_corrected_version_only(self):
        p=self.pending_corrected();old=deepcopy(p);r=accept_current_ball_image(p,'可接受')
        self.assertEqual(p,old);self.assertTrue(r['images'][0]['human_confirmed'])
        self.assertEqual(r['images'][0]['annotations'][0]['xyxy'],[269,113,277,121])
        self.assertEqual(r['pending_ids'],[]);self.assertIsNone(r['next_human_question'])
        self.assertFalse(r['human_confirmation_events'][-1]['other_images_reconfirmed_by_this_reply'])
        with self.assertRaises(ValueError):accept_current_ball_image(r,'可接受')

    def test_stale_display_or_location_only_question_rejected(self):
        p=self.pending_corrected();p['next_human_question']['shown_panel_sha256']='old-v1'
        with self.assertRaises(ValueError):accept_current_ball_image(p,'可接受')
        p=self.pending_corrected();p['next_human_question']['scope']='location only'
        with self.assertRaises(ValueError):accept_current_ball_image(p,'可接受')


if __name__=='__main__':unittest.main()
