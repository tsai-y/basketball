from copy import deepcopy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from development.yolo_train_probe_v1.guided_expansion import accept_image,accept_image_excluding,mark_region_uncertain,partition_readiness,question_overlay
from development.yolo_audit_v1.target_label_pilot import CLASS_NAMES,PROTOCOL


class GuidedExpansionTests(unittest.TestCase):
    def fixture(self):
        meta={'image_id':'E030','image_sha256':'sha','proposals':[{'id':'D1','xyxy':[1,2,4,6],'confidence':.4}]}
        row={**deepcopy(meta),'status':'pending','scene':'unreviewed','annotations':[],
             'proposal_states':{'D1':'pending'},'class_visibility':{n:'unreviewed' for n in CLASS_NAMES},
             'annotation_complete':False,'notes':''}
        draft={'schema':'yolo_target_label_review_v1','pilot_id':'p','classes':CLASS_NAMES,'protocol':PROTOCOL,'images':[row]}
        ai={'image_id':'E030','scene_hint':'court_play','uncertain_regions':[],
            'annotations':[{'id':'P1','class_id':0,'xyxy':[1,1,20,40]},
                           {'id':'B1','class_id':1,'xyxy':[20,20,25,25]}]}
        question={'image_id':'E030','question_id':'E030_whole_image_boxes_v2','response':None,
                  'question':'all boxes and no rim acceptable?','confirmation_scope':'Whole image and all visible objects'}
        event={'image_sha256':'sha','recorded_at_utc':'2026-10-25T03:54:43Z'}
        return draft,ai,meta,question,event

    def test_full_acceptance_records_scope_counts_and_preserves_draft(self):
        draft,ai,meta,q,e=self.fixture();before=deepcopy(draft)
        result=accept_image(draft,draft,ai,meta,q,'可接受',e)
        self.assertEqual(draft,before)
        self.assertTrue(result['images'][0]['annotation_complete'])
        self.assertEqual(result['images'][0]['class_visibility']['rim'],'absent')
        self.assertEqual(result['human_full_image_confirmation_events'][0]['accepted_counts'],
                         {'active_player':1,'ball':1,'rim':0})

    def test_full_acceptance_clears_stale_ball_pending_hint(self):
        draft,ai,meta,q,e=self.fixture();review=deepcopy(draft)
        review['images'][0]['ball_localization_status']='B1_rejected_new_B2_location_reported_exact_box_pending'
        out=accept_image(draft,review,ai,meta,q,'可接受',e)
        self.assertEqual(out['images'][0]['ball_localization_status'],'confirmed_in_whole_image_review')

    def test_identity_reply_or_unresolved_region_cannot_be_full_approval(self):
        draft,ai,meta,q,e=self.fixture()
        with self.assertRaises(ValueError):accept_image(draft,draft,ai,meta,q,'是',e)
        q['question_id']='E030_Q1_identity'
        with self.assertRaises(ValueError):accept_image(draft,draft,ai,meta,q,'可接受',e)
        draft,ai,meta,q,e=self.fixture();ai['uncertain_regions']=[{'id':'Q1'}]
        with self.assertRaises(ValueError):accept_image(draft,draft,ai,meta,q,'可接受',e)

    def test_wrong_image_or_duplicate_approval_rejected(self):
        draft,ai,meta,q,e=self.fixture();e['image_sha256']='foreign'
        with self.assertRaises(ValueError):accept_image(draft,draft,ai,meta,q,'可接受',e)
        e['image_sha256']='sha';out=accept_image(draft,draft,ai,meta,q,'可接受',e)
        with self.assertRaises(ValueError):accept_image(draft,out,ai,meta,q,'可接受',e)

    def test_named_rejection_accepts_only_unchanged_remaining_boxes(self):
        draft,ai,meta,q,e=self.fixture()
        ai['annotations'].append({'id':'P9','class_id':0,'xyxy':[0,50,5,60]})
        before=deepcopy(ai)
        out=accept_image_excluding(draft,draft,ai,meta,q,'P9不是，其餘匡可接受',e,['P9'])
        self.assertEqual(ai,before)
        self.assertEqual([a['id'] for a in out['images'][0]['annotations']],['P1','B1'])
        event=out['human_full_image_confirmation_events'][-1]
        self.assertEqual(event['user_reply'],'P9不是，其餘匡可接受')
        self.assertEqual(event['rejected_annotation_ids'],['P9'])
        self.assertEqual(event['accepted_counts']['active_player'],1)
        self.assertTrue(out['images'][0]['annotation_complete'])

    def test_rejection_without_remaining_acceptance_or_unknown_id_fails(self):
        draft,ai,meta,q,e=self.fixture()
        for answer,ids in [('P1不是',['P1']),('P1不確定，其餘匡可接受',['P1']),
                           ('P9不是，其餘匡可接受',['P9']),('B1不是，其餘匡可接受',['B1'])]:
            with self.subTest(answer=answer),self.assertRaises(ValueError):
                accept_image_excluding(draft,draft,ai,meta,q,answer,e,ids)

    def test_box_acceptance_typo_keeps_exact_reply_and_visible_rim(self):
        draft,ai,meta,q,e=self.fixture()
        ai['annotations'].append({'id':'R1','class_id':2,'xyxy':[30,10,45,15]})
        q['question']='all boxes including rim acceptable?'
        out=accept_image(draft,draft,ai,meta,q,'匡可接受',e)
        event=out['human_full_image_confirmation_events'][0]
        self.assertEqual(event['user_reply'],'匡可接受')
        self.assertEqual(event['accepted_counts']['rim'],1)
        self.assertEqual(out['images'][0]['class_visibility']['rim'],'present')
        q['question_id']='E030_Q1_identity'
        with self.assertRaises(ValueError):accept_image(draft,draft,ai,meta,q,'匡可接受',e)

    def test_annotation_complete_validation_is_not_training_membership(self):
        status={'training_ready_ids':['E017','E030']}
        plan={'records':[{'image_id':'E017','detector_partition':'detector_dev_validation'},
                         {'image_id':'E030','detector_partition':'detector_train'}]}
        result=partition_readiness(status,plan)
        self.assertEqual(result['annotation_complete_images'],2)
        self.assertEqual(result['detector_train_ready_ids'],['E030'])
        self.assertEqual(result['detector_dev_validation_ready_ids'],['E017'])

    def test_confirmation_uses_displayed_overlay_version_not_stale_v2(self):
        with TemporaryDirectory() as folder:
            p=Path(folder);(p/'ai').mkdir()
            for name in ('E022_overlay_v2.png','E022_overlay_v3.png','E015_overlay_v2.png'):
                (p/'ai'/name).write_bytes(b'fixture')
            q={'image_id':'E022','overlay_path':'ai/E022_overlay_v3.png'}
            self.assertEqual(question_overlay(p,'E022',q),p/'ai/E022_overlay_v3.png')
            self.assertEqual(question_overlay(p,'E022',{'image_id':'E022'}),p/'ai/E022_overlay_v2.png')
            q['overlay_path']='ai/E015_overlay_v2.png'
            with self.assertRaises(ValueError):question_overlay(p,'E022',q)

    def test_uncertain_observation_never_becomes_absent_or_complete(self):
        draft,ai,meta,q,e=self.fixture()
        q.update(question_id='E030_Q1_ball_visibility',region_id='Q1')
        before=deepcopy(draft)
        out=mark_region_uncertain(draft,draft,q,'不確定，疑似被球員擋住',e)
        self.assertEqual(draft,before)
        row=out['images'][0]
        self.assertEqual(row['class_visibility']['ball'],'uncertain')
        self.assertEqual(row['annotations'],[])
        self.assertEqual(row['status'],'pending')
        self.assertFalse(row['annotation_complete'])
        event=out['human_region_observation_events'][0]
        self.assertFalse(event['whole_image_confirmed'])
        self.assertFalse(event['occlusion_cause_confirmed'])
        with self.assertRaises(ValueError):mark_region_uncertain(draft,draft,q,'可接受',e)


if __name__=='__main__':unittest.main()
