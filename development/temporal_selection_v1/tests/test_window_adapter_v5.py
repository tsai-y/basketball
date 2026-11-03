from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from development.data_pipeline.pipeline import GateError
from development.data_pipeline.multiwindow_v1 import select_covering_windows
from development.temporal_selection_v1.selector_v5 import ExposurePolicyV5
from development.temporal_selection_v1.window_adapter_v1 import IntervalRequest
from development.temporal_selection_v1.window_adapter_v5 import apply_v5_request,V5DiagnosticThreeWindowAdapter


class V5WindowAdapterTests(unittest.TestCase):
    def item(self):
        return {'video_id':'v','frame_count':146,'windows':select_covering_windows(146),
                'tensor':torch.zeros(3,3,64,2,2),'binary_label':1,'fps':24.,'split':'train',
                'candidate_overlap_frames':torch.tensor([1,2,3]),'policy_version':'original',
                'quality_aware_selection_used':False}

    def record(self):
        return {'video_id':'v','frame_count':146,'original_split':'train','v5':{
            'policy':asdict(ExposurePolicyV5()),'training_use_approved':False,
            'all_segments_including_short':[
                {'start_frame':0,'end_frame_exclusive':37,'inside_proposed_change_span':True},
                {'start_frame':37,'end_frame_exclusive':142,'inside_proposed_change_span':False},
                {'start_frame':142,'end_frame_exclusive':146,'inside_proposed_change_span':False}]}}

    def request(self,interval):
        return IntervalRequest(interval,review_scope='explicit_input_smoke_only',target_identity_confirmed=True)

    def test_cross_scene_request_cannot_be_forced_by_opt_in(self):
        old=self.item();new=apply_v5_request(object(),0,old,self.request((0,100)),self.record(),'a'*64)
        self.assertEqual(new['adaptation']['mode'],'fallback_original')
        self.assertIn('not_exact_frozen_v5',new['adaptation']['reason'])
        self.assertIs(new['tensor'],old['tensor']);self.assertEqual(new['policy_version'],'original')
        self.assertEqual(new['adaptation']['originating_request']['interval'],(0,100))

    def test_identity_and_uncertain_fallback_keep_every_original_field(self):
        for request in (self.request((0,146)),IntervalRequest(uncertainty_reasons=('v5_recovery_unconfirmed',))):
            old=self.item();new=apply_v5_request(object(),0,old,request,self.record(),'a'*64)
            for key in old:self.assertIs(new[key],old[key])
            self.assertTrue(new['adaptation']['fallback_same_tensor_object'])
            self.assertFalse(new['adaptation']['production_crop_adopted'])

    def test_short_span_preserves_original_policy(self):
        old=self.item();new=apply_v5_request(object(),0,old,self.request((142,146)),self.record(),'a'*64)
        self.assertEqual(new['adaptation']['reason'],'interval_shorter_than_66')
        self.assertIs(new['tensor'],old['tensor'])

    def test_crop_uses_absolute_rgb_reference_coordinates_and_v5_provenance(self):
        calls=[]
        def decoder(row,guard,*,selection,expected_frame_count,expected_fps_rational):
            a,b=selection['start_frame'],selection['end_frame_exclusive'];calls.append((a,b))
            clip=np.empty((64,2,2,3),np.uint8)
            clip[:,:,:,0]=np.arange(a,b,dtype=np.uint8)[:,None,None]
            clip[:,:,:,1]=50;clip[:,:,:,2]=200
            return clip,{},selection,146,24.
        cfg={'version':'video_clip_transforms_v1','clip_length':64,'output_height':2,'output_width':2,
             'brightness':[1,1],'contrast':[1,1],'saturation':[1,1],'spatial_scale':[1,1],'enhancement_gamma':1.}
        base=SimpleNamespace(rows=[{'video_id':'v','binary_label':'1'}],split='train',epoch=0,decoder=decoder,guard=None,
            config=SimpleNamespace(raw={'seed':5,'transform':{**cfg,'policy':'reference'}}),
            quality_index=SimpleNamespace(summaries={'v':{'frame_count':'146','fps_rational':'24'}},segments={}))
        old=self.item();new=apply_v5_request(SimpleNamespace(base=base),0,old,self.request((37,142)),self.record(),'a'*64)
        self.assertEqual(calls,[(37,101),(57,121),(78,142)])
        self.assertEqual(new['policy_version'],'fixed_v5_candidate_three_windows_input_smoke_only_v1')
        self.assertEqual(new['binary_label'],1)
        for wi,(a,b) in enumerate(calls):
            torch.testing.assert_close(new['tensor'][wi,0,:,0,0],torch.arange(a,b).float()/255,rtol=0,atol=0)
            self.assertTrue(torch.all(new['tensor'][wi,1]==50/255))
            self.assertTrue(torch.all(new['tensor'][wi,2]==200/255))
        self.assertFalse(new['adaptation']['event_completeness_verified'])
        self.assertFalse(new['adaptation']['production_crop_adopted'])
        self.assertEqual(new['adaptation']['source_report_sha256'],'a'*64)

    def test_wrong_version_or_source_identity_rejected(self):
        for key,value in (('video_id','other'),('frame_count',145)):
            r=self.record();r[key]=value
            with self.assertRaises(ValueError):apply_v5_request(object(),0,self.item(),None,r,'a'*64)
        r=self.record();r['v5']['policy']['luma_peak']=.05
        with self.assertRaises(ValueError):apply_v5_request(object(),0,self.item(),None,r,'a'*64)

    def test_nontrain_and_bad_version_rejected_before_original_constructor(self):
        for split in ('test','validation'):
            with patch('development.temporal_selection_v1.window_adapter_v5.BasketballThreeWindowDataset') as old:
                with self.assertRaises(GateError):V5DiagnosticThreeWindowAdapter(None,split=split,video_ids=[],v5_records={},source_report_sha256='a'*64)
                old.assert_not_called()
        r=self.record();r['v5']['policy']['version']='v4'
        with patch('development.temporal_selection_v1.window_adapter_v5.BasketballThreeWindowDataset') as old:
            with self.assertRaises(ValueError):V5DiagnosticThreeWindowAdapter(None,split='train',video_ids=['v'],v5_records={'v':r},source_report_sha256='a'*64)
            old.assert_not_called()


if __name__=='__main__':unittest.main()
