from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from development.data_pipeline.pipeline import GateError
from development.data_pipeline.multiwindow_v1 import select_covering_windows
from development.temporal_selection_v1.window_adapter_v1 import (
    IntervalRequest, resolve_interval, apply_request, DiagnosticThreeWindowAdapter,
)


class WindowAdapterTests(unittest.TestCase):
    def request(self,interval,**kwargs):
        return IntervalRequest(interval=interval,review_scope='explicit_input_smoke_only',target_identity_confirmed=True,**kwargs)

    def test_translate_exact_half_open_contiguous_windows(self):
        d=resolve_interval(146,self.request((37,142)))
        self.assertEqual(d['mode'],'diagnostic_candidate')
        self.assertEqual(d['relative_windows'],((0,64),(20,84),(41,105)))
        self.assertEqual(d['absolute_windows'],((37,101),(57,121),(78,142)))
        self.assertEqual({i for a,b in d['absolute_windows'] for i in range(a,b)},set(range(37,142)))
        self.assertFalse(d['training_use_approved'])

    def test_coverage_all_supported_lengths_and_limits(self):
        for n in range(66,193):
            d=resolve_interval(240,self.request((10,10+n)))
            self.assertEqual(d['mode'],'diagnostic_candidate')
            self.assertEqual({i for a,b in d['absolute_windows'] for i in range(a,b)},set(range(10,10+n)))
        d=resolve_interval(240,self.request((10,203)))
        self.assertEqual(d['reason'],'three_windows_cannot_cover_interval')

    def test_short_unknown_unreviewed_and_invalid_fall_back(self):
        for interval in ((0,6),(0,64),(0,65)):
            self.assertEqual(resolve_interval(146,self.request(interval))['reason'],'interval_shorter_than_66')
        for interval in ((-1,70),(20,147),(20,20),(True,70),(0,70,80)):
            self.assertEqual(resolve_interval(146,self.request(interval))['mode'],'fallback_original')
        self.assertEqual(resolve_interval(146,IntervalRequest((37,142)))['mode'],'fallback_original')
        r=self.request((0,89),uncertainty_reasons=('source_start_uncertain',))
        self.assertTrue(resolve_interval(146,r)['reason'].startswith('uncertain:'))

    def test_fallback_is_same_tensor_and_all_original_fields_no_redecode(self):
        item={'tensor':torch.arange(24).reshape(2,3,4),'frame_count':146,
              'windows':select_covering_windows(146),'binary_label':0,'video_id':'x',
              'candidate_overlap_frames':torch.tensor([1,2,3]),'policy_version':'original'}
        # A fallback must not even require a base decoder or quality index.
        result=apply_request(object(),0,item,IntervalRequest(uncertainty_reasons=('multiple_events',)))
        for key in item:self.assertIs(result[key],item[key])
        self.assertEqual(result['adaptation']['absolute_windows'],item['windows'])
        self.assertEqual(item['policy_version'],'original')

    def test_identity_same_tensor(self):
        item={'tensor':torch.zeros(1),'frame_count':146,'windows':select_covering_windows(146)}
        result=apply_request(object(),0,item,self.request((0,146)))
        self.assertIs(result['tensor'],item['tensor']);self.assertEqual(result['adaptation']['mode'],'identity_original')

    def test_crop_preserves_rgb_order_absolute_indices_label_and_no_temporal_resample(self):
        calls=[]
        def decoder(row,guard,*,selection,expected_frame_count,expected_fps_rational):
            a,b=selection['start_frame'],selection['end_frame_exclusive'];calls.append((a,b))
            clip=np.empty((64,2,2,3),dtype=np.uint8)
            clip[:,:,:,0]=np.arange(a,b,dtype=np.uint8)[:,None,None]
            clip[:,:,:,1]=50;clip[:,:,:,2]=200
            return clip,{'decoded_frame_indices':list(range(a,b))},selection,146,24.
        cfg={'version':'video_clip_transforms_v1','clip_length':64,'output_height':2,'output_width':2,
             'brightness':[1,1],'contrast':[1,1],'saturation':[1,1],'spatial_scale':[1,1],'enhancement_gamma':1.}
        row={'video_id':'v','binary_label':'1'}
        base=SimpleNamespace(rows=[row],split='train',epoch=0,decoder=decoder,guard=None,
             config=SimpleNamespace(raw={'seed':5,'transform':{**cfg,'policy':'reference'}}),
             quality_index=SimpleNamespace(summaries={'v':{'frame_count':'146','fps_rational':'24'}},segments={}))
        item={'tensor':torch.zeros(3,3,64,2,2),'frame_count':146,'fps':24.,'video_id':'v','binary_label':1,
              'windows':select_covering_windows(146),'split':'train','quality_aware_selection_used':False}
        result=apply_request(SimpleNamespace(base=base),0,item,self.request((37,142)))
        self.assertEqual(calls,[(37,101),(57,121),(78,142)])
        self.assertEqual(result['binary_label'],1)
        self.assertEqual(tuple(result['tensor'].shape),(3,3,64,2,2))
        for i,(a,b) in enumerate(calls):
            torch.testing.assert_close(result['tensor'][i,0,:,0,0],torch.arange(a,b).float()/255,rtol=0,atol=0)
            self.assertTrue(torch.all(result['tensor'][i,1]==50/255))
            self.assertTrue(torch.all(result['tensor'][i,2]==200/255))
        self.assertFalse(result['adaptation']['temporal_resampling_used'])

    def test_nontrain_rejected_before_constructing_original(self):
        for split in ('test','validation'):
            with patch('development.temporal_selection_v1.window_adapter_v1.BasketballThreeWindowDataset') as original:
                with self.assertRaises(GateError):DiagnosticThreeWindowAdapter(None,split=split,video_ids=[])
                original.assert_not_called()


if __name__=='__main__':unittest.main()
