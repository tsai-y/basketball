import unittest
from unittest.mock import patch
import numpy as np
import torch
from .data import ResolutionDataset,check_resolution,decode_pair,normalize
from .probe import config
from development.aca_net_v1.data import normalize_rgb,uniform_indices
from development.aca_net_v1.model import VideoResNet50


class ResolutionTests(unittest.TestCase):
    def test_test_split_rejected_before_path_or_scope(self):
        with patch('development.aca_resolution_v1.data.VideoScope',side_effect=AssertionError('scope accessed')):
            with self.assertRaises(ValueError):ResolutionDataset(source_root='/forbidden',split='test',resolution=(192,256))

    def test_authorization_before_open(self):
        class Forbidden:
            def authorize(self,_):raise ValueError('forbidden')
        with patch('development.aca_resolution_v1.data.gate1a.open_guarded',side_effect=AssertionError('open')):
            with self.assertRaises(ValueError):decode_pair(Forbidden(),'test.mp4')

    def test_only_two_resolutions(self):
        for size in ((120,160),(192,256)):self.assertEqual(check_resolution(size),size)
        for size in ((160,120),(192.,256),(224,224),(50,192,256)):
            with self.assertRaises(ValueError):check_resolution(size)

    def test_control_normalization_bitwise_equal(self):
        rgb=np.random.default_rng(5).integers(0,256,(50,120,160,3),dtype=np.uint8)
        self.assertTrue(torch.equal(normalize(rgb),normalize_rgb(rgb)))

    def test_high_normalization_shape_and_finite(self):
        x=normalize(np.full((50,192,256,3),128,dtype=np.uint8))
        self.assertEqual(tuple(x.shape),(3,50,192,256));self.assertTrue(x.is_contiguous());self.assertTrue(torch.isfinite(x).all())

    def test_invalid_frame_count_dtype_or_shape(self):
        for x in (np.zeros((49,192,256,3),dtype=np.uint8),np.zeros((50,192,256,3)),np.zeros((50,224,224,3),dtype=np.uint8)):
            with self.assertRaises(ValueError):normalize(x)

    def test_uniform_sampling_unchanged(self):
        for n in (50,90,146,180):
            indices=uniform_indices(n);self.assertEqual(indices,tuple(i*(n-1)//49 for i in range(50)))

    def test_meta_both_spatial_shapes_same_parameters(self):
        for h,w in ((120,160),(192,256)):
            with torch.device('meta'):
                m=VideoResNet50(variant='aca',num_classes=2);self.assertEqual(sum(p.numel() for p in m.parameters()),93163870)
                self.assertEqual(tuple(m(torch.empty(1,3,50,h,w)).shape),(1,2))

    def test_protocol_has_no_formal_training(self):
        c=config();self.assertFalse(c['formal_training_enabled']);self.assertEqual(c['engineering_updates_per_resolution'],2)


if __name__=='__main__':unittest.main()
