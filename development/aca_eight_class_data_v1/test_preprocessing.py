"""Synthetic engineering checks only; no dataset clips or model evaluation."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from .preprocessing import (FRAMES,MEAN,STD,exact_time,nearest_index,
                            spatial_normalize,temporal_plan)

ROOT=Path(__file__).resolve().parents[2]


class TemporalContract(unittest.TestCase):
    def test_exact_25fps_matches_legacy_endpoint_indices(self):
        for count in (50,51,145,151,180):
            plan=temporal_plan([Fraction(i,25) for i in range(count)])
            self.assertEqual(plan['source_indices'],[i*(count-1)//49 for i in range(50)])

    def test_native_rate_fixtures_have_unique_in_bounds_indices(self):
        for rate,count in ((Fraction(2997,125),146),(Fraction(25),151),(Fraction(2997,100),180)):
            plan=temporal_plan([i/rate for i in range(count)])
            indices=plan['source_indices']
            self.assertEqual(len(set(indices)),50)
            self.assertTrue(all(0<=i<count for i in indices))
            self.assertTrue(all(a<b for a,b in zip(indices,indices[1:])))
            self.assertLessEqual(Fraction(plan['max_absolute_time_error_seconds']),1/(2*rate))

    def test_display_time_offset_does_not_change_plan(self):
        base=[Fraction(i,25) for i in range(151)]
        self.assertEqual(temporal_plan(base),temporal_plan([v+Fraction(901,7) for v in base]))

    def test_variable_display_intervals_are_used(self):
        times=[Fraction(i,25)+Fraction(i%2,1000) for i in range(151)]
        plan=temporal_plan(times)
        self.assertEqual(len(plan['source_indices']),50)
        self.assertEqual(plan['source_duration_seconds'],'6')
        self.assertEqual(plan['selected_source_seconds'],[str(times[i]) for i in plan['source_indices']])

    def test_nearest_tie_uses_earlier_frame(self):
        self.assertEqual(nearest_index([Fraction(0),Fraction(1,25)],Fraction(1,50)),0)

    def test_virtual_endpoint_not_after_last_source_pts(self):
        times=[Fraction(i,30) for i in range(180)]
        plan=temporal_plan(times)
        self.assertLessEqual(Fraction(plan['target_seconds'][-1]),times[-1])
        self.assertLess(times[-1]-Fraction(plan['target_seconds'][-1]),Fraction(1,25))

    def test_short_source_rejected(self):
        with self.assertRaisesRegex(ValueError,'Fewer than 50'):
            temporal_plan([Fraction(i,25) for i in range(49)])

    def test_insufficient_duration_rejected(self):
        with self.assertRaisesRegex(ValueError,'grid has fewer'):
            temporal_plan([Fraction(i,100) for i in range(100)])

    def test_missing_or_float_timestamp_rejected(self):
        for value in (None,float('nan'),0.04,True,'NaN','1/0'):
            with self.subTest(value=value),self.assertRaises(ValueError):exact_time(value)

    def test_nonmonotonic_and_duplicate_pts_rejected(self):
        base=[Fraction(i,25) for i in range(151)]
        for bad in (list(reversed(base)),base[:70]+[base[69]]+base[71:]):
            with self.assertRaisesRegex(ValueError,'strictly increasing'):temporal_plan(bad)

    def test_large_timestamp_gap_cannot_be_hidden_by_padding(self):
        with self.assertRaisesRegex(ValueError,'repeats a source frame'):
            temporal_plan([Fraction(i,100000) for i in range(49)]+[Fraction(6)])


class SpatialContract(unittest.TestCase):
    def test_known_bgr_color_becomes_correct_rgb_channels(self):
        image=np.empty((50,240,320,3),dtype=np.uint8);image[:]=[0,128,255]
        value=spatial_normalize(image)
        expected=[(1-MEAN[0])/STD[0],(128/255-MEAN[1])/STD[1],(0-MEAN[2])/STD[2]]
        for c in range(3):np.testing.assert_allclose(value[c],expected[c],rtol=0,atol=1e-6)
        self.assertEqual(value.shape,(3,50,120,160))
        self.assertEqual(value.dtype,np.float32);self.assertTrue(value.flags.c_contiguous)

    def test_area_downsampling_preserves_block_average(self):
        checker=((np.indices((240,320)).sum(0)%2)*255).astype(np.uint8)
        image=np.broadcast_to(checker[None,:,:,None],(50,240,320,3)).copy()
        value=spatial_normalize(image)
        for c in range(3):
            np.testing.assert_allclose(value[c],(128/255-MEAN[c])/STD[c],rtol=0,atol=1e-6)

    def test_black_and_white_normalization_bounds(self):
        for pixel in (0,255):
            value=spatial_normalize(np.full((50,240,320,3),pixel,dtype=np.uint8))
            for c in range(3):np.testing.assert_allclose(value[c],(pixel/255-MEAN[c])/STD[c],rtol=0,atol=1e-6)

    def test_invalid_dtype_time_and_size_rejected(self):
        for image in (np.zeros((50,240,320,3),dtype=np.float32),
                      np.zeros((49,240,320,3),dtype=np.uint8),
                      np.zeros((50,120,160,3),dtype=np.uint8)):
            with self.assertRaises(ValueError):spatial_normalize(image)

    def test_repeatability_and_input_not_mutated(self):
        image=np.random.default_rng(20261108).integers(0,256,(50,240,320,3),dtype=np.uint8)
        before=hashlib.sha256(image.tobytes()).hexdigest()
        first=spatial_normalize(image);second=spatial_normalize(image)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(before,hashlib.sha256(image.tobytes()).hexdigest())

    def test_contract_constants_and_dependency_versions_match(self):
        spec=json.loads((ROOT/'docs/research/specs/aca_eight_class_preprocessing_v1.json').read_text())
        self.assertEqual(spec['temporal']['output_frames'],FRAMES)
        self.assertEqual(spec['normalization']['mean'],list(MEAN))
        self.assertEqual(spec['normalization']['std'],list(STD))
        self.assertEqual(spec['runtime']['opencv'],cv2.__version__)
        self.assertEqual(spec['runtime']['numpy'],np.__version__)


if __name__=='__main__':unittest.main()
