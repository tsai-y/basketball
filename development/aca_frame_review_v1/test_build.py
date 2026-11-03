import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from .build import contact_pages, decode_original, key_panel, reconstruct_mainline, timing


class FrameReviewTests(unittest.TestCase):
    def test_aca_endpoints_count_and_mainline_complete_union(self):
        for n in (66,90,120,150,180,192):
            info=timing(n,30.,2.)
            self.assertEqual(len(info['aca_indices']),50)
            self.assertEqual(info['aca_indices'][0],0)
            self.assertEqual(info['aca_indices'][-1],n-1)
            self.assertTrue(info['mainline_union_covers_all_source_frames'])
            self.assertTrue(info['mainline_windows_containing_nearest_source_frame'])
            self.assertIsNone(info['semantic_key_event_presence_confirmed'])

    def test_approximate_time_outside_video_not_silently_accepted_as_exact(self):
        info=timing(90,30.,5.)
        self.assertTrue(info['human_time_outside_frame_timestamp_range'])
        self.assertEqual(info['nearest_source_frame'],89)
        self.assertGreater(info['aca_time_distance_seconds'],2.)

    def test_invalid_timing_and_uncoverable_length_refused(self):
        for args in ((50,30.,1.),(150,0.,1.),(150,30.,float('nan')),(193,30.,2.)):
            with self.assertRaises(ValueError):timing(*args)

    def test_authorization_before_video_open(self):
        class Scope:
            def authorize(self,video_id):raise ValueError('forbidden')
        with patch('development.aca_frame_review_v1.build.gate.open_guarded',side_effect=AssertionError('opened')):
            with self.assertRaises(ValueError):decode_original(Scope(),'test/forbidden.mp4')

    def test_lossless_contact_native_pixels_and_all_indices(self):
        frames=np.random.default_rng(4).integers(0,256,(25,120,160,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            pages=contact_pages(frames,list(range(25)),30.,p,'test',columns=6,per_page=24,approximate_time=5.)
            self.assertEqual([i for page in pages for i in page['source_indices']],list(range(25)))
            with Image.open(p/pages[0]['file']) as image:
                canvas=np.array(image)
            self.assertTrue(np.array_equal(canvas[24:144,2:162],frames[0]))

    def test_native_key_panel_and_no_source_mutation(self):
        source=np.random.default_rng(1).integers(0,256,(90,240,320,3),dtype=np.uint8)
        before=source.copy();info=timing(90,30.,1.)
        aca=np.zeros((50,120,160,3),dtype=np.uint8)
        main=np.zeros((3,64,192,256,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'panel.png';key_panel(source,aca,main,info,p)
            with Image.open(p) as image:
                self.assertEqual(image.size,(1008,858))
        self.assertTrue(np.array_equal(source,before))

    def test_reference_transform_preserves_source_temporal_mapping(self):
        source=np.stack([np.full((240,320,3),i,dtype=np.uint8) for i in range(90)])
        cfg={'seed':20261016,'transform':{'policy':'reference','version':'video_clip_transforms_v1',
            'clip_length':64,'output_height':192,'output_width':256,'brightness':[.95,1.05],
            'contrast':[.95,1.05],'saturation':[.95,1.05],'spatial_scale':[.92,1.0],'enhancement_gamma':.95}}
        windows=timing(90,30.,1.)['mainline_windows_half_open']
        main=reconstruct_mainline(source,windows,cfg,'ft0/synthetic.mp4')
        self.assertEqual(main.shape,(3,64,192,256,3))
        for wi,(a,b) in enumerate(windows):
            self.assertTrue(np.array_equal(main[wi,:,0,0,0],np.arange(a,b,dtype=np.uint8)))


if __name__=='__main__':unittest.main()
