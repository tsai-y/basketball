"""Small geometry/access checks before the fixed16 real-image probe."""
import unittest
import numpy as np

from .crop_probe import candidate


class CropProbeTests(unittest.TestCase):
    def test_temporal_window_is_50_distinct_frames_without_padding(self):
        native=np.zeros((180,240,320,3),dtype=np.uint8)
        native[:,:,:,0]=np.arange(180,dtype=np.uint8)[:,None,None]
        rgb,meta=candidate(native,spatial=False,temporal=True)
        indices=meta['source_indices']
        self.assertEqual((indices[0],indices[-1]),(9,170))
        self.assertEqual(len(set(indices)),50)
        self.assertEqual(rgb[:,0,0,0].tolist(),indices)
        self.assertTrue(np.all(rgb[:,:,:,1:]==0))

    def test_crop_removes_only_the_declared_border(self):
        native=np.zeros((50,240,320,3),dtype=np.uint8)
        native[:,:,:,0]=255
        native[:,12:228,16:304]=[0,80,160]
        rgb,meta=candidate(native,spatial=True,temporal=False)
        self.assertEqual(rgb.shape,(50,120,160,3))
        self.assertEqual(meta['crop_xyxy'],[16,12,304,228])
        self.assertTrue(np.all(rgb==np.array([0,80,160],dtype=np.uint8)))

    def test_short_trimmed_input_rejected_instead_of_looping_frames(self):
        with self.assertRaisesRegex(ValueError,'at least 50'):
            candidate(np.zeros((50,240,320,3),dtype=np.uint8),spatial=False,temporal=True)


if __name__=='__main__':
    unittest.main()
