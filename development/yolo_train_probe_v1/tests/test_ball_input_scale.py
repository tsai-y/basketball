import unittest
from development.yolo_train_probe_v1.ball_input_scale_v1 import parameters_for_scale,same_predictions


class BallScaleTests(unittest.TestCase):
    def setUp(self):self.ref={'imgsz':640,'conf':.25,'iou':.7,'max_det':100,'half':False}
    def test_only_input_size_changes(self):
        other=parameters_for_scale(self.ref,1280)
        self.assertEqual({k for k in other if other[k]!=self.ref[k]},{'imgsz'})
        self.assertEqual(self.ref['imgsz'],640)
    def test_only_two_frozen_scales(self):
        for size in (320,960,1536,True):
            with self.subTest(size=size),self.assertRaises(ValueError):parameters_for_scale(self.ref,size)
    def test_threshold_changes_rejected(self):
        for key,value in [('conf',.1),('iou',.5),('max_det',300),('half',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):parameters_for_scale({**self.ref,key:value},1280)
    def test_unchanged_and_changed_predictions(self):
        p={'xyxy':[1.,2.,3.,4.],'confidence':.5}
        self.assertTrue(same_predictions([p],[p]));self.assertTrue(same_predictions([],[]))
        self.assertFalse(same_predictions([p],[]))
        self.assertFalse(same_predictions([p],[{**p,'xyxy':[1.,2.,3.,5.]}]))
    def test_prediction_order_does_not_imply_change(self):
        a={'xyxy':[1.,2.,3.,4.],'confidence':.5};b={'xyxy':[5.,6.,7.,8.],'confidence':.9}
        self.assertTrue(same_predictions([a,b],[b,a]))


if __name__=='__main__':unittest.main()
