import unittest
import numpy as np
import torch
from torch import nn
from development.yolo_train_probe_v1.ball_negative_pair_v1 import schedule,encode,train_mode,combined_loss


class NegativePairTests(unittest.TestCase):
    def ids(self):return [f'E{i:03d}' for i in range(11)]
    def test_schedule_exact_exposures_and_reproducibility(self):
        s=schedule(self.ids(),30);self.assertEqual(s,schedule(list(reversed(self.ids())),30));self.assertEqual(len(s),330)
        for cycle in range(1,31):self.assertEqual({r['source_image_id'] for r in s if r['cycle']==cycle},set(self.ids()))
    def test_schedule_rejects_unknown_budget_and_duplicates(self):
        for ids,cycles in [(self.ids(),31),(self.ids(),0),(['same']*11,1),(self.ids()[:10],1)]:
            with self.assertRaises(ValueError):schedule(ids,cycles)
    def test_rgb_letterbox_and_target_geometry(self):
        raw=np.zeros((240,320,3),np.uint8);raw[:]=(10,20,30);b=encode(raw,[[0,0,320,240]])
        self.assertEqual(tuple(b['img'].shape),(1,3,640,640))
        torch.testing.assert_close(b['img'][0,:,80,0],torch.tensor([30,20,10])/255)
        torch.testing.assert_close(b['img'][0,:,0,0],torch.full((3,),114/255))
        torch.testing.assert_close(b['bboxes'],torch.tensor([[.5,.5,1.,.75]]))
    def test_empty_crop_targets_have_valid_shapes(self):
        for w,h in [(160,120),(128,96)]:
            b=encode(np.zeros((h,w,3),np.uint8),[])
            self.assertEqual(tuple(b['bboxes'].shape),(0,4));self.assertEqual(tuple(b['cls'].shape),(0,1))
    def test_invalid_pixels_and_boxes_rejected(self):
        with self.assertRaises(ValueError):encode(np.zeros((64,64,3),np.uint8),[])
        with self.assertRaises(ValueError):encode(np.zeros((240,320,3),np.uint8),[[0,0,321,20]])
    def test_feature_eval_head_train_but_all_bn_eval(self):
        model=nn.Module();model.model=nn.ModuleList([nn.Sequential(nn.BatchNorm2d(2),nn.ReLU()),nn.Sequential(nn.Conv2d(2,2,1),nn.BatchNorm2d(2))])
        model.train();train_mode(model)
        self.assertFalse(model.model[0].training);self.assertTrue(model.model[-1].training)
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d)))
    def test_auxiliary_weight_does_not_rescale_positive_gradient(self):
        p=torch.tensor(2.,requires_grad=True);n=torch.tensor(3.,requires_grad=True)
        combined_loss(p,n,.25).backward();self.assertEqual(p.grad.item(),1.);self.assertEqual(n.grad.item(),.25)
        with self.assertRaises(ValueError):combined_loss(p,n,.5)


if __name__=='__main__':unittest.main()
