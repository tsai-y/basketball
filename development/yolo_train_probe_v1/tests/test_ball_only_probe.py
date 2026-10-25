import unittest
from copy import deepcopy
import torch
from torch import nn
from development.yolo_train_probe_v1.ball_only_probe_v1 import ball_annotations,transfer_ball_state,freeze_backbone_eval,clone_ball_model


class BallOnlyTests(unittest.TestCase):
    def states(self):
        source={'shared':torch.arange(4.).reshape(2,2)};target={'shared':torch.zeros(2,2)}
        for i in range(3):
            for suffix,shape in [('weight',(4,2,1,1)),('bias',(4,))]:
                key=f'model.23.cv3.{i}.2.{suffix}'
                source[key]=torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float32).reshape(shape)
                target[key]=torch.zeros((1,)+shape[1:])
        return source,target
    def test_only_accepted_ball_geometry_remapped(self):
        source=[{'id':'P','class_id':0,'xyxy':[0,0,20,50]}, {'id':'B','class_id':1,'xyxy':[1,2,8,9]},
                {'id':'R','class_id':2,'xyxy':[1,1,5,2]}];old=deepcopy(source)
        result=ball_annotations(source);self.assertEqual(source,old);self.assertEqual(len(result),1)
        self.assertEqual(result[0]['class_id'],0);self.assertEqual(result[0]['source_class_id'],1)
        self.assertEqual(result[0]['xyxy'],[1,2,8,9])
    def test_no_invented_negative(self):
        with self.assertRaises(ValueError):ball_annotations([{'class_id':0,'xyxy':[0,0,2,2]}])
    def test_semantic_row_exactly_copied(self):
        source,target=self.states();mapped,audit=transfer_ball_state(source,target,2)
        self.assertEqual(audit['semantic_row_count'],6);self.assertEqual(audit['exact_shared_count'],1)
        self.assertTrue(torch.equal(mapped['shared'],source['shared']))
        for key in target:
            if key!='shared':self.assertTrue(torch.equal(mapped[key],source[key][2:3]))
    def test_mapping_does_not_alias_or_mutate_original(self):
        source,target=self.states();old=source['shared'].clone();mapped,_=transfer_ball_state(source,target,2)
        mapped['shared'].zero_();self.assertTrue(torch.equal(source['shared'],old))
    def test_unexpected_mismatch_and_invalid_class_rejected(self):
        source,target=self.states();target['shared']=torch.zeros(3,3)
        with self.assertRaisesRegex(ValueError,'non-classifier'):transfer_ball_state(source,target,2)
        source,target=self.states()
        with self.assertRaises(ValueError):transfer_ball_state(source,target,7)
    def test_backbone_bn_eval_without_freezing_head_mode(self):
        model=nn.Module();model.model=nn.Sequential(nn.BatchNorm2d(2),nn.BatchNorm2d(2));model.train()
        freeze_backbone_eval(model,1);self.assertFalse(model.model[0].training);self.assertTrue(model.model[1].training)

    def test_checkpoint_clone_preserves_non_tensor_activation(self):
        original=nn.Module();head=nn.Module()
        head.cv3=nn.ModuleList([nn.Sequential(nn.Identity(),nn.Identity(),nn.Conv2d(2,4,1)) for _ in range(3)])
        head.reg_max=16;head.nc=4;head.no=68;head.shape=None;head.end2end=False
        original.model=nn.ModuleList([nn.Sequential(nn.Conv2d(2,2,1),nn.SiLU()),head])
        original.yaml={'nc':4};original.names={i:str(i) for i in range(4)}
        clone,audit=clone_ball_model(original,2)
        self.assertIsInstance(clone.model[0][1],nn.SiLU)
        self.assertEqual(original.model[-1].nc,4);self.assertEqual(clone.model[-1].nc,1)
        self.assertEqual(audit['semantic_row_count'],6)
        for a,b in zip(original.model[-1].cv3,clone.model[-1].cv3):
            self.assertTrue(torch.equal(a[-1].weight[2:3],b[-1].weight))


if __name__=='__main__':unittest.main()
