from copy import deepcopy
from pathlib import Path
import unittest

import torch
from torch import nn

from development.yolo_train_probe_v1.ball_train_v2 import (
    checked_loader_paths, parameter_scope, gradient_summary, smoke_gate, SCHEMA,
)
from development.yolo_train_probe_v1.train_expansion_v1 import training_args


class BallTrainV2Tests(unittest.TestCase):
    def model(self):
        m=nn.Module();m.model=nn.Sequential(nn.Linear(2,2),nn.Linear(2,1))
        for p in m.model[0].parameters():p.requires_grad_(False)
        return m

    def test_loader_allowlist_not_count_only(self):
        checked_loader_paths(['/tmp/train_a','/tmp/train_b'],{'/tmp/train_b','/tmp/train_a'})
        for actual in (['/tmp/train_a','/tmp/dev'],['/tmp/train_a','/tmp/train_a'],['/tmp/train_a']):
            with self.assertRaises(ValueError):checked_loader_paths(actual,{'/tmp/train_a','/tmp/train_b'})

    def test_head_only_and_finite_nonzero_gradients(self):
        m=self.model();m.model(torch.ones(2,2)).sum().backward()
        report=gradient_summary(m,1)
        self.assertGreater(report['tensors_with_nonzero_gradient'],0)
        self.assertEqual(len(parameter_scope(m,1)),2)
        self.assertTrue(all(p.grad is None for p in m.model[0].parameters()))

    def test_trainable_backbone_or_frozen_gradient_fails(self):
        m=self.model();m.model[0].weight.requires_grad_(True)
        with self.assertRaises(ValueError):parameter_scope(m,1)
        m=self.model();m.model(torch.ones(2,2)).sum().backward()
        m.model[0].weight.grad=torch.ones_like(m.model[0].weight)
        with self.assertRaises(ValueError):gradient_summary(m,1)

    def test_missing_zero_or_nonfinite_gradients_fail(self):
        for value in (None,0.,float('nan'),float('inf')):
            m=self.model()
            for p in m.model[1].parameters():p.grad=None if value is None else torch.full_like(p,value)
            with self.assertRaises(ValueError):gradient_summary(m,1)

    def smoke_state(self):
        state=dict(schema=SCHEMA,status='completed',stage='smoke',dataset=str(Path('/tmp/dataset').resolve()),
                   dataset_manifest_sha256='data',initializer_sha256='weights',source_code_sha256='code',
                   epochs_completed=1,optimizer_steps=7,training_image_exposures=14)
        for flag in ('actual_loader_allowlists_checked','all_optimizer_gradients_finite_nonzero',
                     'backbone_parameters_and_BN_buffers_unchanged','frozen_head_parameters_unchanged',
                     'checkpoint_round_trip','head_updated','source_data_unchanged'):state[flag]=True
        return state

    def test_matching_completed_one_epoch_smoke_gate(self):
        s=self.smoke_state();smoke_gate(s,'/tmp/dataset','data','weights','code')
        for field,value in [('stage','ball_only_head_training_v2'),('status','failed'),('epochs_completed',0),
                            ('optimizer_steps',6),('training_image_exposures',13),('initializer_sha256','other'),
                            ('dataset_manifest_sha256','other'),('source_code_sha256','other'),
                            ('checkpoint_round_trip',False),('backbone_parameters_and_BN_buffers_unchanged',False)]:
            with self.subTest(field=field):
                bad=deepcopy(s);bad[field]=value
                with self.assertRaises(ValueError):smoke_gate(bad,'/tmp/dataset','data','weights','code')

    def test_smoke_budget_and_policy_bounds(self):
        args=training_args('/tmp/data','/tmp/run',1,True)
        self.assertEqual(args['batch'],2);self.assertEqual(args['nbs'],2)
        self.assertFalse(args['amp']);self.assertEqual(args['lr0'],.0001)
        for epochs,smoke in ((2,True),(0,False),(101,False)):
            with self.assertRaises(ValueError):training_args('/tmp/data','/tmp/run',epochs,smoke)


if __name__=='__main__':unittest.main()
