from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from development.tanet_v1.probe_bn_behavior_v1 import (
    BNCapture, MODES, bn_condition, compare_logits, render, select_train_cases,
    summarize_outputs, tam_bn_modules,
)
from development.tanet_v1.training import state_digest


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = nn.BatchNorm2d(2)
        self.block = nn.Module()
        self.block.tam = nn.Module()
        self.block.tam.G = nn.Sequential(nn.Identity(),nn.BatchNorm1d(2),nn.ReLU(inplace=True))
        self.block.tam.L = nn.Sequential(nn.Identity(),nn.BatchNorm1d(2),nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(.9)

    def forward(self,x):
        x=self.pretrained(x.unsqueeze(-1)).squeeze(-1)
        return self.dropout(self.block.tam.L(self.block.tam.G(x))).mean(2)


class BNBehaviorContracts(unittest.TestCase):
    def rows(self):
        return [{'video_id':f'{cls}_{i}','original_class':cls,'source_group':f'g{i}',
                 'split':'train','excluded':'0'}
                for cls in ('2p0','2p1','3p0','3p1','ft0','ft1','mp0','mp1') for i in range(4)]

    def test_fixed_class_sampling_is_order_independent_and_ignores_scores(self):
        rows=self.rows();selected=select_train_cases(rows)
        other=list(reversed(rows))
        for row in other:row['untrusted_prediction_confidence']=.99
        again=select_train_cases(other)
        self.assertEqual([r['video_id'] for r in selected],[r['video_id'] for r in again])
        self.assertEqual(len(selected),16)
        for start in range(0,16,2):
            self.assertEqual(len({r['original_class'] for r in selected[start:start+2]}),1)
            self.assertEqual(len({r['source_group'] for r in selected[start:start+2]}),2)

    def test_nontrain_duplicate_and_missing_class_rejected_before_pixels(self):
        rows=self.rows()
        for split in ('validation','test'):
            wrong=[dict(r) for r in rows];wrong[0]['split']=split
            with self.assertRaises(ValueError):select_train_cases(wrong)
        with self.assertRaises(ValueError):select_train_cases(rows+[rows[0]])
        with self.assertRaises(ValueError):select_train_cases(rows[:-4])

    def test_branch_modes_match_functional_batchnorm_without_state_or_dropout_updates(self):
        torch.set_num_threads(1)
        model=Toy().requires_grad_(False).eval()
        x=torch.tensor([[[-2.,1.,4.,7.],[3.,4.,8.,11.]]])
        before=state_digest(model.state_dict())
        with torch.inference_mode():
            for branch in ('saved','G','L','GL'):
                z=model.pretrained(x.unsqueeze(-1)).squeeze(-1)
                for name,bn in [('G',model.block.tam.G[1]),('L',model.block.tam.L[1])]:
                    batch=branch=='GL' or branch==name
                    z=F.batch_norm(z,None if batch else bn.running_mean,None if batch else bn.running_var,
                                   bn.weight,bn.bias,training=batch,eps=bn.eps).relu()
                expected=z.mean(2)
                with bn_condition(model,branch):
                    self.assertFalse(model.dropout.training)
                    self.assertFalse(model.pretrained.training)
                    torch.testing.assert_close(model(x),expected)
                self.assertEqual(state_digest(model.state_dict()),before)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with bn_condition(model,'GL'):raise RuntimeError('intentional')
        self.assertTrue(all(not m.training for m in model.modules()))
        self.assertTrue(all(m.track_running_stats for m in tam_bn_modules(model).values()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_capture_clones_before_inplace_relu_and_records_zero_replay_difference(self):
        model=Toy().requires_grad_(False).eval()
        cap=BNCapture(model,1,'synthetic');cap.mode='saved_b1';cap.ids=['train0']
        x=torch.tensor([[[-3.,-1.,1.,3.],[-2.,0.,2.,4.]]])
        try:
            with torch.inference_mode():model(x)
            self.assertLess(float(cap.baseline[('block.tam.G.1','train0')].min()),0.)
            cap.mode='saved_repeat_b1'
            with torch.inference_mode():model(x)
            self.assertTrue(all(r['relative_L2_output']==0 for r in cap.rows))
            self.assertEqual(len(cap.rows),4)
        finally:cap.close()

    def test_probability_flip_near_boundary_and_saturated_logits(self):
        a=compare_logits(torch.tensor([.01,0.]),torch.tensor([0.,.01]))
        self.assertTrue(a['prediction_flipped']);self.assertLess(a['absolute_PMake_shift'],.01)
        b=compare_logits(torch.tensor([1000.,-1000.]),torch.tensor([-1000.,1000.]))
        self.assertEqual(b['absolute_PMake_shift'],1.)
        s=summarize_outputs([a,b]);self.assertEqual(s['videos'],2);self.assertEqual(s['prediction_flips'],2)
        self.assertEqual(s['shift_at_least']['0.1'],1)
        with self.assertRaises(ValueError):compare_logits(torch.tensor([float('nan'),0.]),torch.zeros(2))

    def test_complete_report_schema_renders_output_and_activation_plots(self):
        row=compare_logits(torch.tensor([0.,1.]),torch.tensor([1.,0.]))
        metrics={name:summarize_outputs([row]*16) for name,_,_ in MODES}
        cp=[{'seed':20261024,'snapshot':tag,'modes':metrics} for tag in ('best','last')]
        activations=[{'seed':20261024,'snapshot':c['snapshot'],'mode':mode,
                      'layer':f'backbone.layer1.{i}.tam.{branch}.1','relative_L2_output':.2}
                     for c in cp for mode in ('G_batch_b1','L_batch_b1','GL_batch_b1')
                     for i in range(16) for branch in ('G','L')]
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'plots').mkdir()
            render(out,{'checkpoints':cp,'limitations':['Synthetic test.']},activations)
            self.assertEqual(len(list((out/'plots').glob('*.png'))),2)
            self.assertIn('saved_b2',(out/'index.html').read_text())


if __name__=='__main__':unittest.main()
