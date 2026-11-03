import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from development.tanet_v1.validate_L_bn_v1 import (
    POLICY,cohort_summary,forward_one,make_scores,paired_change,render,require_replay,validate_policy,
)
from development.tanet_v1.training import state_digest
from development.tanet_v1.tests.test_probe_bn_behavior_v1 import Toy


class ValidationLContracts(unittest.TestCase):
    def scores(self):
        ids=['a','b','c','d'];labels=torch.tensor([0,1,0,1])
        allowed={vid:{'split':'validation','excluded':'0','binary_label':str(int(y)),
                      'source_group':'g'+str(i%2),'original_class':f'2p{int(y)}'}
                 for i,(vid,y) in enumerate(zip(ids,labels))}
        a=torch.tensor([[2.,0.],[0.,2.],[0.,2.],[2.,0.]])
        b=torch.tensor([[2.,0.],[2.,0.],[2.,0.],[2.,0.]])
        sa,ra=make_scores(a,labels,ids,allowed);sb,rb=make_scores(b,labels,ids,allowed)
        return sa,ra,sb,rb

    def test_paired_id_join_counts_and_metrics_reproduce(self):
        a,ra,b,rb=self.scores()
        result,rows=paired_change(ra,list(reversed(rb)))
        self.assertEqual(result['counts'],{'both_correct':1,'both_wrong':1,'L_only_correct':1,'saved_only_correct':1})
        self.assertEqual(result['net_additional_correct'],0)
        self.assertEqual(result['prediction_flips'],2)
        self.assertEqual(a['overall']['metrics']['accuracy'],.5)
        self.assertEqual(b['overall']['metrics']['accuracy'],.5)
        with self.assertRaises(ValueError):paired_change(ra,rb[:-1])
        with self.assertRaises(ValueError):paired_change(ra,rb[:-1]+[rb[0]])

    def test_saved_replay_is_strict_before_candidate_interpretation(self):
        x=torch.tensor([1.,2.]);require_replay(x,x.clone())
        for actual in (x.double(),x+.001,torch.tensor([float('nan'),2.]),torch.ones(1,2)):
            with self.assertRaisesRegex(ValueError,'did not reproduce'):require_replay(actual,x)

    def test_policy_refuses_batch_strategy_checkpoint_and_scope_changes(self):
        plan={'policy':copy.deepcopy(POLICY),
              'checkpoints':[{'seed':s,'snapshot':tag} for s in (20261024,20261025,20261026) for tag in ('best','last')],
              'validation_ids':[f'video{i:04}' for i in range(1547)]}
        validate_policy(plan)
        for mutate in ('batch','strategy','checkpoint','ids'):
            p=copy.deepcopy(plan)
            if mutate=='batch':p['policy']['batch_size']=2
            elif mutate=='strategy':p['policy']['modes']=['saved_b1','GL_batch_b1']
            elif mutate=='checkpoint':p['checkpoints'][1]['snapshot']='best'
            else:p['validation_ids'].pop()
            with self.assertRaises(ValueError):validate_policy(p)

    def test_only_L_forward_keeps_buffers_parameters_and_dropout_unchanged(self):
        class FullShapeToy(nn.Module):
            def __init__(self):super().__init__();self.core=Toy()
            def forward(self,x):return self.core(x[:,:2].mean((3,4)))
        model=FullShapeToy().requires_grad_(False).eval()
        x=torch.arange(50.).view(1,1,50,1,1).expand(1,3,50,192,256)
        before=state_digest(model.state_dict())
        a=forward_one(model,x,'saved');b=forward_one(model,x,'L')
        self.assertFalse(torch.equal(a,b))
        self.assertEqual(state_digest(model.state_dict()),before)
        self.assertTrue(all(not m.training for m in model.modules()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaises(ValueError):forward_one(model,x,'GL')
        with self.assertRaises(ValueError):forward_one(model,x.expand(2,-1,-1,-1,-1),'L')

    def test_all_original_seeds_kept_and_complete_report_schema_renders(self):
        a,_,b,_=self.scores()
        details=[{'seed':seed,'snapshot':tag,'epoch':1,'saved_b1':a,'L_batch_b1':b}
                 for seed in (20261024,20261025,20261026) for tag in ('best','last')]
        agg=cohort_summary(details)
        self.assertEqual(agg['best']['saved_b1']['accuracy']['values'],[.5,.5,.5])
        with self.assertRaises(ValueError):cohort_summary(details[:-1])
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'plots').mkdir()
            render(out,{'details':details,'aggregate':agg,'limitations':['Synthetic test.']})
            self.assertEqual(len(list((out/'plots').glob('*.png'))),3)
            self.assertIn('L_batch_b1',(out/'index.html').read_text())


if __name__=='__main__':unittest.main()
