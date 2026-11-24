"""Bounded architecture/pairing/normalization/loop tests, synthetic inputs only."""
import ast
import copy
import gc
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from . import model as m
from .engine import EngineeringLoop, SCHEMA
from development.aca_eight_class_data_v1 import loop_core as core
from development.aca_eight_class_data_v1.model_adapter import initialized_baseline8, baseline_training_mode
from development.aca_eight_class_data_v1.gpu_readiness import cpu_tree,equal
from development.aca_net_v1.modules import TSCI,dynamic_temporal_conv

EVIDENCE={}

def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def fingerprints(model):
    return {k:digest(v) for k,v in model.state_dict().items()}

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(3,4,1)
        self.bn=nn.BatchNorm2d(4)
        self.aca=TSCI()
        self.fc=nn.Linear(4,8)
    def forward(self,x):
        n,c,t,h,w=x.shape
        f=self.bn(self.conv(x.permute(0,2,1,3,4).reshape(n*t,c,h,w)))
        z=f.reshape(n,t,4,h,w).permute(0,2,1,3,4)
        return self.fc(self.aca(z).mean((2,3,4)))


def fixture():
    g=torch.Generator().manual_seed(41)
    data={'train':[(torch.randn(1,3,4,3,4,generator=g),torch.tensor([i%8])) for i in range(24)],
          'validation':[(torch.randn(1,3,4,3,4,generator=g),torch.tensor([i%8])) for i in range(8)]}
    plan={'schema':SCHEMA,'formal_training_authorized':False,'test_enabled':False,
          'seed':20261024,'effective':16,'patience':5,'max_epochs':2,'checkpoint_every':25,
          'train_ids':[f't{i:02}' for i in range(24)],'validation_ids':[f'v{i:02}' for i in range(8)]}
    def fetch(split,index):
        x,y=data[split][index]
        return {'tensor':x,'targets':y,'records':[{'split':split,'video_id':plan[split+'_ids'][index]}]}
    return plan,fetch

class EngineeringTests(unittest.TestCase):
    def test_meta_shape_and_all_arm_counts(self):
        rows={}
        for arm in m.TOTAL:
            model=m.meta_model(arm); positions=[]; handles=[]
            for stage in ('layer1','layer2','layer3','layer4'):
                for block in getattr(model,stage):
                    handles.append(block.aca.register_forward_pre_hook(lambda _,x: positions.append(list(x[0].shape))))
            y=model(torch.empty(1,3,50,120,160,device='meta'))
            self.assertEqual(list(y.shape),[1,8]); self.assertEqual(len(positions),16)
            expected=[]
            for c,shapes in [(64,[(30,40)]*3),(128,[(30,40)]+[(15,20)]*3),
                             (256,[(15,20)]+[(8,10)]*5),(512,[(8,10)]+[(4,5)]*2)]:
                expected.extend([[1,c,50,h,w] for h,w in shapes])
            self.assertEqual(positions,expected)
            for layer in model.modules():
                if isinstance(layer,m.SeparableLSTA):
                    self.assertEqual(layer.gate_value[0].groups,1)
                    self.assertEqual(layer.gate_control[0].groups,layer.gate_control[0].in_channels)
                    self.assertEqual(layer.gate_output[0].groups,layer.gate_output[0].in_channels)
            rows[arm]={'total':sum(p.numel() for p in model.parameters()),
                       'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                       'positions':positions,'output_shape':list(y.shape)}
            for h in handles:h.remove()
            del model;gc.collect()
        EVIDENCE['meta_shapes_parameters']=rows

    def test_exact_paired_initialization_and_baseline(self):
        old,_=initialized_baseline8(); old_state=fingerprints(old); rng=torch.get_rng_state().clone()
        x=torch.linspace(-1,1,50).reshape(1,1,50,1,1).expand(1,3,50,32,40)
        baseline_training_mode(old)
        with torch.no_grad(): expected=old(x)
        del old;gc.collect()
        arms={}
        for arm in m.TOTAL:
            model,info=m.initialized(arm)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()),arm)
            fs=fingerprints(model)
            shared={k:v for k,v in fs.items() if '.aca.' not in k}
            self.assertEqual(shared,old_state,arm)
            if arm=='baseline':
                with torch.no_grad(): self.assertTrue(torch.equal(expected,model(x)))
            arms[arm]={k:v for k,v in fs.items() if '.aca.' in k}
            del model;gc.collect()
        for arm in ('lsta','tsci'):
            for k,v in arms[arm].items():
                self.assertEqual(v,arms['aca'][k.replace('.aca.',f'.aca.{arm}.')],k)
        EVIDENCE['initialization']={'shared_backbone_head_bitwise_equal':True,'shared_modules_bitwise_equal':True,
                                    'global_cpu_rng_unchanged_by_modules':True,'baseline_logits_bitwise_equal':True,
                                    'old_task_weights_loaded':False}

    def test_dynamic_convolution_reference_and_gradient_isolation(self):
        torch.manual_seed(23)
        x=torch.randn(2,4,5,2,3,dtype=torch.float64,requires_grad=True)
        k=torch.randn(2,4,3,dtype=torch.float64,requires_grad=True)
        actual=dynamic_temporal_conv(x,k)
        expected=torch.zeros_like(actual)
        for n in range(2):
            for c in range(4):
                for t in range(5):
                    for j in range(3):
                        if 0<=t+j-1<5:expected[n,c,t]+=k[n,c,j]*x[n,c,t+j-1]
        self.assertTrue(torch.allclose(actual,expected,rtol=1e-12,atol=1e-12))
        gx,gk=torch.autograd.grad(actual[0].sum(),(x,k))
        self.assertEqual(gx[1].abs().sum().item(),0);self.assertEqual(gk[1].abs().sum().item(),0)
        EVIDENCE['dynamic_reference_and_isolation']=True

    def test_lsta_tsci_gradients_and_norm_lifecycle(self):
        torch.manual_seed(4)
        lsta=m.SeparableLSTA(4); tsci=TSCI()
        holder=nn.Module();holder.backbone_bn=nn.BatchNorm2d(4);holder.aca=m.ParallelACA(lsta,tsci)
        m.training_mode(holder)
        self.assertFalse(holder.backbone_bn.training)
        self.assertTrue(all(g.bn.training for g in (tsci.channel,tsci.height,tsci.width)))
        x=torch.randn(1,4,50,3,4,requires_grad=True); y=holder.aca(x)
        self.assertTrue(torch.equal(y,lsta(x)+tsci(x)))
        (y.square().mean()).backward()
        for name,sub in [('lsta',lsta),('tsci',tsci)]:
            grads=[p.grad for p in sub.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0,name)
        holder.eval(); before=fingerprints(holder)
        with torch.no_grad(): a=holder.aca(x);b=holder.aca(x)
        self.assertTrue(torch.equal(a,b));self.assertEqual(before,fingerprints(holder))
        m.training_mode(holder)
        self.assertTrue(all(p.requires_grad for g in (tsci.channel,tsci.height,tsci.width) for p in g.bn.parameters()))
        self.assertTrue(all(not p.requires_grad for p in holder.backbone_bn.parameters()))
        EVIDENCE['new_bn_lifecycle_and_module_gradients']=True

    def test_loop_ast_matches_frozen_numerical_core(self):
        def tree(f):return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(f))),include_attributes=False)
        self.assertEqual(tree(EngineeringLoop._run),tree(core.FixtureLoop._run))
        EVIDENCE['numerical_loop_ast_equal']=True

    def test_resume_preserves_bn_optimizer_rng_and_selection(self):
        plan,fetch=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            def fresh(path):
                torch.manual_seed(81);return EngineeringLoop(Tiny(),plan,fetch,path,'engineering-fixture')
            continuous=fresh(Path(tmp)/'a.pt');self.assertEqual(continuous.run(),'completed'); expected=continuous._payload()
            for case,stop in [('train',lambda p,s,n:p=='train' and s['epoch']==1 and s['updates']==3),
                                ('validation',lambda p,s,n:p=='validation' and s['epoch']==0 and n==3)]:
                path=Path(tmp)/(case+'.pt');part=fresh(path)
                self.assertEqual(part.run(stop),'paused')
                resumed=fresh(path);resumed.resume();self.assertEqual(resumed.run(),'completed')
                self.assertTrue(equal(expected,resumed._payload()),case)
            self.assertEqual(expected['state']['updates'],4)
            self.assertEqual(expected['model']['aca.channel.bn.num_batches_tracked'].item(),48)
            self.assertEqual(expected['model']['bn.num_batches_tracked'].item(),0)
        EVIDENCE['synthetic_resume_train_and_validation_exact']=True

    def test_tail_divisor_matches_independent_adam(self):
        plan,fetch=fixture();plan['max_epochs']=1
        torch.manual_seed(81); reference=Tiny();m.training_mode(reference)
        opt=torch.optim.Adam([p for p in reference.parameters() if p.requires_grad],lr=1e-4,
                             betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,fused=False)
        order=torch.randperm(24,generator=torch.Generator().manual_seed(20261024)).tolist()
        for start in (0,16):
            ids=order[start:start+16];opt.zero_grad(set_to_none=True)
            for i in ids:
                b=fetch('train',i);(F.cross_entropy(reference(b['tensor']),b['targets'],reduction='sum')/len(ids)).backward()
            nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            opt.step();opt.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(81); actual=EngineeringLoop(Tiny(),plan,fetch,Path(tmp)/'last.pt','tail')
            actual.run()
            self.assertTrue(equal(reference.state_dict(),actual.model.state_dict()))
            self.assertTrue(equal(opt.state_dict(),actual.optimizer.state_dict()))
        EVIDENCE['tail8_independent_adam_exact']=True

    def test_scope_rejects_formal_plan_and_other_seed(self):
        p,f=fixture()
        for patch in ({'formal_training_authorized':True},{'test_enabled':True},{'max_epochs':30},
                      {'seed':20261025},{'schema':'baseline8_formal_checkpoint_v1'}):
            with self.assertRaises(ValueError):EngineeringLoop(Tiny(),{**p,**patch},f,'/tmp/unwritten.pt','x')
        with self.assertRaises(ValueError):m.initialized('aca',20261025)
        EVIDENCE['formal_scope_rejected']=True

if __name__=='__main__':
    import argparse,time
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.report.exists():raise ValueError('Never overwrite evidence')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    start=__import__('time').monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EngineeringTests))
    args.report.write_text(json.dumps({'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,
        'seconds':__import__('time').monotonic()-start,'cuda_initialized':torch.cuda.is_initialized(),
        'real_video_decodes':0,'formal_training_started':False,'evidence':EVIDENCE},indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
