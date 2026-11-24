"""Bounded CPU fixtures and two train-clip CUDA acceptance; never formal results."""
import argparse
import ast
import copy
import gc
import json
from pathlib import Path
import tempfile
import torch
from torch import nn
from . import core, runtime, formal_train as formal
from .data import BinaryCollator, RevisedDataset
from .model import initialized
from development.aca_reconstruction_v2.model import initialized as initialized8, training_mode
from development.adjudicated_input_v1 import adapter as labels
from development.aca_eight_class_data_v1.gpu_readiness import configure, equal, cpu_tree


def rejects(fn):
    try:fn()
    except ValueError:return True
    raise AssertionError('Expected rejection')


def report(plan, fields, name):
    formal.QA.mkdir(exist_ok=True)
    out=formal.QA/name
    core.need(not out.exists(),'Acceptance already exists; inspect it')
    value={'status':'passed','at_taiwan':runtime.now(), 'draft_plan_sha256':core.plan_digest(plan),
           'code_sha256':{k:v for k,v in plan['code_sha256'].items() if k.startswith('development/aca_binary_v1/')},
           'test_video_decodes':0,'formal_training_started':False,**fields}
    runtime.atomic_json(out,value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def cpu():
    configure();plan=formal.draft_plan();checks=[]
    z=torch.tensor([[3.,0.],[0.,3.],[0.,3.],[3.,0.]])
    y=torch.tensor([0,1,0,1]);m=core.BinaryMetrics();m.add(z,y)
    assert m.result()['confusion_rows_true_columns_predicted']==[[1,1],[1,1]]
    assert m.result()['accuracy']==m.result()['macro_f1']==.5
    assert abs(m.result()['ce']-torch.nn.functional.cross_entropy(z.double(),y).item())<1e-12
    assert 'derived_binary' not in m.result();checks.append('direct_binary_metric_and_ce')
    assert rejects(lambda:core.summed_ce(torch.zeros(1,8),torch.tensor([1])))
    assert rejects(lambda:core.summed_ce(torch.zeros(1,2),torch.tensor([2])))
    bad=core.metrics_state(m);bad['schema']='eight_class_metrics_v1';assert rejects(lambda:core.restore_metrics(bad))
    checks.append('reject_old_logits_targets_and_metrics')
    rev=labels.LabelRevision()
    assert all(rev.record(v,split)['binary_label']==plan['reference'][v]['class_id']%2
               for split in ('train','validation') for v in plan[split+'_ids'])
    checks.append('all_8739_effective_class_to_binary_mappings')
    # Fixture loop tests actual update, tail normalization, state, and restart.
    class Tiny(nn.Module):
        def __init__(self):super().__init__();self.fc=nn.Linear(3,2)
        def forward(self,x):return self.fc(x)
    fixture={**plan,'train_ids':[f't{i}' for i in range(24)],'validation_ids':['v0','v1'],
             'max_epochs':2,'checkpoint_every':1,'device':'cpu'}
    def fetch(split,i):
        x=torch.tensor([[i/24.,float(i%3),1.]])
        return {'tensor':x,'targets':torch.tensor([i%2]),'records':[{'video_id':fixture[split+'_ids'][i],'split':split}]}
    torch.manual_seed(71);initial=Tiny().state_dict()
    def make(path):
        model=Tiny();model.load_state_dict(initial);return runtime.TrainingLoop(model,fixture,fetch,path,'fixture')
    with tempfile.TemporaryDirectory() as temp:
        a=make(Path(temp)/'a.pt');manual=make(Path(temp)/'manual.pt')
        # Explicit 16 + 8 reference updates for epoch1, independent of loop aggregation.
        order=torch.randperm(24,generator=torch.Generator().manual_seed(plan['seed'])).tolist()
        for ids in [order[:16],order[16:]]:
            manual.optimizer.zero_grad();xx=torch.cat([fetch('train',i)['tensor'] for i in ids]);yy=torch.cat([fetch('train',i)['targets'] for i in ids])
            torch.nn.functional.cross_entropy(manual.model(xx),yy).backward()
            nn.utils.clip_grad_norm_(manual.model.parameters(),1.);manual.optimizer.step();manual.optimizer.zero_grad()
        a.run(lambda phase,st,seen:phase=='validation')
        assert all(torch.allclose(v,manual.model.state_dict()[k],atol=1e-7,rtol=1e-6) for k,v in a.model.state_dict().items())
        checks.append('effective16_tail8_matches_reference')
        c=make(Path(temp)/'c.pt');assert c.run()=='completed'
        d=make(Path(temp)/'d.pt');assert d.run(lambda phase,st,seen:phase=='train' and st['updates']==1)=='paused'
        e=make(Path(temp)/'d.pt');e.resume();assert e.run()=='completed'
        assert equal(c.model.state_dict(),e.model.state_dict()) and equal(c.optimizer.state_dict(),e.optimizer.state_dict()) and c.state==e.state
        checks.append('continuous_vs_resumed_model_adam_history_equal')
        payload=torch.load(Path(temp)/'d.pt',weights_only=True);payload['schema']='aca_reconstruction_v2_formal_checkpoint_v1';core.atomic_save(payload,Path(temp)/'wrong.pt')
        f=make(Path(temp)/'wrong.pt');before=core.capture_rng();assert rejects(f.resume) and equal(before,core.capture_rng())
        checks.append('reject_eight_class_checkpoint_before_mutation')
        tie=copy.deepcopy(c.state);row=copy.deepcopy(tie['history'][0]);row['epoch']=2;tie['history'][1]=row
        tie.update(best_epoch=1,best_ce=row['validation']['ce'],bad_epochs=1)
        runtime.validate_state(tie,fixture,{})
        tie['best_epoch']=2;assert rejects(lambda:runtime.validate_state(tie,fixture,{}))
        checks.append('equal_ce_keeps_earlier_epoch')
    # Verify non-head parameters/BN are exactly original fresh initialization.
    eight,_=initialized8();binary,info=initialized()
    assert all(torch.equal(v,binary.state_dict()[k]) for k,v in eight.state_dict().items() if not k.startswith('fc.'))
    assert binary.fc.out_features==2 and info['trained_parent_weights_loaded'] is False
    checks.append('all_non_head_initial_weights_and_buffers_equal')
    del eight,binary;gc.collect()
    # AST verifies the audited epoch algorithm changed only its metric symbol.
    src=(formal.ROOT/'development/aca_reconstruction_v2/engine.py').read_text();new=Path(runtime.__file__).read_text()
    def run_ast(text):return next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='_run')
    assert ast.dump(run_ast(src.replace('EightClassMetrics','BinaryMetrics')),include_attributes=False)==ast.dump(run_ast(new),include_attributes=False)
    checks.append('epoch_algorithm_ast_identical_except_binary_metrics')
    report(plan,{'checks':checks,'real_video_decodes':0,'synthetic_only':True,'temporary_checkpoints_removed':True},'cpu_report.json')


def gpu():
    configure();plan=formal.draft_plan();formal.original.assert_idle_gpu(plan['gpu_policy'])
    ids=[next(v for v in plan['train_ids'] if plan['reference'][v]['binary_label']==0),labels.TARGET]
    ds=RevisedDataset(plan['source_root'],'train',ids,plan['reference']);batches=[]
    for i in range(2):
        item=ds[i];batch=BinaryCollator()([item]);ref=plan['reference'][ids[i]]
        assert labels.tensor_sha(batch['tensor'][0])==ref['tensor_sha256']
        assert batch['targets'].tolist()==[ref['binary_label']]
        batches.append(batch)
    model,info=initialized();model.cuda();training_mode(model)
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if '.aca.' not in k and ('.bn' in k or k.startswith('bn') or '.downsample.1.' in k)}
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    before=model.fc.weight.detach().clone();opt.zero_grad()
    for batch in batches:
        z=model(batch['tensor'].cuda());assert z.shape==(1,2)
        (core.summed_ce(z,batch['targets'].cuda())/2).backward()
    assert model.fc.weight.grad is not None and torch.isfinite(model.fc.weight.grad).all()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '.aca.' in n)
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step();opt.zero_grad()
    assert not torch.equal(before,model.fc.weight)
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in frozen.items())
    model.eval();buffers={k:v.clone() for k,v in model.named_buffers()}
    with torch.no_grad():
        for batch in batches:core.summed_ce(model(batch['tensor'].cuda()),batch['targets'].cuda())
    assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in buffers.items())
    peak=torch.cuda.max_memory_allocated();del model,opt;gc.collect();torch.cuda.empty_cache()
    report(plan,{'train_video_ids':ids,'real_train_video_decodes':2,'validation_video_decodes':0,
        'engineering_optimizer_updates':1,'engineering_weights_discarded':True,'formal_initialization':'fresh_ImageNet_V1',
        'output_shape':[1,2],'targets':[0,1],'input_tensor_sha_unchanged':True,'backbone_bn_unchanged':True,
        'validation_buffers_unchanged':True,'peak_allocated_bytes':peak,'scope':'engineering only, no classification performance'},'gpu_report.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);args=p.parse_args()
    cpu() if args.mode=='cpu' else gpu()
