"""Exactly two fresh seed25 runs; immutable parent implementations and numerical loop."""
import argparse
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from development.aca_binary_bn001_v1 import formal_train as fixed
from development.aca_temporal_v1 import formal_train as temporal
from development.aca_binary_v1.data import RevisedDataset, EpochStream as FixedStream
from .data import TemporalDataset, EpochStream as TemporalStream
from .model import initialized

ROOT=fixed.ROOT
O=fixed.ORIGINAL
core=fixed.core
rt=fixed.runtime
need=core.need
CURRENT=fixed.base.CURRENT
CAMPAIGN=CURRENT/'aca_temporal_pair_seed20261025_20261122_v1'
SEED=20261025
ARMS=('fixed','temporal')
PARENTS={'fixed':fixed,'temporal':temporal}
PARENT_SHA={'fixed':'0a9e3de9e28ca1588b44607d020370f8bb664e764dfe62a493eca468b65ac686',
 'temporal':'ee331c184ca8b0cea54550098bd926929e9f1b74398528948c62410a70a2495a'}
_ORIGINAL_CONFIGURE=O.configure

def run_path(arm):
    need(arm in ARMS,'Only two authorized arms')
    return CURRENT/('aca_binary_'+('bn001' if arm=='fixed' else 'temporal50')+'_seed20261025_labelrev1_v1')

def completed(path,plan):
    s=O.read(path/'summary.json');h=O.read(path/'history.json');r=O.read(path/'run.json')
    need(O.read(path/'plan.json')==plan and s['status']=='completed' and s['test_run'] is False,'Parent unfinished')
    need(s['run_id']==r['run_id'] and s['plan_sha256']==r['plan_sha256']==core.plan_digest(plan),'Parent identity mismatch')
    best=min(h,key=lambda x:x['validation']['ce']);bad=len(h)-best['epoch']
    rt.validate_state(dict(epoch=len(h),cursor=0,updates=s['updates'],train_metrics=core.metrics_state(core.BinaryMetrics()),
        history=h,best_epoch=best['epoch'],best_ce=best['validation']['ce'],bad_epochs=bad),plan,{})
    need(s['completed_epochs']==len(h) and s['best_epoch']==best['epoch'] and s['selected_validation']==best['validation']
         and (bad>=plan['patience'] or len(h)==plan['max_epochs'])
         and s['stop_reason']==('early_stop' if bad>=plan['patience'] else 'epoch_cap'),'Completion selection mismatch')
    return dict(directory=str(path),run_id=r['run_id'],plan_sha256=r['plan_sha256'],
        summary_sha256=O.sha(path/'summary.json'),history_sha256=O.sha(path/'history.json'))

def draft_plan(arm):
    need(arm in ARMS,'Unknown arm')
    parent=PARENTS[arm];old=parent.build_plan()
    need(core.plan_digest(old)==PARENT_SHA[arm],'Pinned seed24 plan changed')
    ref=completed(parent.RUN,old)
    p=deepcopy(old)
    p.update(seed=SEED,purpose='bounded_second_seed_paired_confirmation_of_train_temporal_sampling',
        allowed_run_directory=str(run_path(arm)))
    p['head']['initialization_seed']=SEED+3000
    p['paired_confirmation']=dict(campaign_directory=str(CAMPAIGN),arm=arm,seed=SEED,
        order=list(ARMS),authorized_runs=2,seed24_reference=ref,
        only_within_seed_numerical_change='train_uniform50_to_endpoint_stratified50',
        inherited_readiness_scope='seed24_implementation_provenance; new_seed25_acceptance_separate',
        inherited_momentum_and_temporal_experiment_scope='seed24_history_only_not_new_changes',
        seed_override_scope='model_module_head_RNG_data_order_worker_seed_temporal_sampling',
        author_exact=False,test_enabled=False)
    for f in sorted(Path(__file__).parent.glob('*.py')):
        p['code_sha256'][str(f.relative_to(ROOT))]=O.sha(f)
    need({k for k in p if p[k]!=old.get(k)}=={'seed','purpose','allowed_run_directory','head','paired_confirmation','code_sha256'},
        'Unplanned change beyond seed25 routing')
    return p

def build_plan(arm):
    p=draft_plan(arm);qa=O.read(CAMPAIGN/'acceptance.json')
    need(qa['status']=='passed' and qa['test_video_decodes']==0 and qa['draft_hashes'][arm]==core.plan_digest(p),
        'Seed25 acceptance mismatch')
    p['paired_readiness_sha256']=O.sha(CAMPAIGN/'acceptance.json')
    return p

def verify_plan(p,digest):
    arm=p['paired_confirmation']['arm']
    need(core.plan_digest(p)==digest and p==build_plan(arm),'Paired plan/source/runtime mismatch')

def configure():
    _ORIGINAL_CONFIGURE()
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

def execute(arm,path,digest):
    need(arm in ARMS,'Unknown arm')
    def predecessor(p):
        need(p['seed']==SEED and p['paired_confirmation']['arm']==arm,'Wrong seed/arm')
        if arm=='temporal': completed(run_path('fixed'),build_plan('fixed'))
    def model(variant,seed):
        need(seed==SEED,'Wrong formal seed')
        return initialized(variant,seed)
    O.verify_plan=verify_plan
    O.run_path=lambda variant:run_path(arm) if variant=='aca' else (_ for _ in ()).throw(ValueError('Only ACA'))
    O.require_predecessor=predecessor
    O.initialized=model
    O.configure=configure
    O.rt=SimpleNamespace(**{**vars(rt),'VerifiedDataset':RevisedDataset if arm=='fixed' else TemporalDataset,
        'EpochStream':FixedStream if arm=='fixed' else TemporalStream,
        'TrainingLoop':fixed.TrainingLoop if arm=='fixed' else temporal.TrainingLoop})
    return O.execute(path,digest,run_path(arm),resume=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['start']);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    a=p.parse_args();execute(a.arm,a.plan,a.plan_sha256)
if __name__=='__main__':main()
