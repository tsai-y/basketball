"""Fresh seed24 temporal ACA: Adam LR 1e-4 -> 5e-5 only; no resume or queue."""
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from development.aca_temporal_v1 import formal_train as parent

ROOT, ORIGINAL, core, runtime = parent.ROOT, parent.ORIGINAL, parent.core, parent.runtime
need = core.need
PARENT = parent.RUN
PARENT_SHA = 'ee331c184ca8b0cea54550098bd926929e9f1b74398528948c62410a70a2495a'
RUN = PARENT.parent/'aca_binary_temporal50_lr5e5_seed20261024_labelrev1_v1'
PREP = PARENT.parent.parent/'00_admin/aca_temporal50_lr5e5_prepare_20261123_v1'
PREFIX = 'development/aca_temporal_lr5e5_v1/'
LR = 0.00005


def parent_reference(plan):
    need(ORIGINAL.read(PARENT/'plan.json') == plan and core.plan_digest(plan) == PARENT_SHA, 'Parent plan changed')
    s, h, r = (ORIGINAL.read(PARENT/n) for n in ('summary.json','history.json','run.json'))
    need(s['status']=='completed' and s['test_run'] is False and s['run_id']==r['run_id'] and
         s['plan_sha256']==r['plan_sha256']==PARENT_SHA and len(h)==s['completed_epochs']>0, 'Parent not complete')
    best=min(h,key=lambda x:x['validation']['ce']);bad=len(h)-best['epoch']
    runtime.validate_state(dict(epoch=len(h),cursor=0,updates=s['updates'],
        train_metrics=core.metrics_state(core.BinaryMetrics()),history=h,best_epoch=best['epoch'],
        best_ce=best['validation']['ce'],bad_epochs=bad),plan,{})
    need(s['selected_validation']==best['validation'] and s['best_epoch']==best['epoch'] and
         (bad>=plan['patience'] or len(h)==plan['max_epochs']) and
         s['stop_reason']==('early_stop' if bad>=plan['patience'] else 'epoch_cap'), 'Parent selection mismatch')
    return dict(directory=str(PARENT),run_id=r['run_id'],plan_sha256=PARENT_SHA,
                summary_sha256=ORIGINAL.sha(PARENT/'summary.json'),history_sha256=ORIGINAL.sha(PARENT/'history.json'))


def draft_plan():
    old=parent.build_plan();ref=parent_reference(old)
    plan=deepcopy(old)
    need(old['optimizer']['lr']==0.0001 and old['seed']==20261024, 'Unexpected parent LR/seed')
    plan['optimizer']['lr']=LR
    plan.update(purpose='single_seed_temporal_ACA_Adam_lr_1e4_to_5e5_controlled_comparison',
        allowed_run_directory=str(RUN),
        lr_experiment=dict(baseline=ref,only_change='Adam_learning_rate',before=0.0001,after=LR,
            fresh_initialization=True,automatic_followup=False,
            inherited_momentum_and_temporal_experiments='historical_provenance_only_unchanged',
            decision='No CE improvement: stop candidate. CE improves: inspect accuracy/F1/recall tradeoffs before any seed25 authorization.'))
    for p in sorted(Path(__file__).parent.glob('*.py')):
        plan['code_sha256'][str(p.relative_to(ROOT))]=ORIGINAL.sha(p)
    need({k for k in plan if plan[k]!=old.get(k)}==
         {'purpose','allowed_run_directory','optimizer','lr_experiment','code_sha256'},'Unexpected protocol change')
    need({k for k in plan['optimizer'] if plan['optimizer'][k]!=old['optimizer'][k]}=={'lr'},'Optimizer changed beyond LR')
    return plan


def build_plan():
    plan=draft_plan();e=ORIGINAL.read(PREP/'cpu_acceptance.json')
    need(e['status']=='passed' and e['draft_plan_sha256']==core.plan_digest(plan) and
         e['gpu_training_started'] is False and e['video_decodes']==0 and e['formal_optimizer_updates']==0,
         'CPU acceptance mismatch')
    plan['lr_cpu_acceptance_sha256']=ORIGINAL.sha(PREP/'cpu_acceptance.json')
    return plan


def verify_plan(plan,digest):
    need(core.plan_digest(plan)==digest and plan==build_plan(),'LR plan/source/runtime mismatch')


def set_fresh_learning_rate(optimizer, plan):
    expected=dict(name='Adam',lr=LR,betas=[0.9,0.999],eps=1e-8,weight_decay=0.,clip_norm=1.)
    need(plan['optimizer']==expected, 'Unexpected candidate optimizer configuration')
    need(not optimizer.state and len(optimizer.param_groups)==1, 'LR override only before first update')
    g=optimizer.param_groups[0]
    need(g['lr']==0.0001 and g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0. and
         g['amsgrad'] is False and g['foreach'] is False and g['fused'] is False, 'Inherited Adam changed')
    g['lr']=LR
    optimizer.defaults['lr']=LR


class TrainingLoop(parent.TrainingLoop):
    def __init__(self,model,plan,*args,**kwargs):
        need(plan['lr_experiment']['after']==LR and plan['seed']==20261024, 'Wrong LR experiment')
        super().__init__(model,plan,*args,**kwargs)
        need(self.state['updates']==0 and not self.started,'Must use a fresh run')
        set_fresh_learning_rate(self.optimizer,plan)


def initialized(variant='aca',seed=20261024):
    need(variant=='aca' and seed==20261024,'Only ACA seed24')
    model,evidence=parent.initialized(variant,seed)
    evidence={**evidence,'inherited_single_change':evidence.get('single_change'),
        'single_change':dict(field='Adam_learning_rate',before=0.0001,after=LR),
        'optimizer_lr':LR,'initialization_unchanged_from_temporal_parent':True}
    return model,evidence


def run_path(variant):
    need(variant=='aca','Only ACA');return RUN


def predecessor(plan):
    need(parent_reference(parent.build_plan())==plan['lr_experiment']['baseline'],'Reference changed')


def execute(path,digest):
    ORIGINAL.verify_plan=verify_plan;ORIGINAL.run_path=run_path;ORIGINAL.require_predecessor=predecessor
    ORIGINAL.initialized=initialized
    ORIGINAL.rt=SimpleNamespace(**{**vars(runtime),'VerifiedDataset':parent.TemporalDataset,
        'EpochStream':parent.EpochStream,'TrainingLoop':TrainingLoop})
    return ORIGINAL.execute(path,digest,RUN,resume=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('start');q.add_argument('--plan',type=Path,required=True);q.add_argument('--plan-sha256',required=True)
    a=p.parse_args();execute(a.plan,a.plan_sha256)

if __name__=='__main__':main()
