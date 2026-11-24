"""One fresh binary ACA comparison: train temporal sampling only."""
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from development.aca_binary_bn001_v1 import formal_train as parent
from development.aca_binary_bn001_v1.model import initialized
from . import sampling
from .data import TemporalDataset, EpochStream

ROOT=parent.ROOT
ORIGINAL=parent.ORIGINAL
core=parent.core
runtime=parent.runtime
need=core.need
PARENT=parent.RUN
PARENT_SHA='0a9e3de9e28ca1588b44607d020370f8bb664e764dfe62a493eca468b65ac686'
RUN=parent.base.CURRENT/'aca_binary_temporal50_seed20261024_labelrev1_v1'
QA=parent.base.CURRENT/'aca_temporal50_acceptance_20261122_v1'
PREFIX='development/aca_temporal_v1/'
INPUT_CONTRACT=dict(parent_pixel_contract_sha256=parent.base.revised.labels.f.CONTRACT_SHA256,
                    train=sampling.POLICY,validation='exact_unchanged_AdjudicatedDataset_and_BinaryCollator')
INPUT_SHA=core.plan_digest(INPUT_CONTRACT)


def completed_parent(plan):
    s=ORIGINAL.read(PARENT/'summary.json');h=ORIGINAL.read(PARENT/'history.json');r=ORIGINAL.read(PARENT/'run.json')
    need(ORIGINAL.read(PARENT/'plan.json')==plan and core.plan_digest(plan)==PARENT_SHA,'Parent plan mismatch')
    need(s['status']=='completed' and s['test_run'] is False and s['run_id']==r['run_id'] and
         s['plan_sha256']==r['plan_sha256']==PARENT_SHA and len(h)==s['completed_epochs']>0,'Parent completion mismatch')
    best=min(h,key=lambda x:x['validation']['ce']);bad=len(h)-best['epoch']
    runtime.validate_state(dict(epoch=len(h),cursor=0,updates=s['updates'],
        train_metrics=core.metrics_state(core.BinaryMetrics()),history=h,best_epoch=best['epoch'],
        best_ce=best['validation']['ce'],bad_epochs=bad),plan,{})
    need(s['selected_validation']==best['validation'] and s['best_epoch']==best['epoch'] and
         (bad>=plan['patience'] or len(h)==plan['max_epochs']) and
         s['stop_reason']==('early_stop' if bad>=plan['patience'] else 'epoch_cap'),'Parent selection mismatch')
    return dict(directory=str(PARENT),run_id=r['run_id'],plan_sha256=PARENT_SHA,
                summary_sha256=ORIGINAL.sha(PARENT/'summary.json'),history_sha256=ORIGINAL.sha(PARENT/'history.json'))


def draft_plan():
    old=parent.build_plan();ref=completed_parent(old)
    plan=deepcopy(old)
    plan.update(purpose='single_seed_train_only_temporal_sampling_controlled_comparison',allowed_run_directory=str(RUN),
        input_contract_sha256=INPUT_SHA,temporal_input_contract=INPUT_CONTRACT,temporal_sampling=sampling.POLICY,
        temporal_experiment=dict(baseline=ref,only_change='train_uniform50_to_endpoint_stratified50',
            initialization='same_fresh_parameters_buffers_and_RNG',validation_pixels='identical_fixed_uniform50',
            reference_tensor_sha_scope='fixed_uniform_baseline_for_train_and_validation; new_train_SHA_recorded_per_epoch',
            inherited_momentum_experiment='historical_provenance_only_BN_0.01_is_unchanged',
            author_exact_sampling_verified=False))
    for p in sorted(Path(__file__).parent.glob('*.py')):
        plan['code_sha256'][str(p.relative_to(ROOT))]=ORIGINAL.sha(p)
    need({k for k in plan if plan[k]!=old.get(k)}=={'purpose','allowed_run_directory','input_contract_sha256',
        'temporal_input_contract','temporal_sampling','temporal_experiment','code_sha256'},'Unexpected protocol change')
    return plan


def build_plan():
    plan=draft_plan()
    visual=ORIGINAL.read(QA/'visual_review.json')
    need(visual['status']=='passed_bounded_candidate_review' and visual['new_blockers']==0 and
         len(visual['records'])==32,'Visual acceptance incomplete')
    for row in visual['records']:
        need(ORIGINAL.sha(QA/f"{row['review_index']:02d}.jpg")==row['image_sha256'],'Reviewed image changed')
    for name in ('cpu_report.json','gpu_report.json'):
        evidence=ORIGINAL.read(QA/name)
        need(evidence['status']=='passed' and evidence['test_video_decodes']==0 and
             evidence['draft_plan_sha256']==core.plan_digest(plan) and
             evidence['code_sha256']=={k:v for k,v in plan['code_sha256'].items() if k.startswith(PREFIX)},
             'Temporal acceptance/code mismatch')
    plan['temporal_readiness']={n:ORIGINAL.sha(QA/n) for n in
        ('cpu_report.json','gpu_report.json','visual_review.json','evidence.json')}
    return plan


def verify_plan(plan,digest):
    need(core.plan_digest(plan)==digest and plan==build_plan(),'Temporal plan identity mismatch')


class TrainingLoop(parent.TrainingLoop):
    def __init__(self,model,plan,fetch,*args,**kwargs):
        need(plan['temporal_sampling']==sampling.POLICY and plan['input_contract_sha256']==INPUT_SHA,'Wrong input policy')
        def guarded(split,index):
            batch=fetch(split,index)
            if split=='train':
                need(batch.get('temporal_sampling_version')==sampling.VERSION,'Fixed input incorrectly routed to train')
                need(all(r['metadata']['epoch_index']==self.state['epoch'] for r in batch['records']), 'Stale temporal epoch')
            else:
                need('temporal_sampling_version' not in batch,'Validation augmentation forbidden')
            return batch
        super().__init__(model,plan,guarded,*args,**kwargs)


def run_path(variant):
    need(variant=='aca','Only ACA seed24');return RUN


def predecessor(plan):
    need(completed_parent(parent.build_plan())==plan['temporal_experiment']['baseline'],'Parent changed')


def execute(path,digest):
    ORIGINAL.verify_plan=verify_plan;ORIGINAL.run_path=run_path;ORIGINAL.require_predecessor=predecessor
    ORIGINAL.initialized=initialized
    ORIGINAL.rt=SimpleNamespace(**{**vars(runtime),'VerifiedDataset':TemporalDataset,
        'EpochStream':EpochStream,'TrainingLoop':TrainingLoop})
    return ORIGINAL.execute(path,digest,RUN,resume=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('start');q.add_argument('--plan',type=Path,required=True);q.add_argument('--plan-sha256',required=True)
    a=p.parse_args()
    if a.command=='prepare':
        plan=build_plan();gpu=ORIGINAL.assert_idle_gpu(plan['gpu_policy']);out=ORIGINAL.safe_new_directory(a.output)
        runtime.atomic_json(out/'plan.json',plan)
        runtime.atomic_json(out/'review.json',dict(status='prepared_not_started',plan_sha256=core.plan_digest(plan),
            run_directory=str(RUN),gpu_processes=gpu,only_change='train temporal sampling',test_run=False))
        print(core.plan_digest(plan),flush=True)
    else:execute(a.plan,a.plan_sha256)

if __name__=='__main__':main()
