"""Single explicit MixNet-I3D run, audited shared numerical loop, no auto-restart."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import timm
from development.aca_reconstruction_v2 import formal_train as shell
from development.aca_eight_class_data_v1 import formal_train as base
from development.aca_binary_v1 import runtime, core
from development.aca_binary_v1.data import guard_fetch
from development.aca_temporal_v1.data import TemporalDataset, EpochStream
from development.aca_temporal_v1 import sampling
from development.adjudicated_input_v1 import adapter as labels
from development.aca_four_arm_campaign_v1 import gpu_guard
from .model import initialized, training_mode, validate_bn, BN_POLICY, WEIGHTS, WEIGHT_SHA, need

ROOT = Path(__file__).resolve().parents[2]
CURRENT = Path('/home/tsai/projects/basketball/outputs/01_current')
RUN = CURRENT/'mixnet_i3d_binary50_seed20261024_labelrev1_v1'
QA = CURRENT/'mixnet_binary50_formal_readiness_20261123_v1'
PRETRAIN = CURRENT/'mixnet_binary50_pretrained_acceptance_20261123_v1'
PARENT = CURRENT/'aca_binary_temporal50_seed20261024_labelrev1_v1'
PARENT_SHA = 'ee331c184ca8b0cea54550098bd926929e9f1b74398528948c62410a70a2495a'
SCHEMA = 'mixnet_i3d_binary50_formal_v1'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def source_hashes(parent):
    # Pin actual files in this worktree, including unchanged shared loop/data code.
    hashes = {}
    for name, expected in parent['code_sha256'].items():
        p = ROOT/name
        need(sha(p) == expected, 'Shared source changed: '+name)
        hashes[str(p)] = expected
    for folder in ('mixnet_i3d','mixnet_binary50_v1','mixnet_binary50_formal_v1'):
        for p in sorted((ROOT/'development'/folder).glob('*.py')):
            hashes[str(p)] = sha(p)
    # Some historical data contract resources resolve against the original tree.
    for name, expected in labels.PINNED_CODE.items():
        p=labels.f.ROOT/name
        need(sha(p)==expected,'Pinned data source changed')
        hashes[str(p)]=expected
    return dict(sorted(hashes.items()))

def draft_plan():
    parent=read(PARENT/'plan.json')
    need(core.plan_digest(parent)==PARENT_SHA, 'Comparison plan changed')
    reference=read(PARENT/'summary.json'); history=read(PARENT/'history.json')
    need(reference['status']=='completed' and reference['plan_sha256']==PARENT_SHA and
         len(history)==reference['completed_epochs'] and reference['selected_validation']==min(history,key=lambda r:r['validation']['ce'])['validation'], 'Comparison result invalid')
    rev=labels.LabelRevision(); manifest=rev._manifest
    need(parent['reference']==manifest['reference'] and parent['train_ids']==manifest['train_ids'] and
         parent['validation_ids']==manifest['validation_ids'], 'Data protocol changed')
    audit=read(PRETRAIN/'report.json')
    need(audit['status']=='PASS_PRETRAINED_CPU_ONLY' and audit['weights']['sha256']==WEIGHT_SHA,
         'Pretrained CPU audit missing')
    need(all(sha(p)==h for p,h in audit['source_sha256'].items()),'Pretrained-audited source changed')
    need(sha(WEIGHTS)==WEIGHT_SHA,'Weights changed')
    keys=('seed','max_epochs','patience','effective','checkpoint_every','device','class_order','train_ids',
          'validation_ids','reference','train_count','validation_count','updates_per_epoch','tail_videos',
          'source_root','data_version','label_revision_sha256','pixel_contract_parent_sha256','label_patch_sha256',
          'task','loss','prediction','optimizer','selection','temporal_sampling','temporal_input_contract','input_contract_sha256')
    plan={k:parent[k] for k in keys}
    plan.update(schema=SCHEMA, variant='mixnet_i3d',purpose='single_seed_system_comparison_pure_MixNet_I3D_vs_ACA_not_pure_backbone_causality',
        allowed_run_directory=str(RUN),normalization_policy=BN_POLICY,
        head=dict(in_features=1536,out_features=2,initialization_seed=20264024),
        initialization=dict(name='timm/mixnet_s.ft_in1k',path=str(WEIGHTS),sha256=WEIGHT_SHA,load='strict_CPU_then_existing_3D_inflation',
                            head='fresh_independent_CPU_seed_plus_3000',trained_weights_used=False),
        comparison=dict(run=str(PARENT),plan_sha256=PARENT_SHA,summary_sha256=sha(PARENT/'summary.json'),
                        changed=['backbone','attention_removed','pretraining_source','BN_policy'],author_exact=False),
        cpu_pretrained_evidence=dict(path=str(PRETRAIN/'report.json'),sha256=sha(PRETRAIN/'report.json')),
        runtime={**base.runtime_identity(),'timm':timm.__version__},
        source_inventory=base.source_inventory(plan['source_root'],plan['reference']),
        code_sha256=source_hashes(parent),gpu_policy=gpu_guard.make_policy(),
        test_enabled=False,automatic_restart=False,automatic_next_arm=False,
        checkpoint='single_atomic_last.pt_current_model_Adam_RNG_history_best_model',
        notifications='no_hourly_reports')
    need(plan['source_inventory']==parent['source_inventory'],'Source inventory changed')
    return plan

def build_plan():
    plan=draft_plan()
    for name in ('cpu_report.json','gpu_report.json'):
        result=read(QA/name)
        need(result['status']=='passed' and result['draft_plan_sha256']==core.plan_digest(plan), 'Readiness does not match current draft')
    plan['readiness']={name:sha(QA/name) for name in ('cpu_report.json','gpu_report.json')}
    return plan

def verify_plan(plan, digest):
    need(core.plan_digest(plan)==digest and plan==build_plan(),'Plan/code/data/runtime changed')

def install_runtime():
    # Process-local hooks only; never edit shared ACA source files.
    core.training_mode=training_mode
    runtime.training_mode=training_mode

class TrainingLoop(runtime.TrainingLoop):
    def __init__(self, model, plan, fetch, *args, **kwargs):
        need(plan['schema']==SCHEMA and plan['normalization_policy']==BN_POLICY and
             plan['temporal_sampling']==sampling.POLICY and plan['seed']==20261024,'Wrong formal model/input plan')
        labels.require_revision_identity(plan)
        binary_fetch=guard_fetch(fetch,plan)
        def guarded(split,index):
            batch=binary_fetch(split,index)
            need(tuple(batch['tensor'].shape)==(1,3,50,120,160),'Wrong single clip shape')
            if split=='train':
                need(batch.get('temporal_sampling_version')==sampling.VERSION and
                     all(r['metadata']['epoch_index']==self.state['epoch'] for r in batch['records']), 'Stale train temporal sampling')
            else:
                need('temporal_sampling_version' not in batch,'Validation augmentation forbidden')
            return batch
        super().__init__(model,plan,guarded,*args,**kwargs)
        validate_bn(model)
        need({id(p) for g in self.optimizer.param_groups for p in g['params']}==
             {id(p) for p in model.parameters() if p.requires_grad},'Optimizer parameter coverage mismatch')

def execute(path,digest):
    install_runtime()
    shell.verify_plan=verify_plan
    shell.run_path=lambda variant: RUN if variant=='mixnet_i3d' else None
    shell.require_predecessor=lambda plan: None  # validated comparison inside build_plan
    shell.initialized=initialized
    shell.rt=SimpleNamespace(**{**vars(runtime),'VerifiedDataset':TemporalDataset,'EpochStream':EpochStream,'TrainingLoop':TrainingLoop})
    return shell.execute(path,digest,RUN,resume=False)

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    s=sub.add_parser('start');s.add_argument('--plan',type=Path,required=True);s.add_argument('--plan-sha256',required=True)
    a=p.parse_args()
    if a.command=='prepare':
        plan=build_plan();gpu_guard.assert_idle_gpu(plan['gpu_policy'])
        target=QA/'formal_plan.json';need(not target.exists(),'Prepared plan already exists')
        runtime.atomic_json(target,plan)
        print(core.plan_digest(plan),flush=True)
    else:execute(a.plan,a.plan_sha256)

if __name__=='__main__':main()
