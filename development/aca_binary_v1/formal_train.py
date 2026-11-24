"""Explicit single-run binary ACA prepare/start/resume; no automatic continuation."""
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from development.aca_revised_formal_v1 import formal_train as revised
from development.aca_reconstruction_v2 import formal_train as original, training_runtime as old_runtime
from . import runtime, core
from .data import RevisedDataset, EpochStream, guard_fetch
from .model import initialized

ROOT = revised.ROOT
CURRENT = revised.CURRENT
RUN = CURRENT/'aca_binary_seed20261024_labelrev1_v1'
PARENT = revised.RUN
PARENT_SHA = '9c405ffba9496844b2b4a86dea36f972192a5280b7c6288987cce593a07d373d'
QA = CURRENT/'aca_binary_readiness_20261121_v1'
SCHEMA = 'aca_direct_binary_formal_plan_v1'
need = core.need

class TrainingLoop(runtime.TrainingLoop):
    def __init__(self, model, plan, fetch, checkpoint, run_id, *, publish=None):
        revised.labels.require_revision_identity(plan)
        need(plan['schema'] == SCHEMA and plan['task'] == 'basketball51_direct_binary', 'Binary formal plan required')
        super().__init__(model, plan, guard_fetch(fetch, plan), checkpoint, run_id, publish=publish)


def draft_plan():
    # Parent equality includes frozen source inventory and current host runtime.
    previous = original.rt
    try:
        original.rt = old_runtime
        parent = revised.build_plan()
        need(core.plan_digest(parent) == PARENT_SHA, 'Revised parent changed')
        predecessor = original.completed_result(PARENT, parent)
    finally:
        original.rt = previous
    plan = deepcopy(parent)
    plan.update(schema=SCHEMA, execution_schema=SCHEMA, task='basketball51_direct_binary',
                purpose='single_seed_direct_binary_ACA_vs_eight_class_task_setting_not_pure_head_causality',
                class_order=['Miss','Make'], source_eight_class_order=parent['class_order'],
                class_to_binary=[0,1,0,1,0,1,0,1], allowed_run_directory=str(RUN),
                parent_plan_sha256=PARENT_SHA, completed_eight_class_reference=predecessor,
                prediction='two_logits_argmax_tie_Miss', loss='unweighted_two_class_cross_entropy_no_smoothing',
                optimizer={'name':'Adam','lr':0.0001,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.,'clip_norm':1.},
                head={'in_features':2048,'out_features':2,'initialization_seed':20264024},
                selection='strict_lowest_complete_binary_validation_CE_ties_keep_earlier')
    for p in sorted(Path(__file__).parent.glob('*.py')):
        plan['code_sha256'][str(p.relative_to(ROOT))] = original.sha(p)
    return plan


def build_plan():
    plan = draft_plan()
    for name in ['cpu_report.json','gpu_report.json']:
        path = QA/name; evidence = original.read(path)
        need(evidence['status']=='passed' and evidence['test_video_decodes']==0, 'Readiness failed')
        need(evidence['draft_plan_sha256']==core.plan_digest(plan), 'Readiness plan differs')
        need(evidence['code_sha256']=={k:v for k,v in plan['code_sha256'].items() if k.startswith('development/aca_binary_v1/')}, 'Readiness code differs')
        plan_key = 'binary_'+name.replace('.json','')
        # QA hashes are appended after checking against the same draft.
    plan['binary_readiness']={name:original.sha(QA/name) for name in ['cpu_report.json','gpu_report.json']}
    return plan


def verify_plan(plan, expected_hash):
    revised.labels.require_revision_identity(plan)
    need(core.plan_digest(plan)==expected_hash and plan==build_plan(), 'Binary plan/runtime/source mismatch')


def run_path(variant):
    need(variant=='aca','Only ACA');return RUN


def predecessor(plan):
    need(original.read(PARENT/'summary.json')['status']=='completed', 'Parent unfinished')
    need(original.sha(PARENT/'summary.json')==plan['completed_eight_class_reference']['summary_sha256'], 'Parent result changed')


def execute(path, digest, *, resume=False):
    original.verify_plan=verify_plan
    original.run_path=run_path
    original.require_predecessor=predecessor
    original.initialized=initialized
    original.rt=SimpleNamespace(**{**vars(runtime), 'VerifiedDataset':RevisedDataset,
                                  'EpochStream':EpochStream,'TrainingLoop':TrainingLoop})
    return original.execute(path,digest,RUN,resume=resume)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,required=True)
    for name in ['start','resume']:
        p=sub.add_parser(name);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        plan=build_plan();gpu=original.assert_idle_gpu(plan['gpu_policy']);out=original.safe_new_directory(args.output)
        runtime.atomic_json(out/'plan.json',plan)
        runtime.atomic_json(out/'review.json',{'status':'prepared_not_started','plan_sha256':core.plan_digest(plan),
            'run_directory':str(RUN),'gpu_processes':gpu,'initialization':'fresh_ImageNet_V1','test_run':False})
        print(core.plan_digest(plan),flush=True)
    else:execute(args.plan,args.plan_sha256,resume=args.command=='resume')

if __name__=='__main__':main()
