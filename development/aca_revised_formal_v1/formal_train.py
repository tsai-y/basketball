"""Bind the accepted label overlay to the unchanged formal ACA numerical loop.

Default CLI preparation never trains. Only explicit start/resume may use CUDA.
No old source files, run plans, checkpoints, or GPU policies are modified.
"""
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

from development.adjudicated_input_v1 import adapter as labels
from development.aca_reconstruction_v2 import formal_train as original
from development.aca_reconstruction_v2 import training_runtime as runtime
from development.aca_eight_class_data_v1 import training_runtime as base_runtime

ROOT = labels.f.ROOT
CURRENT = labels.CURRENT
RUN = CURRENT / 'aca_reconstruction_v2_aca_seed20261024_labelrev1_v1'
EXECUTION_SCHEMA = 'aca_label_revision_formal_launch_v1'
PARENT_SHA = '94d447a702270b2be9aabac9813164bb45d8ff2d899d26e5130fafe1d8c23913'
QA = labels.REVISION_DIR / 'cpu_acceptance_v1/report.json'
INTEGRATION = CURRENT / 'aca_labelrev1_integration_20261121_v1/report.json'
need = original.need


def parent_plan():
    # Original build_plan must retain the original completed-arm directory rules.
    # This restores only this process-local routing hook while deriving its plan.
    current = original.run_path
    try:
        original.run_path = _ORIGINAL_RUN_PATH
        result = original.build_plan('/home/tsai/projects/basketball', 'aca')
    finally:
        original.run_path = current
    need(original.core.plan_digest(result) == PARENT_SHA, 'Original frozen ACA plan changed')
    return result


def data_plan(parent):
    revision = labels.LabelRevision()
    manifest = revision._manifest
    qa = original.read(QA)
    need(qa['status'] == 'passed' and qa['unit_tests_run'] == 12
         and qa['label_revision_sha256'] == labels.MANIFEST_SHA
         and qa['real_video_id'] == labels.TARGET and qa['batch_targets'] == [1]
         and qa['tensor_matches_prior_reference'], 'Missing label loader CPU acceptance')
    for name, expected in qa['code_sha256'].items():
        need(original.sha(ROOT / 'development/adjudicated_input_v1' / name) == expected,
             'Accepted label adapter code changed')
    plan = deepcopy(parent)
    plan.update(
        execution_schema=EXECUTION_SCHEMA,
        purpose='single_ACA_label_revision_run_not_same_dataset_ablation_comparison',
        data_version=labels.VERSION, label_revision_sha256=labels.MANIFEST_SHA,
        pixel_contract_parent_sha256=labels.f.CONTRACT_SHA256,
        label_patch_sha256=labels.PATCH_SHA,
        parent_plan_sha256=PARENT_SHA,
        reference=deepcopy(manifest['reference']),
        train_ids=list(manifest['train_ids']), validation_ids=list(manifest['validation_ids']),
        allowed_run_directory=str(RUN), execution_order=['aca'],
        automatic_next_arm=False, automatic_restart=False,
        initialization='fresh_fixed_ImageNet_V1_no_trained_weights',
        label_loader_acceptance={'path': str(QA), 'sha256': original.sha(QA)},
    )
    hashes = deepcopy(parent['code_sha256'])
    for directory in ('adjudicated_input_v1', 'aca_revised_formal_v1'):
        for p in sorted((ROOT / 'development' / directory).glob('*.py')):
            hashes[str(p.relative_to(ROOT))] = original.sha(p)
    plan['code_sha256'] = dict(sorted(hashes.items()))
    labels.require_revision_identity(plan)
    return plan


def build_plan():
    plan = data_plan(parent_plan())
    qa = original.read(INTEGRATION)
    need(qa['status'] == 'passed' and qa['optimizer_updates'] == 0
         and qa['real_worker_decodes'] == 1 and qa['worker_batch_target'] == 1
         and qa['worker_tensor_sha256'] == plan['reference'][labels.TARGET]['tensor_sha256']
         and qa['old_plan_rejected'] and qa['stale_identity_rejected'],
         'Missing formal integration CPU acceptance')
    for name, expected in qa['code_sha256'].items():
        need(original.sha(ROOT/name) == expected, 'Integration-tested source changed')
    plan['formal_integration_acceptance'] = {'path': str(INTEGRATION), 'sha256': original.sha(INTEGRATION)}
    return plan


def verify_plan(plan, expected_hash):
    labels.require_revision_identity(plan)
    need(plan.get('execution_schema') == EXECUTION_SCHEMA
         and original.core.plan_digest(plan) == expected_hash, 'Revised plan identity mismatch')
    need(plan == build_plan(), 'Code, runtime, data, or numerical protocol changed')


def new_run_path(variant):
    need(variant == 'aca', 'Only the single requested ACA arm is available')
    return RUN


class RevisedDataset(labels.AdjudicatedDataset):
    def __init__(self, source_root, split, ids, reference):
        super().__init__(source_root=source_root, split=split, video_ids=ids)
        need(reference == self.revision._manifest['reference'], 'Run reference differs from label revision')


class RevisedEpochStream(base_runtime.EpochStream):
    def __init__(self, datasets, plan, state_getter):
        labels.require_revision_identity(plan)
        super().__init__(datasets, plan, state_getter,
                         collate=labels.AdjudicatedCollator(), workers=2)


class RevisedTrainingLoop(runtime.TrainingLoop):
    def __init__(self, model, plan, fetch, checkpoint, run_id, *, publish=None):
        labels.require_revision_identity(plan)
        need(plan.get('execution_schema') == EXECUTION_SCHEMA and plan['variant'] == 'aca',
             'Only revised ACA plans accepted')
        revision = labels.LabelRevision()

        def guarded_fetch(split, index):
            batch = fetch(split, index)
            labels.require_revision_identity(batch)
            need(batch['label_revision_sha256'] == plan['label_revision_sha256'],
                 'Batch/plan revision mismatch')
            vid = plan[split + '_ids'][index]
            ref = revision.record(vid, split)
            need(batch['targets'].tolist() == [ref['class_id']]
                 and batch['binary_targets'].tolist() == [ref['binary_label']]
                 and batch['records'][0]['original_class'] == ref['original_class']
                 and batch['records'][0]['effective_class'] == ref['effective_class'],
                 'Effective training target mismatch')
            return batch

        super().__init__(model, plan, guarded_fetch, checkpoint, run_id, publish=publish)

    def _payload(self, state=None, best=None):
        payload = super()._payload(state, best)
        payload['data_version'] = self.plan['data_version']
        payload['label_revision_sha256'] = self.plan['label_revision_sha256']
        return payload

    def resume(self):
        # Before torch.load or mutating any model/RNG, confirm new plan data identity.
        labels.require_revision_identity(self.plan)
        # Parent resume then checks the entire embedded plan/hash and run_id before
        # loading weights, so historical checkpoints cannot cross this boundary.
        return super().resume()


_ORIGINAL_RUN_PATH = original.run_path


def _predecessor(plan):
    labels.require_revision_identity(plan)
    parent = original.read(CURRENT / 'aca_reconstruction_v2_aca_seed20261024_v1/plan.json')
    need(original.core.plan_digest(parent) == PARENT_SHA, 'Wrong historical parent')
    original.completed_result(CURRENT / 'aca_reconstruction_v2_aca_seed20261024_v1', parent)


def execute(plan_path, expected_hash, *, resume=False):
    # Reuse the original execute/locks/reporting/numerical implementation. Only
    # explicit dependency routing changes in this one launch process.
    original.verify_plan = verify_plan
    original.run_path = new_run_path
    original.require_predecessor = _predecessor
    original.rt = SimpleNamespace(**{**vars(runtime),
        'VerifiedDataset': RevisedDataset, 'EpochStream': RevisedEpochStream,
        'TrainingLoop': RevisedTrainingLoop})
    return original.execute(plan_path, expected_hash, RUN, resume=resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--output', type=Path, required=True)
    for command in ('start', 'resume'):
        p = sub.add_parser(command)
        p.add_argument('--plan', type=Path, required=True)
        p.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = build_plan()
        observed = original.assert_idle_gpu(plan['gpu_policy'])
        path = original.safe_new_directory(args.output)
        runtime.atomic_json(path/'plan.json', plan)
        runtime.atomic_json(path/'review.json', {
            'status': 'prepared_not_started', 'at_taiwan': runtime.now(),
            'plan_sha256': original.core.plan_digest(plan), 'gpu_processes': observed,
            'run_directory': str(RUN), 'model_instances': 0, 'video_decodes': 0,
            'optimizer_updates': 0, 'automatic_restart': False})
        print(original.core.plan_digest(plan), flush=True)
    else:
        execute(args.plan, args.plan_sha256, resume=args.command == 'resume')


if __name__ == '__main__':
    main()
