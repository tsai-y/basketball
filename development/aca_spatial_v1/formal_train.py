"""Single-seed direct-binary ACA formal run: add clip-consistent spatial aug only.

Compares directly against completed aca_binary_temporal50_seed20261024_labelrev1_v1.
No existing modules, splits, labels, or historical runs are modified.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from development.aca_reconstruction_v2 import formal_train as shell
from development.aca_eight_class_data_v1 import formal_train as base
from development.aca_binary_v1 import core, runtime
from development.aca_binary_v1.data import guard_fetch
from development.aca_binary_bn001_v1.model import initialized, validate_momentum
from development.aca_temporal_v1 import sampling
from development.aca_four_arm_campaign_v1 import gpu_guard
from development.adjudicated_input_v1 import adapter as labels

from . import augment
from .data import SpatialDataset, EpochStream

ROOT = Path(__file__).resolve().parents[2]
CURRENT = Path('/home/tsai/projects/basketball/outputs/01_current')
PARENT = CURRENT / 'aca_binary_temporal50_seed20261024_labelrev1_v1'
PARENT_SHA = 'ee331c184ca8b0cea54550098bd926929e9f1b74398528948c62410a70a2495a'
CPU_QA = CURRENT / 'aca_spatial_cpu_acceptance_20261123_v2'
QA = CURRENT / 'aca_spatial_formal_readiness_20261123_v1'
RUN = CURRENT / 'aca_binary_spatial_temporal50_seed20261024_labelrev1_v1'
SCHEMA = 'aca_binary_spatial_temporal50_formal_v1'
ANTIGRAVITY_EXE = Path('/opt/antigravity/antigravity')
need = core.need


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def make_gpu_policy():
    """Extend immutable gpu_guard policy with pinned Antigravity GUI gpu-process only."""
    base_policy = gpu_guard.make_policy()
    gui_allowlist = []
    if ANTIGRAVITY_EXE.is_file() and not ANTIGRAVITY_EXE.is_symlink():
        gui_allowlist.append({
            'path': str(ANTIGRAVITY_EXE.resolve(strict=True)),
            'required_cmdline_flag': '--type=gpu-process',
            'sha256': sha(ANTIGRAVITY_EXE),
            'reason': 'Active Antigravity desktop IDE GUI renderer (non-compute)',
        })
    return {
        **base_policy,
        'schema': 'aca_gpu_desktop_policy_v2_antigravity_gui',
        'base_policy_schema': base_policy['schema'],
        'pinned_gui_gpu_processes': gui_allowlist,
    }


def verify_gpu_policy(policy):
    need(policy.get('schema') == 'aca_gpu_desktop_policy_v2_antigravity_gui', 'Unexpected GPU policy schema')
    base_view = {k: v for k, v in policy.items() if k not in ('base_policy_schema', 'pinned_gui_gpu_processes')}
    base_view['schema'] = policy['base_policy_schema']
    gpu_guard.verify_policy(base_view)
    for row in policy.get('pinned_gui_gpu_processes', []):
        p = Path(row['path'])
        need(p.is_file() and not p.is_symlink() and str(p.resolve(strict=True)) == row['path'],
             'Pinned GUI executable path changed')
        need(sha(p) == row['sha256'], 'Pinned GUI executable SHA256 changed')
        need(row['required_cmdline_flag'] == '--type=gpu-process', 'Unexpected GUI cmdline flag')


def _is_pinned_gui(pid, pinned_guis):
    try:
        exe = str((Path('/proc') / str(pid) / 'exe').resolve(strict=True))
        raw = (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\0', b' ')
        args = raw.decode('utf-8', errors='replace').split()
    except (FileNotFoundError, PermissionError):
        return False
    for row in pinned_guis:
        if exe == row['path'] and row['required_cmdline_flag'] in args:
            return True
    return False


def assert_idle_gpu(policy):
    verify_gpu_policy(policy)
    query = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True,
    )
    extra = {r['path'] for r in policy['extra_executables']}
    pinned_guis = policy.get('pinned_gui_gpu_processes', [])
    observed = []
    for raw_pid in query.split():
        pid = int(raw_pid)
        cls = gpu_guard.classify(pid, extra)
        if cls not in ('self', 'exited', 'desktop') and _is_pinned_gui(pid, pinned_guis):
            cls = 'desktop_pinned_gui'
        observed.append({'pid': pid, 'classification': cls})
    busy = [r for r in observed if r['classification'] not in ('self', 'exited', 'desktop', 'desktop_pinned_gui')]
    need(not busy, f'Other GPU compute processes active: {busy}; none were stopped')
    free = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        text=True,
    ).split()
    need(len(free) == 1 and int(free[0]) >= policy['minimum_free_gpu_MiB'], 'Insufficient free GPU memory')
    need(shutil.disk_usage(gpu_guard.FOLDER).free >= 8 * 1024**3, 'Insufficient checkpoint space')
    return observed


def completed_parent():
    parent = read(PARENT / 'plan.json')
    need(core.plan_digest(parent) == PARENT_SHA, 'Parent temporal50 plan changed')
    s = read(PARENT / 'summary.json')
    h = read(PARENT / 'history.json')
    r = read(PARENT / 'run.json')
    need(s['status'] == 'completed' and s['test_run'] is False and s['run_id'] == r['run_id'] and
         s['plan_sha256'] == r['plan_sha256'] == PARENT_SHA and len(h) == s['completed_epochs'] > 0,
         'Parent completion mismatch')
    best = min(h, key=lambda x: x['validation']['ce'])
    need(s['selected_validation'] == best['validation'] and s['best_epoch'] == best['epoch'],
         'Parent best selection mismatch')
    ref = {
        'directory': str(PARENT),
        'run_id': r['run_id'],
        'plan_sha256': PARENT_SHA,
        'summary_sha256': sha(PARENT / 'summary.json'),
        'history_sha256': sha(PARENT / 'history.json'),
        'best_epoch': best['epoch'],
        'selected_validation': best['validation'],
    }
    return parent, ref


def source_hashes(parent):
    hashes = {}
    for name, expected in parent['code_sha256'].items():
        p = ROOT / name
        need(sha(p) == expected, 'Shared source changed: ' + name)
        hashes[name] = expected
    for p in sorted((ROOT / 'development/aca_spatial_v1').glob('*.py')):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    for name, expected in labels.PINNED_CODE.items():
        p = labels.f.ROOT / name
        need(sha(p) == expected, 'Pinned data source changed: ' + name)
    return dict(sorted(hashes.items()))


def verified_cpu_acceptance():
    report_path = CPU_QA / 'cpu/cpu_report.json'
    report = read(report_path)
    need(report['status'] == 'PASS_CPU_ONLY' and
         report['policy'] == augment.POLICY and
         report['policy_sha256'] == augment.POLICY_SHA and
         report['baseline_run'] == PARENT.name,
         'Spatial CPU acceptance mismatch')
    for path_str, expected in report['new_source_sha256'].items():
        need(sha(path_str) == expected, 'CPU-audited aca_spatial_v1 source changed: ' + path_str)
    for path_str, expected in report['protected_source_sha256'].items():
        need(sha(path_str) == expected, 'Protected source changed: ' + path_str)
    return {
        'cpu_report_path': str(report_path),
        'cpu_report_sha256': sha(report_path),
        'visual_review_sha256': sha(CPU_QA / 'visual_review.json'),
        'policy_sha256': augment.POLICY_SHA,
        'user_accepted_residual_highlights_at_095_105': True,
    }


def draft_plan():
    parent, ref = completed_parent()
    rev = labels.LabelRevision()
    manifest = rev._manifest
    need(parent['reference'] == manifest['reference'] and
         parent['train_ids'] == manifest['train_ids'] and
         parent['validation_ids'] == manifest['validation_ids'],
         'Data protocol changed')
    cpu_evidence = verified_cpu_acceptance()
    spatial_contract = {
        'parent_temporal_input_contract_sha256': parent['input_contract_sha256'],
        'temporal_train': sampling.POLICY,
        'spatial_train': augment.POLICY,
        'validation': 'exact_unchanged_AdjudicatedDataset_and_BinaryCollator',
    }
    plan = deepcopy(parent)
    plan.update(
        schema=SCHEMA,
        purpose='single_seed_train_only_clip_consistent_spatial_aug_controlled_comparison',
        allowed_run_directory=str(RUN),
        spatial_augmentation=augment.POLICY,
        spatial_policy_sha256=augment.POLICY_SHA,
        spatial_input_contract=spatial_contract,
        input_contract_sha256=core.plan_digest(spatial_contract),
        spatial_experiment={
            'baseline': ref,
            'only_change': 'train_add_clip_flip_color_light_v2_on_top_of_endpoint_stratified50',
            'initialization': 'same_fresh_ACA_BN001_parameters_buffers_and_RNG',
            'validation_pixels': 'identical_fixed_uniform50_zero_augmentation',
            'cpu_acceptance': cpu_evidence,
            'author_exact_augmentation_verified': False,
        },
        runtime=base.runtime_identity(),
        source_inventory=base.source_inventory(plan['source_root'], plan['reference']),
        code_sha256=source_hashes(parent),
        gpu_policy=make_gpu_policy(),
        test_enabled=False,
        automatic_restart=False,
        automatic_next_arm=False,
        notifications='no_hourly_reports',
    )
    need(plan['source_inventory'] == parent['source_inventory'], 'Source inventory changed')
    return plan


def build_plan():
    plan = draft_plan()
    gpu_ev = read(QA / 'gpu_report.json')
    need(gpu_ev['status'] == 'passed' and
         gpu_ev['draft_plan_sha256'] == core.plan_digest(plan) and
         gpu_ev['validation_video_decodes'] == 0 and
         gpu_ev['test_video_decodes'] == 0 and
         all(gpu_ev['continuous_vs_resumed_equal'].values()),
         'GPU readiness missing or does not match current draft')
    plan['spatial_gpu_readiness'] = {
        'gpu_report_path': str(QA / 'gpu_report.json'),
        'gpu_report_sha256': sha(QA / 'gpu_report.json'),
    }
    return plan


def verify_plan(plan, digest):
    need(core.plan_digest(plan) == digest and plan == build_plan(),
         'Spatial plan/code/data/runtime identity mismatch')


class TrainingLoop(runtime.TrainingLoop):
    def __init__(self, model, plan, fetch, *args, **kwargs):
        need(plan['schema'] == SCHEMA and
             plan['temporal_sampling'] == sampling.POLICY and
             plan['spatial_augmentation'] == augment.POLICY and
             plan['seed'] == 20261024,
             'Wrong formal spatial plan')
        labels.require_revision_identity(plan)
        validate_momentum(model)
        binary_fetch = guard_fetch(fetch, plan)

        def guarded(split, index):
            batch = binary_fetch(split, index)
            need(tuple(batch['tensor'].shape) == (1, 3, 50, 120, 160), 'Wrong single clip shape')
            if split == 'train':
                need(batch.get('temporal_sampling_version') == sampling.VERSION and
                     batch.get('spatial_policy_sha256') == augment.POLICY_SHA and
                     all(r['metadata']['epoch_index'] == self.state['epoch'] for r in batch['records']),
                     'Stale or missing train spatial/temporal augmentation')
            else:
                need('temporal_sampling_version' not in batch and
                     'spatial_policy_sha256' not in batch and
                     all('spatial_augmentation' not in r['metadata'] for r in batch['records']),
                     'Validation augmentation forbidden')
            return batch

        super().__init__(model, plan, guarded, *args, **kwargs)
        validate_momentum(model)


def run_path(variant):
    need(variant == 'aca', 'Only ACA seed24 supported')
    return RUN


def predecessor(plan):
    _, ref = completed_parent()
    need(ref == plan['spatial_experiment']['baseline'], 'Parent temporal50 run changed')


def execute(path, digest, *, resume=False):
    shell.assert_idle_gpu = assert_idle_gpu
    shell.verify_plan = verify_plan
    shell.run_path = run_path
    shell.require_predecessor = predecessor
    shell.initialized = initialized
    shell.rt = SimpleNamespace(**{
        **vars(runtime),
        'VerifiedDataset': SpatialDataset,
        'EpochStream': EpochStream,
        'TrainingLoop': TrainingLoop,
    })
    return shell.execute(path, digest, RUN, resume=resume)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    s = sub.add_parser('start')
    s.add_argument('--plan', type=Path, required=True)
    s.add_argument('--plan-sha256', required=True)
    r = sub.add_parser('resume')
    r.add_argument('--plan-sha256', required=True)
    a = p.parse_args()
    if a.command == 'prepare':
        plan = build_plan()
        observed = assert_idle_gpu(plan['gpu_policy'])
        target = QA / 'formal_plan.json'
        need(not target.exists(), 'Prepared plan already exists')
        runtime.atomic_json(target, plan)
        runtime.atomic_json(QA / 'review.json', {
            'status': 'prepared_not_started',
            'plan_sha256': core.plan_digest(plan),
            'run_directory': str(RUN),
            'gpu_processes': observed,
            'only_change': 'train clip_flip_color_light_v2',
            'test_run': False,
        })
        print(core.plan_digest(plan), flush=True)
    elif a.command == 'start':
        execute(a.plan, a.plan_sha256, resume=False)
    else:
        execute(RUN / 'plan.json', a.plan_sha256, resume=True)


if __name__ == '__main__':
    main()
