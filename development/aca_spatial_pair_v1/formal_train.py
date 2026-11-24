"""Paired seed20261025 formal run: clip_flip_color_light_v2 on top of temporal50.

Compares 1:1 against completed aca_binary_temporal50_seed20261025_labelrev1_v1.
No existing modules, splits, labels, or historical runs are modified.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from development.aca_reconstruction_v2 import formal_train as shell
from development.aca_eight_class_data_v1 import formal_train as base
from development.aca_binary_v1 import core, runtime
from development.aca_binary_v1.data import guard_fetch
from development.aca_binary_bn001_v1.model import validate_momentum
from development.aca_temporal_v1 import sampling
from development.aca_temporal_pair_v1.model import initialized as pair_initialized
from development.aca_four_arm_campaign_v1 import gpu_guard
from development.aca_spatial_v1 import augment, formal_train as spatial24
from development.adjudicated_input_v1 import adapter as labels

import shutil
import subprocess

from .data import SEED, SpatialDataset25, EpochStream25

ROOT = Path(__file__).resolve().parents[2]
CURRENT = Path('/home/tsai/projects/basketball/outputs/01_current')
PARENT_TEMPORAL25 = CURRENT / 'aca_binary_temporal50_seed20261025_labelrev1_v1'
PARENT_TEMPORAL25_SHA = '7648128de39039f9aac16ea316a3eca8c9e7d2412de37adc70995037122857a4'
SEED24_SPATIAL = CURRENT / 'aca_binary_spatial_temporal50_seed20261024_labelrev1_v1'
SEED24_SPATIAL_SHA = 'a3f8bf576c42e420a94960fe4bb3af38e68baf9d5cc3c391a97544d5ec000096'
QA = CURRENT / 'aca_spatial_pair_seed20261025_readiness_20261124_v1'
RUN = CURRENT / 'aca_binary_spatial_temporal50_seed20261025_labelrev1_v1'
SCHEMA = 'aca_binary_spatial_temporal50_seed25_formal_v1'
_ORIGINAL_CONFIGURE = shell.configure
need = core.need


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


EXTRA_DESKTOP_PACKAGES = {
    '/usr/libexec/xdg-desktop-portal-gnome': 'xdg-desktop-portal-gnome',
}


def make_gpu_policy():
    policy = spatial24.make_gpu_policy()
    policy['schema'] = 'aca_gpu_desktop_policy_v3_antigravity_and_portal'
    policy['extra_desktop_packages'] = [
        gpu_guard.executable_pin(path, pkg)
        for path, pkg in EXTRA_DESKTOP_PACKAGES.items()
    ]
    policy['user_authorization_note'] = (
        'User explicitly authorized A1 full execution in Antigravity on 2026-11-24 '
        'with Antigravity desktop GUI (--type=gpu-process) pinned by executable SHA256 '
        'and root-owned dpkg package xdg-desktop-portal-gnome pinned via executable_pin; '
        'immutable gpu_guard.py untouched and all unknown compute remains blocked.'
    )
    return policy


def assert_idle_gpu(policy):
    need(policy.get('schema') == 'aca_gpu_desktop_policy_v3_antigravity_and_portal',
         'Unexpected GPU policy schema')
    base_view = {
        k: v for k, v in policy.items()
        if k not in ('extra_desktop_packages', 'user_authorization_note')
    }
    base_view['schema'] = 'aca_gpu_desktop_policy_v2_antigravity_gui'
    spatial24.verify_gpu_policy(base_view)
    extra_pkgs = policy.get('extra_desktop_packages', [])
    need(len(extra_pkgs) == len(EXTRA_DESKTOP_PACKAGES), 'Extra desktop package count mismatch')
    for row, (path, pkg) in zip(extra_pkgs, EXTRA_DESKTOP_PACKAGES.items()):
        need(row == gpu_guard.executable_pin(path, pkg), 'Extra desktop package fingerprint changed')

    query = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True,
    )
    extra = {r['path'] for r in policy['extra_executables']} | {r['path'] for r in extra_pkgs}
    pinned_guis = policy.get('pinned_gui_gpu_processes', [])
    observed = []
    for raw_pid in query.split():
        pid = int(raw_pid)
        cls = gpu_guard.classify(pid, extra)
        if cls not in ('self', 'exited', 'desktop') and spatial24._is_pinned_gui(pid, pinned_guis):
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


def completed_run(directory, expected_plan_sha):
    plan = read(directory / 'plan.json')
    need(core.plan_digest(plan) == expected_plan_sha, f'Plan changed: {directory}')
    s = read(directory / 'summary.json')
    h = read(directory / 'history.json')
    r = read(directory / 'run.json')
    need(s['status'] == 'completed' and s['test_run'] is False and
         s['run_id'] == r['run_id'] and s['plan_sha256'] == r['plan_sha256'] == expected_plan_sha and
         len(h) == s['completed_epochs'] > 0, f'Run completion invalid: {directory}')
    best = min(h, key=lambda x: x['validation']['ce'])
    need(s['selected_validation'] == best['validation'] and s['best_epoch'] == best['epoch'],
         f'Run best selection mismatch: {directory}')
    ref = {
        'directory': str(directory),
        'run_id': r['run_id'],
        'plan_sha256': expected_plan_sha,
        'summary_sha256': sha(directory / 'summary.json'),
        'history_sha256': sha(directory / 'history.json'),
        'best_epoch': best['epoch'],
        'selected_validation': best['validation'],
    }
    return plan, ref


def source_hashes(parent_plan):
    hashes = {}
    for name, expected in parent_plan['code_sha256'].items():
        p = ROOT / name
        need(sha(p) == expected, 'Shared source changed: ' + name)
        hashes[name] = expected
    for folder in ('aca_spatial_v1', 'aca_spatial_pair_v1'):
        for p in sorted((ROOT / 'development' / folder).glob('*.py')):
            hashes[str(p.relative_to(ROOT))] = sha(p)
    for name, expected in labels.PINNED_CODE.items():
        p = labels.f.ROOT / name
        need(sha(p) == expected, 'Pinned data source changed: ' + name)
    return dict(sorted(hashes.items()))


def draft_plan():
    parent25, ref_temporal25 = completed_run(PARENT_TEMPORAL25, PARENT_TEMPORAL25_SHA)
    _, ref_spatial24 = completed_run(SEED24_SPATIAL, SEED24_SPATIAL_SHA)
    need(parent25['seed'] == SEED, 'Expected seed25 temporal parent')
    rev = labels.LabelRevision()
    manifest = rev._manifest
    need(parent25['reference'] == manifest['reference'] and
         parent25['train_ids'] == manifest['train_ids'] and
         parent25['validation_ids'] == manifest['validation_ids'],
         'Data protocol changed')
    cpu24_evidence = spatial24.verified_cpu_acceptance()
    spatial_contract = {
        'parent_temporal_input_contract_sha256': parent25['input_contract_sha256'],
        'temporal_train': sampling.POLICY,
        'spatial_train': augment.POLICY,
        'seed': SEED,
        'validation': 'exact_unchanged_AdjudicatedDataset_and_BinaryCollator',
    }
    plan = deepcopy(parent25)
    plan.update(
        schema=SCHEMA,
        seed=SEED,
        purpose='paired_second_seed_confirmation_of_clip_consistent_spatial_aug_on_temporal50',
        allowed_run_directory=str(RUN),
        spatial_augmentation=augment.POLICY,
        spatial_policy_sha256=augment.POLICY_SHA,
        spatial_input_contract=spatial_contract,
        input_contract_sha256=core.plan_digest(spatial_contract),
        spatial_pair_experiment={
            'seed': SEED,
            'within_seed_baseline': ref_temporal25,
            'seed24_spatial_pair': ref_spatial24,
            'only_within_seed_change': 'train_add_clip_flip_color_light_v2_on_top_of_endpoint_stratified50',
            'seed_override_scope': 'model_module_head_RNG_data_order_worker_seed_temporal_sampling_spatial_sampling',
            'initialization': 'fresh_ACA_BN001_seed20261025_head_seed_20264025',
            'validation_pixels': 'identical_fixed_uniform50_zero_augmentation',
            'seed24_cpu_acceptance': cpu24_evidence,
            'two_seed_statistical_significance_claimed': False,
            'historical_test_contact_disclosed': True,
            'author_exact_augmentation_verified': False,
        },
        runtime=base.runtime_identity(),
        source_inventory=base.source_inventory(plan['source_root'], plan['reference']),
        code_sha256=source_hashes(parent25),
        gpu_policy=make_gpu_policy(),
        test_enabled=False,
        automatic_restart=False,
        automatic_next_arm=False,
        notifications='no_hourly_reports',
    )
    need(plan['head']['initialization_seed'] == SEED + 3000, 'Wrong seed25 head seed')
    need(plan['source_inventory'] == parent25['source_inventory'], 'Source inventory changed')
    return plan


def build_plan():
    plan = draft_plan()
    draft_sha = core.plan_digest(plan)
    cpu_ev = read(QA / 'cpu_report.json')
    gpu_ev = read(QA / 'gpu_report.json')
    need(cpu_ev['status'] == 'passed' and cpu_ev['draft_plan_sha256'] == draft_sha and
         cpu_ev['validation_video_decodes'] == 0 and cpu_ev['test_video_decodes'] == 0,
         'Seed25 CPU readiness missing or stale')
    need(gpu_ev['status'] == 'passed' and gpu_ev['draft_plan_sha256'] == draft_sha and
         gpu_ev['validation_video_decodes'] == 0 and gpu_ev['test_video_decodes'] == 0 and
         all(gpu_ev['train_boundary_continuous_vs_resumed_equal'].values()) and
         all(gpu_ev['post_selection_continuous_vs_resumed_equal'].values()),
         'Seed25 GPU readiness missing or stale')
    plan['spatial_pair_readiness'] = {
        'cpu_report_path': str(QA / 'cpu_report.json'),
        'cpu_report_sha256': sha(QA / 'cpu_report.json'),
        'gpu_report_path': str(QA / 'gpu_report.json'),
        'gpu_report_sha256': sha(QA / 'gpu_report.json'),
    }
    return plan


def verify_plan(plan, digest):
    need(core.plan_digest(plan) == digest and plan == build_plan(),
         'Seed25 spatial plan/code/data/runtime identity mismatch')


def configure():
    _ORIGINAL_CONFIGURE()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def initialized(variant='aca', seed=SEED):
    need(variant == 'aca' and seed == SEED, 'Only ACA seed20261025 permitted')
    return pair_initialized(variant, seed)


class TrainingLoop25(runtime.TrainingLoop):
    def __init__(self, model, plan, fetch, *args, **kwargs):
        need(plan['schema'] == SCHEMA and
             plan['temporal_sampling'] == sampling.POLICY and
             plan['spatial_augmentation'] == augment.POLICY and
             plan['seed'] == SEED,
             'Wrong formal seed25 spatial plan')
        labels.require_revision_identity(plan)
        validate_momentum(model)
        binary_fetch = guard_fetch(fetch, plan)

        def guarded(split, index):
            batch = binary_fetch(split, index)
            need(tuple(batch['tensor'].shape) == (1, 3, 50, 120, 160), 'Wrong single clip shape')
            if split == 'train':
                need(batch.get('temporal_sampling_version') == sampling.VERSION and
                     batch.get('spatial_policy_sha256') == augment.POLICY_SHA and
                     batch.get('spatial_seed') == SEED and
                     all(r['metadata']['temporal_seed'] == SEED and
                         r['metadata']['spatial_augmentation']['seed'] == SEED and
                         r['metadata']['epoch_index'] == self.state['epoch']
                         for r in batch['records']),
                     'Stale or wrong-seed train spatial/temporal augmentation')
            else:
                need('temporal_sampling_version' not in batch and
                     'spatial_policy_sha256' not in batch and
                     all('spatial_augmentation' not in r['metadata'] for r in batch['records']),
                     'Validation augmentation forbidden')
            return batch

        super().__init__(model, plan, guarded, *args, **kwargs)
        validate_momentum(model)


def run_path(variant):
    need(variant == 'aca', 'Only ACA seed25 supported')
    return RUN


def predecessor(plan):
    _, ref_t25 = completed_run(PARENT_TEMPORAL25, PARENT_TEMPORAL25_SHA)
    _, ref_s24 = completed_run(SEED24_SPATIAL, SEED24_SPATIAL_SHA)
    need(ref_t25 == plan['spatial_pair_experiment']['within_seed_baseline'], 'Seed25 temporal parent changed')
    need(ref_s24 == plan['spatial_pair_experiment']['seed24_spatial_pair'], 'Seed24 spatial pair changed')


def execute(path, digest, *, resume=False):
    shell.assert_idle_gpu = assert_idle_gpu
    shell.verify_plan = verify_plan
    shell.run_path = run_path
    shell.require_predecessor = predecessor
    shell.initialized = initialized
    shell.configure = configure
    shell.rt = SimpleNamespace(**{
        **vars(runtime),
        'VerifiedDataset': SpatialDataset25,
        'EpochStream': EpochStream25,
        'TrainingLoop': TrainingLoop25,
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
            'seed': SEED,
            'run_directory': str(RUN),
            'gpu_processes': observed,
            'only_within_seed_change': 'train clip_flip_color_light_v2',
            'historical_test_contact_disclosed': True,
            'test_run': False,
        })
        print(core.plan_digest(plan), flush=True)
    elif a.command == 'start':
        execute(a.plan, a.plan_sha256, resume=False)
    else:
        execute(RUN / 'plan.json', a.plan_sha256, resume=True)


if __name__ == '__main__':
    main()
