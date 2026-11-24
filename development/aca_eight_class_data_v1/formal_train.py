"""Fixed single-baseline plan/start/resume/stop CLI; default action never trains.

prepare writes a reviewable plan without model creation, decoding or updates.
start and resume are separate explicit commands. Test and alternate protocols
have no CLI paths. Run this module from the original ACA worktree.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import traceback
import uuid

import cv2
import numpy as np
import torch
import torchvision

from . import loop_core as core
from . import training_runtime as rt
from .adapter import CLASSES
from .foundation import ROOT, CONTRACT_SHA256, FoundationScope, audit_rows, load_contract, baseline
from .model_adapter import SPEC_SHA256, initialized_baseline8, load_model_contract
from .gpu_readiness import sha, configure, equal
from development.aca_four_arm_campaign_v1.gpu_guard import make_policy, assert_idle_gpu

PLAN_SCHEMA = 'baseline8_single_formal_plan_v1'
AUDIT = Path('/home/tsai/projects/basketball/outputs/01_current/input_foundation_20261115_v1/full_audit_v1')
AUDIT_SHA = 'f6254f19f3c0ecadce034ef37958eb15ac51d80c47d8324be59f3652ba789f90'
OUTPUT_ROOT = Path('/home/tsai/projects/basketball/outputs/01_current')
need = core.need


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    spec = load_model_contract()
    paths = set(spec['source_sha256']) | set(load_contract()['immutable_dependencies_sha256'])
    paths.update('development/aca_eight_class_data_v1/' + name for name in (
        'formal_train.py', 'training_runtime.py', 'loop_core.py', 'model_adapter.py', 'gpu_readiness.py'))
    paths.update(('development/aca_four_arm_campaign_v1/gpu_guard.py',
                  'development/aca_four_arm_campaign_v1/queue.py'))
    return {p: sha(ROOT/p) for p in sorted(paths)}


def audit_reference():
    need(sha(AUDIT/'records.jsonl') == AUDIT_SHA, 'Full input audit changed')
    rows, _ = audit_rows(load_contract())
    expected = {r['video_id']: r for r in rows if r['audit_split'] in ('train', 'validation')}
    reference = {}
    for line in (AUDIT/'records.jsonl').read_text().splitlines():
        row = json.loads(line)
        vid = row['video_id']
        need(vid in expected and vid not in reference and row['status'] == 'passed', 'Audit coverage/status mismatch')
        old = expected[vid]
        need(row['split'] == old['audit_split'] and row['class_id'] == old['class_id'] and
             row['binary_label'] == int(old['binary_label']) and row['source_group'] == old['source_group'], 'Audit label mismatch')
        reference[vid] = {k: row[k] for k in ('split', 'class_id', 'binary_label', 'source_group', 'source_sha256', 'tensor_sha256')}
        reference[vid]['source_size_bytes'] = row['metadata']['source_size_bytes']
    need(set(reference) == set(expected) and len(reference) == 8739, 'Incomplete audit')
    return reference


def source_inventory(source_root, reference):
    """Descriptor/stat checks for allowed train/validation only; no video decoding."""
    identities = []
    for split in ('train', 'validation'):
        scope = FoundationScope(source_root=source_root, split=split)
        for row in scope.rows:
            vid = row['video_id']
            _, path = scope.authorize(vid)
            fd = baseline.gate1a.open_guarded(path)
            try:
                info = os.fstat(fd)
                need(info.st_size == reference[vid]['source_size_bytes'], 'Source size changed: ' + vid)
                identities.append([vid, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns])
            finally:
                os.close(fd)
    return {'count': len(identities), 'sha256': core.plan_digest(identities),
            'total_bytes': sum(x[3] for x in identities), 'method': 'guarded_open_fstat_no_video_bytes'}


def runtime_identity():
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    need(len(gpu) == 1, 'Exactly one GPU required')
    return {'python_executable': str(Path(sys.executable).resolve()), 'python': platform.python_version(),
            'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
            'opencv': cv2.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': gpu[0], 'precision': 'FP32_AMP_TF32_off', 'deterministic': True,
            'cpu_threads': 4, 'workers': 2, 'worker_start': 'spawn', 'prefetch': 2, 'pin_memory': False,
            'cublas_workspace_config': ':4096:8'}


def build_plan(source_root):
    source_root = Path(source_root).resolve(strict=True)
    need(source_root == Path('/home/tsai/projects/basketball'), 'Unexpected source root')
    spec = load_model_contract()
    reference = audit_reference()
    ids = {split: sorted(vid for vid, r in reference.items() if r['split'] == split)
           for split in ('train', 'validation')}
    need(len(ids['train']) == 7192 and len(ids['validation']) == 1547, 'Full cohort required')
    evidence = ROOT/'docs/research/input_processing/eight_class_gpu_readiness_qa_v1.json'
    qa = read(evidence)
    need(qa['status'] == 'passed' and all(qa['continuous_vs_resumed_equal'].values()) and
         qa['workers_equal_serial_tensor_sha'], 'Missing successful GPU wiring evidence')
    need(all(sha(ROOT/p) == h for p, h in qa['source_sha256'].items()), 'Readiness dependency changed')
    weights = spec['initialization']
    need(sha(ROOT/weights['pretrained_path']) == weights['pretrained_sha256'], 'ImageNet weight identity changed')
    policy = make_policy()
    return {'schema': PLAN_SCHEMA, 'purpose': 'single_eight_class_project_baseline_not_author_exact_ACA',
            'spec_sha256': SPEC_SHA256, 'input_contract_sha256': CONTRACT_SHA256,
            'seed': 20261024, 'max_epochs': 30, 'patience': 5, 'effective': 16, 'checkpoint_every': 25,
            'device': 'cuda', 'variant': 'baseline', 'class_order': list(CLASSES),
            'train_ids': ids['train'], 'validation_ids': ids['validation'], 'reference': reference,
            'train_count': 7192, 'validation_count': 1547, 'updates_per_epoch': 450, 'tail_videos': 8,
            'source_root': str(source_root), 'input_audit_records_sha256': AUDIT_SHA,
            'source_inventory': source_inventory(source_root, reference),
            'code_sha256': source_hashes(), 'runtime': runtime_identity(), 'gpu_policy': policy,
            'readiness_evidence_sha256': sha(evidence), 'test_enabled': False,
            'selection': 'strict_lowest_complete_validation_CE_ties_keep_earlier',
            'checkpoint': 'single_atomic_last.pt_includes_current_and_best_weights',
            'prepared_only': True, 'formal_training_started': False}


def verify_plan(plan, expected_hash):
    need(plan.get('schema') == PLAN_SCHEMA and core.plan_digest(plan) == expected_hash, 'Plan schema/hash mismatch')
    need(plan == build_plan(plan['source_root']), 'Code/data/runtime/protocol identity changed; do not launch or resume')


def safe_new_directory(path):
    path = Path(path).absolute()
    need(path.parent.resolve() == OUTPUT_ROOT.resolve() and not path.is_symlink(), 'Use one new directory under outputs/01_current')
    path.mkdir(exist_ok=False)
    return path


def prepare(source_root, output):
    plan = build_plan(source_root)
    processes = assert_idle_gpu(plan['gpu_policy'])
    output = safe_new_directory(output)
    rt.atomic_json(output/'plan.json', plan)
    rt.atomic_json(output/'review.json', {'status': 'prepared_not_started', 'plan_sha256': core.plan_digest(plan),
                                         'created_at_taiwan': rt.now(), 'gpu_processes': processes,
                                         'train_count': 7192, 'validation_count': 1547, 'seed': 20261024,
                                         'max_epochs': 30, 'patience': 5, 'model_instances': 0,
                                         'video_decodes': 0, 'optimizer_updates': 0, 'test_source_access': 0,
                                         'formal_training_started': False})
    print(json.dumps({'prepared': str(output), 'plan_sha256': core.plan_digest(plan), 'training_started': False}))


@contextmanager
def run_lock(directory, name='run.lock'):
    with (directory/name).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process holds this run; nothing was stopped') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_identity():
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': os.getpid(), 'process_start_ticks': int(stat[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'cmdline': Path('/proc/self/cmdline').read_bytes().split(b'\0')[:-1] and
                       Path('/proc/self/cmdline').read_bytes().replace(b'\0', b' ').decode().strip()}


class Reporter:
    def __init__(self, directory, attempt, plan):
        self.directory, self.attempt, self.plan = directory, attempt, plan
        self.committed, self.last_key = None, None

    def event(self, kind, **fields):
        row = {'at_taiwan': rt.now(), 'event': kind, 'attempt': self.attempt.name, **fields}
        with (self.directory/'progress.log').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def commit(self, kind, state):
        self.committed = state
        rt.atomic_json(self.directory/'history.json', state['history'])
        self.event(kind, completed_epochs=state['epoch'], train_cursor=state['cursor'], updates=state['updates'],
                   best_epoch=state['best_epoch'], best_ce=state['best_ce'], bad_epochs=state['bad_epochs'])
        self.progress('checkpoint', state, 0, force=True)

    def progress(self, phase, state, seen, *, force=False, status='running'):
        key = (phase, state['epoch'], state['updates'], seen if phase == 'validation' else state['cursor'], status)
        if key == self.last_key and not force:
            return
        self.last_key = key
        saved = self.committed or state
        row = {'at_taiwan': rt.now(), 'attempt': self.attempt.name, 'status': status, 'phase': phase,
               'epoch_index': state['epoch'], 'max_epochs': 30, 'train_cursor': state['cursor'], 'train_count': 7192,
               'validation_seen': seen if phase == 'validation' else None, 'validation_count': 1547,
               'optimizer_updates': state['updates'], 'checkpoint_updates': saved['updates'],
               'completed_epochs': len(state['history']), 'latest_validation': state['history'][-1]['validation'] if state['history'] else None,
               'best_epoch': state['best_epoch'], 'best_ce': state['best_ce'], 'bad_epochs': state['bad_epochs'],
               'patience': 5, 'plan_sha256': core.plan_digest(self.plan), 'authoritative_checkpoint': 'last.pt'}
        rt.atomic_json(self.directory/'run_state.json', row)
        if phase != 'checkpoint':
            self.event('progress', **{k: row[k] for k in ('phase', 'epoch_index', 'train_cursor', 'validation_seen',
                                                         'optimizer_updates', 'checkpoint_updates', 'status')})


class StopController:
    def __init__(self, path, reporter):
        self.path, self.reporter, self.requested = path, reporter, False

    def signal(self, signum, frame):
        self.requested = True

    def __call__(self, phase, state, seen):
        stop = self.requested or self.path.exists()
        if stop or (phase == 'train' and state['updates'] % 25 == 0) or (phase == 'validation' and seen % 25 == 0):
            self.reporter.progress(phase, state, seen)
        return stop


def execute(plan_path, expected_hash, directory, *, resume):
    plan = read(plan_path)
    verify_plan(plan, expected_hash)  # Fail before a model, decoder or output mutation.
    assert_idle_gpu(plan['gpu_policy'])
    directory = Path(directory).absolute()
    if resume:
        need(directory.parent.resolve() == OUTPUT_ROOT.resolve() and not directory.is_symlink(), 'Unexpected run path')
        identity = read(directory/'run.json')
        need(identity['plan_sha256'] == expected_hash and read(directory/'plan.json') == plan, 'Run/plan mismatch')
        need((directory/'last.pt').is_file(), 'No committed checkpoint; do not guess a restart')
        need(not (directory/'summary.json').exists(), 'Run already completed; nothing to resume')
    else:
        directory = safe_new_directory(directory)
        identity = {'run_id': uuid.uuid4().hex, 'plan_sha256': expected_hash, 'created_at_taiwan': rt.now()}
        rt.atomic_json(directory/'run.json', identity)
        rt.atomic_json(directory/'plan.json', plan)
    with run_lock(OUTPUT_ROOT.parent/'00_admin', 'baseline8_gpu.lock'), run_lock(directory):
        assert_idle_gpu(plan['gpu_policy'])
        attempt = directory / ('attempt_' + uuid.uuid4().hex[:12])
        attempt.mkdir()
        reporter = Reporter(directory, attempt, plan)
        loop, stream = None, None
        stop = StopController(directory/'STOP.json', reporter)
        old_handlers = {}
        try:
            if resume:
                (directory/'STOP.json').unlink(missing_ok=True)
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, stop.signal)
            launcher = {**process_identity(), 'run_id': identity['run_id'], 'plan_sha256': expected_hash,
                        'attempt': attempt.name, 'resume': resume, 'at_taiwan': rt.now()}
            rt.atomic_json(attempt/'launcher.json', launcher)
            rt.atomic_json(directory/'launcher.json', launcher)
            reporter.event('explicit_resume' if resume else 'explicit_start', run_id=identity['run_id'])
            rt.atomic_json(directory/'run_state.json', {'status': 'initializing', 'phase': 'restore' if resume else 'initialization',
                'at_taiwan': rt.now(), 'attempt': attempt.name, 'plan_sha256': expected_hash,
                'optimizer_updates': None if resume else 0, 'authoritative_checkpoint': 'last.pt'})
            configure()
            model, init = initialized_baseline8()
            model.cuda()
            datasets = {split: rt.VerifiedDataset(plan['source_root'], split, plan[split+'_ids'], plan['reference'])
                        for split in ('train', 'validation')}
            stream = rt.EpochStream(datasets, plan, lambda: loop.state)
            loop = rt.TrainingLoop(model, plan, stream, directory/'last.pt', identity['run_id'], publish=reporter.commit)
            rt.atomic_json(attempt/'initialization.json', init)
            if resume:
                loop.resume()
            result = loop.run(stop)
            effective_state = loop.committed if result == 'paused' else loop.state
            reporter.progress('paused' if result == 'paused' else 'completed', effective_state, 0, force=True, status=result)
            if result == 'completed':
                state = loop.state
                need(state['best_epoch'] is not None, 'Completed without full validation')
                rt.atomic_json(directory/'summary.json', {'status': 'completed', 'run_id': identity['run_id'],
                    'plan_sha256': expected_hash, 'completed_epochs': state['epoch'], 'updates': state['updates'],
                    'stop_reason': 'early_stop' if state['bad_epochs'] >= 5 else 'epoch_cap',
                    'best_epoch': state['best_epoch'], 'selected_validation': state['history'][state['best_epoch']-1]['validation'],
                    'test_run': False, 'best_weights': 'last.pt:best_model'})
            rt.atomic_json(attempt/'outcome.json', {'status': result, 'at_taiwan': rt.now()})
        except BaseException as exc:
            rt.atomic_json(attempt/'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                                                  'at_taiwan': rt.now(), 'last_checkpoint_preserved': (directory/'last.pt').exists()})
            if loop is not None:
                reporter.progress('failed', loop.committed, 0, force=True, status='failed')
            else:
                rt.atomic_json(directory/'run_state.json', {'status': 'failed', 'phase': 'initialization',
                    'attempt': attempt.name, 'at_taiwan': rt.now(), 'failure': str(attempt/'failure.json')})
            reporter.event('failed', error=repr(exc), failure=str(attempt/'failure.json'))
            raise
        finally:
            if stream is not None:
                stream.close()
            for sig, previous in old_handlers.items():
                signal.signal(sig, previous)


def request_stop(directory):
    directory = Path(directory).absolute()
    need(directory.parent.resolve() == OUTPUT_ROOT.resolve() and not directory.is_symlink(), 'Unexpected run path')
    identity = read(directory/'run.json')
    # No PID signal is sent. The matching run checks its own marker at a safe boundary.
    rt.atomic_json(directory/'STOP.json', {'run_id': identity['run_id'], 'requested_at_taiwan': rt.now()})
    print('Stop requested; train saves at the next optimizer boundary, validation remains uncommitted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('start')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p = commands.add_parser('resume')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p = commands.add_parser('stop')
    p.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source_root, args.output)
    elif args.command == 'stop':
        request_stop(args.run_dir)
    else:
        execute(args.run_dir/'plan.json' if args.command == 'resume' else args.plan,
                args.plan_sha256, args.run_dir, resume=args.command == 'resume')


if __name__ == '__main__':
    main()
