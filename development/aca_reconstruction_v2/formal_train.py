"""Fixed single-seed reconstruction plan/start/resume/stop CLI; default action never trains.

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

from development.aca_eight_class_data_v1 import loop_core as core
from . import training_runtime as rt
from development.aca_eight_class_data_v1.adapter import CLASSES
from development.aca_eight_class_data_v1.foundation import ROOT, CONTRACT_SHA256, FoundationScope, audit_rows, load_contract, baseline
from development.aca_eight_class_data_v1.model_adapter import SPEC_SHA256, initialized_baseline8, load_model_contract
from development.aca_eight_class_data_v1.gpu_readiness import sha, configure, equal
from development.aca_four_arm_campaign_v1.gpu_guard import make_policy, assert_idle_gpu

PLAN_SCHEMA = rt.PLAN_SCHEMA
AUDIT = Path('/home/tsai/projects/basketball/outputs/01_current/input_foundation_20261115_v1/full_audit_v1')
AUDIT_SHA = 'f6254f19f3c0ecadce034ef37958eb15ac51d80c47d8324be59f3652ba789f90'
OUTPUT_ROOT = Path('/home/tsai/projects/basketball/outputs/01_current')
need = core.need


def read(path):
    return json.loads(Path(path).read_text())


from development.aca_eight_class_data_v1 import formal_train as baseline_formal
from .model import initialized, contract, PROPOSAL_SHA
BASELINE_RUN=OUTPUT_ROOT/'baseline8_formal_seed20261024_v1'
BASELINE_HASH='7934894ec6d8252b1b03ebd918ee2cc973e92676af3a66c7e0a8f4f7fca39318'
EVIDENCE=OUTPUT_ROOT/'aca_reconstruction_v2_engineering_20261118_v1/gpu_attempt2'
ARMS=('lsta','tsci','aca')

def run_path(arm):
    need(arm in ARMS, 'Only the three unfinished seed24 arms may start')
    return OUTPUT_ROOT/f'aca_reconstruction_v2_{arm}_seed20261024_v1'


def source_hashes():
    hashes=baseline_formal.source_hashes()
    for p in (ROOT/'development/aca_reconstruction_v2').glob('*.py'):
        hashes[str(p.relative_to(ROOT))]=sha(p)
    for p,h in contract()['source_hashes'].items():
        path=Path(p)
        if path.is_relative_to(ROOT):
            hashes[str(path.relative_to(ROOT))]=sha(path)
    hashes['docs/research/specs/aca_reconstruction_proposal_v2.json']=PROPOSAL_SHA
    return dict(sorted(hashes.items()))


def completed_result(directory, plan):
    summary=read(directory/'summary.json'); history=read(directory/'history.json'); identity=read(directory/'run.json')
    need(summary['status']=='completed' and summary['test_run'] is False and
         identity['run_id']==summary['run_id'] and identity['plan_sha256']==summary['plan_sha256']==core.plan_digest(plan),
         'Completed run identity mismatch')
    need(read(directory/'plan.json')==plan and len(history)==summary['completed_epochs']>0, 'Completed history mismatch')
    best=min(history,key=lambda r:r['validation']['ce'])
    bad=len(history)-best['epoch']
    state={'epoch':len(history),'cursor':0,'updates':summary['updates'],
           'train_metrics':core.metrics_state(core.EightClassMetrics()),'history':history,
           'best_epoch':best['epoch'],'best_ce':best['validation']['ce'],'bad_epochs':bad}
    rt.validate_state(state,plan,{})
    need(summary['best_epoch']==best['epoch'] and summary['selected_validation']==best['validation'] and
         (bad>=plan['patience'] or len(history)==plan['max_epochs']) and
         summary['stop_reason']==('early_stop' if bad>=plan['patience'] else 'epoch_cap'), 'Incomplete or inconsistent selection')
    return {'directory':str(directory),'run_id':identity['run_id'],'plan_sha256':core.plan_digest(plan),
            'summary_sha256':sha(directory/'summary.json'),'history_sha256':sha(directory/'history.json'),
            'best_epoch':best['epoch'],'selected_validation':best['validation']}


def require_predecessor(plan):
    index=ARMS.index(plan['variant'])
    if index:
        arm=ARMS[index-1]
        completed_result(run_path(arm),build_plan(plan['source_root'],arm))


def build_plan(source_root, arm):
    need(arm in ARMS, 'No baseline rerun, other seed, or test arm')
    proposal=contract()
    need(all(sha(Path(p))==h for p,h in proposal['source_hashes'].items()), 'Frozen proposal dependency changed')
    # Exact original baseline plan equality establishes the common protocol/runtime.
    plan=baseline_formal.build_plan(source_root)
    baseline_plan=read(BASELINE_RUN/'plan.json')
    need(core.plan_digest(baseline_plan)==BASELINE_HASH and plan==baseline_plan,
         'Baseline common plan changed; stop for compatibility review')
    reuse=completed_result(BASELINE_RUN,baseline_plan)
    evidence=read(EVIDENCE/'report.json'); engineering_plan=read(EVIDENCE/'plan.json')
    need(evidence['status']=='passed' and evidence['formal_training_started'] is False and
         evidence['test_videos_opened']==0 and evidence['temporary_checkpoints_removed'] and
         all(evidence['continuous_vs_resumed_equal'].values()) and
         evidence['workers_equal_serial_tensor_sha'] and evidence['real_optimizer_updates_engineering_only']==4 and
         evidence['plan_sha256']==core.plan_digest(engineering_plan) and engineering_plan['proposal_sha256']==PROPOSAL_SHA,
         'Missing bounded ACA GPU evidence')
    need(all(sha(ROOT/p)==h for p,h in engineering_plan['code_sha256'].items()), 'GPU-tested implementation changed')
    plan.update(schema=PLAN_SCHEMA,purpose='explicit_assumption_ACA_reconstruction_single_seed_ablation_not_author_exact',
                variant=arm,proposal_sha256=PROPOSAL_SHA,code_sha256=source_hashes(),baseline_reuse=reuse,
                normalization_policy=contract()['normalization_policy'],
                reconstruction_gpu_evidence={'path':str(EVIDENCE/'report.json'),'sha256':sha(EVIDENCE/'report.json'),
                    'scope':'ACA only real CUDA; all four arms CPU/meta; not classification evidence'},
                allowed_run_directory=str(run_path(arm)),execution_order=list(ARMS),
                automatic_next_arm=False,automatic_restart=False)
    return plan


def verify_plan(plan, expected_hash):
    need(plan.get('schema')==PLAN_SCHEMA and core.plan_digest(plan)==expected_hash,'Plan schema/hash mismatch')
    need(plan==build_plan(plan['source_root'],plan['variant']), 'Code/data/runtime/protocol identity changed; do not launch or resume')


def safe_new_directory(path):
    path = Path(path).absolute()
    need(path.parent.resolve() == OUTPUT_ROOT.resolve() and not path.is_symlink(), 'Use one new directory under outputs/01_current')
    path.mkdir(exist_ok=False)
    return path


def prepare(source_root, output, arm):
    plan = build_plan(source_root, arm)
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
    need(directory == run_path(plan['variant']) and not directory.is_symlink(), 'Wrong arm run directory')
    require_predecessor(plan)
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
            model, init = initialized(plan['variant'],plan['seed'])
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
    need(directory in [run_path(a) for a in ARMS], 'Not a reconstruction run')
    identity = read(directory/'run.json')
    need(read(directory/'plan.json')['schema']==PLAN_SCHEMA, 'Wrong run schema')
    # No PID signal is sent. The matching run checks its own marker at a safe boundary.
    rt.atomic_json(directory/'STOP.json', {'run_id': identity['run_id'], 'requested_at_taiwan': rt.now()})
    print('Stop requested; train saves at the next optimizer boundary, validation remains uncommitted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
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
        prepare(args.source_root, args.output, args.arm)
    elif args.command == 'stop':
        request_stop(args.run_dir)
    else:
        execute(args.run_dir/'plan.json' if args.command == 'resume' else args.plan,
                args.plan_sha256, args.run_dir, resume=args.command == 'resume')


if __name__ == '__main__':
    main()
