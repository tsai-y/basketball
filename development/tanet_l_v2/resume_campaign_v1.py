"""Explicit, single-attempt recovery of the reboot-interrupted final L-v2 seed.

Keep the pinned engine and campaign plan unchanged. Fresh engineering readiness
precedes the original --resume entry, which restores model/Adam/RNG at an epoch
boundary. Completed seeds are never retrained. Failure stops without retries.
"""
import argparse
import fcntl
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

from . import campaign
from .config import FOLDER, need, sha256
from .training import atomic_save

read, stamp = campaign.read, campaign.stamp


def verify_inactive(identity, boot_id):
    need(identity['boot_id'] != boot_id, 'Recovery is restricted to the documented host reboot')
    try:
        actual = campaign.old.process_identity(identity['pid'])
    except FileNotFoundError:
        return
    need(any(actual[k] != identity[k] for k in ('boot_id', 'process_start_ticks', 'cmdline')),
         'Original process still alive; do not resume twice')


def validate_checkpoint(saved, best, contract, history):
    cfg = contract['recipe']
    for cp in (saved, best):
        need(cp['kind'] == 'formal' and cp['contract_sha256'] == contract['contract_sha256'],
             'Wrong checkpoint purpose/contract')
        need(cp['runtime']['torch'] == str(torch.__version__) and cp['runtime']['cuda'] == torch.version.cuda,
             'Resume runtime changed')
    need(saved['history'] == history and saved['epoch'] == history[-1]['epoch'] and
         saved['global_step'] == history[-1]['global_step'], 'Checkpoint/history boundary mismatch')
    need(saved['epoch'] < cfg['max_epochs'] and saved['stop_state']['wait'] < cfg['patience'],
         'Completed/early-stopped training must not extend')
    need(best['epoch'] == min(history, key=lambda h: h['validation']['CE'])['epoch'] and
         best['stop_state']['best_ce'] == saved['stop_state']['best_ce'], 'Best checkpoint mismatch')
    need(cfg['max_epochs'] == 15 and cfg['patience'] == 5 and cfg['test_access'] is False and
         cfg['evaluation_policy'] == campaign.POLICY, 'Pinned recipe changed')


def verify_completed(hashes):
    need(all(sha256(path) == value for path, value in hashes.items()), 'Completed seed artifact changed')


def prepare(output):
    output = Path(output).resolve()
    need(output.parent == (FOLDER/'artifacts').resolve() and output.name.startswith('three_seed_'), 'Foreign campaign')
    plan, state = read(output/'plan.json'), read(output/'state.json')
    need(state['plan_sha256'] == sha256(output/'plan.json') and state['active_seed'] == 20261026 and
         not (output/'summary.json').exists(), 'Wrong campaign/recovery scope')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    verify_inactive(state['supervisor'], boot)
    interrupted = state['jobs'][-1]
    need(interrupted['seed'] == 20261026 and interrupted['stage'] == 'training' and
         'returncode' not in interrupted, 'Final seed is not an interrupted training job')
    verify_inactive(interrupted, boot)
    campaign.verify_sources(plan)
    item = plan['queue'][-1]
    need(item['seed'] == 20261026, 'Final queue seed changed')
    run = campaign.old.output_path(interrupted['stdout_log'], 'OUTPUT: ')
    need(run.parent == (Path(item['package'])/'artifacts').resolve(), 'Foreign training output')
    contract = importlib.import_module(item['module']+'.training').contract()[0]
    need(contract == read(run/'contract.json') and contract['contract_sha256'] == item['contract_sha256'],
         'Training source/data/recipe identity changed')
    saved = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
    best = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    validate_checkpoint(saved, best, contract, read(run/'history.json'))
    completed = {}
    need(set(state['formal_runs']) == {'20261024', '20261025'}, 'Completed seed set changed')
    for seed, path in state['formal_runs'].items():
        folder = Path(path)
        need(read(folder/'run_state.json')['status'] == 'completed', 'Prior seed not complete')
        for name in ('contract.json', 'history.json', 'run_state.json', 'best.pt', 'last.pt', 'progress.log'):
            completed[str(folder/name)] = sha256(folder/name)
    session = Path(tempfile.mkdtemp(prefix='resume_', dir=output))
    backup = session/'before_resume'
    backup.mkdir()
    for name in ('state.json', 'launcher.json', 'supervisor.log'):
        shutil.copy2(output/name, backup/('campaign_'+name))
    for name in ('last.pt', 'best.pt', 'history.json', 'contract.json', 'progress.log', 'run_state.json'):
        shutil.copy2(run/name, backup/name)
    shutil.copy2(__file__, session/'resume_campaign_v1.py')
    recovery = {'schema': 'explicit_reboot_resume_final_L_seed_v1', 'created_at': stamp(),
        'user_explicitly_authorized_resume': True, 'campaign': str(output), 'run': str(run),
        'plan_sha256': sha256(output/'plan.json'), 'before_state_sha256': sha256(output/'state.json'),
        'resume_from_epoch': saved['epoch'], 'next_epoch_replayed': saved['epoch']+1,
        'resume_global_step': saved['global_step'], 'stop_state': saved['stop_state'],
        'max_epochs': 15, 'patience': 5, 'test_used': False, 'new_boot_id': boot,
        'last_sha256': sha256(run/'last.pt'), 'best_sha256': sha256(run/'best.pt'),
        'completed_artifacts_sha256': completed, 'script_sha256': sha256(__file__),
        'backup_sha256': {str(p.relative_to(session)): sha256(p) for p in backup.iterdir()},
        'failure_policy': 'stop_without_retry_or_automatic_resume'}
    atomic_save(session/'resume_plan.json', recovery, json_format=True)
    print(session, flush=True)
    return session


def run_child(output, session, state, item, stage, command):
    log = output/f'seed{item["seed"]}_{stage}_{session.name}.log'
    env = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONDONTWRITEBYTECODE': '1'}
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=campaign.ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT)
        job = {'seed': item['seed'], 'stage': stage, 'attempt': session.name, 'command': command,
               'stdout_log': str(log), 'started_at': stamp(), **campaign.old.process_identity(child.pid)}
        state['jobs'].append(job)
        state.update(status='running', active_seed=item['seed'], active_stage=stage)
        atomic_save(output/'state.json', state, json_format=True)
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    job.update(returncode=code, ended_at=stamp())
    atomic_save(output/'state.json', state, json_format=True)
    need(code == 0, f'Seed {item["seed"]} {stage} failed ({code}); see {log}')
    result = campaign.old.output_path(log, 'OUTPUT: ')
    need(result.parent == (Path(item['package'])/'artifacts').resolve(), 'Foreign child output')
    job['output'] = str(result)
    atomic_save(output/'state.json', state, json_format=True)
    return result


def execute(session):
    session = Path(session).resolve()
    recovery = read(session/'resume_plan.json')
    output, run = Path(recovery['campaign']), Path(recovery['run'])
    need(session.parent == output and session.name.startswith('resume_'), 'Foreign recovery session')
    need(recovery['user_explicitly_authorized_resume'] is True and recovery['script_sha256'] == sha256(__file__),
         'Unapproved/changed recovery script')
    with (campaign.old.FOLDER/'artifacts/multiseed_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (session/'started.claim').open('x') as claim:
            claim.write(stamp()+'\n')
        need(sha256(output/'plan.json') == recovery['plan_sha256'] and
             sha256(output/'state.json') == recovery['before_state_sha256'], 'Campaign changed before recovery')
        plan, state = read(output/'plan.json'), read(output/'state.json')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        need(boot == recovery['new_boot_id'], 'Host rebooted again; inspect before resuming')
        verify_inactive(state['supervisor'], boot)
        verify_inactive(state['jobs'][-1], boot)
        verify_completed(recovery['completed_artifacts_sha256'])
        campaign.verify_sources(plan)
        need(sha256(run/'last.pt') == recovery['last_sha256'] and sha256(run/'best.pt') == recovery['best_sha256'],
             'Recovery checkpoints changed')
        state['jobs'][-1].update(returncode=None, status='interrupted_by_host_reboot',
            termination_observed_at=stamp(), output=str(run), resume_attempt=session.name)
        state.update(status='running', supervisor=campaign.old.process_identity(os.getpid()),
                     active_stage='resume_preparation', resumed_at=stamp())
        state.setdefault('manual_resume_attempts', []).append(str(session))
        atomic_save(output/'state.json', state, json_format=True)
        try:
            item = plan['queue'][-1]
            ready = run_child(output, session, state, item, 'readiness', campaign.old.command_for(item, 'readiness'))
            report = read(ready/'report.json')
            need(report['contract_sha256'] == item['contract_sha256'] and
                 report['status'] == 'GPU_readiness_passed_formal_start_pending' and all(report['checks'].values()),
                 'Fresh reboot readiness failed')
            campaign.verify_sources(plan)
            # The old active.lock is evidence from the now-absent boot, not a live lock.
            if (run/'active.lock').exists():
                (run/'active.lock').rename(session/'stale_active.lock')
            command = campaign.old.command_for(item, 'training', ready/'report.json') + ['--resume', str(run/'last.pt')]
            resumed = run_child(output, session, state, item, 'training', command)
            need(resumed == run and read(run/'run_state.json')['status'] == 'completed', 'Resumed run incomplete/foreign')
            state['formal_runs']['20261026'] = str(run)
            state.update(active_seed=None, active_stage='summary')
            atomic_save(output/'state.json', state, json_format=True)
            campaign.verify_sources(plan)
            verify_completed(recovery['completed_artifacts_sha256'])
            campaign.summarize(output, plan, state['formal_runs'])
            state.update(status='completed', active_seed=None, active_stage=None,
                         completed_at=stamp(), summary='summary.json', test_used=False)
            atomic_save(output/'state.json', state, json_format=True)
        except BaseException as exc:
            state.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}', failed_at=stamp())
            atomic_save(output/'state.json', state, json_format=True)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--run', type=Path)
    args = parser.parse_args()
    prepare(args.prepare) if args.prepare else execute(args.run)
