"""User-authorized continuation of untouched queued jobs; completed runs stay frozen."""
import argparse
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from . import queue as q
from .gpu_guard import assert_idle_gpu, load_policy, make_policy


def stream_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def continuation_jobs(state):
    jobs = copy.deepcopy(state['jobs'])
    expected = [(s, v) for s in q.SEEDS for v in q.VARIANTS]
    q.require([(j['seed'], j['variant']) for j in jobs] == expected, 'Changed queue order')
    seen_pending = False
    for j in jobs:
        q.require(j['id'] == f"{j['variant']}_seed{j['seed']}", 'Changed job identity')
        q.require(j['status'] in ('completed', 'queued'), 'Only untouched queued jobs can continue')
        if j['status'] == 'queued':
            seen_pending = True
            q.require(not any(k in j for k in ('output', 'process', 'stdout_log', 'returncode')),
                      'Queued job already has an attempt; explicit checkpoint recovery needed')
        else:
            q.require(not seen_pending, 'Completed jobs must form the preserved prefix')
    q.require(seen_pending, 'No queued jobs remain')
    return jobs


def verify_preserved(plan):
    for path, expected in plan['preserved_files'].items():
        q.require(stream_sha(path) == expected, f'Preserved prior artifact changed: {path}')


def prepare(previous):
    old = q.read(previous / 'plan.json')
    state = q.read(previous / 'state.json')
    launcher = q.read(previous / 'launcher.json')
    q.require(state['status'] == 'failed' and not q.alive(state['supervisor']),
              'Previous queue must be confirmed stopped')
    q.require(state['plan_sha256'] == launcher['plan_sha256'] == q.sha(previous / 'plan.json'),
              'Previous plan identity mismatch')
    q.verify_pins(old)
    jobs = continuation_jobs(state)
    policy = make_policy()
    assert_idle_gpu(policy)
    # Revalidate current data/runtime/readiness against the immutable common campaign.
    from development.aca_four_arm_v1.experiment import campaign_plan, verify_preflight, ArmConfig
    q.require(q.digest(campaign_plan(old['source_root'])) == old['campaign_sha256'],
              'Common data/source/runtime plan changed')
    preserved = {str(previous / name): stream_sha(previous / name)
                 for name in ('plan.json', 'state.json', 'launcher.json', 'failure.json', 'supervisor.log')}
    for j in jobs:
        verify_preflight(j['readiness_report'], old['campaign'], ArmConfig(variant=j['variant'], seed=j['seed']))
        if j['status'] == 'completed':
            q.require(not q.alive(j['process']), 'Previous trainer still alive')
            result = q.validate_completed(Path(j['output']), j, old['campaign'])
            q.require(result == j['result'], 'Previous result mirror changed')
            for name in ('plan.json', 'campaign.json', 'history.json', 'summary.json', 'last.pt', 'best.pt'):
                path = Path(j['output']) / name
                preserved[str(path)] = stream_sha(path)
        else:
            q.require(not list(q.TRAIN_ARTIFACTS.glob(f"formal_{j['variant']}_seed{j['seed']}_*")),
                      'Pending arm already has a formal run; inspect before continuing')
    plan = copy.deepcopy(old)
    plan.update(schema='aca_four_arm_explicit_continuation_v1', created_at=q.now(), jobs=jobs,
                previous_output=str(previous), previous_plan_file_sha256=q.sha(previous / 'plan.json'),
                preserved_files=preserved, gpu_policy=policy,
                reason='User authorized remaining jobs after desktop GPU check stopped the prior queue',
                completed_runs_reused_only_as_completed_results=True, automatic_retry=False,
                automatic_resume=False,
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=q.ROOT, text=True).strip())
    for path in q.FOLDER.glob('*.py'):
        plan['pinned_files'][path.relative_to(q.ROOT).as_posix()] = q.sha(path)
    return plan


def supervise(output, lock_fd):
    q.require(lock_fd is not None, 'Inherited campaign lock required')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = q.read(output / 'plan.json')
    q.verify_pins(plan)
    verify_preserved(plan)
    policy_path = output / 'gpu_policy.json'
    policy_sha = q.sha(policy_path)
    policy = load_policy(policy_path, policy_sha)
    q.require(policy == plan['gpu_policy'], 'Scheduling policy plan mismatch')
    parent = q.identity(os.getpid())
    launcher = {'created_at': q.now(), 'supervisor': parent, 'plan_sha256': q.sha(output / 'plan.json'),
                'campaign_sha256': plan['campaign_sha256'], 'output': str(output),
                'previous_output': plan['previous_output'], 'gpu_policy_sha256': policy_sha}
    q.atomic(output / 'launcher.json', launcher)
    state = {'status': 'running', 'supervisor': parent, 'plan_sha256': launcher['plan_sha256'],
             'jobs': plan['jobs'], 'active_job': None, 'started_at': q.now(),
             'previous_output': plan['previous_output']}
    child = None
    stopped = False

    def save():
        state['updated_at'] = q.now()
        q.atomic(output / 'state.json', state)

    def on_stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_stop)
    save()
    try:
        for job in state['jobs']:
            if job['status'] == 'completed':
                q.require(q.validate_completed(Path(job['output']), job, plan['campaign']) == job['result'],
                          'Preserved completion changed')
                continue
            state['active_job'] = job['id']
            save()
            q.require(not stopped, 'Stop requested; no automatic continuation')
            q.verify_pins(plan)
            q.require(q.sha(policy_path) == policy_sha, 'Scheduling policy changed')
            observed = assert_idle_gpu(policy)
            cmd = [plan['python'], '-u', '-B', '-m', 'development.aca_four_arm_campaign_v1.train_entry',
                   '--gpu-policy', str(policy_path), '--gpu-policy-sha256', policy_sha,
                   '--source-root', plan['source_root'], '--start-training', '--variant', job['variant'],
                   '--seed', str(job['seed']), '--readiness-report', job['readiness_report'],
                   '--accept-candidate-architecture']
            log = output / (job['id'] + '.log')
            job.update(status='starting', started_at=q.now(), stdout_log=str(log), command_line=cmd,
                       gpu_processes_at_launch=observed)
            save()
            with log.open('x') as stream:
                child = subprocess.Popen(cmd, cwd=q.ROOT, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
                job.update(status='running', process=q.identity(child.pid))
                print(f"{q.now()} START {job['id']} PID={child.pid}", flush=True)
                save()
                while child.poll() is None:
                    job['progress'] = q.snapshot(job)
                    save()
                    time.sleep(10)
                job['returncode'] = child.returncode
            job['ended_at'] = q.now()
            job['progress'] = q.snapshot(job)
            q.require(child.returncode == 0 and 'output' in job, 'Trainer failed/stopped without complete output')
            job['result'] = q.validate_completed(Path(job['output']), job, plan['campaign'])
            job['status'] = 'completed'
            save()
            print(f"{q.now()} COMPLETE {job['id']} best_epoch={job['result']['best_epoch']}", flush=True)
            child = None
        verify_preserved(plan)
        q.atomic(output / 'summary.json', q.aggregate(state['jobs']))
        state.update(status='completed', active_job=None, completed_at=q.now())
        save()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait()
        active = next((j for j in state['jobs'] if j['id'] == state['active_job']), None)
        if active is not None and active['status'] != 'completed':
            active.update(status='interrupted' if stopped else 'failed', ended_at=q.now(),
                          returncode=child.returncode if child else None)
        failure = {'error': repr(exc), 'at': q.now(), 'active_job': state['active_job'],
                   'completed_jobs': sum(j['status'] == 'completed' for j in state['jobs']),
                   'automatic_retry': False, 'automatic_resume': False}
        q.atomic(output / 'failure.json', failure)
        state.update(status='interrupted' if stopped else 'failed', failure=failure)
        save()
        raise


def launch(previous):
    with (q.ARTIFACTS / 'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q.require(not list(previous.glob('continuation_*/plan.json')),
                  'A continuation already exists; inspect that latest attempt before proceeding')
        plan = prepare(previous)
        output = Path(tempfile.mkdtemp(prefix='continuation_', dir=previous))
        q.atomic(output / 'plan.json', plan)
        q.atomic(output / 'gpu_policy.json', plan['gpu_policy'])
        for name in plan['pinned_files']:
            target = output / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(q.ROOT / name, target)
        env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONDONTWRITEBYTECODE': '1'}
        cmd = [plan['python'], '-u', '-B', '-m', 'development.aca_four_arm_campaign_v1.continuation',
               '--supervise', str(output), '--lock-fd', str(lock.fileno())]
        q.atomic(output / 'launch_request.json', {'at': q.now(), 'command_line': cmd,
                                                'plan_sha256': q.sha(output / 'plan.json')})
        with (output / 'supervisor.log').open('x') as stream:
            process = subprocess.Popen(cmd, cwd=q.ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        q.atomic(output / 'spawn.json', {'at': q.now(), 'supervisor': q.identity(process.pid)})
    print(f'OUTPUT: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--accept-candidate-architecture', action='store_true')
    parser.add_argument('--supervise', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise:
        supervise(args.supervise.resolve(), args.lock_fd)
    elif args.continue_from:
        q.require(args.accept_candidate_architecture, 'Explicit candidate architecture acknowledgement required')
        previous = args.continue_from.resolve()
        q.require(previous.is_relative_to(q.ARTIFACTS.resolve()), 'Previous run must be inside campaign artifacts')
        launch(previous)
    else:
        print('No training started. Explicit --continue-from required.')


if __name__ == '__main__':
    main()
