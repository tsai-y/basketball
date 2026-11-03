"""Explicit launch only; no retries, resume, test, or protocol changes."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
ARTIFACTS = FOLDER / 'artifacts'
TRAIN_ARTIFACTS = ROOT / 'development/aca_four_arm_v1/artifacts'
READINESS = ROOT / 'docs/research/results/aca_four_arm_readiness_v1.json'
VARIANTS = ('baseline', 'lsta', 'tsci', 'aca')
SEEDS = (20261024, 20261025, 20261026)
METRICS = ('loss', 'accuracy', 'macro_f1', 'miss_recall', 'make_recall')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.json-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def stat_fields(text):
    # comm can contain spaces and parentheses; fields after its final ')' start at #3.
    fields = text[text.rfind(')') + 2:].split()
    return fields[0], int(fields[19])


def identity(pid):
    proc = Path('/proc') / str(pid)
    state, ticks = stat_fields((proc / 'stat').read_text())
    return {'pid': pid, 'process_start_ticks': ticks,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'command_line': (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
            'process_state': state}


def alive(saved):
    try:
        current = identity(saved['pid'])
        return current['process_state'] != 'Z' and all(current[k] == saved[k] for k in
            ('pid', 'process_start_ticks', 'boot_id', 'command_line'))
    except (FileNotFoundError, ProcessLookupError):
        return False


def ordered_jobs(reports):
    return [{'id': f'{variant}_seed{seed}', 'variant': variant, 'seed': seed,
             'readiness_report': reports[seed], 'status': 'queued'}
            for seed in SEEDS for variant in VARIANTS]


def checked_output(log, job, parent=TRAIN_ARTIFACTS):
    outputs = re.findall(r'^OUTPUT: (.+)$', Path(log).read_text(errors='replace'), re.MULTILINE)
    require(len(outputs) <= 1, 'Ambiguous OUTPUT lines; no automatic resume is allowed')
    if not outputs:
        return None
    output = Path(outputs[0]).resolve()
    require(output.parent == parent.resolve() and
            output.name.startswith(f"formal_{job['variant']}_seed{job['seed']}_"),
            'Unexpected formal output path')
    return output


def validate_history(history):
    require(isinstance(history, list) and len(history) <= 30, 'Invalid history length')
    best = None
    bad = 0
    for index, row in enumerate(history, 1):
        value = row['validation']
        require(row['epoch'] == index and row['train']['count'] == 7193 and
                value['count'] == 1547, 'Incomplete epoch or validation')
        require(all(math.isfinite(value[key]) for key in METRICS), 'Nonfinite validation')
        improved = best is None or value['loss'] < best['validation']['loss']
        require(row['selected_as_best'] is improved, 'Incorrect minimum CE selection')
        require(bad < 5, 'History continued beyond patience')
        bad = 0 if improved else bad + 1
        if improved:
            best = row
    return best, bad


def validate_completed(output, job, campaign):
    from development.aca_four_arm_v1.experiment import ArmConfig, arm_plan
    expected = arm_plan(campaign, ArmConfig(variant=job['variant'], seed=job['seed']))
    require(read(output / 'plan.json') == expected, 'Completed run has mismatched arm/seed/plan')
    require(read(output / 'campaign.json') == campaign, 'Completed campaign differs')
    result = read(output / 'summary.json')
    history = read(output / 'history.json')
    best, bad = validate_history(history)
    epochs = len(history)
    require(result['status'] == 'completed' and result['model'] == job['variant'] and
            result['output'] == str(output) and 1 <= epochs <= 30 and result['epochs'] == epochs and
            result['optimizer_steps'] == 450 * epochs and result['test_videos_opened'] == 0 and
            result['best_epoch'] == best['epoch'] and result['best_validation'] == best['validation'] and
            result['stop_reason'] == ('early_stop' if bad >= 5 else 'max_epochs') and
            (epochs == 30 or bad >= 5), 'Missing/inconsistent final selection or termination')
    require(all((output / name).is_file() and not (output / name).is_symlink()
                for name in ('best.pt', 'last.pt')), 'Missing formal checkpoints')
    return result


def snapshot(job):
    output = checked_output(job['stdout_log'], job)
    if output is None:
        return {'phase': 'initialization', 'epoch': 1, 'train_videos': 0, 'optimizer_updates': 0}
    job['output'] = str(output)
    log = Path(job['stdout_log']).read_text(errors='replace')
    matches = re.findall(r'^four-arm (\w+) epoch=(\d+) cursor=(\d+)/7193 steps=(\d+) ', log, re.MULTILINE)
    epoch, cursor, steps = (1, 0, 0)
    if matches:
        variant, epoch, cursor, steps = matches[-1]
        require(variant == job['variant'], 'Log arm mismatch')
        epoch, cursor, steps = int(epoch), int(cursor), int(steps)
    history = read(output / 'history.json') if (output / 'history.json').exists() else []
    best, bad = validate_history(history)
    if len(history) >= epoch:
        epoch, cursor, steps = len(history) + 1, 0, 450 * len(history)
    return {'phase': 'validation' if cursor == 7193 else 'train',
            'epoch': min(epoch, 30), 'max_epochs': 30, 'train_videos': cursor,
            'train_total': 7193, 'optimizer_updates': steps,
            'progress_precision': 'last logged checkpoint, every 25 optimizer updates',
            'latest_complete_validation': history[-1] if history else None,
            'best_epoch': best['epoch'] if best else None, 'early_stop_count': bad}


def aggregate(jobs):
    require(len(jobs) == 12 and all(j['status'] == 'completed' for j in jobs),
            'All twelve formal runs must complete before aggregation')
    rows = [{'variant': j['variant'], 'seed': j['seed'], 'output': j['output'],
             **j['result']} for j in jobs]
    arms = {}
    for variant in VARIANTS:
        group = [r for r in rows if r['variant'] == variant]
        require(sorted(r['seed'] for r in group) == list(SEEDS), 'Missing/duplicate seed')
        arms[variant] = {key: {'mean': statistics.mean(r['best_validation'][key] for r in group),
                              'sample_sd': statistics.stdev(r['best_validation'][key] for r in group)}
                         for key in METRICS}
    return {'status': 'completed', 'completed_at': now(), 'rows': rows, 'by_variant': arms,
            'evaluation': 'validation, strict minimum CE within each run; not test',
            'paper_exact_reproduction': False, 'test_videos_opened': 0,
            'prior_project_test_exposure': True, 'automatic_variant_selection': False}


def verify_pins(plan):
    for path, expected in plan['pinned_files'].items():
        require(sha(ROOT / path) == expected, f'Changed pinned file: {path}')


def prepare(source_root):
    from development.aca_four_arm_v1.experiment import (ArmConfig, campaign_plan, verify_preflight,
                                                      assert_idle_gpu)
    assert_idle_gpu()
    require(shutil.disk_usage(FOLDER).free >= 40 * 1024**3, 'Need 40 GiB free before campaign')
    evidence = read(READINESS)
    campaign = campaign_plan(source_root)
    require(digest(campaign) == evidence['campaign_sha256'], 'Readiness fingerprint is stale')
    require(campaign['arms'] == evidence['configurations'], 'Readiness configurations changed')
    reports = {}
    pins = {READINESS.relative_to(ROOT).as_posix(): sha(READINESS), **campaign['source_code_sha256']}
    cpu = evidence['cpu_preparation']
    for name, expected in (('report.json', cpu['report_sha256']), ('plan.json', cpu['plan_sha256'])):
        path = ROOT / cpu['path'] / name
        require(sha(path) == expected, 'CPU preparation changed')
        pins[path.relative_to(ROOT).as_posix()] = expected
    require(digest(read(ROOT / cpu['path'] / 'plan.json')) == digest(campaign),
            'CPU preparation plan mismatch')
    for row in evidence['gpu_readiness_reports']:
        path = ROOT / row['path']
        require(row['seed'] in SEEDS and row['seed'] not in reports and sha(path) == row['sha256'],
                'Readiness SHA/seed mismatch')
        reports[row['seed']] = str(path)
        pins[row['path']] = row['sha256']
    require(set(reports) == set(SEEDS), 'Need all three readiness reports')
    jobs = ordered_jobs(reports)
    require([asdict(ArmConfig(variant=j['variant'], seed=j['seed'])) for j in jobs] == campaign['arms'],
            'Queue order/config mismatch')
    for job in jobs:
        verify_preflight(job['readiness_report'], campaign,
                         ArmConfig(variant=job['variant'], seed=job['seed']))
    for path in FOLDER.glob('*.py'):
        pins[path.relative_to(ROOT).as_posix()] = sha(path)
    return {'schema': 'aca_four_arm_sequential_queue_v1', 'created_at': now(),
            'campaign_sha256': digest(campaign), 'campaign': campaign, 'jobs': jobs,
            'python': str(Path(sys.executable).absolute()), 'worktree': str(ROOT),
            'pinned_files': pins, 'source_root': str(Path(source_root).resolve()),
            'retry': False, 'resume': False, 'test': False, 'parallel_jobs': 1,
            'order': 'seed outer, baseline/lsta/tsci/aca inner',
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}


def supervisor(output, lock_fd):
    require(lock_fd is not None, 'Supervisor requires inherited exclusive lock')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(output / 'plan.json')
    verify_pins(plan)
    parent = identity(os.getpid())
    launcher = {'created_at': now(), 'supervisor': parent, 'plan_sha256': sha(output / 'plan.json'),
                'campaign_sha256': plan['campaign_sha256'], 'output': str(output)}
    atomic(output / 'launcher.json', launcher)
    state = {'status': 'running', 'supervisor': parent, 'plan_sha256': launcher['plan_sha256'],
             'jobs': plan['jobs'], 'active_job': None, 'started_at': now()}
    stop = {'requested': False}
    child = None

    def on_stop(signum, frame):
        stop['requested'] = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_stop)

    def save():
        state['updated_at'] = now()
        atomic(output / 'state.json', state)

    try:
        from development.aca_four_arm_v1.experiment import assert_idle_gpu
        for job in state['jobs']:
            require(not stop['requested'], 'Supervisor stopped; queue will not continue automatically')
            verify_pins(plan)
            assert_idle_gpu()
            require(shutil.disk_usage(output).free >= 8 * 1024**3, 'Insufficient checkpoint space')
            cmd = [plan['python'], '-u', '-B', '-m', 'development.aca_four_arm_v1.train',
                   '--source-root', plan['source_root'], '--start-training', '--variant', job['variant'],
                   '--seed', str(job['seed']), '--readiness-report', job['readiness_report'],
                   '--accept-candidate-architecture']
            log = output / (job['id'] + '.log')
            job.update(status='starting', stdout_log=str(log), started_at=now(), command_line=cmd)
            state['active_job'] = job['id']
            save()
            with log.open('x') as stream:
                child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                         stderr=subprocess.STDOUT)
                job.update(status='running', process=identity(child.pid))
                save()
                print(f"{now()} START {job['id']} PID={child.pid}", flush=True)
                while child.poll() is None:
                    job['progress'] = snapshot(job)
                    save()
                    time.sleep(10)
                job['returncode'] = child.returncode
            job['ended_at'] = now()
            job['progress'] = snapshot(job)
            require(child.returncode == 0, f"{job['id']} exited {child.returncode}")
            require('output' in job, 'Trainer exited without a formal output')
            job['result'] = validate_completed(Path(job['output']), job, plan['campaign'])
            job['status'] = 'completed'
            save()
            print(f"{now()} COMPLETE {job['id']} best_epoch={job['result']['best_epoch']}", flush=True)
            child = None
        atomic(output / 'summary.json', aggregate(state['jobs']))
        state.update(status='completed', active_job=None, completed_at=now())
        save()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait()  # Let the trainer commit a full optimizer boundary; never kill/restart it.
        active = next((j for j in state['jobs'] if j['id'] == state['active_job']), None)
        if active is not None and active['status'] != 'completed':
            active.update(status='interrupted' if stop['requested'] else 'failed',
                          returncode=child.returncode if child else None, ended_at=now())
        failure = {'error': repr(exc), 'at': now(), 'active_job': state['active_job'],
                   'completed_jobs': sum(j['status'] == 'completed' for j in state['jobs']),
                   'automatic_retry': False, 'automatic_resume': False}
        atomic(output / 'failure.json', failure)
        state.update(status='interrupted' if stop['requested'] else 'failed', failure=failure)
        save()
        raise


def launch(source_root):
    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / 'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(source_root)
        for previous in ARTIFACTS.glob('campaign_*/plan.json'):
            require(read(previous)['campaign_sha256'] != plan['campaign_sha256'],
                    f'Campaign already launched/prepared at {previous.parent}; do not duplicate')
        output = Path(tempfile.mkdtemp(prefix='campaign_', dir=ARTIFACTS))
        atomic(output / 'plan.json', plan)
        for name in plan['pinned_files']:
            target = output / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONDONTWRITEBYTECODE': '1'}
        cmd = [plan['python'], '-u', '-B', '-m', 'development.aca_four_arm_campaign_v1.queue',
               '--supervise', str(output), '--lock-fd', str(lock.fileno())]
        atomic(output / 'launch_request.json', {'at': now(), 'command_line': cmd,
                                               'plan_sha256': sha(output / 'plan.json')})
        with (output / 'supervisor.log').open('x') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(lock.fileno(),))
        # Closing the parent's FD preserves the child's flock; explicitly unlocking would not.
        atomic(output / 'spawn.json', {'at': now(), 'supervisor': identity(process.pid)})
    print(f'OUTPUT: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--accept-candidate-architecture', action='store_true')
    parser.add_argument('--supervise', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.supervise:
        supervisor(args.supervise.resolve(), args.lock_fd)
    elif args.launch:
        require(args.source_root is not None and args.accept_candidate_architecture,
                'Explicit source root and acknowledgement of unverified candidate architecture required')
        launch(args.source_root)
    else:
        print('No training started. Explicit --launch and --accept-candidate-architecture required.')


if __name__ == '__main__':
    main()
