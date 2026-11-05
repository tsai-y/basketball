"""Explicit recovery using original-version Ubuntu driver libraries, without system installation."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from . import queue as q
from . import continuation as c
from .gpu_guard import assert_idle_gpu, make_policy


def eligible_jobs(state, failure):
    """Only reset a confirmed NVML pre-launch failure, never a started training job."""
    q.require(state['status'] == 'failed' and
              failure['error'].startswith("CalledProcessError(18, ['nvidia-smi',") and
              failure['active_job'] == state['active_job'], 'Not a confirmed NVML launch-check failure')
    jobs = copy.deepcopy(state['jobs'])
    target = next(j for j in jobs if j['id'] == state['active_job'])
    q.require(target['status'] == 'failed' and target.get('returncode') is None and
              not any(k in target for k in ('process', 'output', 'stdout_log', 'started_at', 'command_line')),
              'Failed job has training evidence; automatic checkpoint recovery is prohibited')
    original = copy.deepcopy(target)
    target['status'] = 'queued'
    for key in ('ended_at', 'returncode'):
        target.pop(key, None)
    normalized = c.continuation_jobs({'jobs': jobs})
    q.require(sum(j['status'] == 'completed' for j in jobs) == failure['completed_jobs'],
              'Completed count differs from failure record')
    return normalized, {'job': target['id'], 'original_entry': original,
                        'normalized_entry': copy.deepcopy(target),
                        'reason': 'Explicit user authorization; failed solely before subprocess creation'}


def verify_runtime(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    m = q.read(manifest_path)
    root = Path(m['root']).resolve()
    q.require(root.is_relative_to(q.ARTIFACTS.resolve()) and root == manifest_path.parent / 'root',
              'Recovered runtime must be isolated inside campaign artifacts')
    q.require(m['schema'] == 'aca_original_driver_runtime_v1' and m['version'] == '595.91.07' and
              m['version'] in Path('/proc/driver/nvidia/version').read_text(),
              'Recovered runtime no longer matches the loaded kernel driver')
    for name, expected in m['files_sha256'].items():
        path = root / name
        q.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root) and
                  c.stream_sha(path) == expected, f'Recovered runtime file changed: {name}')
    for name, target in m['symlinks'].items():
        path = root / name
        q.require(path.is_symlink() and os.readlink(path) == target and
                  path.resolve(strict=True).is_relative_to(root), 'Recovered runtime symlink changed')
    libdir = root / 'usr/lib/x86_64-linux-gnu'
    q.require(os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] == str(libdir) and
              Path(shutil.which('nvidia-smi')).resolve() == root / 'usr/bin/nvidia-smi',
              'Launch process must already use the recovered runtime environment')
    return m


def prepare(previous, manifest_path):
    runtime = verify_runtime(manifest_path)
    old = q.read(previous / 'plan.json')
    state = q.read(previous / 'state.json')
    launcher = q.read(previous / 'launcher.json')
    failure = q.read(previous / 'failure.json')
    q.require(not q.alive(state['supervisor']), 'Previous supervisor still alive')
    q.require(state['plan_sha256'] == launcher['plan_sha256'] == q.sha(previous / 'plan.json'),
              'Previous plan identity changed')
    q.verify_pins(old)
    c.verify_preserved(old)
    jobs, reset = eligible_jobs(state, failure)
    policy = make_policy()
    observed = assert_idle_gpu(policy)
    from development.aca_four_arm_v1.experiment import campaign_plan, ArmConfig, verify_preflight
    current = campaign_plan(old['source_root'])
    q.require(q.digest(current) == old['campaign_sha256'],
              'Original common campaign/runtime/readiness fingerprint was not restored')
    preserved = dict(old.get('preserved_files', {}))
    for name in ('plan.json', 'state.json', 'launcher.json', 'failure.json', 'supervisor.log'):
        path = previous / name
        preserved[str(path)] = c.stream_sha(path)
    for j in jobs:
        verify_preflight(j['readiness_report'], current, ArmConfig(variant=j['variant'], seed=j['seed']))
        if j['status'] == 'completed':
            q.require(not q.alive(j['process']), 'Previous trainer still alive')
            result = q.validate_completed(Path(j['output']), j, old['campaign'])
            q.require(result == j['result'], 'Preserved result changed')
            for name in ('plan.json', 'campaign.json', 'history.json', 'summary.json', 'best.pt', 'last.pt'):
                path = Path(j['output']) / name
                preserved[str(path)] = c.stream_sha(path)
        else:
            q.require(not list(q.TRAIN_ARTIFACTS.glob(f"formal_{j['variant']}_seed{j['seed']}_*")),
                      'Pending job already has a formal run')
    plan = copy.deepcopy(old)
    plan.update(schema='aca_four_arm_original_runtime_recovery_v1', created_at=q.now(), jobs=jobs,
                previous_output=str(previous), previous_plan_file_sha256=q.sha(previous / 'plan.json'),
                preserved_files=preserved, gpu_policy=policy, explicit_prelaunch_reset=reset,
                runtime_manifest=str(manifest_path), runtime_manifest_sha256=q.sha(manifest_path),
                runtime=runtime, gpu_processes_at_recovery=observed,
                reason='User authorized repair and continuation after system NVIDIA user-space update',
                automatic_retry=False, automatic_resume=False,
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=q.ROOT, text=True).strip())
    for path in q.FOLDER.glob('*.py'):
        plan['pinned_files'][path.relative_to(q.ROOT).as_posix()] = q.sha(path)
    plan['pinned_files'][str(manifest_path)] = q.sha(manifest_path)
    for name, expected in runtime['files_sha256'].items():
        plan['pinned_files'][str(Path(runtime['root']) / name)] = expected
    return plan


def launch(previous, manifest_path):
    with (q.ARTIFACTS / 'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q.require(not list(previous.glob('continuation_*/plan.json')), 'Continuation already exists')
        plan = prepare(previous, manifest_path)
        output = Path(tempfile.mkdtemp(prefix='continuation_runtime_', dir=previous))
        q.atomic(output / 'plan.json', plan)
        q.atomic(output / 'gpu_policy.json', plan['gpu_policy'])
        # Snapshot source/readiness separately; runtime binaries are pinned in-place, not recopied.
        for name in plan['pinned_files']:
            if Path(name).is_absolute():
                continue
            target = output / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(q.ROOT / name, target)
        env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONDONTWRITEBYTECODE': '1'}
        cmd = [plan['python'], '-u', '-B', '-m', 'development.aca_four_arm_campaign_v1.continuation',
               '--supervise', str(output), '--lock-fd', str(lock.fileno())]
        q.atomic(output / 'launch_request.json', {'at': q.now(), 'command_line': cmd,
                  'plan_sha256': q.sha(output / 'plan.json'),
                  'runtime_environment': {k: env[k] for k in ('PATH', 'LD_LIBRARY_PATH')}})
        with (output / 'supervisor.log').open('x') as stream:
            process = subprocess.Popen(cmd, cwd=q.ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
        q.atomic(output / 'spawn.json', {'at': q.now(), 'supervisor': q.identity(process.pid)})
    print(f'OUTPUT: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--runtime-manifest', type=Path)
    parser.add_argument('--accept-candidate-architecture', action='store_true')
    args = parser.parse_args()
    if args.continue_from:
        q.require(args.runtime_manifest and args.accept_candidate_architecture,
                  'Explicit recovered runtime and candidate acknowledgement required')
        previous = args.continue_from.resolve()
        q.require(previous.is_relative_to(q.ARTIFACTS.resolve()), 'Previous attempt is outside artifacts')
        launch(previous, args.runtime_manifest.resolve())
    else:
        print('No training started. Explicit recovery launch required.')


if __name__ == '__main__':
    main()
