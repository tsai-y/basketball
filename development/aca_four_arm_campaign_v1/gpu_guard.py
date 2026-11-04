"""Scheduling-only desktop exceptions; unknown compute workloads remain blocked."""
import os
from pathlib import Path
import shutil
import subprocess

from .queue import FOLDER, read, require, sha

LEGACY_DESKTOP = {'Discord', 'gnome-shell', 'Xwayland', 'chrome', 'codex', 'Codex',
                  'gnome-remote-desktop-daemon', 'nautilus', 'gnome-text-editor', 'ptyxis'}
EXTRA_PACKAGES = {'/usr/bin/papers': 'papers',
                  '/usr/bin/gnome-control-center': 'gnome-control-center'}


def executable_pin(path, package):
    path = Path(path)
    require(str(path.resolve(strict=True)) == str(path), 'Desktop executable path changed')
    require(all(p.stat().st_uid == 0 and not p.stat().st_mode & 0o022
                for p in (path, *path.parents)), 'Desktop executable or parent is not root-protected')
    owner = subprocess.check_output(['dpkg-query', '-S', str(path)], text=True).strip()
    require(owner == f'{package}: {path}', 'Unexpected desktop package owner')
    return {'path': str(path), 'package': package, 'sha256': sha(path)}


def make_policy():
    return {'schema': 'aca_gpu_desktop_policy_v1',
            'legacy_desktop_names': sorted(LEGACY_DESKTOP),
            'extra_executables': [executable_pin(path, pkg) for path, pkg in EXTRA_PACKAGES.items()],
            'scope': 'GPU launch scheduling only; common mathematical trainer unchanged',
            'unknown_compute': 'block; never terminate other processes',
            'minimum_free_gpu_MiB': 6144}


def verify_policy(policy):
    require(policy['schema'] == 'aca_gpu_desktop_policy_v1' and
            policy['legacy_desktop_names'] == sorted(LEGACY_DESKTOP) and
            policy['minimum_free_gpu_MiB'] == 6144, 'GPU policy changed')
    require(len(policy['extra_executables']) == len(EXTRA_PACKAGES), 'Desktop policy count mismatch')
    for row, (path, pkg) in zip(policy['extra_executables'], EXTRA_PACKAGES.items()):
        require(row == executable_pin(path, pkg), 'Desktop package/executable fingerprint changed')


def classify(pid, extra_paths):
    if pid == os.getpid():
        return 'self'
    try:
        path = (Path('/proc') / str(pid) / 'exe').resolve(strict=True)
    except FileNotFoundError:
        return 'exited'
    except PermissionError:
        return 'unknown_permission_denied'
    if path.name in LEGACY_DESKTOP or str(path) in extra_paths:
        return 'desktop'
    return str(path)


def assert_idle_gpu(policy):
    verify_policy(policy)
    query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], text=True)
    extra = {r['path'] for r in policy['extra_executables']}
    observed = [{'pid': int(pid), 'classification': classify(int(pid), extra)} for pid in query.split()]
    busy = [r for r in observed if r['classification'] not in ('self', 'exited', 'desktop')]
    require(not busy, f'Other GPU compute processes active: {busy}; none were stopped')
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'], text=True).split()
    require(len(free) == 1 and int(free[0]) >= policy['minimum_free_gpu_MiB'], 'Insufficient free GPU memory')
    require(shutil.disk_usage(FOLDER).free >= 8 * 1024**3, 'Insufficient checkpoint space')
    return observed


def load_policy(path, expected_sha):
    require(Path(path).is_file() and not Path(path).is_symlink() and sha(path) == expected_sha,
            'Missing/changed scheduling policy')
    policy = read(path)
    verify_policy(policy)
    return policy
