import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import queue as q, runtime_continuation as r


def fixture():
    jobs = q.ordered_jobs({s: str(s) for s in q.SEEDS})
    for j in jobs[:4]:
        j['status'] = 'completed'
    jobs[4].update(status='failed', returncode=None, ended_at='fixture')
    state = {'status': 'failed', 'jobs': jobs, 'active_job': jobs[4]['id']}
    failure = {'error': "CalledProcessError(18, ['nvidia-smi', '--query-compute-apps=pid'])",
               'active_job': state['active_job'], 'completed_jobs': 4}
    return state, failure


class RuntimeContinuationTests(unittest.TestCase):
    def test_only_unstarted_nvml_failure_resets_and_preserves_four(self):
        state, failure = fixture()
        original = copy.deepcopy(state)
        jobs, reset = r.eligible_jobs(state, failure)
        self.assertEqual(state, original)
        self.assertEqual(jobs[:4], original['jobs'][:4])
        self.assertEqual(sum(j['status'] == 'queued' for j in jobs), 8)
        self.assertEqual(jobs[4]['id'], 'baseline_seed20261025')
        self.assertNotIn('returncode', jobs[4])
        self.assertEqual(reset['original_entry'], original['jobs'][4])

    def test_never_resets_a_started_or_non_nvml_failure(self):
        for key in ('process', 'output', 'stdout_log', 'started_at', 'command_line'):
            state, failure = fixture()
            state['jobs'][4][key] = None
            with self.assertRaises(RuntimeError):
                r.eligible_jobs(state, failure)
        state, failure = fixture()
        failure['error'] = 'CUDA out of memory'
        with self.assertRaises(RuntimeError):
            r.eligible_jobs(state, failure)
        state, failure = fixture()
        state['jobs'][4]['returncode'] = 1
        with self.assertRaises(RuntimeError):
            r.eligible_jobs(state, failure)

    def test_runtime_pins_env_kernel_and_files(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); root = out / 'root'; libdir = root / 'usr/lib/x86_64-linux-gnu'
            libdir.mkdir(parents=True); (root / 'usr/bin').mkdir()
            binary = root / 'usr/bin/nvidia-smi'; binary.write_text('fixture')
            library = libdir / 'libnvidia-ml.so.595.91.07'; library.write_text('fixture')
            link = libdir / 'libnvidia-ml.so.1'; link.symlink_to(library.name)
            manifest = {'schema': 'aca_original_driver_runtime_v1', 'version': '595.91.07', 'root': str(root),
                        'files_sha256': {str(p.relative_to(root)): r.c.stream_sha(p) for p in (binary, library)},
                        'symlinks': {str(link.relative_to(root)): library.name}}
            path = out / 'manifest.json'; q.atomic(path, manifest)
            actual_read_text = Path.read_text
            def read_text(p, *args, **kwargs):
                return 'NVRM 595.91.07' if str(p) == '/proc/driver/nvidia/version' else actual_read_text(p, *args, **kwargs)
            with patch.object(q, 'ARTIFACTS', out), patch.object(Path, 'read_text', read_text), \
                 patch.object(r.shutil, 'which', return_value=str(binary)), \
                 patch.dict(os.environ, {'LD_LIBRARY_PATH': str(libdir)}):
                self.assertEqual(r.verify_runtime(path), manifest)
                with patch.dict(os.environ, {'LD_LIBRARY_PATH': '/wrong'}):
                    with self.assertRaises(RuntimeError): r.verify_runtime(path)
                library.write_text('changed')
                with self.assertRaises(RuntimeError): r.verify_runtime(path)


if __name__ == '__main__':
    unittest.main()
