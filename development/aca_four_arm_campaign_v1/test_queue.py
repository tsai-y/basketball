import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import queue as q


def history():
    rows = []
    for epoch in range(1, 7):
        rows.append({'epoch': epoch, 'train': {'count': 7193},
                     'validation': {'count': 1547, 'loss': .5 if epoch == 1 else .6,
                                    'accuracy': .8, 'macro_f1': .79,
                                    'miss_recall': .75, 'make_recall': .85},
                     'selected_as_best': epoch == 1})
    return rows


class QueueTests(unittest.TestCase):
    def test_prepare_accepts_json_tuple_roundtrip_but_rejects_stale_plan(self):
        from dataclasses import asdict
        from development.aca_four_arm_v1.experiment import ArmConfig
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / 'queue'
            package.mkdir()
            cpu = root / 'cpu'
            cpu.mkdir()
            campaign = {'arms': [asdict(ArmConfig(variant=v, seed=s))
                                 for s in q.SEEDS for v in q.VARIANTS],
                        'source_code_sha256': {}, 'normalization': {'mean': (.1, .2, .3)}}
            q.atomic(cpu / 'plan.json', campaign)
            q.atomic(cpu / 'report.json', {})
            reports = []
            for seed in q.SEEDS:
                path = root / f'{seed}.json'
                q.atomic(path, {})
                reports.append({'seed': seed, 'path': path.name, 'sha256': q.sha(path)})
            evidence = {'campaign_sha256': q.digest(campaign), 'configurations': campaign['arms'],
                        'gpu_readiness_reports': reports, 'cpu_preparation': {
                            'path': 'cpu', 'plan_sha256': q.sha(cpu / 'plan.json'),
                            'report_sha256': q.sha(cpu / 'report.json')}}
            q.atomic(root / 'readiness.json', evidence)
            with patch.object(q, 'ROOT', root), patch.object(q, 'FOLDER', package), \
                 patch.object(q, 'READINESS', root / 'readiness.json'), \
                 patch.object(q.shutil, 'disk_usage', return_value=Mock(free=100 * 1024**3)), \
                 patch('development.aca_four_arm_v1.experiment.assert_idle_gpu'), \
                 patch('development.aca_four_arm_v1.experiment.campaign_plan', return_value=campaign), \
                 patch('development.aca_four_arm_v1.experiment.verify_preflight') as gate, \
                 patch.object(q.subprocess, 'check_output', return_value='fixture'):
                self.assertEqual(len(q.prepare(root)['jobs']), 12)
                self.assertEqual(gate.call_count, 12)
                campaign['normalization']['mean'] = (.4, .2, .3)
                with self.assertRaisesRegex(RuntimeError, 'fingerprint is stale'):
                    q.prepare(root)

    def test_exact_order_all_twelve(self):
        jobs = q.ordered_jobs({seed: str(seed) for seed in q.SEEDS})
        self.assertEqual([(j['seed'], j['variant']) for j in jobs],
                         [(s, v) for s in q.SEEDS for v in q.VARIANTS])
        self.assertEqual(len(set(j['id'] for j in jobs)), 12)

    def test_proc_identity_and_reused_pid_detection(self):
        current = q.identity(os.getpid())
        self.assertTrue(q.alive(current))
        for field in ('boot_id', 'process_start_ticks', 'command_line'):
            other = copy.deepcopy(current)
            other[field] = 'wrong'
            self.assertFalse(q.alive(other))
        fields = ['S'] + ['0'] * 18 + ['12345']
        self.assertEqual(q.stat_fields('99 (worker ) with spaces) ' + ' '.join(fields)), ('S', 12345))

    def test_output_rejects_old_arm_or_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log = root / 'stdout.log'
            job = {'variant': 'aca', 'seed': 20261024}
            log.write_text('initializing\n')
            self.assertIsNone(q.checked_output(log, job, root))
            log.write_text(f'OUTPUT: {root}/formal_aca_seed20261024_new\n')
            self.assertEqual(q.checked_output(log, job, root).name, 'formal_aca_seed20261024_new')
            log.write_text(f'OUTPUT: {root}/formal_lsta_seed20261024_old\n')
            with self.assertRaises(RuntimeError):
                q.checked_output(log, job, root)
            log.write_text(f'OUTPUT: {root}/formal_aca_seed20261024_new\n' * 2)
            with self.assertRaises(RuntimeError):
                q.checked_output(log, job, root)

    def test_full_validation_strict_ce_tie_and_patience(self):
        rows = history()
        self.assertEqual(q.validate_history(rows)[1], 5)
        rows[1]['validation']['loss'] = .5
        self.assertEqual(q.validate_history(rows)[0]['epoch'], 1)
        rows[1]['selected_as_best'] = True
        with self.assertRaises(RuntimeError):
            q.validate_history(rows)
        rows = history()
        rows[0]['validation']['count'] = 8
        with self.assertRaises(RuntimeError):
            q.validate_history(rows)
        rows = history()
        rows.append({**rows[-1], 'epoch': 7})
        with self.assertRaises(RuntimeError):
            q.validate_history(rows)

    def test_completion_needs_real_summary_and_matching_seed(self):
        from development.aca_four_arm_v1.experiment import ArmConfig, arm_plan
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            job = {'variant': 'baseline', 'seed': 20261024}
            campaign = {'train_count': 7193, 'validation_count': 1547}
            q.atomic(out / 'campaign.json', campaign)
            q.atomic(out / 'plan.json', arm_plan(campaign, ArmConfig(**job)))
            q.atomic(out / 'history.json', history())
            # A clean return code without a summary is not completion.
            with self.assertRaises(FileNotFoundError):
                q.validate_completed(out, job, campaign)
            result = {'status': 'completed', 'model': 'baseline', 'output': str(out),
                      'epochs': 6, 'optimizer_steps': 2700, 'test_videos_opened': 0,
                      'best_epoch': 1, 'best_validation': history()[0]['validation'],
                      'stop_reason': 'early_stop'}
            q.atomic(out / 'summary.json', result)
            for name in ('best.pt', 'last.pt'):
                (out / name).write_text('fixture only; never loaded as a model')
            self.assertEqual(q.validate_completed(out, job, campaign), result)
            with self.assertRaises(RuntimeError):
                q.validate_completed(out, {**job, 'seed': 20261025}, campaign)
            result['optimizer_steps'] = 2699
            q.atomic(out / 'summary.json', result)
            with self.assertRaises(RuntimeError):
                q.validate_completed(out, job, campaign)

    def test_aggregate_requires_all_three_seeds_and_sample_sd(self):
        jobs = q.ordered_jobs({s: str(s) for s in q.SEEDS})
        for j in jobs:
            j.update(status='completed', output=j['id'], result={
                'best_epoch': 1, 'best_validation': {k: j['seed'] - 20261023 for k in q.METRICS}})
        result = q.aggregate(jobs)
        self.assertEqual(result['by_variant']['aca']['accuracy'], {'mean': 2, 'sample_sd': 1.0})
        jobs[-1]['status'] = 'queued'
        with self.assertRaises(RuntimeError):
            q.aggregate(jobs)

    def test_no_default_launch(self):
        result = subprocess.run([os.sys.executable, '-B', '-m', q.__name__], cwd=q.ROOT,
                                capture_output=True, text=True, check=True)
        self.assertIn('No training started', result.stdout)

    def test_queue_advances_after_early_stop_and_fails_without_retry(self):
        # Only the orchestration is exercised: fake child processes never run training.
        class Child:
            pid = os.getpid()
            returncode = 0
            def poll(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            jobs = q.ordered_jobs({s: str(s) for s in q.SEEDS})
            plan = {'jobs': jobs, 'campaign_sha256': 'fixture', 'campaign': {},
                    'source_root': str(out), 'python': os.sys.executable}
            q.atomic(out / 'plan.json', plan)
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                with (out / 'lock').open('w') as lock, \
                     patch.object(q, 'verify_pins'), \
                     patch('development.aca_four_arm_v1.experiment.assert_idle_gpu'), \
                     patch.object(q.subprocess, 'Popen', return_value=Child()) as popen, \
                     patch.object(q, 'snapshot', side_effect=lambda j: j.update(output=str(out)) or {}), \
                     patch.object(q, 'validate_completed', side_effect=[{'best_epoch': 1},
                                                                       RuntimeError('no complete validation')]):
                    with self.assertRaisesRegex(RuntimeError, 'no complete validation'):
                        q.supervisor(out, lock.fileno())
                    self.assertEqual(popen.call_count, 2)
                state = q.read(out / 'state.json')
                self.assertEqual([j['status'] for j in state['jobs'][:3]], ['completed', 'failed', 'queued'])
                self.assertEqual(state['failure']['completed_jobs'], 1)
                self.assertFalse(state['failure']['automatic_resume'])
                self.assertFalse((out / 'summary.json').exists())
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


if __name__ == '__main__':
    unittest.main()
