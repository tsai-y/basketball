import copy
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from . import continuation as c, gpu_guard as g, train_entry
from . import queue as q


def jobs():
    result = q.ordered_jobs({s: str(s) for s in q.SEEDS})
    metric = {k: .5 for k in q.METRICS}
    for j in result[:2]:
        j.update(status='completed', output='/fixture/' + j['id'],
                 result={'best_epoch': 1, 'best_validation': metric})
    return result


class ContinuationTests(unittest.TestCase):
    def test_only_untouched_suffix_is_eligible(self):
        source = {'jobs': jobs()}
        saved = copy.deepcopy(source)
        self.assertEqual(c.continuation_jobs(source), source['jobs'])
        self.assertEqual(source, saved)
        for key, value in (('status', 'failed'), ('output', '/old'), ('returncode', 0)):
            changed = copy.deepcopy(source)
            changed['jobs'][2][key] = value
            with self.assertRaises(RuntimeError):
                c.continuation_jobs(changed)
        changed = copy.deepcopy(source)
        changed['jobs'][2], changed['jobs'][3] = changed['jobs'][3], changed['jobs'][2]
        with self.assertRaises(RuntimeError):
            c.continuation_jobs(changed)

    def test_unknown_workload_still_blocks_and_free_memory_checked(self):
        policy = {'extra_executables': [{'path': '/usr/bin/papers'}], 'minimum_free_gpu_MiB': 6144}
        with patch.object(g, 'verify_policy'), patch.object(g.shutil, 'disk_usage', return_value=Mock(free=100 * 1024**3)):
            with patch.object(g.subprocess, 'check_output', side_effect=['1\n2\n']), \
                 patch.object(g, 'classify', side_effect=['desktop', '/usr/bin/python']):
                with self.assertRaisesRegex(RuntimeError, 'Other GPU compute'):
                    g.assert_idle_gpu(policy)
            with patch.object(g.subprocess, 'check_output', side_effect=['1\n2\n', '9000\n']), \
                 patch.object(g, 'classify', return_value='desktop'):
                self.assertEqual(len(g.assert_idle_gpu(policy)), 2)
            with patch.object(g.subprocess, 'check_output', side_effect=['1\n', '1000\n']), \
                 patch.object(g, 'classify', return_value='desktop'):
                with self.assertRaisesRegex(RuntimeError, 'free GPU memory'):
                    g.assert_idle_gpu(policy)

    def test_desktop_name_alone_does_not_grant_new_exception(self):
        with patch.object(Path, 'resolve', return_value=Path('/tmp/papers')):
            self.assertEqual(g.classify(12345, {'/usr/bin/papers'}), '/tmp/papers')
        with patch.object(Path, 'resolve', return_value=Path('/usr/bin/papers')):
            self.assertEqual(g.classify(12345, {'/usr/bin/papers'}), 'desktop')

    def test_desktop_package_sha_and_root_permissions_are_required(self):
        real = g.make_policy()
        g.verify_policy(real)
        changed = copy.deepcopy(real)
        changed['extra_executables'][0]['sha256'] = 'wrong'
        with self.assertRaises(RuntimeError):
            g.verify_policy(changed)
        with patch.object(g.subprocess, 'check_output', return_value='wrong-package: /usr/bin/papers'):
            with self.assertRaisesRegex(RuntimeError, 'package owner'):
                g.executable_pin('/usr/bin/papers', 'papers')
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'papers'
            p.write_text('untrusted')
            p.chmod(0o777)
            with self.assertRaisesRegex(RuntimeError, 'root-protected'):
                g.executable_pin(p, 'papers')

    def test_adapter_changes_only_scheduling_binding_and_forwards_cli(self):
        from development.aca_four_arm_v1 import train
        original = train.assert_idle_gpu
        functions = (train.start_training, train.initialized, train.engine.fit)
        argv = ['adapter', '--gpu-policy', '/fixture', '--gpu-policy-sha256', 'fixture',
                '--source-root', '/source', '--start-training', '--variant', 'tsci', '--seed', '20261024']

        def check():
            self.assertEqual((train.start_training, train.initialized, train.engine.fit), functions)
            self.assertEqual(sys.argv[1:], argv[5:])
            train.assert_idle_gpu()

        with patch.object(sys, 'argv', argv), patch.object(train_entry, 'load_policy', return_value={'fixture': True}), \
             patch.object(train_entry, 'assert_idle_gpu') as guard, patch.object(train, 'main', side_effect=check):
            train_entry.main()
            guard.assert_called_once_with({'fixture': True})
            self.assertIs(train.assert_idle_gpu, original)
            self.assertEqual(sys.argv, argv)
        with patch.object(sys, 'argv', argv), patch.object(train_entry, 'load_policy', return_value={}), \
             patch.object(train, 'main', side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):
                train_entry.main()
            self.assertIs(train.assert_idle_gpu, original)

    def test_preserved_file_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prior.json'
            path.write_text('original')
            plan = {'preserved_files': {str(path): c.stream_sha(path)}}
            c.verify_preserved(plan)
            path.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'Preserved prior artifact'):
                c.verify_preserved(plan)

    def test_completed_prefix_is_skipped_then_all_ten_are_aggregated(self):
        self.simulate(False)

    def test_gpu_failure_points_to_pending_job_and_starts_nothing(self):
        self.simulate(True)

    def simulate(self, fail):
        class Child:
            pid = os.getpid()
            returncode = 0
            def poll(self): return 0

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            initial = jobs()
            result = initial[0]['result']
            plan = {'jobs': initial, 'campaign_sha256': 'fixture', 'campaign': {},
                    'source_root': '/fixture', 'python': sys.executable,
                    'previous_output': '/fixture', 'gpu_policy': {}}
            q.atomic(out / 'plan.json', plan)
            q.atomic(out / 'gpu_policy.json', {})
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                with (out / 'lock').open('w') as lock, \
                     patch.object(q, 'verify_pins'), patch.object(c, 'verify_preserved'), \
                     patch.object(c, 'load_policy', return_value={}), \
                     patch.object(c, 'assert_idle_gpu', return_value=[], side_effect=RuntimeError('busy') if fail else None), \
                     patch.object(q, 'validate_completed', return_value=result), \
                     patch.object(q, 'snapshot', side_effect=lambda j: j.update(output=str(out)) or {}), \
                     patch.object(c.subprocess, 'Popen', return_value=Child()) as popen:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'busy'):
                            c.supervise(out, lock.fileno())
                        self.assertEqual(popen.call_count, 0)
                        state = q.read(out / 'state.json')
                        self.assertEqual(state['failure']['active_job'], 'tsci_seed20261024')
                        self.assertEqual(state['failure']['completed_jobs'], 2)
                        self.assertEqual([j['status'] for j in state['jobs'][:3]], ['completed', 'completed', 'failed'])
                    else:
                        c.supervise(out, lock.fileno())
                        self.assertEqual(popen.call_count, 10)
                        commands = [call.args[0] for call in popen.call_args_list]
                        self.assertEqual([(cmd[cmd.index('--seed') + 1], cmd[cmd.index('--variant') + 1]) for cmd in commands],
                                         [(str(j['seed']), j['variant']) for j in initial[2:]])
                        self.assertEqual(q.read(out / 'state.json')['jobs'][:2], initial[:2])
                        self.assertEqual(len(q.read(out / 'summary.json')['rows']), 12)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


if __name__ == '__main__':
    unittest.main()
