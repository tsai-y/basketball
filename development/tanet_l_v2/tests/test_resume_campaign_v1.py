import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from development.tanet_l_v2 import resume_campaign_v1 as resume
from development.tanet_l_v2.training import recipe


class ExplicitResumeContracts(unittest.TestCase):
    def checkpoint_pair(self):
        history = [{'epoch': 3, 'global_step': 1350, 'validation': {'CE': .5}},
                   {'epoch': 4, 'global_step': 1800, 'validation': {'CE': .6}}]
        runtime = {'torch': str(torch.__version__), 'cuda': torch.version.cuda}
        common = {'kind': 'formal', 'contract_sha256': 'pinned', 'runtime': runtime}
        last = {**common, 'epoch': 4, 'global_step': 1800, 'history': history,
                'stop_state': {'best_ce': .5, 'wait': 1}}
        best = {**common, 'epoch': 3, 'stop_state': {'best_ce': .5}}
        return last, best, {'recipe': recipe(), 'contract_sha256': 'pinned'}, history

    def test_epoch_boundary_and_existing_best_are_required(self):
        args = self.checkpoint_pair()
        resume.validate_checkpoint(*args)
        for target, key, value in ((0, 'kind', 'engineering_only'), (0, 'contract_sha256', 'other'),
                                   (0, 'global_step', 2207), (1, 'epoch', 4)):
            wrong = copy.deepcopy(args)
            wrong[target][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                resume.validate_checkpoint(*wrong)

    def test_completed_early_stopped_or_changed_runtime_recipes_are_refused(self):
        for change in ('completed', 'stopped', 'runtime', 'extended', 'test'):
            last, best, contract, history = copy.deepcopy(self.checkpoint_pair())
            if change == 'completed':
                last['epoch'] = history[-1]['epoch'] = 15
            elif change == 'stopped':
                last['stop_state']['wait'] = 5
            elif change == 'runtime':
                last['runtime']['torch'] = 'changed'
            elif change == 'extended':
                contract['recipe']['max_epochs'] = 20
            else:
                contract['recipe']['test_access'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                resume.validate_checkpoint(last, best, contract, history)

    def test_reboot_and_inactive_original_identity_required(self):
        old = {'pid': 123, 'boot_id': 'before', 'process_start_ticks': 10, 'cmdline': 'old'}
        with patch.object(resume.campaign.old, 'process_identity', side_effect=FileNotFoundError):
            resume.verify_inactive(old, 'after')
            with self.assertRaisesRegex(ValueError, 'documented host reboot'):
                resume.verify_inactive(old, 'before')
        reused = {**old, 'boot_id': 'after', 'process_start_ticks': 20, 'cmdline': 'unrelated'}
        with patch.object(resume.campaign.old, 'process_identity', return_value=reused):
            resume.verify_inactive(old, 'after')
        with patch.object(resume.campaign.old, 'process_identity', return_value=old):
            with self.assertRaisesRegex(ValueError, 'still alive'):
                resume.verify_inactive(old, 'after')

    def test_failed_child_is_recorded_once_without_retry_and_logs_are_unique(self):
        class Child:
            pid = 123
            def wait(self):
                return 9
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = root/'resume_example'
            session.mkdir()
            state = {'jobs': []}
            item = {'seed': 20261026, 'package': str(root)}
            with patch.object(resume.subprocess, 'Popen', return_value=Child()) as popen, \
                 patch.object(resume.campaign.old, 'process_identity', return_value={'pid': 123}):
                with self.assertRaisesRegex(ValueError, 'failed'):
                    resume.run_child(root, session, state, item, 'training', ['python', '--resume', 'last.pt'])
                popen.assert_called_once()
            saved = json.loads((root/'state.json').read_text())
            self.assertEqual(saved['jobs'][0]['returncode'], 9)
            self.assertEqual(saved['jobs'][0]['stage'], 'training')
            self.assertIn('resume_example', saved['jobs'][0]['stdout_log'])
            self.assertTrue(Path(saved['jobs'][0]['stdout_log']).is_file())

    def test_completed_seed_hash_guard_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'completed.pt'
            path.write_bytes(b'completed seed')
            hashes = {str(path): resume.sha256(path)}
            resume.verify_completed(hashes)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Completed seed artifact changed'):
                resume.verify_completed(hashes)


if __name__ == '__main__':
    unittest.main()
