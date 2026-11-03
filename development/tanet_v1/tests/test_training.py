from contextlib import redirect_stderr, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from development.tanet_v1 import train
from development.tanet_v1.training import (atomic_save, checkpoint, collate, configure_runtime,
    eval_batch, load_checkpoint, metrics, recipe, state_digest, stopping, train_group, training_mode)


class TrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.cfg = recipe()
        torch.manual_seed(7)
        self.model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
        self.x = torch.randn(25, 4)
        self.y = torch.arange(25) % 2

    def batch(self, x, y):
        return {'tensor': x, 'binary_label': y, 'video_id': [str(i) for i in range(len(y))]}

    def test_accumulated_updates_and_nine_video_tail_match_full_group_means(self):
        reference = copy.deepcopy(self.model)
        actual_optim = torch.optim.SGD(self.model.parameters(), lr=.01)
        ref_optim = torch.optim.SGD(reference.parameters(), lr=.01)
        for start, end in [(0, 16), (16, 25)]:
            batches = [self.batch(self.x[i:i+1], self.y[i:i+1]) for i in range(start, end)]
            train_group(self.model, actual_optim, batches, device=torch.device('cpu'),
                        group_videos=end-start, cfg=self.cfg)
            ref_optim.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(reference(self.x[start:end]), self.y[start:end])
            loss.backward()
            nn.utils.clip_grad_norm_(reference.parameters(), self.cfg['max_grad_norm'])
            ref_optim.step()
        for a, b in zip(self.model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)

    def test_checkpoint_adam_dropout_rng_resume_is_exact(self):
        configure_runtime(self.cfg, cuda=False)
        model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(.5), nn.Linear(8, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        batches = [self.batch(self.x[i:i+1], self.y[i:i+1]) for i in range(2)]
        def update(m, o):
            return train_group(m, o, batches, device=torch.device('cpu'), group_videos=2, cfg=self.cfg)
        update(model, optimizer)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'resume.pt'
            payload = checkpoint(model, optimizer, contract_sha256='same_contract', kind='formal',
                epoch=1, global_step=1, stop_state={}, history=[])
            atomic_save(path, payload)
            expected = update(model, optimizer)
            expected_model = state_digest(model.state_dict())
            expected_optimizer = state_digest(optimizer.state_dict())
            other = nn.Sequential(nn.Linear(4, 8), nn.Dropout(.5), nn.Linear(8, 2))
            other_optim = torch.optim.Adam(other.parameters(), lr=.001)
            load_checkpoint(path, other, other_optim, contract_sha256='same_contract', kind='formal')
            resumed = update(other, other_optim)
            self.assertEqual(expected['CE'], resumed['CE'])
            self.assertEqual(expected['last_logits'], resumed['last_logits'])
            self.assertEqual(expected_model, state_digest(other.state_dict()))
            self.assertEqual(expected_optimizer, state_digest(other_optim.state_dict()))

    def test_engineering_or_changed_contract_checkpoint_rejected_before_loading(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'engineering.pt'
            atomic_save(path, checkpoint(self.model, optimizer, contract_sha256='old', kind='engineering_only',
                epoch=0, global_step=1, stop_state={}, history=[]))
            with self.assertRaisesRegex(ValueError, 'purpose mismatch'):
                load_checkpoint(path, self.model, optimizer, contract_sha256='old', kind='formal')
            with self.assertRaisesRegex(ValueError, 'recipe mismatch'):
                load_checkpoint(path, self.model, optimizer, contract_sha256='new', kind='engineering_only')

    def test_bn_stats_policy_and_validation_state(self):
        model = nn.Sequential(nn.BatchNorm2d(3), nn.Flatten(), nn.Linear(12, 2))
        training_mode(model)
        self.assertTrue(model.training)
        self.assertFalse(model[0].training)
        self.assertTrue(model[0].weight.requires_grad)
        tam_bn = nn.BatchNorm1d(3)
        wrapper = nn.ModuleList([model, tam_bn])
        training_mode(wrapper)
        self.assertTrue(tam_bn.training)
        before = state_digest(model.state_dict())
        eval_batch(model, self.batch(torch.randn(2, 3, 2, 2), torch.tensor([0, 1])), torch.device('cpu'))
        self.assertEqual(before, state_digest(model.state_dict()))

    def test_macro_f1_is_not_only_make_f1(self):
        result = metrics(torch.tensor([[3., 0.], [2., 0.], [0., 2.], [0., 3.]]), torch.tensor([0, 1, 1, 1]))
        self.assertEqual(result['confusion_matrix_Miss_Make'], [[1, 0], [1, 2]])
        self.assertAlmostEqual(result['macro_f1'], (2/3+4/5)/2)
        self.assertNotEqual(result['macro_f1'], result['per_class_f1'][1])

    def test_real_minimum_selects_best_but_small_gain_does_not_reset_patience(self):
        state = {'best_ce': .5, 'early_anchor': .5, 'wait': 2}
        state, selected, stopped = stopping(.49995, state, self.cfg)
        self.assertTrue(selected)
        self.assertEqual(state['wait'], 3)
        self.assertFalse(stopped)
        state, _, _ = stopping(.49, state, self.cfg)
        self.assertEqual(state['wait'], 0)
        for _ in range(5):
            state, _, stopped = stopping(.6, state, self.cfg)
        self.assertTrue(stopped)
        with self.assertRaises(ValueError):
            stopping(float('nan'), state, self.cfg)

    def test_nonfinite_input_does_not_update_parameters(self):
        before = state_digest(self.model.state_dict())
        optimizer = torch.optim.Adam(self.model.parameters())
        with self.assertRaisesRegex(ValueError, 'Invalid normalized'):
            train_group(self.model, optimizer, [self.batch(self.x[:1]*float('nan'), self.y[:1])],
                        device=torch.device('cpu'), group_videos=1, cfg=self.cfg)
        self.assertEqual(before, state_digest(self.model.state_dict()))

    def test_metadata_dry_run_never_decodes_creates_optimizer_or_output(self):
        with patch('development.tanet_v1.data.decode', side_effect=AssertionError('Decoded video')), \
             patch.object(train, 'run_training', side_effect=AssertionError('Started formal training')), \
             patch.object(torch.optim, 'Adam', side_effect=AssertionError('Created optimizer')), \
             patch.object(tempfile, 'mkdtemp', side_effect=AssertionError('Created output')), \
             redirect_stdout(io.StringIO()):
            result = train.main(['--dry-run'])
        self.assertEqual(result['video_opens'], 0)
        self.assertFalse(result['output_files_written'])
        self.assertFalse(result['formal_training_started'])
        self.assertEqual(result['plan']['optimizer_steps_per_epoch'], 450)
        self.assertEqual(result['plan']['last_optimizer_group_videos'], 9)

    def test_explicit_modes_required_before_metadata(self):
        for argv in [[], ['--start'], ['--dry-run', '--resume', 'last.pt']]:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), \
                 patch.object(train, 'contract', side_effect=AssertionError('Read metadata')):
                with self.assertRaises(SystemExit):
                    train.main(argv)

    def test_formal_entry_and_epoch_boundary_resume_match_continuous_toy_run(self):
        from types import SimpleNamespace
        from torch.utils.data import Dataset
        class ToyVideos(Dataset):
            def __init__(self, *, split, config):
                self.count = 25 if split == 'train' else 5
                self.scope = SimpleNamespace(rows=[{'video_id': f'{split}_{i}'} for i in range(self.count)])
            def __len__(self):
                return self.count
            def __getitem__(self, i):
                return {'tensor': torch.tensor([i/25., .1, .2, .3]), 'binary_label': i % 2,
                        'video_id': self.scope.rows[i]['video_id']}
        cfg = {**self.cfg, 'max_epochs': 2}
        plan = {'recipe': cfg, 'contract_sha256': 'toy_epoch_resume'}
        args = SimpleNamespace(readiness_report='unused_mock_report', resume=None)
        def new_model(cfg, device):
            return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(.5), nn.Linear(8, 2))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            (root/'artifacts').mkdir()
            with patch.object(train, 'FOLDER', root), \
                 patch.object(train, 'readiness_gate', return_value={'environment': {'gpu': 'toy'}}), \
                 patch.object(train, 'configure_runtime', side_effect=lambda c: configure_runtime(c, cuda=False)), \
                 patch.object(train.torch.cuda, 'get_device_name', return_value='toy'), \
                 patch.object(train, 'initialized_model', side_effect=new_model), \
                 patch.object(train, 'BasketballTANetDataset', ToyVideos):
                continuous = train.run_training(args, plan)
                original_save = train.atomic_save
                def interrupt_after_epoch1(path, payload, **kwargs):
                    original_save(path, payload, **kwargs)
                    if Path(path).name == 'last.pt' and payload['epoch'] == 1:
                        raise InterruptedError('simulated epoch-boundary interruption')
                with patch.object(train, 'atomic_save', side_effect=interrupt_after_epoch1):
                    with self.assertRaises(InterruptedError):
                        train.run_training(args, plan)
                interrupted = next(p for p in (root/'artifacts').iterdir() if p != continuous)
                self.assertFalse((interrupted/'active.lock').exists())
                resumed = train.run_training(SimpleNamespace(readiness_report='unused_mock_report',
                    resume=interrupted/'last.pt'), plan)
            a = torch.load(continuous/'last.pt', weights_only=True)
            b = torch.load(resumed/'last.pt', weights_only=True)
            self.assertEqual(a['epoch'], 2)
            self.assertEqual(a['global_step'], 4)
            self.assertEqual(state_digest(a['model']), state_digest(b['model']))
            self.assertEqual(state_digest(a['optimizer']), state_digest(b['optimizer']))
            self.assertEqual(a['validation']['metrics'], b['validation']['metrics'])
            self.assertEqual(a['validation']['metrics']['videos'], 5)


if __name__ == '__main__':
    unittest.main()
