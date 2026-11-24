"""Bounded CPU fixtures: no video decoder, pretrained network, GPU or real training."""
import copy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import loop_core as core


class Tiny(nn.Module):
    def __init__(self, stochastic=False, tied_validation=False):
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.stochastic, self.tied_validation = stochastic, tied_validation

    def forward(self, x):
        if self.training and self.stochastic:
            x = F.dropout(x, p=.2, training=True) * (1 + .01 * (random.random() + np.random.rand()))
        y = self.fc(x)
        return y * 0 if not self.training and self.tied_validation else y


class Fixture:
    def __init__(self, count=24):
        g = torch.Generator().manual_seed(123)
        self.x = {'train': torch.randn(count, 4, generator=g),
                  'validation': torch.randn(10, 4, generator=g)}
        self.y = {split: torch.arange(len(x)) % 8 for split, x in self.x.items()}
        self.ids = {split: [f'{split}_{i:03d}' for i in range(len(x))] for split, x in self.x.items()}
        self.calls = []

    def fetch(self, split, index):
        self.calls.append((split, index))
        return {'tensor': self.x[split][index:index+1], 'targets': self.y[split][index:index+1],
                'records': [{'split': split, 'video_id': self.ids[split][index]}]}

    def identity(self, model):
        h = hashlib.sha256(Path(__file__).read_bytes())
        for split in ('train', 'validation'):
            h.update(self.x[split].numpy().tobytes())
            h.update(self.y[split].numpy().tobytes())
        h.update(repr((model.stochastic, model.tied_validation)).encode())
        return h.hexdigest()


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


class LoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setUp(self):
        seed()
        self.temp = tempfile.TemporaryDirectory(prefix='aca8_loop_fixture_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = Fixture()

    def loop(self, name='last.pt', epochs=2, stochastic=False, tied=False):
        model = Tiny(stochastic, tied)
        plan = core.fixture_plan(self.data.ids['train'], self.data.ids['validation'],
                                 model_identity=self.data.identity(model), max_epochs=epochs)
        return core.FixtureLoop(model, plan, self.data.fetch, self.root / name)

    def test_micro_accumulation_tail8_matches_independent_physical_batches(self):
        loop = self.loop(epochs=1)
        reference = copy.deepcopy(loop.model)
        optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0., amsgrad=False, foreach=False, fused=False)
        order = torch.randperm(24, generator=torch.Generator().manual_seed(20261024))
        for indices in (order[:16], order[16:]):
            optimizer.zero_grad(set_to_none=True)
            # Independent mean CE over physical batch, including the eight-item tail.
            F.cross_entropy(reference(self.data.x['train'][indices]), self.data.y['train'][indices]).backward()
            nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        self.assertEqual(loop.run(), 'completed')
        for a, b in zip(loop.model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=3e-8)
        self.assertEqual(loop.state['updates'], 2)
        self.assertEqual(loop.state['history'][0]['train']['count'], 24)
        self.assertEqual(loop.state['history'][0]['validation']['count'], 10)
        self.assertEqual([i for s, i in self.data.calls if s == 'train'], order.tolist())
        self.assertEqual([i for s, i in self.data.calls if s == 'validation'], list(range(10)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_train_resume_exact_model_adam_metrics_history_and_all_rng(self):
        full = self.loop('full.pt', stochastic=True)
        full.run()
        expected = full._payload()
        seed()
        partial = self.loop('resume.pt', stochastic=True)
        self.assertEqual(partial.run(lambda phase, state, seen: phase == 'train' and state['updates'] == 1), 'paused')
        self.assertEqual(partial.state['cursor'], 16)
        restored = self.loop('resume.pt', stochastic=True)
        restored.resume()
        restored.run()
        actual = restored._payload()
        self.assertTrue(tree_equal(actual, expected))

    def test_partial_validation_restarts_full_without_selection_or_extra_updates(self):
        full = self.loop('full.pt', stochastic=True)
        full.run()
        expected = full._payload()
        seed()
        partial = self.loop('resume.pt', stochastic=True)
        self.assertEqual(partial.run(lambda phase, state, seen: phase == 'validation' and seen == 3), 'paused')
        saved = torch.load(partial.checkpoint, weights_only=True)
        self.assertEqual(saved['state']['history'], [])
        self.assertIsNone(saved['best_model'])
        self.assertEqual(saved['state']['cursor'], 24)
        self.data.calls.clear()
        restored = self.loop('resume.pt', stochastic=True)
        restored.resume()
        restored.run()
        self.assertEqual(self.data.calls[:10], [('validation', i) for i in range(10)])
        self.assertTrue(tree_equal(restored._payload(), expected))

    def test_ties_keep_first_best_and_stop_at_patience5(self):
        loop = self.loop(epochs=30, tied=True)
        self.assertEqual(loop.run(), 'completed')
        self.assertEqual((loop.state['epoch'], loop.state['best_epoch'], loop.state['bad_epochs']), (6, 1, 5))
        self.assertEqual(loop.state['updates'], 12)
        saved = torch.load(loop.checkpoint, weights_only=True)
        self.assertTrue(tree_equal(saved['best_model'], loop.best_model))
        self.assertFalse(tree_equal(loop.model.state_dict(), loop.best_model))
        restored = self.loop(epochs=30, tied=True)
        restored.resume()
        self.assertEqual(restored.run(), 'completed')
        self.assertEqual(restored.state['updates'], 12)

    def test_metrics_reject_corruption_and_return_independent_copies(self):
        metric = core.EightClassMetrics()
        metric.add(torch.zeros(2, 8), torch.tensor([0, 7]))
        state = core.metrics_state(metric)
        restored = core.restore_metrics(state)
        self.assertEqual(metric.result(), restored.result())
        state['matrix'][0][0] = -1
        self.assertEqual(metric._matrix[0][0], 1)
        with self.assertRaises(ValueError):
            core.restore_metrics(state)
        state = core.metrics_state(metric)
        state['loss_sum'] = float('nan')
        with self.assertRaises(ValueError):
            core.restore_metrics(state)

    def test_resume_rejects_identity_boundary_metrics_and_optimizer_before_mutation(self):
        loop = self.loop()
        loop.run(lambda phase, state, seen: phase == 'train' and state['updates'] == 1)
        good = torch.load(loop.checkpoint, weights_only=True)
        mutations = [lambda p: p.update(schema='binary_v1'),
                     lambda p: p.update(plan_sha256='different'),
                     lambda p: p['state'].update(cursor=17),
                     lambda p: p['state']['train_metrics'].update(count=15),
                     lambda p: p['optimizer']['param_groups'][0].update(lr=.01),
                     lambda p: p['model']['fc.weight'].fill_(float('nan')),
                     lambda p: p['model'].update({'fc.weight': p['model']['fc.weight'].double()})]
        for change in mutations:
            bad = copy.deepcopy(good)
            change(bad)
            core.atomic_save(bad, loop.checkpoint)
            restored = self.loop()
            before = restored._payload()
            with self.assertRaises(ValueError):
                restored.resume()
            self.assertTrue(tree_equal(before, restored._payload()))

    def test_nonfinite_or_wrong_id_preserves_last_commit_and_blocks_reentry(self):
        for malformed in ('nonfinite', 'wrong_id'):
            with self.subTest(malformed=malformed):
                loop = self.loop(malformed + '.pt')
                original = loop.fetch
                calls = 0
                def bad_fetch(split, index):
                    nonlocal calls
                    calls += 1
                    batch = original(split, index)
                    if calls == 17:
                        if malformed == 'nonfinite':
                            batch['tensor'] = torch.full((1, 4), float('nan'))
                        else:
                            batch['records'][0]['video_id'] = 'wrong'
                    return batch
                loop.fetch = bad_fetch
                with self.assertRaises(ValueError):
                    loop.run()
                saved = torch.load(loop.checkpoint, weights_only=True)
                self.assertEqual(saved['state']['updates'], 0)  # cadence25: replay unsaved work
                self.assertEqual(saved['state']['history'], [])
                with self.assertRaises(ValueError):
                    loop.run()

    def test_atomic_replace_and_validation_commit_failure_preserve_consistent_old_state(self):
        path = self.root / 'atomic.pt'
        core.atomic_save({'generation': 1}, path)
        with patch.object(core.os, 'replace', side_effect=OSError('injected before replace')):
            with self.assertRaises(OSError):
                core.atomic_save({'generation': 2}, path)
        self.assertEqual(torch.load(path, weights_only=True), {'generation': 1})
        self.assertEqual(list(self.root.glob('.atomic.pt.*')), [])
        loop = self.loop(epochs=1)
        original = core.atomic_save
        def fail_validation(payload, destination):
            if payload['state']['epoch'] == 1:
                raise OSError('injected before validation commit')
            original(payload, destination)
        with patch.object(core, 'atomic_save', side_effect=fail_validation):
            with self.assertRaises(OSError):
                loop.run()
        old = torch.load(loop.checkpoint, weights_only=True)
        self.assertEqual(old['state']['history'], [])
        self.assertIsNone(old['best_model'])
        restored = self.loop(epochs=1)
        restored.resume()
        restored.run()
        new = torch.load(loop.checkpoint, weights_only=True)
        self.assertEqual(new['state']['best_epoch'], 1)
        self.assertEqual(new['state']['best_ce'], new['state']['history'][0]['validation']['ce'])
        self.assertTrue(tree_equal(new['model'], new['best_model']))


if __name__ == '__main__':
    unittest.main()
