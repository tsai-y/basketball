"""Bounded synthetic checks for the new formal adapter; never real model/data training."""
import copy
import hashlib
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from . import loop_core as core
from . import training_runtime as rt
from . import formal_train as formal
from .gpu_readiness import equal

DEVICE = os.environ.get('ACA8_TEST_DEVICE', 'cpu')


def seed():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed_all(42)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.drop = nn.Dropout(.2)

    def forward(self, x):
        if self.training:
            x = x * (1 + .01 * (random.random() + np.random.rand()))
        return self.fc(self.drop(x))


class TinyData(Dataset):
    def __init__(self, split, n):
        self.split, self.n = split, n
        self.x = torch.randn(n, 4, generator=torch.Generator().manual_seed(91))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {'tensor': self.x[i], 'targets': i % 8, 'video_id': f'{self.split}_{i:03d}', 'split': self.split}


def tiny_collate(items):
    item = items[0]
    return {'tensor': item['tensor'][None], 'targets': torch.tensor([item['targets']]),
            'records': [{'video_id': item['video_id'], 'split': item['split']}]}


class FormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if DEVICE == 'cuda':
            assert torch.cuda.is_available()

    def setUp(self):
        seed()
        self.tmp = tempfile.TemporaryDirectory(prefix='baseline8_formal_fixture_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.datasets = {'train': TinyData('train', 24), 'validation': TinyData('validation', 8)}
        self.plan = {'seed': 20261024, 'effective': 16, 'max_epochs': 2, 'patience': 5, 'checkpoint_every': 25,
                     **{split+'_ids': [f'{split}_{i:03d}' for i in range(len(ds))] for split, ds in self.datasets.items()}}

    def loop(self, name, run_id='fixed_fixture_run', workers=0):
        model = Tiny().to(DEVICE)
        loop = rt.TrainingLoop(model, self.plan, None, self.root/name, run_id)
        stream = rt.EpochStream(self.datasets, self.plan, lambda: loop.state, collate=tiny_collate, workers=workers)
        loop.fetch = stream
        self.addCleanup(stream.close)
        return loop

    def test_cuda_general_resume_after_validation_interrupt_and_later_epoch_train(self):
        full = self.loop('full.pt')
        self.assertEqual(full.run(), 'completed')
        expected = full._payload()
        stops = [('validation', lambda phase, state, seen: phase == 'validation' and state['epoch'] == 0 and seen == 3),
                 ('later_train', lambda phase, state, seen: phase == 'train' and state['epoch'] == 1 and state['cursor'] == 16)]
        for name, predicate in stops:
            seed()
            first = self.loop(name+'.pt')
            self.assertEqual(first.run(predicate), 'paused')
            first.fetch.close()
            saved = torch.load(first.checkpoint, weights_only=True)
            self.assertEqual(len(saved['state']['history']), 0 if name == 'validation' else 1)
            restored = self.loop(name+'.pt')
            restored.resume()
            self.assertEqual(restored.run(), 'completed')
            self.assertTrue(equal(expected, restored._payload()), name)

    def test_resume_rejects_other_run_and_inconsistent_history_without_mutation(self):
        first = self.loop('resume.pt')
        first.run(lambda phase, state, seen: phase == 'train' and state['epoch'] == 1)
        first.fetch.close()
        good = torch.load(first.checkpoint, weights_only=True)
        mutations = [lambda p: p.update(run_id='wrong_run'),
                     lambda p: p.update(schema='baseline8_cuda_readiness_only_v1'),
                     lambda p: p['state'].update(updates=3),
                     lambda p: p['state']['history'][0]['validation'].update(count=7),
                     lambda p: p['state'].update(best_epoch=2),
                     lambda p: p['optimizer']['state'][0]['exp_avg_sq'].fill_(-1)]
        for mutate in mutations:
            bad = copy.deepcopy(good)
            mutate(bad)
            core.atomic_save(bad, first.checkpoint)
            restored = self.loop('resume.pt')
            before = restored._payload()
            with self.assertRaises(ValueError):
                restored.resume()
            self.assertTrue(equal(before, restored._payload()))

    def test_two_spawn_worker_orders_across_epochs_and_restarted_validation(self):
        holder = {'epoch': 0, 'cursor': 16}
        stream = rt.EpochStream(self.datasets, self.plan, lambda: holder, collate=tiny_collate, workers=2)
        self.addCleanup(stream.close)
        for epoch, cursor in ((0,16), (1,0)):
            holder.update(epoch=epoch, cursor=cursor)
            order = torch.randperm(24, generator=torch.Generator().manual_seed(20261024+epoch)).tolist()[cursor:]
            for index in order:
                batch = stream('train', index)
                self.assertEqual(batch['records'][0]['video_id'], self.plan['train_ids'][index])
            holder['cursor'] = 24
            for index in range(8):
                self.assertEqual(stream('validation', index)['records'][0]['video_id'], self.plan['validation_ids'][index])
        self.assertIsNone(stream.iterator)

    def test_stop_marker_and_signal_only_request_boundary_stop(self):
        class Reporter:
            def progress(self, *args):
                pass
        path = self.root/'STOP.json'
        stop = formal.StopController(path, Reporter())
        state = {'updates': 1}
        self.assertFalse(stop('train', state, 16))
        rt.atomic_json(path, {'requested': True})
        self.assertTrue(stop('validation', state, 3))
        path.unlink()
        stop.signal(15, None)
        self.assertTrue(stop('train', state, 16))

    def test_checkpoint_remains_authoritative_if_report_publish_fails(self):
        loop = self.loop('last.pt')
        def fail(kind, state):
            if state['epoch'] == 1:
                raise OSError('injected report IO failure after checkpoint commit')
        loop.publish = fail
        with self.assertRaises(OSError):
            loop.run()
        loop.fetch.close()
        saved = torch.load(loop.checkpoint, weights_only=True)
        self.assertEqual(saved['state']['epoch'], 1)
        published = []
        restored = self.loop('last.pt')
        restored.publish = lambda kind,state: published.append(copy.deepcopy(state))
        restored.resume()
        self.assertEqual(published[0], saved['state'])
        restored.run()
        self.assertEqual(restored.state['epoch'], 2)

    def test_same_run_lock_blocks_duplicate_writer(self):
        with formal.run_lock(self.root):
            with self.assertRaises(ValueError):
                with formal.run_lock(self.root):
                    self.fail('duplicate acquired')

    def test_full_metadata_binding_and_test_rejection_without_source_decode(self):
        reference = formal.audit_reference()
        self.assertEqual(len(reference), 8739)
        self.assertEqual(sum(r['split']=='train' for r in reference.values()), 7192)
        self.assertEqual(sum(r['split']=='validation' for r in reference.values()), 1547)
        with self.assertRaises(ValueError):
            rt.VerifiedDataset(object(), 'test', [], {})
        # Wrong approval hash fails before attempting resource/model/data setup.
        with patch.object(formal, 'build_plan', side_effect=AssertionError('must not build')):
            with self.assertRaises(ValueError):
                formal.verify_plan({'schema':formal.PLAN_SCHEMA}, 'wrong')


if __name__ == '__main__':
    unittest.main()
