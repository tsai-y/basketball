import contextlib
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import resnet50

from development.aca_net_v1.readiness import ToyDataset, nested_equal
from development.aca_net_v1.train import Metrics, epoch_order, initial_progress
from . import engine, experiment as exp, preflight, train


class FourArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_rejects_unreviewed_variants_seeds_and_protocol(self):
        for kwargs in ({'variant': 'dw3d'}, {'seed': 1}, {'seed': True}, {'micro_batch': 2},
                       {'lr': .001}, {'weight_decay': .01}, {'epochs': 31}, {'workers': -1}):
            with self.assertRaises(ValueError): exp.ArmConfig(**kwargs)
        self.assertEqual([exp.ArmConfig(variant=v).variant for v in exp.VARIANTS], list(exp.VARIANTS))

    def test_named_rng_streams_distinguish_component_block_and_seed(self):
        values = [exp.component_seed(s, b, c) for s in exp.SEEDS
                  for b in ('layer1.0', 'layer1.1') for c in ('lsta', 'tsci')]
        self.assertEqual(len(set(values)), len(values))

    def test_all_four_initializations_pair_shared_components_and_preserve_RNG(self):
        # Synthetic complete torchvision weights only test transfer and pairing.
        state = resnet50(weights=None).state_dict()
        before_rng = torch.get_rng_state().clone()
        rows = {}
        for v in exp.VARIANTS:
            model, _ = exp.initialized(v, 20261025, state)
            rows[v] = exp.component_digests(model)
            self.assertEqual(sum(p.numel() for p in model.parameters()), exp.PARAMETERS[v])
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    self.assertFalse(m.training)
                    self.assertTrue(all(not p.requires_grad for p in m.parameters()))
            del model
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(len({d['head'] for d in rows.values()}), 1)
        self.assertEqual(len({d['backbone'] for d in rows.values()}), 1)
        self.assertEqual(rows['lsta']['lsta'], rows['aca']['lsta'])
        self.assertEqual(rows['tsci']['tsci'], rows['aca']['tsci'])

    def test_CPU_resume_for_each_real_module_variant_with_dropout(self):
        for v in exp.VARIANTS:
            with self.subTest(variant=v), tempfile.TemporaryDirectory() as directory, \
                 contextlib.redirect_stdout(io.StringIO()):
                result = preflight.synthetic_resume_probe(Path(directory), v)
                self.assertTrue(result['bitwise_equal'])

    def test_tail9_matches_actual_full_batch_mean_reference(self):
        class DoubleData(ToyDataset):
            def __getitem__(self, index):
                item = super().__getitem__(index); item['tensor'] = item['tensor'].double(); return item
        cfg = exp.ArmConfig(epochs=1, patience=1, workers=0)
        data = DoubleData(25)
        model = nn.Linear(3, 2).double(); reference = copy.deepcopy(model)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        ref = torch.optim.Adam(reference.parameters(), lr=cfg.lr)
        order = epoch_order(25, cfg.seed, 0)
        for start in (0, 16):
            items = [data[i] for i in order[start:start+16]]
            ref.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(reference(torch.stack([r['tensor'] for r in items])),
                                        torch.tensor([r['binary_label'] for r in items])).backward()
            nn.utils.clip_grad_norm_(reference.parameters(), cfg.clip_norm); ref.step()
        plan = exp.arm_plan({'train_count': 25, 'validation_count': 5}, cfg, synthetic=True)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            result = engine.fit(model, opt, data, DoubleData(5), cfg, plan, Path(d), initial_progress(), device='cpu')
        self.assertEqual(result['optimizer_steps'], 2)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_checkpoint_cross_variant_and_seed_rejected_before_mutation(self):
        cfg = exp.ArmConfig(workers=0)
        plan = exp.arm_plan({'train_count': 19, 'validation_count': 5}, cfg)
        model = nn.Linear(3, 2); opt = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'last.pt'; engine.save_checkpoint(p, model, opt, initial_progress(), plan)
            before = model.weight.detach().clone()
            for changes in ({'variant': 'lsta'}, {'seed': 20261024}):
                other = {**plan, 'config': {**plan['config'], **changes}}
                with self.assertRaises(ValueError): engine.load_checkpoint(p, model, opt, other)
                self.assertTrue(torch.equal(before, model.weight))

    def test_formal_loader_rejects_synthetic_and_real_engineering_checkpoints(self):
        cfg = exp.ArmConfig(workers=0); campaign = {'train_count': 19, 'validation_count': 5}
        model = nn.Linear(3, 2); opt = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'x.pt'
            engine.save_checkpoint(p, model, opt, initial_progress(), exp.arm_plan(campaign, cfg, synthetic=True))
            with self.assertRaises(ValueError): engine.load_checkpoint(p, model, opt, exp.arm_plan(campaign, cfg))
            torch.save({'schema': 'aca_four_arm_real_engineering_v1', 'engineering_only': True}, p)
            with self.assertRaises(ValueError): engine.load_checkpoint(p, model, opt, exp.arm_plan(campaign, cfg))

    def test_invalid_cursor_rejected(self):
        cfg = exp.ArmConfig(workers=0); plan = exp.arm_plan({'train_count': 19, 'validation_count': 5}, cfg)
        model = nn.Linear(3, 2); opt = torch.optim.Adam(model.parameters())
        progress = initial_progress(); progress['cursor'] = 2
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.pt'; engine.save_checkpoint(p, model, opt, progress, plan)
            with self.assertRaises(ValueError): engine.load_checkpoint(p, model, opt, plan)

    def test_nonfinite_gradient_aborts_before_optimizer_step(self):
        cfg = exp.ArmConfig(workers=0); model = nn.Linear(3, 2)
        model.weight.register_hook(lambda g: g*float('nan'))
        optimizer = torch.optim.Adam(model.parameters())
        batch = {'tensor': torch.ones(1, 3), 'binary_label': torch.tensor([1]),
                 'video_id': ['x'], 'original_class': ['2p1']}
        with patch.object(optimizer, 'step', side_effect=AssertionError('must not update')):
            with self.assertRaisesRegex(ValueError, 'gradient'):
                engine.optimizer_update(model, optimizer, iter([batch]), ['x'], 1, cfg, Metrics(), device='cpu')

    def test_loss_tie_keeps_earlier_best_and_stops_by_patience(self):
        cfg = exp.ArmConfig(epochs=3, patience=1, workers=0)
        plan = exp.arm_plan({'train_count': 19, 'validation_count': 5}, cfg, synthetic=True)
        m=nn.Linear(3,2); o=torch.optim.Adam(m.parameters(),lr=cfg.lr)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(engine, 'validation_pass', return_value={'count':5, 'loss':.7}):
            result = engine.fit(m,o,ToyDataset(19),ToyDataset(5),cfg,plan,Path(d),initial_progress(),device='cpu')
        self.assertEqual(result['epochs'],2); self.assertEqual(result['best_epoch'],1)

    def test_interrupted_validation_does_not_repeat_train_updates(self):
        cfg=exp.ArmConfig(epochs=1,patience=1,workers=0)
        plan=exp.arm_plan({'train_count':25,'validation_count':5},cfg,synthetic=True)
        m=nn.Linear(3,2); o=torch.optim.Adam(m.parameters(),lr=cfg.lr)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            folder=Path(d)
            with patch.object(engine,'validation_pass',return_value=None):
                result=engine.fit(m,o,ToyDataset(25),ToyDataset(5),cfg,plan,folder,initial_progress(),device='cpu')
            self.assertEqual(result['status'],'stopped_validation_will_repeat')
            progress=engine.load_checkpoint(folder/'last.pt',m,o,plan)
            with patch.object(o,'step',side_effect=AssertionError('must not repeat training')):
                engine.fit(m,o,ToyDataset(25),ToyDataset(5),cfg,plan,folder,progress,device='cpu')

    def test_readiness_rejects_other_seed_stale_and_missing_arm(self):
        campaign={'id':'fixed'}; cfg=exp.ArmConfig()
        good={'schema':exp.PREFLIGHT_SCHEMA,'status':'passed','campaign_sha256':engine.canonical_hash(campaign),
              'seed':cfg.seed,'formal_training_started':False,'engineering_only':True,
              'checkpoint_may_initialize_formal_training':False,
              'arms':{'baseline':{'status':'passed','checkpoint_roundtrip_bitwise_equal':True,
                                  'synthetic_resume_bitwise_equal':True}}}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'report.json'; path.write_text(json.dumps(good)); exp.verify_preflight(path,campaign,cfg)
            for bad in ({**good,'seed':20261024},{**good,'campaign_sha256':'stale'},
                        {**good,'status':'cpu_preparation_passed'},{**good,'arms':{}}):
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): exp.verify_preflight(path,campaign,cfg)

    def test_default_CLI_does_not_train_prepare_or_decode(self):
        with patch.object(sys,'argv',['fourarm','--source-root','/unused']), \
             patch.object(train,'campaign_plan',return_value={'formal_training_started':False}), \
             patch.object(train,'start_training',side_effect=AssertionError('no launch')), \
             patch.object(train,'prepare',side_effect=AssertionError('no preparation')), \
             contextlib.redirect_stdout(io.StringIO()):
            train.main()

    def test_training_acknowledgement_checked_before_GPU_query(self):
        with patch.object(train,'assert_idle_gpu',side_effect=AssertionError('no GPU query')):
            with self.assertRaises(ValueError): train.start_training('/unused',exp.ArmConfig(),'/unused')

    def test_environment_versions_are_pinned_without_CUDA_context(self):
        with patch.object(exp.subprocess, 'run', return_value=SimpleNamespace(stdout='GPU, uuid, driver, 16303 MiB\n')), \
             patch.object(torch.cuda, 'init', side_effect=AssertionError('no CUDA init')):
            value=exp.environment()
        for key in ('python','torch','torchvision','opencv','numpy','cudnn','cuda_build','GPU'):
            self.assertIn(key,value)

    def test_other_compute_process_blocks_preflight(self):
        with patch.object(exp.subprocess,'run',return_value=SimpleNamespace(stdout='999999\n')), \
             patch.object(Path,'resolve',return_value=Path('/usr/bin/python')):
            with self.assertRaisesRegex(ValueError,'Other GPU compute'): exp.assert_idle_gpu()


if __name__ == '__main__': unittest.main()
