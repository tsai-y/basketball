import ast
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from development.tanet_v1 import training as original
from development.tanet_v1.probe_bn_behavior_v1 import bn_condition
from development.tanet_l_v2 import campaign, train, training
from development.tanet_l_v2.config import FOLDER
from development.tanet_l_v2.eval_policy import POLICY, assert_evaluation_mode, per_video_L_stats


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = nn.BatchNorm2d(2)
        self.block = nn.Module()
        self.block.tam = nn.Module()
        self.block.tam.G = nn.Sequential(nn.Identity(), nn.BatchNorm1d(2), nn.ReLU())
        self.block.tam.L = nn.Sequential(nn.Identity(), nn.BatchNorm1d(2), nn.ReLU())
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.pretrained(x.unsqueeze(-1)).squeeze(-1)
        return self.dropout(self.block.tam.L(self.block.tam.G(x))).mean(2)


def batch(x, y=0):
    return {'tensor': x, 'binary_label': torch.tensor([y] * len(x)),
            'video_id': [f'toy{i}' for i in range(len(x))]}


class LPolicyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.model = Toy()
        self.x = torch.tensor([[[-2., 1., 4., 7.], [3., 4., 8., 11.]]])
        self.cpu = torch.device('cpu')

    def test_exact_diagnostic_and_functional_equivalence_no_state_rng_or_gradient_update(self):
        model = self.model.eval()
        before = training.state_digest(model.state_dict())
        rng = training.state_digest(training.capture_rng())
        with torch.no_grad():
            z = model.pretrained(self.x.unsqueeze(-1)).squeeze(-1)
            bn = model.block.tam.G[1]
            z = F.batch_norm(z, bn.running_mean, bn.running_var, bn.weight, bn.bias,
                             training=False, eps=bn.eps).relu()
            bn = model.block.tam.L[1]
            expected = F.batch_norm(z, None, None, bn.weight, bn.bias, training=True, eps=bn.eps).relu().mean(2)
            with bn_condition(model, 'L'):
                diagnostic = model(self.x)
        actual, y = training.eval_batch(model, batch(self.x), self.cpu)
        self.assertTrue(torch.equal(actual, diagnostic))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(y.tolist(), [0])
        self.assertEqual(before, training.state_digest(model.state_dict()))
        self.assertEqual(rng, training.state_digest(training.capture_rng()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(not m.training for m in model.modules()))
        self.assertTrue(model.block.tam.L[1].track_running_stats)
        with torch.no_grad():
            self.assertFalse(torch.equal(actual, model(self.x)))

    def test_exception_restores_flags_and_multiple_videos_rejected_before_forward(self):
        self.model.eval()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with per_video_L_stats(self.model):
                assert_evaluation_mode(self.model)
                raise RuntimeError('intentional')
        self.assertTrue(all(not m.training for m in self.model.modules()))
        self.assertTrue(self.model.block.tam.L[1].track_running_stats)
        with patch.object(self.model, 'forward', side_effect=AssertionError('No forward allowed')):
            with self.assertRaisesRegex(ValueError, 'exactly one video'):
                training.eval_batch(self.model, batch(self.x.repeat(2, 1, 1)), self.cpu)
        with self.assertRaisesRegex(ValueError, 'Missing TAM'):
            with per_video_L_stats(nn.Linear(2, 2).eval()):
                pass

    def test_original_training_adam_rng_and_bn_updates_are_bitwise_identical(self):
        reference = copy.deepcopy(self.model)
        a = training.optimizer_for(self.model, training.recipe())
        b = original.optimizer_for(reference, original.recipe())
        for step in range(2):
            samples = [batch(self.x, 0), batch(self.x * .7 + .3, 1)]
            rng = training.capture_rng()
            actual = training.train_group(self.model, a, samples, device=self.cpu,
                                           group_videos=2, cfg=training.recipe())
            training.restore_rng(rng)
            expected = original.train_group(reference, b, samples, device=self.cpu,
                                             group_videos=2, cfg=original.recipe())
            self.assertEqual(actual['CE'], expected['CE'])
            self.assertEqual(actual['last_logits'], expected['last_logits'])
            self.assertEqual(training.state_digest(self.model.state_dict()), training.state_digest(reference.state_dict()))
            self.assertEqual(training.state_digest(a.state_dict()), training.state_digest(b.state_dict()))
            # Intervening new-policy validation must not change the next update.
            training.eval_batch(self.model, batch(self.x), self.cpu)
        self.assertEqual(int(self.model.pretrained.num_batches_tracked), 0)
        for branch in ('G', 'L'):
            self.assertEqual(int(getattr(self.model.block.tam, branch)[1].num_batches_tracked), 4)

    def test_source_and_recipe_changes_are_limited_to_evaluation_contract(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        before = functions(campaign.old.FOLDER/'training.py')
        after = functions(FOLDER/'training.py')
        self.assertEqual(set(before), set(after))
        self.assertEqual({n for n in before if before[n] != after[n]}, {'recipe', 'contract', 'eval_batch'})
        for name in campaign.old.FILES:
            if name not in ('training.py', 'readiness.py', 'train_config.json'):
                self.assertEqual((FOLDER/name).read_bytes(), (campaign.old.FOLDER/name).read_bytes(), name)
        cfg = original.recipe()
        self.assertEqual(training.recipe(), {**cfg, 'schema': 'tanet_uniform50_L_eval_recipe_v2', 'evaluation_policy': POLICY})
        self.assertEqual((FOLDER/'train.py').read_bytes(), (campaign.old.FOLDER/'train.py').read_bytes())

    def test_clones_differ_only_by_seed_and_refuse_overwrite_or_recipe_change(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)/'clone'
            campaign.clone(destination, 20261024)
            cfg = json.loads((destination/'train_config.json').read_text())
            self.assertEqual(cfg['seed'], 20261024)
            self.assertEqual(cfg['evaluation_policy'], POLICY)
            with self.assertRaisesRegex(ValueError, 'overwritten'):
                campaign.clone(destination, 20261024)
            cfg['max_epochs'] = 20
            (destination/'train_config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'beyond seed'):
                campaign.check_package(destination, 20261024)
        for seed in (0, True, '20261024'):
            with self.assertRaisesRegex(ValueError, 'declared seeds'):
                campaign.expected_json('config.json', seed)

    def test_summary_requires_the_pinned_L_contract_for_every_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queue, runs = [], {}
            for seed in campaign.SEEDS:
                run = root/str(seed)
                run.mkdir()
                runs[str(seed)] = str(run)
                queue.append({'seed': seed, 'contract_sha256': str(seed)})
                (run/'contract.json').write_text(json.dumps({'contract_sha256': str(seed),
                    'recipe': {'evaluation_policy': POLICY}}))
            historical = root/'historical.json'
            historical.write_text('{"historical_only": true}')
            plan = {'queue': queue, 'historical_TANet_summary': str(historical),
                    'limitations': ['Exploratory validation comparison']}
            def existing_aggregator(output, plan, runs):
                (output/'index.html').write_text('TANet三seed<h2>三seed描述統計</h2>')
                return {'limitations': ['No test'], 'seeds': list(campaign.SEEDS)}
            with patch.object(campaign.old, 'summarize', side_effect=existing_aggregator) as aggregator:
                result = campaign.summarize(root, plan, runs)
                self.assertEqual(result['evaluation_policy'], POLICY)
                self.assertTrue(result['same_validation_used_to_choose_policy'])
                self.assertTrue(result['historical_TANet']['historical_only'])
                self.assertIn('探索性比較', (root/'index.html').read_text())
                aggregator.assert_called_once()
            wrong = Path(runs['20261026'])/'contract.json'
            wrong.write_text(json.dumps({'contract_sha256': '20261026', 'recipe': {'evaluation_policy': 'saved'}}))
            with patch.object(campaign.old, 'summarize') as aggregator:
                with self.assertRaisesRegex(ValueError, 'policy/contract'):
                    campaign.summarize(root, plan, runs)
                aggregator.assert_not_called()

    def test_epoch_selection_and_interrupted_resume_use_same_L_rule(self):
        class Videos(Dataset):
            def __init__(self, *, split, config):
                self.count = 5 if split == 'train' else 3
                self.scope = SimpleNamespace(rows=[{'video_id': f'{split}{i}'} for i in range(self.count)])
            def __len__(self):
                return self.count
            def __getitem__(self, i):
                return {'tensor': torch.tensor([[-2., i + 1., 4., 7.], [3., 4., i + 8., 11.]]),
                        'binary_label': i % 2, 'video_id': self.scope.rows[i]['video_id']}
        cfg = {**training.recipe(), 'max_epochs': 2}
        plan = {'recipe': cfg, 'contract_sha256': 'toy_L_policy'}
        args = SimpleNamespace(readiness_report='mock', resume=None)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            (root/'artifacts').mkdir()
            with patch.object(train, 'FOLDER', root), \
                 patch.object(train, 'readiness_gate', return_value={'environment': {'gpu': 'toy'}}), \
                 patch.object(train, 'configure_runtime', side_effect=lambda c: training.configure_runtime(c, cuda=False)), \
                 patch.object(torch.cuda, 'get_device_name', return_value='toy'), \
                 patch.object(train, 'initialized_model', side_effect=lambda c, d: Toy()), \
                 patch.object(train, 'BasketballTANetDataset', Videos):
                continuous = train.run_training(args, plan)
                save = train.atomic_save
                def interrupt(path, payload, **kwargs):
                    save(path, payload, **kwargs)
                    if Path(path).name == 'last.pt' and payload['epoch'] == 1:
                        raise InterruptedError('boundary')
                with patch.object(train, 'atomic_save', side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        train.run_training(args, plan)
                interrupted = next(p for p in (root/'artifacts').iterdir() if p != continuous)
                resumed = train.run_training(SimpleNamespace(readiness_report='mock', resume=interrupted/'last.pt'), plan)
            a = torch.load(continuous/'last.pt', weights_only=True)
            b = torch.load(resumed/'last.pt', weights_only=True)
            for key in ('model', 'optimizer', 'rng', 'validation'):
                self.assertEqual(training.state_digest(a[key]), training.state_digest(b[key]), key)
            best = torch.load(continuous/'best.pt', weights_only=True)
            self.assertEqual(best['epoch'], min(a['history'], key=lambda h: h['validation']['CE'])['epoch'])
            model = Toy().eval()
            model.load_state_dict(best['model'])
            ds = Videos(split='validation', config=None)
            logits = torch.cat([training.eval_batch(model, training.collate([ds[i]]), self.cpu)[0] for i in range(3)])
            self.assertTrue(torch.equal(logits, best['validation']['logits']))

    def test_campaign_runs_all_three_in_order_and_stops_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                output = root/'artifacts/three_seed_toy'
                output.mkdir(parents=True)
                queue = [{'seed': s, 'contract_sha256': str(s)} for s in campaign.SEEDS]
                plan = {'seeds': list(campaign.SEEDS), 'queue': queue, 'evaluation_policy': POLICY,
                        'max_epochs': 15, 'formal_training_explicitly_authorized': True,
                        'test_access': False, 'fresh_ImageNet_initialization_each_seed': True}
                training.atomic_save(output/'plan.json', plan, json_format=True)
                training.atomic_save(output/'state.json', {'status': 'prepared', 'jobs': [],
                    'plan_sha256': campaign.sha256(output/'plan.json')}, json_format=True)
                called = []
                def child(out, state, item, stage, command):
                    called.append((item['seed'], stage))
                    if fail and item['seed'] == 20261025 and stage == 'readiness':
                        raise RuntimeError('simulated failure')
                    p = output/f'{item["seed"]}_{stage}'
                    p.mkdir()
                    value = {'contract_sha256': item['contract_sha256'], 'recipe': {'evaluation_policy': POLICY},
                             'status': 'GPU_readiness_passed_formal_start_pending', 'checks': {'ok': True}}
                    training.atomic_save(p/('report.json' if stage == 'readiness' else 'run_state.json'),
                        value if stage == 'readiness' else {'status': 'completed'}, json_format=True)
                    return p
                with patch.object(campaign, 'FOLDER', root), patch.object(campaign.old, 'FOLDER', root), \
                     patch.object(campaign.old, 'process_identity', return_value={'pid': 123}), \
                     patch.object(campaign, 'verify_sources'), patch.object(campaign.old, 'command_for', return_value=[]), \
                     patch.object(campaign.old, 'run_child', side_effect=child), \
                     patch.object(campaign, 'summarize') as summary:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                            campaign.execute(output)
                        summary.assert_not_called()
                    else:
                        campaign.execute(output)
                        summary.assert_called_once()
                        self.assertEqual(set(summary.call_args.args[2]), set(map(str, campaign.SEEDS)))
                final = json.loads((output/'state.json').read_text())
                self.assertEqual(final['status'], 'failed_or_interrupted' if fail else 'completed')
                expected = [(s, stage) for s in campaign.SEEDS for stage in ('readiness', 'training')]
                self.assertEqual(called, expected[:3] if fail else expected)


if __name__ == '__main__':
    unittest.main()
