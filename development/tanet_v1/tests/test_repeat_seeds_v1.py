import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import csv
import torch

from development.tanet_v1 import repeat_seeds_v1 as repeats


class RepeatSeedContracts(unittest.TestCase):
    def test_isolated_engines_are_identical_and_only_json_seed_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)/'repeat'
            original = {name: repeats.sha256(repeats.FOLDER/name) for name in repeats.FILES}
            repeats.clone_seed(repeats.FOLDER, destination, 20261024)
            self.assertEqual(original, {name: repeats.sha256(repeats.FOLDER/name) for name in repeats.FILES})
            for name in ('config.json', 'train_config.json'):
                before, after = repeats.read(repeats.FOLDER/name), repeats.read(destination/name)
                self.assertEqual(after['seed'], 20261024)
                after['seed'] = before['seed']
                self.assertEqual(before, after)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                repeats.clone_seed(repeats.FOLDER, destination, 20261024)
            cfg = repeats.read(destination/'train_config.json')
            cfg['max_epochs'] = 20
            (destination/'train_config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'Only seed'):
                repeats.check_clone(repeats.FOLDER, destination, 20261024)

    def test_undeclared_seed_rejected_and_original_config_not_mutated(self):
        cfg = {'seed': 20261025, 'optimizer': {'name': 'Adam'}}
        unchanged = copy.deepcopy(cfg)
        for seed in (20261025, 0, '20261024', True):
            with self.assertRaises(ValueError):
                repeats.seed_config(cfg, seed)
        changed = repeats.seed_config(cfg, 20261026)
        changed['optimizer']['name'] = 'other'
        self.assertEqual(cfg, unchanged)

    def test_fresh_readiness_required_before_formal_command(self):
        item = {'module': 'development.tanet_seed20261024_v1'}
        self.assertEqual(repeats.command_for(item, 'readiness')[-2:],
                         ['development.tanet_seed20261024_v1.readiness', '--run'])
        with self.assertRaises(ValueError):
            repeats.command_for(item, 'training')
        command = repeats.command_for(item, 'training', Path('/fresh/report.json'))
        self.assertEqual(command[-3:], ['--start', '--readiness-report', '/fresh/report.json'])
        self.assertNotIn('--resume', command)

    def test_output_ambiguity_and_duplicate_seed_statistics_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'stdout.log'
            log.write_text('not ready\n')
            with self.assertRaises(ValueError):
                repeats.output_path(log, 'OUTPUT: ')
            log.write_text('OUTPUT: /first\nOUTPUT: /second\n')
            with self.assertRaises(ValueError):
                repeats.output_path(log, 'OUTPUT: ')
        with self.assertRaises(ValueError):
            repeats.distribution([.8, .9])
        stats = repeats.distribution([.7, .8, .9])
        self.assertAlmostEqual(stats['mean'], .8)
        self.assertAlmostEqual(stats['sample_sd'], .1)

    def test_failed_first_job_stops_queue_and_cannot_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            output = base/'artifacts/multiseed_test'
            output.mkdir(parents=True)
            plan = {'new_seeds': list(repeats.NEW_SEEDS), 'seeds': list(repeats.SEEDS),
                    'formal_training_explicitly_authorized': True, 'test_access': False,
                    'original_seed20261025': '/original',
                    'queue': [{'seed': seed, 'module': 'example'} for seed in repeats.NEW_SEEDS]}
            (output/'plan.json').write_text(json.dumps(plan))
            (output/'state.json').write_text(json.dumps({'status': 'prepared', 'jobs': []}))
            with patch.object(repeats, 'FOLDER', base), patch.object(repeats, 'verify_sources'), \
                    patch.object(repeats, 'process_identity', return_value={'pid': 1}), \
                    patch.object(repeats, 'run_child', side_effect=RuntimeError('probe failed')) as child:
                with self.assertRaisesRegex(RuntimeError, 'probe failed'):
                    repeats.execute(output)
                self.assertEqual(child.call_count, 1)
                self.assertEqual(repeats.read(output/'state.json')['status'], 'failed_or_interrupted')
                with self.assertRaisesRegex(ValueError, 'no automatic restart'):
                    repeats.execute(output)
                self.assertEqual(child.call_count, 1)

    def test_existing_mainline_recipes_are_same_except_seed(self):
        runs = repeats.validate_mainlines()
        self.assertEqual(set(runs), {'20261024', '20261025', '20261026'})

    def test_completed_summary_joins_full_validation_and_reports_all_seeds(self):
        from development.tanet_v1.training import metrics
        n = 1547
        ids = [f'video{i}' for i in range(n)]
        labels = torch.arange(n) % 2
        logits = torch.stack([1.-labels.float(), labels.float()], dim=1)
        scores = metrics(logits, labels)
        canonical = {vid: {'split': 'validation', 'excluded': '0', 'binary_label': str(int(y)),
                          'source_group': 'synthetic', 'original_class': f'2p{y}'} for vid, y in zip(ids, labels)}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'out'; out.mkdir()
            runs, mains = {}, {}
            for seed in repeats.SEEDS:
                run, main = Path(folder)/f'tanet{seed}', Path(folder)/f'main{seed}'
                run.mkdir(); main.mkdir()
                runs[str(seed)], mains[str(seed)] = str(run), str(main)
                (run/'contract.json').write_text(json.dumps({'recipe': {'seed': seed}, 'contract_sha256': 'synthetic'}))
                (run/'run_state.json').write_text(json.dumps({'status': 'completed'}))
                (run/'history.json').write_text(json.dumps([{'epoch': 1, 'validation': scores},
                    {'epoch': 2, 'validation': {**scores, 'CE': scores['CE']+.1}}]))
                torch.save({'kind': 'formal', 'epoch': 1, 'contract_sha256': 'synthetic',
                            'validation': {'logits': logits, 'labels': labels, 'video_ids': ids}}, run/'best.pt')
                with (main/'validation_predictions.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['video_id', 'true_label', 'predicted_label', 'p_make'])
                    writer.writeheader()
                    # Deliberately reversed order to exercise ID joins.
                    writer.writerows({'video_id': vid, 'true_label': int(y), 'predicted_label': int(y),
                                      'p_make': .8 if y else .2} for vid, y in reversed(list(zip(ids, labels))))
                (main/'summary.json').write_text(json.dumps({'best_epoch': 3, 'validation': {
                    'loss': .4, 'confusion_matrix': scores['confusion_matrix_Miss_Make']}}))
            with patch('development.tanet_v1.data.VideoScope') as scope:
                scope.return_value.allowed = canonical
                result = repeats.summarize(out, {'mainline_runs': mains}, runs)
            self.assertEqual([d['seed'] for d in result['details']], list(repeats.SEEDS))
            self.assertEqual(result['aggregate']['TANet']['accuracy']['mean'], 1.)
            self.assertEqual(result['paired_accuracy_difference']['values'], [0., 0., 0.])
            self.assertTrue(all(d['paired_counts'] == {'both_correct': n} for d in result['details']))
            self.assertEqual(len(list(out.glob('paired_seed*.csv'))), 3)
            self.assertFalse(result['test_used'])


if __name__ == '__main__':
    unittest.main()
