import copy
from pathlib import Path
import tempfile
import unittest

from development.tanet_l_v2 import diagnose_errors_v1 as audit


class ErrorAuditContracts(unittest.TestCase):
    def fixture(self):
        allowed, paired = {}, {s: [] for s in audit.SEEDS}
        for bucket in audit.BUCKETS:
            for y in (0, 1):
                for shot in audit.SHOTS:
                    for i in range(4):
                        vid = f'{bucket}_{shot}{y}_{i}'
                        allowed[vid] = {'split': 'validation', 'excluded': '0', 'binary_label': str(y),
                                        'source_group': f'g{i}', 'original_class': f'{shot}{y}'}
                        for j, seed in enumerate(audit.SEEDS):
                            correct = bucket == 'all_correct' or bucket == 'seed_disagreement' and j < 1+i % 2
                            pred = y if correct else 1-y
                            paired[seed].append({'video_id': vid, 'split': 'validation', 'true_label': y,
                                'source_group': f'g{i}', 'shot_type': shot,
                                'tanet_prediction': pred, 'main_prediction': pred,
                                'tanet_p_make': .8 if pred else .2, 'main_p_make': .8 if pred else .2})
        return allowed, paired

    def rows(self):
        allowed, paired = self.fixture()
        return audit.join_seeds(paired, allowed)

    def test_full_id_join_is_order_independent_and_classifies_all_patterns(self):
        allowed, paired = self.fixture()
        expected = audit.join_seeds(paired, allowed)
        actual = audit.join_seeds({s: list(reversed(rows)) for s, rows in paired.items()}, allowed)
        self.assertEqual(actual, expected)
        d = audit.distribution(actual)
        self.assertEqual(d['support'], 96)
        self.assertEqual((d['common_wrong'], d['seed_disagreement'], d['all_correct']), (32, 32, 32))
        self.assertEqual(d['correct_seed_counts'], {'0': 32, '1': 16, '2': 16, '3': 32})

    def test_foreign_test_missing_duplicate_label_and_nonfinite_predictions_rejected(self):
        for change in ('test', 'missing', 'duplicate', 'label', 'probability'):
            allowed, paired = copy.deepcopy(self.fixture())
            if change == 'test':
                allowed[next(iter(allowed))]['split'] = 'test'
            elif change == 'missing':
                paired[audit.SEEDS[0]].pop()
            elif change == 'duplicate':
                paired[audit.SEEDS[0]].append(paired[audit.SEEDS[0]][0])
            elif change == 'label':
                paired[audit.SEEDS[0]][0]['true_label'] = 1
            else:
                paired[audit.SEEDS[0]][0]['tanet_p_make'] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.join_seeds(paired, allowed)

    def test_rates_use_video_denominators_and_preserve_zero_support_class(self):
        rows = self.rows()
        selected = [next(r for r in rows if r['bucket'] == b and r['true_label'] == 0)
                    for b in ('common_wrong', 'all_correct', 'seed_disagreement')]
        d = audit.distribution(selected)
        self.assertEqual(d['common_wrong_rate'], 1/3)
        self.assertEqual(d['support'], 3)
        self.assertEqual(d['by_seed'][str(audit.SEEDS[0])]['L']['per_class']['Make']['recall'], None)
        groups = audit.summarize_groups(rows)
        self.assertEqual(sum(r['support'] for r in groups['shot_label']), len(rows))
        self.assertEqual(sum(r['common_wrong'] for r in groups['source']), 32)

    def test_fixed24_quotas_ignore_confidence_and_row_order_and_spread_sources(self):
        rows = self.rows()
        selected, strata = audit.select_cases(rows)
        wrong = copy.deepcopy(list(reversed(rows)))
        for row in wrong:
            for s in audit.SEEDS:
                row[f'L_{s}_p_make'] = .9999
        again, _ = audit.select_cases(wrong)
        self.assertEqual([r['video_id'] for r in selected], [r['video_id'] for r in again])
        self.assertEqual(len(selected), 24)
        self.assertEqual([sum(r['bucket'] == b for r in selected) for b in audit.BUCKETS], [12, 8, 4])
        self.assertEqual([sum(r['true_label'] == y for r in selected) for y in (0, 1)], [12, 12])
        for st in strata:
            cell = [r for r in selected if (r['bucket'], r['true_label'], r['shot_type']) ==
                    (st['bucket'], st['true_label'], st['shot_type'])]
            self.assertEqual(len({r['source_group'] for r in cell}), st['selected'])

    def test_missing_stratum_stops_instead_of_replacing_case(self):
        rows = [r for r in self.rows() if not (r['bucket'] == 'common_wrong' and r['original_class'] == 'mp1')]
        with self.assertRaisesRegex(ValueError, 'Missing fixed stratum'):
            audit.select_cases(rows)
        with self.assertRaises(ValueError):
            audit.select_cases(self.rows()+[self.rows()[0]])

    def test_sampling_gaps_and_exact_endpoints(self):
        indices = list(audit.uniform_indices(192))
        self.assertEqual((indices[0], indices[-1]), (0, 191))
        d = audit.temporal_gaps(indices, 30.)
        self.assertEqual(d['max_sample_step_frames'], 4)
        self.assertEqual(d['max_omitted_consecutive_frames'], 3)
        self.assertAlmostEqual(d['max_time_between_samples_seconds'], 4/30)
        with self.assertRaises(ValueError):
            audit.temporal_gaps([0]*50, 30.)

    def test_complete_numeric_and_case_schema_renders_page_and_scientific_plot(self):
        rows = self.rows()
        cases, _ = audit.select_cases(rows)
        cases = [{**c, 'frame_count': 50, 'fps': 30., 'tanet_indices': list(range(50)),
                  'sampling_gaps': audit.temporal_gaps(list(range(50)), 30.)} for c in cases]
        report = {'cases': cases, 'distributions': audit.summarize_groups(rows)}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out/'plots').mkdir()
            audit.render(out, report)
            audit.plot_groups(out, report['distributions'])
            page = (out/'index.html').read_text()
            self.assertEqual(page.count('<section class="case"'), 24)
            self.assertIn('human_review_notes.json', page)
            self.assertIn('最近的TANet抽樣影格', page)
            self.assertTrue((out/'plots/common_errors.png').is_file())


if __name__ == '__main__':
    unittest.main()
