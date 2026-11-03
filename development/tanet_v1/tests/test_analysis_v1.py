import copy
import unittest
from unittest.mock import Mock, patch

from development.tanet_v1.analyze_v1 import (
    PATTERNS, aggregate, align_predictions, best_epoch, decode_native, select_cases,
)


class FrozenAnalysisTests(unittest.TestCase):
    def fixture(self):
        allowed, ta, ma = {}, [], []
        for y in (0, 1):
            for pattern in PATTERNS:
                for i in range(3):
                    vid = f'{y}_{pattern}_{i}'
                    allowed[vid] = {'video_id': vid, 'split': 'validation', 'excluded': '0',
                        'binary_label': str(y), 'source_group': f'group{i}', 'original_class': f'2p{y}'}
                    a = y if pattern in ('both_correct', 'tanet_only_correct') else 1-y
                    b = y if pattern in ('both_correct', 'main_only_correct') else 1-y
                    ta.append({'video_id': vid, 'true_label': y, 'prediction': a, 'p_make': .8 if a else .2})
                    ma.append({'video_id': vid, 'true_label': y, 'prediction': b, 'p_make': .8 if b else .2})
        return allowed, ta, ma

    def test_id_join_is_order_independent_and_preserves_paired_outcomes(self):
        allowed, ta, ma = self.fixture()
        rows = align_predictions(ta, list(reversed(ma)), allowed)
        self.assertEqual({p: sum(r['pattern'] == p for r in rows) for p in PATTERNS}, {p: 6 for p in PATTERNS})
        self.assertEqual(aggregate(rows, 'tanet_prediction')['confusion_matrix'], [[6, 6], [6, 6]])
        self.assertEqual(aggregate(rows, 'tanet_prediction')['macro_f1'], .5)

    def test_duplicate_missing_wrong_label_and_test_metadata_fail_closed(self):
        allowed, ta, ma = self.fixture()
        for badta, badma, badallowed in [(ta+[ta[0]], ma, allowed), (ta[:-1], ma, allowed)]:
            with self.assertRaises(ValueError):
                align_predictions(badta, badma, badallowed)
        wrong = copy.deepcopy(ta)
        wrong[0]['true_label'] = 1-wrong[0]['true_label']
        with self.assertRaises(ValueError):
            align_predictions(wrong, ma, allowed)
        wrong_allowed = copy.deepcopy(allowed)
        wrong_allowed[ta[0]['video_id']]['split'] = 'test'
        with self.assertRaises(ValueError):
            align_predictions(ta, ma, wrong_allowed)

    def test_fixed16_selection_ignores_order_and_confidence_and_spreads_sources(self):
        allowed, ta, ma = self.fixture()
        rows = align_predictions(ta, ma, allowed)
        selected, strata = select_cases(rows)
        other = copy.deepcopy(list(reversed(rows)))
        for row in other:
            row['tanet_p_make'], row['main_p_make'] = .99, .01
        again, _ = select_cases(other)
        self.assertEqual([r['video_id'] for r in selected], [r['video_id'] for r in again])
        self.assertEqual(len(selected), 16)
        for cell in strata:
            group = [r for r in selected if r['true_label'] == cell['true_label'] and r['pattern'] == cell['pattern']]
            self.assertEqual(len({r['source_group'] for r in group}), 2)

    def test_missing_stratum_is_reported_without_replacement(self):
        allowed, ta, ma = self.fixture()
        rows = [r for r in align_predictions(ta, ma, allowed) if r['pattern'] != 'both_wrong']
        selected, strata = select_cases(rows)
        self.assertEqual(len(selected), 12)
        self.assertTrue(all(s['selected'] == 0 for s in strata if s['pattern'] == 'both_wrong'))

    def test_checkpoint_selection_stays_minimum_loss_not_highest_accuracy(self):
        history = [{'epoch': 1, 'validation': {'CE': .5, 'accuracy': .7}},
                   {'epoch': 2, 'validation': {'CE': .6, 'accuracy': .8}}]
        self.assertEqual(best_epoch(history), 1)

    def test_test_video_rejected_before_any_path_access(self):
        guard = Mock()
        guard.split = 'validation'
        with patch('development.tanet_v1.analyze_v1.cv2.VideoCapture', side_effect=AssertionError('Opened video')):
            with self.assertRaises(ValueError):
                decode_native({'split': 'test', 'video_id': 'forbidden'}, guard)
        guard.identity.assert_not_called()
        guard.open.assert_not_called()


if __name__ == '__main__':
    unittest.main()
