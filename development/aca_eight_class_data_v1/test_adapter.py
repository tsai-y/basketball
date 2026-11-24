"""Bounded CPU synthetic checks; video decoding is forbidden throughout tests."""
import math
import unittest
from unittest.mock import patch
import torch

from . import adapter as a


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.decode_guard = patch.object(a.f.baseline, 'decode_uniform50',
                                        side_effect=AssertionError('No video decode allowed'))
        self.decode_guard.start()
        self.addCleanup(self.decode_guard.stop)

    @staticmethod
    def item(cls=3, video_id='synthetic_a', split='train'):
        return {'tensor': torch.zeros(3, 50, 120, 160), 'class_id': cls,
                'binary_label': a.CLASS_TO_BINARY[cls], 'original_class': a.CLASSES[cls],
                'video_id': video_id, 'split': split, 'source_group': 'synthetic',
                'metadata': {'input_contract_sha256': a.f.CONTRACT_SHA256}}

    def test_real_metadata_counts_and_forbidden_ids_without_decoding(self):
        for split, expected in [('train', 7192), ('validation', 1547)]:
            ds = a.EightClassDataset(source_root='/nonexistent-no-video', split=split)
            self.assertEqual(len(ds), expected)
        with patch.object(a.f, 'FoundationDataset', side_effect=AssertionError('No access')):
            with self.assertRaises(ValueError):
                a.EightClassDataset(source_root=object(), split='test')
        rows, _ = a.f.audit_rows(a.f.load_contract())
        for split in ('validation', 'test', 'excluded'):
            vid = next(r['video_id'] for r in rows if r['audit_split'] == split)
            with self.assertRaises(ValueError):
                a.EightClassDataset(source_root='/nonexistent-no-video', split='train', video_ids=[vid])

    def test_delegate_preserves_tensor_and_collation_uses_eight_class_targets(self):
        ds = a.EightClassDataset(source_root='/nonexistent-no-video', split='train')
        first, second = self.item(), self.item(6, 'synthetic_b')
        with patch.object(a.f.FoundationDataset, '__getitem__', return_value=first):
            self.assertIs(ds[0], first)
        batch = a.collate_eight_class([first, second])
        self.assertEqual(batch['targets'].tolist(), [3, 6])
        self.assertEqual(batch['binary_targets'].tolist(), [1, 0])
        self.assertEqual(tuple(batch['tensor'].shape), (2, 3, 50, 120, 160))
        self.assertTrue(torch.equal(batch['tensor'][0], first['tensor']))
        self.assertEqual(batch['records'][0]['video_id'], 'synthetic_a')
        loss = a.eight_class_ce(torch.zeros(2, 8), batch['targets'])
        self.assertAlmostEqual(loss.item(), math.log(8), places=6)

    def test_wrong_label_provenance_shape_and_mixed_batches_rejected(self):
        item = self.item()
        for change in [{'class_id': 1}, {'class_id': True}, {'class_id': 8},
                       {'binary_label': 0}, {'split': 'test'},
                       {'metadata': {'input_contract_sha256': 'wrong'}},
                       {'tensor': torch.zeros(3, 49, 120, 160)}]:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                a.collate_eight_class([dict(item, **change)])
        with self.assertRaises(ValueError):
            a.collate_eight_class([item, dict(item)])
        with self.assertRaises(ValueError):
            a.collate_eight_class([item, dict(item, split='validation', video_id='b')])
        with self.assertRaises(ValueError):
            a.collate_eight_class([])

    def test_hand_computed_ce_gradient_confusion_f1_and_tail_weighting(self):
        targets = torch.arange(8)
        predictions = torch.tensor([2, 0, 2, 3, 4, 5, 6, 7])
        probs = torch.full((8, 8), .2 / 7, dtype=torch.float64)
        probs[torch.arange(8), predictions] = .8
        logits = probs.log().requires_grad_()
        loss = a.eight_class_ce(logits, targets)
        expected_ce = -(6 * math.log(.8) + 2 * math.log(.2 / 7)) / 8
        self.assertAlmostEqual(loss.item(), expected_ce)
        gradient = torch.autograd.grad(loss, logits)[0]
        expected_gradient = probs.clone()
        expected_gradient[torch.arange(8), targets] -= 1
        torch.testing.assert_close(gradient, expected_gradient / 8)
        metrics = a.EightClassMetrics()
        metrics.add(logits[:3], targets[:3])
        metrics.add(logits[3:7], targets[3:7])
        metrics.add(logits[7:], targets[7:])  # Unequal tail must be sample weighted.
        result = metrics.result()
        expected_matrix = [[0]*8 for _ in range(8)]
        for t, p in enumerate([2, 0, 2, 3, 4, 5, 6, 7]): expected_matrix[t][p] = 1
        self.assertEqual(result['confusion_rows_true_columns_predicted'], expected_matrix)
        self.assertAlmostEqual(result['ce'], expected_ce)
        self.assertEqual(result['accuracy'], .75)
        self.assertAlmostEqual(result['macro_f1'], 17/24)
        self.assertEqual(result['derived_binary']['confusion_rows_true_columns_predicted'], [[4, 0], [1, 3]])
        self.assertEqual(result['derived_binary']['accuracy'], 7/8)
        whole = a.EightClassMetrics(); whole.add(logits, targets)
        self.assertAlmostEqual(whole.result()['ce'], result['ce'])
        # Report objects cannot mutate accumulated state.
        result['confusion_rows_true_columns_predicted'][0][0] = 99
        self.assertEqual(metrics.result()['count'], 8)
        self.assertEqual(metrics.result()['confusion_rows_true_columns_predicted'], expected_matrix)

    def test_binary_argmax_mapping_is_not_probability_sum(self):
        probabilities = torch.tensor([[.30, .18, .02, .17, .01, .16, .01, .15]], dtype=torch.float64)
        self.assertGreater(probabilities[:, 1::2].sum().item(), .5)
        metrics = a.EightClassMetrics(); metrics.add(probabilities.log(), torch.tensor([1]))
        self.assertEqual(metrics.result()['derived_binary']['confusion_rows_true_columns_predicted'], [[0, 0], [1, 0]])

    def test_ties_absent_classes_and_invalid_batches_are_explicit(self):
        metrics = a.EightClassMetrics()
        with self.assertRaises(ValueError): metrics.result()
        metrics.add(torch.zeros(1, 8), torch.tensor([0]))
        self.assertEqual(metrics.result()['macro_f1'], 1/8)
        before = metrics.result()
        cases = [(torch.zeros(1, 2), torch.tensor([0])),
                 (torch.zeros(0, 8), torch.tensor([], dtype=torch.long)),
                 (torch.zeros(1, 8), torch.tensor([8])),
                 (torch.zeros(1, 8), torch.tensor([-1])),
                 (torch.zeros(1, 8), torch.tensor([0.])),
                 (torch.zeros(1, 8), torch.tensor([True])),
                 (torch.zeros(1, 8), torch.tensor([0, 1])),
                 (torch.full((1, 8), float('nan')), torch.tensor([0])),
                 (torch.full((1, 8), float('inf')), torch.tensor([0]))]
        for logits, targets in cases:
            with self.assertRaises(ValueError): metrics.add(logits, targets)
            with self.assertRaises(ValueError): a.eight_class_ce(logits, targets)
            self.assertEqual(metrics.result(), before)


if __name__ == '__main__':
    unittest.main()
