"""CPU-only regression tests; no test video bytes or GPU usage."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('isolated_test_evaluate', Path(__file__).with_name('evaluate.py'))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def row(p=.5, y=0, split='test'):
    import math
    return dict(video_id='x', split=split, label=y, original_class='ft0', shot_type='ft',
                source_group='v1', p_make=p, log_probabilities=[math.log(1-p), math.log(p)])


class FrozenTest(unittest.TestCase):
    def test_tie_is_miss(self):
        self.assertEqual(ev.fuse([row()] * 3)['prediction'], 0)

    def test_probability_average_not_logits(self):
        r = ev.fuse([row(.9), row(.2), row(.2)])
        self.assertAlmostEqual(r['p_make'], 1.3/3)
        self.assertEqual(r['prediction'], 0)

    def test_stable_loss(self):
        import math
        r = ev.fuse([row(.8), row(.2), row(.5)])
        self.assertAlmostEqual(r['ce_loss'], math.log(2))

    def test_metadata_mismatch_rejected(self):
        for field, value in [('video_id', 'foreign'), ('label', 1), ('split', 'validation'), ('source_group', 'v2')]:
            bad = row(); bad[field] = value
            with self.assertRaises(ValueError):
                ev.fuse([row(), bad, row()])

    def test_nonfinite_probability_rejected(self):
        with self.assertRaises(ValueError):
            ev.fuse([row(), row(), dict(row(), p_make=float('nan'))])

    def test_exactly_three(self):
        with self.assertRaises(ValueError):
            ev.fuse([row(), row()])

    def test_test_scope_needs_claim_before_source_access(self):
        scope = ev.EvaluationScope('test')
        self.assertEqual(len(scope.rows), 1539)
        with patch.object(ev.gate1a, 'open_guarded', side_effect=AssertionError('Should never open')):
            with self.assertRaises(ValueError):
                scope.authorize(scope.rows[0]['video_id'])
            with self.assertRaises(ValueError):
                scope.authorize('../train.mp4')

    def test_no_test_subset(self):
        with self.assertRaises(ValueError):
            ev.EvaluationScope('test', [])

    def test_original_guard_still_rejects_test(self):
        with self.assertRaises(ValueError):
            ev.VideoScope(source_root=ev.SOURCE, split='test')

    def test_single_tie_rules_preserved(self):
        import torch
        source = dict(video_id='x', split='validation', binary_label='0', original_class='ft0', source_group='v1')
        self.assertEqual(ev.prediction('dense', torch.tensor([[0., 0.]]), source)['prediction'], 0)
        self.assertEqual(ev.prediction('mainline', torch.tensor([0.]), source)['prediction'], 1)


if __name__ == '__main__':
    unittest.main()
