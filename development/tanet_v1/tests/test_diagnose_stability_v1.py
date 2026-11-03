import copy
import math
import unittest
from pathlib import Path
import tempfile

import torch

from development.tanet_v1.diagnose_stability_v1 import (
    bn_statistics, confidence_metrics, frozen_bn_check, grouped_metrics, plots, render, rows_from_logits, validate_ids,
)
from development.tanet_v1.training import metrics


class StabilityDiagnostics(unittest.TestCase):
    def fixture(self, logits=None):
        logits = torch.tensor([[2., 0.], [0., 2.]]) if logits is None else logits
        labels = torch.tensor([0, 1])
        allowed = {str(i): {'split': 'validation', 'excluded': '0', 'binary_label': str(i),
                            'source_group': 'synthetic', 'original_class': f'2p{i}'} for i in (0, 1)}
        saved = {'validation': {'logits': logits, 'labels': labels, 'video_ids': ['0', '1'],
                                'metrics': metrics(logits, labels)}}
        return saved, allowed

    def test_stable_nll_for_saturated_wrong_probabilities(self):
        saved, allowed = self.fixture(torch.tensor([[-1000., 1000.], [-1000., 1000.]]))
        rows = rows_from_logits(saved, allowed)
        self.assertEqual(rows[0]['nll'], 2000.)
        self.assertEqual(rows[0]['confidence'], 1.)
        self.assertTrue(all(math.isfinite(r['nll']) for r in rows))
        out = confidence_metrics(rows)
        self.assertEqual(out['high_confidence']['0.9']['errors'], 1)
        self.assertEqual(out['high_confidence']['0.9']['count'], 2)
        self.assertEqual(out['reliability_bins'][-1]['count'], 2)

    def test_fixed_reliability_bins_and_no_error_denominators(self):
        saved, allowed = self.fixture()
        rows = rows_from_logits(saved, allowed)
        rows[0].update(confidence=.9, p_make=.1)
        rows[1].update(confidence=1., p_make=1.)
        out = confidence_metrics(rows)
        self.assertEqual(out['reliability_bins'][-1]['count'], 2)
        self.assertAlmostEqual(out['ece10_toplabel'], .05)
        self.assertIsNone(out['wrong_confidence'])
        self.assertIsNone(out['high_confidence']['0.9']['share_of_all_errors'])
        self.assertAlmostEqual(out['binary_brier'], .005)

    def test_canonical_ids_labels_and_split_fail_closed(self):
        saved, allowed = self.fixture()
        with self.assertRaises(ValueError): validate_ids(['0','0'],[0,0],allowed)
        with self.assertRaises(ValueError): validate_ids(['0'],[0],allowed)
        with self.assertRaises(ValueError): validate_ids(['0','1'],[1,1],allowed)
        allowed['0']['split'] = 'test'
        with self.assertRaises(ValueError): rows_from_logits(saved,allowed)

    def test_id_reordering_preserves_per_video_rows_and_rejects_nonfinite(self):
        saved, allowed = self.fixture()
        expected = rows_from_logits(saved,allowed)
        v = saved['validation'];v['logits']=v['logits'].flip(0);v['labels']=v['labels'].flip(0);v['video_ids'].reverse()
        self.assertEqual(rows_from_logits(saved,allowed),expected)
        v['logits'][0,0]=float('nan')
        with self.assertRaises(ValueError): rows_from_logits(saved,allowed)

    def bn_state(self):
        state={}
        for i in range(16):
            for branch in ('G','L'):
                name=f'backbone.layer1.{i}.tam.{branch}.1'
                for key,value in [('running_mean',[0.,1.]),('running_var',[0.,4.]),
                                  ('weight',[1.,2.]),('bias',[0.,0.])]:
                    state[name+'.'+key]=torch.tensor(value)
                state[name+'.num_batches_tracked']=torch.tensor(7193)
        return state

    def test_bn_counters_gain_zero_variance_and_state_unchanged(self):
        state=self.bn_state();before=copy.deepcopy(state)
        rows,raw=bn_statistics(state,7193)
        self.assertEqual(len(rows),32)
        self.assertTrue(all(r['variance_le_1e-8_count']==1 for r in rows))
        self.assertAlmostEqual(rows[0]['eval_gain_abs']['max'],1/math.sqrt(1e-5))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in state.items()))
        with self.assertRaises(ValueError):bn_statistics(state,7194)
        state[rows[0]['layer']+'.running_var'][0]=-1
        with self.assertRaises(ValueError):bn_statistics(state,7193)

    def test_frozen_bn_legacy_zero_counters_and_changed_buffer(self):
        state,imagenet={},{}
        for i in range(53):
            for key,value in [('running_mean',torch.zeros(2)),('running_var',torch.ones(2))]:
                state[f'backbone.bn{i}.{key}']=value.clone();imagenet[f'bn{i}.{key}']=value.clone()
            state[f'backbone.bn{i}.num_batches_tracked']=torch.tensor(0)
        result=frozen_bn_check(state,imagenet)
        self.assertTrue(result['all_bitwise_equal'])
        self.assertEqual(result['legacy_missing_ImageNet_BN_counters_initialized_to_zero'],53)
        state['backbone.bn0.running_var'][0]=2.
        self.assertEqual(frozen_bn_check(state,imagenet)['different'],['backbone.bn0.running_var'])

    def test_grouped_schema_flows_through_all_report_plots_and_html(self):
        saved,allowed=self.fixture()
        grouped=grouped_metrics(rows_from_logits(saved,allowed))
        snapshot={'epoch':1,'saved_CE':saved['validation']['metrics']['CE'],**grouped}
        bn,_=bn_statistics(self.bn_state(),7193)
        report={'details':[{'seed':20261024,'snapshots':{'best':snapshot,'last':snapshot},
                            'mainline':{'saved_BCE':.3,**grouped}}],
                'histories':{'20261024':[{'epoch':1,'train_CE':.2,'validation':{'CE':.3}}]},
                'limitations':['Synthetic integration test.']}
        bn_rows=[{'seed':20261024,'snapshot':tag,'layer':r['layer'],
                  'variance_median':r['running_var']['median'],'gain_p95':r['eval_gain_abs']['p95']}
                 for tag in ('best','last') for r in bn]
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'plots').mkdir()
            plots(out,report,bn_rows);render(out,report)
            self.assertEqual(len(list((out/'plots').glob('*.png'))),3)
            self.assertIn('Seed 20261024',(out/'index.html').read_text())


if __name__ == '__main__':
    unittest.main()
