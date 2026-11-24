"""CPU metadata and failure-path checks; all real video decoding forbidden."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import adapter as a


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.revision = a.LabelRevision()
        cls.manifest = json.loads((a.REVISION_DIR / 'manifest.json').read_text())
        cls.patch = json.loads((a.REVISION_DIR / 'label_patch.json').read_text())
        cls.decision = json.loads((a.REVISION_DIR.parent / 'adjudication15_confirmed_v1.json').read_text())
        cls.parent = json.loads((a.CURRENT / 'baseline8_formal_seed20261024_v1/plan.json').read_text())
        cls.rows, _ = a.f.audit_rows(a.f.load_contract())

    def setUp(self):
        guard = patch.object(a.f.baseline, 'decode_uniform50',
                             side_effect=AssertionError('Unit checks cannot decode videos'))
        guard.start()
        self.addCleanup(guard.stop)

    def bundle(self, manifest=None, correction=None, decision=None):
        a._validate_bundle(self.manifest if manifest is None else manifest,
                           self.patch if correction is None else correction, self.parent,
                           self.decision if decision is None else decision, self.rows)

    def synthetic_original(self):
        r = self.revision.record(a.TARGET, 'train')
        return dict(video_id=a.TARGET, original_class='3p1', class_id=3, binary_label=1,
                    source_group=r['source_group'], split='train', tensor=torch.zeros(3, 50, 120, 160),
                    metadata={'input_contract_sha256': a.f.CONTRACT_SHA256,
                              'source_size_bytes': r['source_size_bytes']})

    def revised_synthetic(self):
        ds = a.AdjudicatedDataset(source_root='/no-video', split='train', video_ids=[a.TARGET])
        old = self.synthetic_original()
        ref = self.revision.record(a.TARGET, 'train')
        with patch.object(a.f.baseline, 'guarded_source_sha256', return_value=ref['source_sha256']), \
             patch.object(a.f.FoundationDataset, '__getitem__', return_value=old), \
             patch.object(a, 'tensor_sha', return_value=ref['tensor_sha256']):
            item = ds[0]
        return old, item, ref

    def test_all_8739_targets_and_order(self):
        changed = []
        for split, count in [('train', 7192), ('validation', 1547)]:
            ids = self.revision.ids(split)
            self.assertEqual(ids, tuple(self.parent[split + '_ids']))
            self.assertEqual(len(ids), count)
            for vid in ids:
                r = self.revision.record(vid, split)
                p = self.parent['reference'][vid]
                self.assertEqual(r['original_class_id'], p['class_id'])
                self.assertEqual(r['binary_label'], p['binary_label'])
                if r['class_id'] != p['class_id']:
                    changed.append((vid, p['class_id'], r['class_id']))
        self.assertEqual(changed, [(a.TARGET, 3, 1)])

    def test_test_split_rejected_before_any_metadata_or_source_access(self):
        with patch.object(a, 'LabelRevision', side_effect=AssertionError('No metadata access')):
            with self.assertRaises(ValueError):
                a.AdjudicatedDataset(source_root=object(), split='test')

    def test_cross_split_excluded_unknown_and_duplicate_requests_rejected(self):
        for split in ('validation', 'test', 'excluded'):
            vid = next(r['video_id'] for r in self.rows if r['audit_split'] == split)
            with self.subTest(split=split), self.assertRaises(ValueError):
                a.AdjudicatedDataset(source_root='/no-video', split='train', video_ids=[vid])
        for ids in ([a.TARGET, a.TARGET], ['nonexistent']):
            with self.assertRaises(ValueError):
                a.AdjudicatedDataset(source_root='/no-video', split='train', video_ids=ids)

    def test_manifest_file_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ['manifest.json', 'label_patch.json']:
                (Path(d) / name).write_bytes((a.REVISION_DIR / name).read_bytes())
            path = Path(d) / 'manifest.json'
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'Pinned file changed'):
                a.LabelRevision(d)

    def test_wrong_original_source_patch_and_unconfirmed_decision_rejected(self):
        for key, value in [('expected_original_class', '3p0'), ('source_sha256', 'wrong'),
                           ('split', 'validation'), ('effective_class_id', True)]:
            p = deepcopy(self.patch); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.bundle(correction=p)
        with self.assertRaises(ValueError):
            self.bundle(correction=[self.patch, self.patch])
        d = deepcopy(self.decision); d['status'] = 'pending'
        with self.assertRaises(ValueError):
            self.bundle(decision=d)

    def test_extra_correction_duplicate_id_and_changed_input_rejected(self):
        for mutation in ('extra_correction', 'duplicate_id', 'source_sha', 'tensor_sha', 'original_label'):
            m = deepcopy(self.manifest)
            if mutation == 'duplicate_id':
                m['train_ids'].append(a.TARGET)
            elif mutation == 'extra_correction':
                m['reference'][m['validation_ids'][0]]['class_id'] = 6
            elif mutation == 'original_label':
                m['reference'][a.TARGET]['class_id'] = 3
            else:
                m['reference'][a.TARGET][mutation + '256'] = 'changed'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.bundle(manifest=m)

    def test_source_sha_failure_precedes_decode(self):
        ds = a.AdjudicatedDataset(source_root='/no-video', split='train', video_ids=[a.TARGET])
        with patch.object(a.f.baseline, 'guarded_source_sha256', return_value='wrong'), \
             patch.object(a.f.FoundationDataset, '__getitem__', side_effect=AssertionError('No decode')):
            with self.assertRaisesRegex(ValueError, 'Source SHA'):
                ds[0]

    def test_overlay_preserves_pixels_original_item_and_batch_uses_effective_target(self):
        old, item, ref = self.revised_synthetic()
        self.assertIs(item['tensor'], old['tensor'])
        self.assertEqual(old['class_id'], 3)
        self.assertNotIn('effective_class', old)
        self.assertNotIn('data_version', old['metadata'])
        self.assertEqual(item['class_id'], 1)
        self.assertEqual(item['original_class'], '3p1')
        self.assertEqual(item['effective_class'], '2p1')
        collator = a.AdjudicatedCollator()
        with patch.object(a, 'tensor_sha', return_value=ref['tensor_sha256']):
            batch = collator([item])
        self.assertEqual(batch['targets'].tolist(), [1])
        self.assertEqual(batch['binary_targets'].tolist(), [1])
        self.assertTrue(torch.equal(batch['tensor'][0], old['tensor']))
        self.assertEqual(batch['records'][0]['original_class'], '3p1')

    def test_old_collator_cannot_silently_accept_revised_target(self):
        old, item, _ = self.revised_synthetic()
        self.assertEqual(a.old_adapter.collate_eight_class([old])['targets'].tolist(), [3])
        with self.assertRaises(ValueError):
            a.old_adapter.collate_eight_class([item])
        with self.assertRaises(ValueError):
            a.AdjudicatedCollator()([old])

    def test_wrong_effective_target_revision_provenance_tensor_and_duplicates_rejected(self):
        _, item, ref = self.revised_synthetic()
        collator = a.AdjudicatedCollator()
        for key, value in [('class_id', 3), ('original_class', '2p1'), ('effective_class', '3p1')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                collator([dict(item, **{key: value})])
        bad = dict(item, metadata=dict(item['metadata'], label_revision_sha256='old'))
        with self.assertRaises(ValueError):
            collator([bad])
        with self.assertRaisesRegex(ValueError, 'Frozen input tensor'):
            collator([item])  # Synthetic zero tensor is NOT the pinned real clip.
        with patch.object(a, 'tensor_sha', return_value=ref['tensor_sha256']):
            with self.assertRaisesRegex(ValueError, 'Duplicate batch'):
                collator([item, item])

    def test_record_mutation_does_not_change_revision(self):
        r = self.revision.record(a.TARGET, 'train'); r['class_id'] = 3
        self.assertEqual(self.revision.record(a.TARGET, 'train')['class_id'], 1)

    def test_historical_run_resume_identity_rejected(self):
        with self.assertRaises(ValueError):
            a.require_revision_identity(self.parent)
        good = dict(data_version=a.VERSION, label_revision_sha256=a.MANIFEST_SHA,
                    pixel_contract_parent_sha256=a.f.CONTRACT_SHA256)
        a.require_revision_identity(good)
        with self.assertRaises(ValueError):
            a.require_revision_identity(dict(good, label_revision_sha256='old'))


if __name__ == '__main__':
    unittest.main()
