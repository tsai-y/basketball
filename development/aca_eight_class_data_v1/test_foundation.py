"""Metadata/access-boundary checks; full real decoding is the separate audit."""
import csv
import hashlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from . import foundation as f


class FoundationContractTests(unittest.TestCase):
    def test_reviewed_cohort_and_labels_are_reproducible(self):
        contract = f.load_contract()
        rows, raw = f.audit_rows(contract)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), contract['audit_manifest_sha256'])
        self.assertEqual(len(rows), 10311)
        for row in rows:
            self.assertEqual(contract['classes'][row['class_id']], row['original_class'])
            self.assertEqual(contract['class_to_binary'][row['class_id']], int(row['binary_label']))
            if not row['excluded']:
                self.assertEqual(row['original_split'], row['audit_split'])

    def test_test_split_rejected_before_any_contract_or_baseline_access(self):
        with patch.object(f, 'load_contract', side_effect=AssertionError('Must not read metadata')), \
             patch.object(f.baseline, 'VideoScope', side_effect=AssertionError('Must not construct path')):
            with self.assertRaisesRegex(ValueError, 'Only train/validation'):
                f.FoundationScope(source_root=object(), split='test')

    def test_cross_split_and_quarantine_rejected_before_source_access(self):
        scope = f.FoundationScope(source_root='/nonexistent-unused-source', split='train')
        rows, _ = f.audit_rows(scope.contract)
        forbidden = [next(r['video_id'] for r in rows if r['audit_split'] == 'validation'),
                     next(r['video_id'] for r in rows if r['audit_split'] == 'test'),
                     next(r['video_id'] for r in rows if r['audit_split'] == 'excluded'),
                     '2p1/2p1_v166_005133_x264.mp4', '../test.mp4']
        with patch.object(scope._baseline, 'authorize', side_effect=AssertionError('Forbidden path access')):
            for video_id in forbidden:
                with self.subTest(video_id=video_id), self.assertRaises(ValueError):
                    scope.authorize(video_id)

    def test_tampered_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'contract.json'
            path.write_bytes(f.CONTRACT_PATH.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                f.load_contract(path)


if __name__ == '__main__':
    unittest.main()
