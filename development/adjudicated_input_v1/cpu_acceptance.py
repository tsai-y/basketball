"""Bounded acceptance: metadata tests, then exactly one adjudicated train decode.

Run from the ACA worktree with python -B -m development.adjudicated_input_v1.cpu_acceptance.
No model, optimizer, weights, GPU work, validation media, or test media is used.
"""
import datetime
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from . import adapter as a
from . import test_adapter


OUT = a.REVISION_DIR / 'cpu_acceptance_v1'


def save(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    a.need(not OUT.exists(), 'Acceptance output exists; inspect it instead of overwriting')
    OUT.mkdir()
    torch.set_num_threads(4)
    verification = json.loads((a.REVISION_DIR / 'verification.json').read_text())
    protected = dict(verification['protected_artifact_hashes'])
    for name in ('manifest.json', 'label_patch.json', 'verification.json', 'build_revision.py', 'README.md'):
        p = a.REVISION_DIR / name
        protected[str(p)] = a.digest(p.read_bytes())
    for name, sha in a.PINNED_CODE.items():
        protected[str(a.f.ROOT / name)] = sha
    for path, sha in protected.items():
        a.need(a.digest(Path(path).read_bytes()) == sha, f'Pre-check changed file: {path}')
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_adapter)
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (OUT / 'unit_tests.log').write_text(log.getvalue())
    if not result.wasSuccessful():
        save('failure.json', {'stage': 'CPU_unit_checks', 'tests_run': result.testsRun,
                              'failures': [(str(t), msg) for t, msg in result.failures],
                              'errors': [(str(t), msg) for t, msg in result.errors]})
        raise RuntimeError('CPU checks failed; no real clip decoded')

    decoded = []
    originals = []
    real_decode = a.f.baseline.decode_uniform50
    real_getitem = a.f.FoundationDataset.__getitem__

    def bounded_decode(scope, video_id):
        a.need(scope.split == 'train' and video_id == a.TARGET and not decoded,
               'Only one decode of the already-adjudicated train clip is permitted')
        decoded.append(video_id)
        return real_decode(scope, video_id)

    def observe_original(dataset, index):
        item = real_getitem(dataset, index)
        originals.append(item)
        return item

    try:
        ds = a.AdjudicatedDataset(source_root='/home/tsai/projects/basketball',
                                  split='train', video_ids=[a.TARGET])
        collator = a.AdjudicatedCollator()
        with patch.object(a.f.baseline, 'decode_uniform50', bounded_decode), \
             patch.object(a.f.FoundationDataset, '__getitem__', observe_original):
            batches = list(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                      collate_fn=collator))
        a.need(len(batches) == len(originals) == len(decoded) == 1, 'Unexpected decode/batch count')
        batch, old = batches[0], originals[0]
        record = batch['records'][0]
        expected = ds.revision.record(a.TARGET, 'train')
        a.need(batch['targets'].tolist() == [1] and batch['binary_targets'].tolist() == [1],
               'Wrong effective batch target')
        a.need(old['class_id'] == 3 and old['original_class'] == '3p1'
               and 'data_version' not in old['metadata'], 'Original delegate was mutated')
        a.need(record['original_class'] == '3p1' and record['effective_class'] == '2p1',
               'Wrong provenance')
        a.need(torch.equal(batch['tensor'][0], old['tensor'])
               and a.tensor_sha(batch['tensor'][0]) == expected['tensor_sha256'],
               'Pixel change introduced')
        # Exercise the unchanged old collator on the SAME decoded original item.
        a.need(a.old_adapter.collate_eight_class([old])['targets'].tolist() == [3],
               'Old collator semantics changed')
        for path, sha in protected.items():
            a.need(a.digest(Path(path).read_bytes()) == sha, f'Historical file changed: {path}')
        report = {
            'schema': 'adjudicated_input_cpu_acceptance_v1',
            'created_at_taiwan': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'status': 'passed', 'data_version': a.VERSION,
            'label_revision_sha256': a.MANIFEST_SHA,
            'unit_tests_run': result.testsRun, 'unit_failures': len(result.failures),
            'unit_errors': len(result.errors), 'metadata_rows_checked': 8739,
            'metadata_unchanged_train_targets': 7191, 'metadata_unchanged_validation_targets': 1547,
            'real_video_decodes': 1, 'real_video_id': a.TARGET, 'real_split': 'train',
            'original_class': '3p1', 'original_class_id': 3,
            'effective_class': '2p1', 'batch_targets': batch['targets'].tolist(),
            'binary_targets': batch['binary_targets'].tolist(),
            'batch_shape': list(batch['tensor'].shape), 'dtype': str(batch['tensor'].dtype),
            'device': str(batch['tensor'].device),
            'selected_indices': old['metadata']['selected_indices'],
            'source_sha256': record['metadata']['source_sha256'],
            'tensor_sha256': a.tensor_sha(batch['tensor'][0]),
            'tensor_equal_to_frozen_delegate': True, 'tensor_matches_prior_reference': True,
            'original_delegate_and_old_collator_target': 3,
            'historical_files_unchanged': True, 'protected_artifact_hashes': protected,
            'code_sha256': {p.name: a.digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')},
            'validation_video_decodes': 0, 'test_video_decodes': 0, 'new_visual_samples': 0,
            'model_forward_calls': 0, 'optimizer_updates': 0,
            'training_launcher_connected': False,
            'limitations': ['CPU loader/collator acceptance only; not classification evidence',
                           'All target metadata checked, but only the adjudicated train clip decoded',
                           'Old run plans/checkpoints cannot silently adopt this data revision'],
        }
        save('report.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'unit_tests_run', 'metadata_rows_checked',
                          'real_video_decodes', 'original_class', 'effective_class', 'batch_targets',
                          'tensor_matches_prior_reference', 'training_launcher_connected')}, ensure_ascii=False))
    except Exception as exc:
        save('failure.json', {'stage': 'bounded_real_CPU_load', 'error': repr(exc), 'decoded': decoded})
        raise


if __name__ == '__main__':
    main()
