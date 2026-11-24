"""Keep frozen RGB loading, but use an explicitly adjudicated effective target.

Original class and file identity remain provenance. The revised class_id is never
derived from the video ID, directory, or filename. No old module is patched.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from development.aca_eight_class_data_v1 import adapter as old_adapter
from development.aca_eight_class_data_v1 import foundation as f

CLASSES = old_adapter.CLASSES
CLASS_TO_BINARY = old_adapter.CLASS_TO_BINARY
VERSION = 'diagnostic32_label_revision_v1'
TARGET = '3p1/3p1_v150_003223_x264.mp4'
CURRENT = Path('/home/tsai/projects/basketball/outputs/01_current')
REVISION_DIR = CURRENT / 'input_diagnostic32_20261119_v1/label_revision_v1'
MANIFEST_SHA = '1d2df00e61e4aa0f8646f03315e24574e77c28c4246c976b58bf54c0b74aa85f'
PATCH_SHA = 'd32342a2b97f4e6c399be3d548b20b19339092769dc54f22b57ff68cbf8965a6'
DECISION_SHA = '1fff1be87b40cadb8129a1b298bc8666a68756df1b3df44cfe627dcffb1fce77'
PARENT_PLAN_SHA = '7934894ec6d8252b1b03ebd918ee2cc973e92676af3a66c7e0a8f4f7fca39318'
PINNED_CODE = {
    'development/aca_eight_class_data_v1/foundation.py': '75c7734b526c5847e817d990f3f7f8be036c83bbd8a2bcc7b0d4b65c44311eec',
    'development/aca_eight_class_data_v1/adapter.py': 'e48d666f24c41633395f0b1eacac9a0ba2a170760743d0576329f2bf12ee3252',
    'development/aca_net_v1/data.py': '2e53e886e0eb369294476e098638f24891981eccde3f83c48a6f553c5b752af4',
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensor_sha(tensor):
    return digest(tensor.detach().numpy().tobytes())


def _read_pinned(path, expected):
    raw = Path(path).read_bytes()
    need(digest(raw) == expected, f'Pinned file changed: {path}')
    return json.loads(raw)


def _validate_bundle(manifest, patch, parent, decision, original_rows):
    """Semantic guards supplement file hashes; no source paths are opened."""
    need(isinstance(patch, dict), 'Exactly one correction object required')
    need(manifest['schema'] == 'adjudicated_train_validation_reference_v1'
         and manifest['data_version'] == VERSION, 'Wrong revision schema/version')
    need(manifest['label_patch_sha256'] == PATCH_SHA, 'Patch identity mismatch')
    need(manifest['pixel_contract_parent_sha256'] == f.CONTRACT_SHA256,
         'Pixel contract mismatch')
    need(manifest['test_records_included'] is False, 'Test records forbidden')
    need(tuple(manifest['class_order']) == CLASSES == tuple(parent['class_order']),
         'Class order mismatch')
    need(patch['data_version'] == VERSION and patch['video_id'] == TARGET
         and patch['split'] == 'train', 'Wrong correction target/split')
    need(patch['expected_original_class'] == '3p1'
         and patch['expected_original_class_id'] == 3
         and patch['effective_class'] == '2p1'
         and type(patch['effective_class_id']) is int and patch['effective_class_id'] == 1
         and type(patch['binary_label']) is int and patch['binary_label'] == 1,
         'Wrong correction labels')
    need(patch['decision_sha256'] == DECISION_SHA
         and decision['status'] == 'confirmed_by_user'
         and decision['video_id'] == TARGET and decision['original_label'] == '3p1'
         and decision['adjudicated_label'] == '2p1', 'User decision mismatch')
    ids = []
    for split, count in [('train', 7192), ('validation', 1547)]:
        current = manifest[split + '_ids']
        need(current == parent[split + '_ids'] and len(current) == count
             and len(set(current)) == count, 'Duplicate, reordered or changed split IDs')
        ids.extend(current)
    need(len(set(ids)) == 8739 and set(manifest['reference']) == set(ids),
         'Unknown, cross-split or duplicate IDs')
    need(TARGET in manifest['train_ids'] and TARGET not in manifest['validation_ids'],
         'Correction outside train')
    rows = {r['video_id']: r for r in original_rows}
    changed = []
    for vid in ids:
        before, after = parent['reference'][vid], manifest['reference'][vid]
        need(set(before) == set(after), 'Reference fields changed')
        need(type(after['class_id']) is int and 0 <= after['class_id'] < 8,
             'Invalid effective target')
        need(type(after['binary_label']) is int
             and after['binary_label'] == CLASS_TO_BINARY[after['class_id']],
             'Invalid binary target')
        original = rows[vid]
        need(original['original_class'] == CLASSES[before['class_id']]
             and original['audit_split'] == before['split']
             and original['source_group'] == before['source_group']
             and int(original['binary_label']) == before['binary_label'],
             'Original metadata mismatch')
        for key in before:
            if before[key] != after[key]:
                changed.append((vid, key))
    need(changed == [(TARGET, 'class_id')], 'Only one adjudicated class_id may change')
    need(parent['reference'][TARGET]['class_id'] == 3
         and manifest['reference'][TARGET]['class_id'] == 1, 'Target reversal')
    need(patch['source_sha256'] == parent['reference'][TARGET]['source_sha256']
         == decision['source_sha256'], 'Correction source SHA mismatch')


class LabelRevision:
    """Pinned metadata loaded once per dataset/collator; no video access."""
    def __init__(self, revision_directory=REVISION_DIR):
        for name, expected in PINNED_CODE.items():
            need(digest((f.ROOT / name).read_bytes()) == expected, f'Frozen loader changed: {name}')
        contract = f.load_contract()
        rows, _ = f.audit_rows(contract)
        directory = Path(revision_directory)
        manifest = _read_pinned(directory / 'manifest.json', MANIFEST_SHA)
        patch = _read_pinned(directory / 'label_patch.json', PATCH_SHA)
        # Decision location is fixed independently of the supplied patch path.
        decision = _read_pinned(REVISION_DIR.parent / 'adjudication15_confirmed_v1.json', DECISION_SHA)
        parent = json.loads((CURRENT / 'baseline8_formal_seed20261024_v1/plan.json').read_text())
        need(digest(json.dumps(parent, sort_keys=True, separators=(',', ':')).encode())
             == PARENT_PLAN_SHA, 'Historical parent plan changed')
        _validate_bundle(manifest, patch, parent, decision, rows)
        self._manifest = manifest
        self._original = {r['video_id']: r for r in rows
                          if r['audit_split'] in ('train', 'validation')}

    def ids(self, split):
        need(split in ('train', 'validation'), 'Only train/validation accepted')
        return tuple(self._manifest[split + '_ids'])

    def record(self, video_id, split):
        need(split in ('train', 'validation'), 'Only train/validation accepted')
        ref = self._manifest['reference'].get(video_id)
        need(ref is not None and ref['split'] == split, 'Unknown/cross-split/test/excluded ID')
        original = self._original[video_id]
        return dict(video_id=video_id, original_class=original['original_class'],
                    original_class_id=CLASSES.index(original['original_class']),
                    effective_class=CLASSES[ref['class_id']],
                    video_path=original['video_path'], **deepcopy(ref))


def _validate_item(item, revision):
    need(isinstance(item, dict) and {'video_id', 'split', 'original_class', 'original_class_id',
         'effective_class', 'class_id', 'binary_label', 'source_group', 'tensor', 'metadata'} <= set(item),
         'Expected a versioned adjudicated item')
    ref = revision.record(item['video_id'], item['split'])
    need(item['original_class'] == ref['original_class']
         and item['original_class_id'] == ref['original_class_id']
         and item['effective_class'] == ref['effective_class']
         and type(item['class_id']) is int and item['class_id'] == ref['class_id']
         and item['source_group'] == ref['source_group'], 'Effective target or provenance mismatch')
    # Validate frozen pixel/binary rules using the ORIGINAL target, without changing item.
    old_adapter._validate_item(dict(item, class_id=ref['original_class_id']))
    metadata = item['metadata']
    need(metadata.get('data_version') == VERSION
         and metadata.get('label_revision_sha256') == MANIFEST_SHA
         and metadata.get('label_patch_sha256') == PATCH_SHA, 'Missing/wrong label revision')
    need(metadata.get('source_sha256') == ref['source_sha256']
         and metadata.get('tensor_sha256') == ref['tensor_sha256']
         and metadata.get('source_size_bytes') == ref['source_size_bytes'], 'Input identity mismatch')
    need(not item['tensor'].requires_grad and tensor_sha(item['tensor']) == ref['tensor_sha256'],
         'Frozen input tensor changed')


def require_revision_identity(run_identity):
    """Future run/resume wiring must opt in; historical identities are rejected."""
    need(run_identity.get('data_version') == VERSION
         and run_identity.get('label_revision_sha256') == MANIFEST_SHA
         and run_identity.get('pixel_contract_parent_sha256') == f.CONTRACT_SHA256,
         'Run/resume data identity does not match the adjudicated revision')


class AdjudicatedDataset(Dataset):
    def __init__(self, *, source_root, split, video_ids=None, revision_directory=REVISION_DIR):
        need(split in ('train', 'validation'), 'Only train/validation accepted')
        self.revision = LabelRevision(revision_directory)
        ids = self.revision.ids(split) if video_ids is None else tuple(video_ids)
        need(len(ids) == len(set(ids)), 'Duplicate requested IDs')
        for vid in ids:
            self.revision.record(vid, split)
        self._ids, self.split = ids, split
        self._base = f.FoundationDataset(source_root=source_root, split=split, video_ids=ids)

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, index):
        vid = self._ids[index]
        ref = self.revision.record(vid, self.split)
        source_sha = f.baseline.guarded_source_sha256(self._base.scope, vid)
        need(source_sha == ref['source_sha256'], 'Source SHA changed before decode')
        item = self._base[index]
        old_adapter._validate_item(item)
        need(item['video_id'] == vid and item['split'] == self.split
             and item['class_id'] == ref['original_class_id']
             and item['original_class'] == ref['original_class'], 'Original delegate target mismatch')
        metadata = dict(item['metadata'], data_version=VERSION,
                        label_revision_sha256=MANIFEST_SHA, label_patch_sha256=PATCH_SHA,
                        source_sha256=source_sha, tensor_sha256=tensor_sha(item['tensor']))
        revised = dict(item, original_class_id=ref['original_class_id'],
                       effective_class=ref['effective_class'], class_id=ref['class_id'],
                       metadata=metadata)
        _validate_item(revised, self.revision)
        return revised


class AdjudicatedCollator:
    def __init__(self, revision_directory=REVISION_DIR):
        self.revision = LabelRevision(revision_directory)

    def __call__(self, items):
        need(bool(items), 'Empty batch')
        for item in items:
            _validate_item(item, self.revision)
        need(len({x['split'] for x in items}) == 1, 'Mixed splits')
        need(len({x['video_id'] for x in items}) == len(items), 'Duplicate batch IDs')
        return dict(
            tensor=torch.stack([x['tensor'] for x in items]),
            targets=torch.tensor([x['class_id'] for x in items], dtype=torch.long),
            binary_targets=torch.tensor([x['binary_label'] for x in items], dtype=torch.long),
            records=[{k: deepcopy(x[k]) for k in ('video_id', 'original_class', 'original_class_id',
                     'effective_class', 'source_group', 'split', 'metadata')} for x in items],
            data_version=VERSION, label_revision_sha256=MANIFEST_SHA,
            pixel_contract_parent_sha256=f.CONTRACT_SHA256)
