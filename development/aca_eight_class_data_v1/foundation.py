"""Frozen native-frame baseline inputs; metadata overlay only, no training.

The inactive virtual25 candidate is deliberately not imported here. All pixel
operations are delegated to the original baseline implementation.
"""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from development.aca_net_v1 import data as baseline

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / 'docs/research/specs/input_foundation_v1.json'
CONTRACT_SHA256 = '544f7de95104bc38f5168996ab40e1a46020971ec97923a5236f7b4f1780549e'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load_contract(path=CONTRACT_PATH):
    raw = Path(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == CONTRACT_SHA256, 'Foundation contract changed')
    contract = json.loads(raw)
    for name, expected in contract['immutable_dependencies_sha256'].items():
        need(baseline.file_sha256(ROOT / name) == expected, f'Frozen dependency changed: {name}')
    return contract


def audit_rows(contract):
    """Derive all splits from pinned metadata; do not inspect any source path."""
    path = ROOT / 'data/processed/dataset_manifest.csv'
    need(baseline.file_sha256(path) == contract['original_manifest_sha256'], 'Manifest changed')
    with path.open(newline='', encoding='utf-8') as f:
        original = list(csv.DictReader(f))
    deny = set(contract['quarantined_ambiguous_ids'])
    need(deny <= {r['video_id'] for r in original}, 'Unknown quarantine ID')
    rows = []
    for row in original:
        excluded = row['split'] == 'excluded' or row['video_id'] in deny
        rows.append(dict(
            video_id=row['video_id'], video_path=row['video_path'],
            original_class=row['original_class'], class_id=contract['classes'].index(row['original_class']),
            binary_label=row['binary_label'], source_group=row['source_group'],
            original_split=row['split'], audit_split='excluded' if excluded else row['split'],
            excluded=int(excluded), exclusion_reason='ambiguous_eight_class_label'
            if row['video_id'] in deny else row['exclusion_reason']))
    need(dict(Counter(r['audit_split'] for r in rows)) == contract['counts'], 'Cohort count mismatch')
    groups = {}
    for row in rows:
        if not row['excluded']:
            groups.setdefault(row['source_group'], set()).add(row['audit_split'])
    need(all(len(v) == 1 for v in groups.values()), 'Source group crosses splits')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    serialized = stream.getvalue().encode('utf-8')
    need(hashlib.sha256(serialized).hexdigest() == contract['audit_manifest_sha256'],
         'Derived cohort differs from reviewed cohort')
    return rows, serialized


class FoundationScope:
    def __init__(self, *, source_root, split, video_ids=None):
        # Reject test before contract reads, path conversion, or baseline access.
        need(split in ('train', 'validation'), 'Only train/validation accepted')
        self.contract = load_contract()
        rows, _ = audit_rows(self.contract)
        allowed_ids = {r['video_id'] for r in rows if r['audit_split'] == split}
        self._baseline = baseline.VideoScope(source_root=source_root, split=split)
        self._allowed = {r['video_id']: dict(r) for r in self._baseline.rows if r['video_id'] in allowed_ids}
        need(set(self._allowed) == allowed_ids, 'Baseline/overlay scope mismatch')
        requested = sorted(allowed_ids) if video_ids is None else list(video_ids)
        need(len(requested) == len(set(requested)) and all(i in self._allowed for i in requested),
             'Duplicate, cross-split, test or excluded requested ID')
        self.rows = tuple(dict(self._allowed[i]) for i in requested)
        self.split = split

    def authorize(self, video_id):
        need(video_id in self._allowed, 'Unknown/cross-split/test/excluded ID rejected before path access')
        return self._baseline.authorize(video_id)


class FoundationDataset(Dataset):
    """One deterministic full-clip tensor plus eight-class and binary labels."""
    def __init__(self, *, source_root, split, video_ids=None):
        self.scope = FoundationScope(source_root=source_root, split=split, video_ids=video_ids)

    def __len__(self):
        return len(self.scope.rows)

    def __getitem__(self, index):
        video_id = self.scope.rows[index]['video_id']
        rgb, metadata, row = baseline.decode_uniform50(self.scope, video_id)
        tensor = baseline.normalize_rgb(rgb)
        need(tuple(tensor.shape) == (3, 50, 120, 160) and tensor.dtype == torch.float32 and
             tensor.device.type == 'cpu' and tensor.is_contiguous() and
             torch.isfinite(tensor).all().item(), 'Invalid foundation tensor')
        metadata = dict(metadata, times_are_nominal_index_over_fps_not_verified_pts=True,
                        input_contract_sha256=CONTRACT_SHA256)
        return dict(tensor=tensor, class_id=self.scope.contract['classes'].index(row['original_class']),
                    binary_label=int(row['binary_label']), video_id=video_id,
                    original_class=row['original_class'], source_group=row['source_group'],
                    split=row['split'], metadata=metadata)
