"""Train-only original-class balancing, with reproducible draw audits."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler


CLASSES = ("2p0", "2p1", "3p0", "3p1", "ft0", "ft1", "mp0", "mp1")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def aligned_groups(video_ids, labels, *, split, manifest_path, manifest_sha256):
    """Read split metadata only; validation is supported for final diagnostics."""
    if split not in ("train", "validation"):
        raise ValueError("Only train/validation metadata are allowed")
    require(len(video_ids) > 0 and len(set(video_ids)) == len(video_ids), "Invalid video IDs")
    require(tuple(labels.shape) == (len(video_ids),), "Label shape mismatch")
    require(set(labels.tolist()) <= {0., 1.}, "Invalid binary labels")
    path = Path(manifest_path)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == manifest_sha256,
            "Manifest identity mismatch")
    with path.open(newline="", encoding="utf-8") as stream:
        selected = [r for r in csv.DictReader(stream)
                    if r["split"] == split and r["excluded"] == "0"]
    rows = {r["video_id"]: r for r in selected}
    require(len(rows) == len(selected), "Duplicate manifest IDs")
    require(set(video_ids) == set(rows), "IDs must match the full requested split")
    groups = []
    for video_id, label in zip(video_ids, labels.tolist()):
        row = rows[video_id]
        group = row["original_class"]
        require(group in CLASSES, "Unknown original class")
        require(int(row["binary_label"]) == int(group[-1]) == label,
                "Original-class/binary-label mismatch")
        groups.append(group)
    return groups


class AuditedWeightedSampler(WeightedRandomSampler):
    """WeightedRandomSampler plus the actual last epoch's draw counts."""

    def __init__(self, weights, groups, *, seed):
        super().__init__(weights, num_samples=len(groups), replacement=True,
                         generator=torch.Generator().manual_seed(seed))
        self.groups = tuple(groups)
        self.last_counts = {}
        self.last_unique_count = 0

    def __iter__(self):
        indices = list(super().__iter__())
        self.last_counts = dict(Counter(self.groups[i] for i in indices))
        self.last_unique_count = len(set(indices))
        yield from indices

    def preview(self):
        """Preview epoch one without consuming any sampler RNG state."""
        state = self.generator.get_state()
        counts, unique = self.last_counts, self.last_unique_count
        try:
            list(self)
            return {"draw_counts": self.last_counts.copy(),
                    "unique_videos": self.last_unique_count}
        finally:
            self.generator.set_state(state)
            self.last_counts, self.last_unique_count = counts, unique


def make_balanced_sampler(video_ids, labels, *, manifest_path,
                          manifest_sha256, seed, split="train"):
    # Reject other splits before reading paths. No validation/test frequencies enter weights.
    if split != "train":
        raise ValueError("Balanced sampling is train-only")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    groups = aligned_groups(video_ids, labels, split="train", manifest_path=manifest_path,
                            manifest_sha256=manifest_sha256)
    counts = Counter(groups)
    require(set(counts) == set(CLASSES), "Train must contain all eight original classes")
    weights = torch.tensor([1. / counts[g] for g in groups], dtype=torch.float64)
    sampler = AuditedWeightedSampler(weights, groups, seed=seed)
    info = {
        "method": "inverse_train_original_class_frequency",
        "weight_formula": "1 / train_count[original_class]",
        "class_counts": {g: counts[g] for g in CLASSES},
        "expected_class_probability": {g: 1. / len(CLASSES) for g in CLASSES},
        "replacement": True, "num_samples_per_epoch": len(groups),
        "manifest_sha256": manifest_sha256,
        "aligned_mapping_sha256": hashlib.sha256(
            json.dumps(list(zip(video_ids, groups))).encode()).hexdigest(),
        "loss_weighting_used": False,
    }
    return sampler, info
