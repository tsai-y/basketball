"""Reuse the fixed RGB three-window dataset; expose only training inputs."""

from collections import Counter
import hashlib
import json

import torch
from torch.utils.data import Dataset

from development.data_pipeline.multiwindow_v1 import BasketballThreeWindowDataset
from development.temporal_baseline_v1.train_temporal import require


class RGBTrainingDataset(Dataset):
    def __init__(self, gate_config, *, split):
        if split not in ("train", "validation"):
            raise ValueError("Only train/validation are allowed; test rejected before access")
        self.source = BasketballThreeWindowDataset(gate_config, split=split, epoch=0)
        self.rows = self.source.base.rows
        self.split = split

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        item = self.source[index]
        x = item["tensor"]
        require(tuple(x.shape) == (3, 3, 64, 192, 256) and x.dtype == torch.float32,
                "Unexpected RGB sample contract")
        require(item["video_id"] == self.rows[index]["video_id"] and item["split"] == self.split,
                "Sample identity mismatch")
        require(int(item["binary_label"]) == int(self.rows[index]["binary_label"]), "Sample label mismatch")
        require(item["quality_aware_selection_used"] is False, "Expected fixed three windows")
        require(torch.isfinite(x).all().item() and x.min() >= 0 and x.max() <= 1, "Invalid RGB range")
        return {"tensor": x, "binary_label": int(item["binary_label"]), "video_id": item["video_id"]}


def check_split_metadata(train, validation):
    """Check the complete split lists without decoding all videos."""
    info = {}
    for dataset, count, counts in ((train, 7193, {0: 3421, 1: 3772}),
                                    (validation, 1547, {0: 740, 1: 807})):
        ids = [r["video_id"] for r in dataset.rows]
        require(len(dataset) == len(ids) == count and len(set(ids)) == count,
                f"Unexpected {dataset.split} IDs/count")
        require(all(r["split"] == dataset.split and r["excluded"] == "0" for r in dataset.rows),
                "Out-of-split/excluded sample")
        actual_counts = Counter(int(r["binary_label"]) for r in dataset.rows)
        require(dict(actual_counts) == counts, "Label distribution changed")
        info[dataset.split] = {"videos": count, "label_counts": dict(actual_counts),
                               "ordered_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    require(set(r["video_id"] for r in train.rows).isdisjoint(r["video_id"] for r in validation.rows),
            "Overlapping train/validation video IDs")
    require(set(r["source_group"] for r in train.rows).isdisjoint(r["source_group"] for r in validation.rows),
            "Overlapping train/validation source groups")
    info["source_groups_disjoint"] = True
    return info
