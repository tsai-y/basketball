"""CPU-only pre-training checks; never steps an optimizer or writes checkpoints."""

from collections import Counter
import csv
import json

import torch
from torch import nn

from development.linear_baseline_v1.cached_features import load_vectors
from development.temporal_baseline_v1.cached_features import ROOT, load_sequences
from development.temporal_baseline_v1.model import WindowTemporalTransformer


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    torch.manual_seed(20261024)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cache = ROOT / (
        "development/mixnet_i3d/artifacts/feature_cache_v1/67b6400e0982bb6d"
    )

    print("Loading and checking full train/validation caches...", flush=True)
    train_x, train_y, train_ids, run_id = load_sequences(cache, split="train")
    val_x, val_y, val_ids, val_run_id = load_sequences(cache, split="validation")
    require(run_id == val_run_id, "Train/validation cache identities differ")
    require(set(train_ids).isdisjoint(val_ids), "Overlapping train/validation IDs")
    require(tuple(train_x.shape) == (7193, 3, 8, 1536), "Unexpected train shape")
    require(tuple(val_x.shape) == (1547, 3, 8, 1536), "Unexpected validation shape")
    require(not train_x.requires_grad and not val_x.requires_grad, "Features must stay frozen")

    with (ROOT / "data/processed/dataset_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = {row["video_id"]: row for row in csv.DictReader(stream)}
    train_groups = {rows[video_id]["source_group"] for video_id in train_ids}
    val_groups = {rows[video_id]["source_group"] for video_id in val_ids}
    require(train_groups.isdisjoint(val_groups), "Overlapping source groups")

    # Compare with the unmodified Linear reader on the same real videos.
    for split, x, y, ids in (
        ("train", train_x, train_y, train_ids),
        ("validation", val_x, val_y, val_ids),
    ):
        pooled, old_y, old_ids, old_run_id = load_vectors(cache, split=split, limit=2)
        require(old_run_id == run_id and old_ids == ids[:2], "Baseline sample identity differs")
        torch.testing.assert_close(old_y, y[:2], rtol=0, atol=0)
        torch.testing.assert_close(pooled, x[:2].mean(dim=(1, 2)))

    model = WindowTemporalTransformer().cpu()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        val_logits = model(val_x[:2])
    require(tuple(val_logits.shape) == (2,), "Unexpected validation output shape")
    require(torch.isfinite(val_logits).all().item(), "Non-finite validation output")

    # One Miss and one Make from train only. Backward checks gradients, not learning.
    indices = torch.tensor([
        int((train_y == label).nonzero(as_tuple=True)[0][0]) for label in (0, 1)
    ])
    model.train()
    logits = model(train_x[indices])
    loss = nn.BCEWithLogitsLoss()(logits, train_y[indices])
    require(torch.isfinite(loss).item(), "Non-finite preflight loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None, f"Missing gradient: {name}")
        require(torch.isfinite(parameter.grad).all().item(), f"Non-finite gradient: {name}")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    try:
        load_sequences(cache, split="test")
    except ValueError:
        pass
    else:
        raise RuntimeError("test split was not rejected")

    print(json.dumps({
        "status": "PASS",
        "cache_run_id": run_id,
        "train_shape": list(train_x.shape),
        "validation_shape": list(val_x.shape),
        "train_labels": dict(Counter(int(v) for v in train_y.tolist())),
        "validation_labels": dict(Counter(int(v) for v in val_y.tolist())),
        "source_groups_disjoint": True,
        "baseline_feature_consistency": True,
        "real_validation_forward": "PASS",
        "real_train_backward": "PASS",
        "model_weights_unchanged": True,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "test_rejected": True,
        "optimizer_steps": 0,
        "training_started": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
