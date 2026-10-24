"""Read frozen video features without pooling away windows or time."""

import csv
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_sequences(cache_root, *, split, limit=None):
    """Return CPU features [N,3,8,1536], float labels, IDs, and cache run ID."""
    # Reject test and invalid arguments before touching any input paths.
    if split not in ("train", "validation"):
        raise ValueError("Only train/validation are allowed")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")

    cache_root = Path(cache_root)

    def read(path):
        return torch.load(path, map_location="cpu", weights_only=True)

    protocol = read(cache_root / "run.pt")
    run_id = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    _require(
        protocol["schema_version"] == "three_window_feature_cache_v1",
        "Unsupported feature-cache schema",
    )
    _require(protocol["quality_weighting_used"] is False, "Unexpected quality weighting")
    _require(
        protocol["quality_aware_selection_used"] is False,
        "Expected fixed-window features",
    )

    index = read(cache_root / split / "index.pt")
    _require(index["run_id"] == run_id, "Index/cache run ID mismatch")
    _require(index["split"] == split, "Index split mismatch")
    video_ids = index["video_ids"]
    _require(index["count"] == len(video_ids) > 0, "Invalid or empty split index")
    _require(len(set(video_ids)) == len(video_ids), "Duplicate index video IDs")

    manifest = ROOT / "data/processed/dataset_manifest.csv"
    _require(
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        == protocol["input_identities"]["manifest_sha256"],
        "Manifest identity mismatch",
    )
    with manifest.open(newline="", encoding="utf-8") as stream:
        manifest_rows = [
            row for row in csv.DictReader(stream)
            if row["split"] == split and row["excluded"] == "0"
        ]
    rows = {row["video_id"]: row for row in manifest_rows}
    _require(len(rows) == len(manifest_rows), "Duplicate manifest video IDs")
    _require(video_ids == list(rows), "Index/manifest video IDs differ")
    if limit is not None:
        video_ids = video_ids[:limit]

    sequences, labels = [], []
    for video_id in video_ids:
        filename = hashlib.sha256(video_id.encode()).hexdigest() + ".pt"
        record = read(cache_root / split / filename)
        _require(record["run_id"] == run_id, "Record/cache run ID mismatch")
        _require(record["video_id"] == video_id, "Record video ID mismatch")
        _require(record["split"] == split, "Record split mismatch")
        label = int(rows[video_id]["binary_label"])
        _require(label in (0, 1), "Invalid binary label")
        _require(record["binary_label"] == label, "Record label mismatch")
        _require(
            record["quality_aware_selection_used"] is False,
            "Record is not fixed-window sampling",
        )

        features = record["features"]
        _require(isinstance(features, torch.Tensor), "Features must be a tensor")
        _require(tuple(features.shape) == (3, 8, 1536), "Feature shape mismatch")
        _require(features.dtype == torch.float32, "Feature dtype mismatch")
        _require(not features.requires_grad, "Cached features must be frozen")
        _require(torch.isfinite(features).all().item(), "Non-finite features")
        digest = hashlib.sha256(
            features.contiguous().numpy().tobytes()
        ).hexdigest()
        _require(digest == record["features_sha256"], "Feature hash mismatch")

        # No mean(), reshuffling, or additional RGB normalization here.
        sequences.append(features)
        labels.append(label)

    return (
        torch.stack(sequences),
        torch.tensor(labels, dtype=torch.float32),
        video_ids,
        run_id,
    )
