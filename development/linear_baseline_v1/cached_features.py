import csv
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def load_vectors(cache_root, *, split, limit=None):
    # 必須在讀取任何檔案之前拒絕 test。
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

    assert protocol["schema_version"] == "three_window_feature_cache_v1"
    assert protocol["quality_weighting_used"] is False

    index = read(cache_root / split / "index.pt")
    assert index["run_id"] == run_id
    assert index["split"] == split
    assert index["count"] == len(index["video_ids"])
    assert len(set(index["video_ids"])) == index["count"]

    # 核對資料清單版本，以及影片的切分與標籤。
    manifest = ROOT / "data/processed/dataset_manifest.csv"
    expected_hash = protocol["input_identities"]["manifest_sha256"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == expected_hash

    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = {
            row["video_id"]: row
            for row in csv.DictReader(stream)
            if row["split"] == split and row["excluded"] == "0"
        }

    assert index["video_ids"] == list(rows)
    video_ids = index["video_ids"]
    if limit is not None:
        video_ids = video_ids[:limit]

    vectors, labels = [], []
    for video_id in video_ids:
        filename = hashlib.sha256(video_id.encode()).hexdigest() + ".pt"
        record = read(cache_root / split / filename)

        assert record["run_id"] == run_id
        assert record["video_id"] == video_id
        assert record["split"] == split
        assert record["binary_label"] == int(rows[video_id]["binary_label"])

        features = record["features"]
        assert tuple(features.shape) == (3, 8, 1536)
        assert features.dtype == torch.float32
        assert torch.isfinite(features).all().item()

        digest = hashlib.sha256(
            features.contiguous().numpy().tobytes()
        ).hexdigest()
        assert digest == record["features_sha256"]

        vectors.append(features.mean(dim=(0, 1)))
        labels.append(record["binary_label"])

    return (
        torch.stack(vectors),
        torch.tensor(labels, dtype=torch.float32),
        video_ids,
        run_id,
    )
