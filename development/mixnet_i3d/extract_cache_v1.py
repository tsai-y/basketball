import argparse
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import timm
import torch

from development.data_pipeline.pipeline import load_gate_config
from development.data_pipeline.multiwindow_v1 import (
    BasketballThreeWindowDataset,
    select_covering_windows,
)
from development.mixnet_i3d.backbone import MixNetI3DBackbone
from development.mixnet_i3d.temporal_features import ThreeWindowFeatureExtractor


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
WEIGHT_SHA = "d8a8162d92ebb5270770d745c7a424126329f3a69a43edeeba0d3a99d72416d4"


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def save_new(payload, path):
    """先寫暫存檔，再建立完整檔名；已有檔案時不覆寫。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.pt")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def validate(record, row, split, run_id):
    assert record["run_id"] == run_id, "Cache version mismatch"
    assert record["video_id"] == row["video_id"], "Video ID mismatch"
    assert record["binary_label"] == int(row["binary_label"]), "Label mismatch"
    assert record["split"] == split, "Split mismatch"
    assert record["windows"] == select_covering_windows(record["frame_count"])
    assert record["quality_aware_selection_used"] is False

    features = record["features"]
    assert tuple(features.shape) == (3, 8, 1536), "Feature shape mismatch"
    assert features.dtype == torch.float32, "Feature dtype mismatch"
    assert torch.isfinite(features).all().item(), "Non-finite features"
    assert tensor_sha(features) == record["features_sha256"], "Feature hash mismatch"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    weight_path = FOLDER / "artifacts/frozen_extractor_v1.pt"
    assert file_sha(weight_path) == WEIGHT_SHA, "Frozen checkpoint changed"
    frozen = load(weight_path)
    assert frozen["schema_version"] == "frozen_extractor_v1"
    assert frozen["timm_version"] == timm.__version__, "timm version changed"

    for name, expected in frozen["source_sha256"].items():
        assert file_sha(FOLDER / name) == expected, f"Model code changed: {name}"

    config = load_gate_config()
    dataset = BasketballThreeWindowDataset(config, split=args.split, epoch=0)

    # 固定推論設定；不使用 AMP 或 TF32。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    sources = [
        *FOLDER.glob("*.py"),
        ROOT / "development/data_pipeline/pipeline.py",
        ROOT / "development/data_pipeline/multiwindow_v1.py",
        ROOT / "development/data_quality/transforms/video_clip_transforms.py",
        ROOT / "development/data_quality/samplers/quality_aware_clip_sampler.py",
        ROOT / "development/data_quality/scripts/audit_quality_aware_sampler_v1.py",
        ROOT / "scripts/audit_i3d_temporal_feasibility.py",
    ]

    protocol = {
        "schema_version": "three_window_feature_cache_v1",
        "weight_sha256": WEIGHT_SHA,
        "gate_config_sha256": config.sha256,
        "input_identities": config.raw["identities"],
        "source_sha256": {
            str(path.relative_to(ROOT)): file_sha(path)
            for path in sorted(sources)
        },
        "versions": {
            "torch": str(torch.__version__),
            "timm": timm.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "cuda": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
        },
        "device": torch.cuda.get_device_name(0),
        "precision": "float32_no_amp_no_tf32",
        "window_policy": "three_window_coverage_v1",
        "clips_per_video": 3,
        "epoch": 0,
        "transform": "reference",
        "quality_aware_selection_used": False,
        "quality_weighting_used": False,
    }
    encoded = json.dumps(protocol, sort_keys=True).encode()
    run_id = hashlib.sha256(encoded).hexdigest()
    cache_root = FOLDER / "artifacts/feature_cache_v1" / run_id[:16]
    manifest_path = cache_root / "run.pt"

    if manifest_path.exists():
        assert load(manifest_path) == protocol, "Run manifest mismatch"
    else:
        save_new(protocol, manifest_path)

    # 先建架構，再完整載入固定擷取器權重。
    old = timm.create_model(frozen["model_name"], pretrained=False)
    extractor = ThreeWindowFeatureExtractor(
        MixNetI3DBackbone(old),
        mean=frozen["mean"],
        std=frozen["std"],
    )
    extractor.load_state_dict(frozen["state_dict"], strict=True)
    extractor = extractor.cuda().eval()

    total = len(dataset)
    count = total if args.limit is None else min(args.limit, total)
    saved = skipped = 0
    print("Cache root:", cache_root)
    print(f"Split: {args.split}; requested: {count}/{total}", flush=True)

    for index in range(count):
        row = dataset.base.rows[index]
        filename = hashlib.sha256(row["video_id"].encode()).hexdigest() + ".pt"
        path = cache_root / args.split / filename

        if path.exists():
            validate(load(path), row, args.split, run_id)
            skipped += 1
        else:
            item = dataset[index]
            features = extractor(
                item["tensor"].unsqueeze(0).cuda()
            )[0].cpu().contiguous()

            record = {
                "run_id": run_id,
                "video_id": item["video_id"],
                "binary_label": item["binary_label"],
                "split": item["split"],
                "features": features,
                "features_sha256": tensor_sha(features),
                "windows": item["windows"],
                "frame_count": int(item["frame_count"]),
                "fps": float(item["fps"]),
                "quality_aware_selection_used": False,
                "candidate_overlap_frames": item["candidate_overlap_frames"],
                "quality_metadata_available": item["quality_metadata_available"],
            }
            validate(record, row, args.split, run_id)
            save_new(record, path)
            validate(load(path), row, args.split, run_id)
            saved += 1

        if index < 2 or (index + 1) % 100 == 0 or index + 1 == count:
            print(
                f"{index + 1}/{count}: saved={saved}, skipped={skipped}",
                flush=True,
            )

    # 只有不限制數量、且全部驗證完成，才產生完整索引。
    if args.limit is None:
        index_record = {
            "run_id": run_id,
            "split": args.split,
            "video_ids": [row["video_id"] for row in dataset.base.rows],
            "count": total,
        }
        index_path = cache_root / args.split / "index.pt"
        if index_path.exists():
            assert load(index_path) == index_record, "Index mismatch"
        else:
            save_new(index_record, index_path)
        print("Complete split index:", index_path)

    print(f"DONE: saved={saved}, skipped={skipped}")


if __name__ == "__main__":
    main()
