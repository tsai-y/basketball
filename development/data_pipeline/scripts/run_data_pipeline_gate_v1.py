#!/usr/bin/env python3
"""Dry-run or execute the bounded Data Pipeline Integration Gate v1 smoke."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
from torch.utils.data import DataLoader

from development.data_pipeline.pipeline import (
    AccessLedger,
    BasketballClipDataset,
    DEFAULT_CONFIG,
    TrainQualityIndex,
    load_gate_config,
)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def rank(seed: int, video_id: str, purpose: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{video_id}\0{purpose}".encode()).digest()


def smoke_ids(config) -> tuple[list[str], list[str]]:
    quality = TrainQualityIndex(config)
    fallback = sorted(
        video_id for video_id, row in quality.summaries.items() if row["has_clean_64"] == "false"
    )
    if len(fallback) != 5:
        raise ValueError("Expected exactly five fallback train videos")
    ordinary = min(
        (video_id for video_id, row in quality.summaries.items() if row["has_clean_64"] == "true"),
        key=lambda video_id: (rank(config.raw["seed"], video_id, "train_smoke"), video_id),
    )

    import csv
    with config.paths["manifest"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validation = []
    for label in ("0", "1"):
        pool = [row["video_id"] for row in rows if row["split"] == "validation" and
                row["excluded"] == "0" and row["binary_label"] == label]
        validation.append(min(pool, key=lambda video_id: (
            rank(config.raw["seed"], video_id, "validation_smoke"), video_id)))
    return fallback + [ordinary], validation


def make_datasets(config):
    train_ids, validation_ids = smoke_ids(config)
    ledger = AccessLedger()
    train = BasketballClipDataset(config, split="train", video_ids=train_ids, ledger=ledger)
    validation = BasketballClipDataset(
        config, split="validation", video_ids=validation_ids, ledger=ledger
    )
    return train, validation, ledger


def dry_run(config) -> dict[str, Any]:
    train, validation, ledger = make_datasets(config)
    stats = ledger.statistics()
    if any(value for split in stats.values() for value in split.values()):
        raise ValueError("Dry-run unexpectedly accessed a video")
    return {
        "schema_version": "basketball.data-pipeline-gate-v1.dry-run",
        "status": "DRY_RUN_PASS",
        "config_sha256": config.sha256,
        "samples": {"train": len(train), "validation": len(validation), "total": len(train) + len(validation)},
        "quality_sidecar": {"train_used": True, "validation_used": validation.quality_index is not None},
        "split_access_statistics": stats,
        "test_video_open_count": stats["test"]["opens"],
        "test_video_decode_count": stats["test"]["decodes"],
    }


def smoke(config) -> dict[str, Any]:
    train, validation, ledger = make_datasets(config)
    cv2.setNumThreads(1)
    samples = []
    required = {
        "tensor", "binary_label", "video_id", "split", "source_group", "shot_type", "fps",
        "start_frame", "end_frame_exclusive", "normalized_start", "candidate_overlap",
        "fallback", "fallback_reason", "sampler_policy", "sampler_version",
        "transform_policy", "transform_version", "normalization", "config_identity",
        "manifest_identity", "split_identity", "sidecar_identity",
    }
    for dataset in (train, validation):
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=lambda batch: batch)
        for batch in loader:
            sample = batch[0]
            if not required <= sample.keys():
                raise ValueError("Sample contract is incomplete")
            tensor = sample["tensor"]
            if tuple(tensor.shape) != (3, 64, 192, 256) or str(tensor.dtype) != "torch.float32":
                raise ValueError("Tensor shape/dtype mismatch")
            if not tensor.is_contiguous() or float(tensor.min()) < 0 or float(tensor.max()) > 1:
                raise ValueError("Tensor contiguity/range mismatch")
            if sample["decoding"]["decoded_frame_indices"] != list(
                range(sample["start_frame"], sample["end_frame_exclusive"])
            ):
                raise ValueError("Decoder/sampler interval mismatch")
            samples.append({
                "video_id": sample["video_id"],
                "split": sample["split"],
                "shape": list(tensor.shape),
                "fallback": sample["fallback"],
                "candidate_overlap": sample["candidate_overlap"],
                "sidecar_used": sample["sidecar_identity"]["used"],
            })
            del tensor, sample, batch

    fallback_samples = [sample for sample in samples if sample["fallback"]]
    if len(fallback_samples) != 5:
        raise ValueError("All five fallback videos were not returned")
    if any(sample["sidecar_used"] for sample in samples if sample["split"] == "validation"):
        raise ValueError("Validation used train quality sidecar")
    stats = ledger.statistics()
    if stats["test"] != {"opens": 0, "decodes": 0}:
        raise ValueError("Test access occurred")
    split_counts = Counter(sample["split"] for sample in samples)
    return {
        "schema_version": "basketball.data-pipeline-gate-v1.smoke",
        "status": "SMOKE_PASS",
        "config_sha256": config.sha256,
        "passed_items": [
            "pytorch_dataloader",
            "reference_tensor_contract",
            "decoder_sampler_half_open_interval",
            "five_fallback_videos",
            "validation_without_train_quality_sidecar",
            "zero_test_video_access",
            "no_model_no_optimizer_no_training",
        ],
        "sample_count": len(samples),
        "samples_by_split": dict(sorted(split_counts.items())),
        "shape": [3, 64, 192, 256],
        "dtype": "torch.float32",
        "fallback_count": len(fallback_samples),
        "split_access_statistics": stats,
        "test_video_open_count": stats["test"]["opens"],
        "test_video_decode_count": stats["test"]["decodes"],
        "samples": samples,
        "training": {"model_created": False, "optimizer_created": False, "optimizer_steps": 0},
    }


def validate_output_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    output = path.absolute()
    if output.is_relative_to(ROOT):
        allowed = (ROOT / "development/data_pipeline/artifacts").absolute()
        if not output.is_relative_to(allowed) or output == allowed:
            raise ValueError("Repository output is restricted to development/data_pipeline/artifacts")
    if any(parent.is_symlink() for parent in [output, *output.parents] if parent.exists()):
        raise ValueError("Symlink output path rejected")
    if output.exists():
        raise ValueError("Refusing to overwrite existing output")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional new JSON path; existing files are never overwritten")
    args = parser.parse_args(argv)
    config = load_gate_config(args.config)
    output = validate_output_path(args.output)
    result = dry_run(config) if args.dry_run else smoke(config)
    rendered = dumps(result)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
