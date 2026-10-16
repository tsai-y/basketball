#!/usr/bin/env python3
"""Run I3D Checkpoint Compatibility Gate 1C forward-only smoke."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from development.data_pipeline.pipeline import AccessLedger, BasketballClipDataset, load_gate_config
from development.data_pipeline.scripts.run_data_pipeline_gate_v1 import smoke_ids
from development.i3d_gate.gate import (
    DEFAULT_CONFIG, gap, load_acquisition_manifest, load_config, load_model_strict,
    normalize_i3d_rgb, sha256_file, tensor_digest, tensor_summary,
)


def one_forward(model, value: torch.Tensor) -> tuple[dict[str, torch.Tensor], float]:
    started = time.perf_counter()
    with torch.inference_mode():
        endpoints = model.extract_endpoints(value, ("Conv3d_1a_7x7", "Mixed_5c"))
        endpoints["GAP"] = gap(endpoints["Mixed_5c"])
    return endpoints, time.perf_counter() - started


def check_path(model, value: torch.Tensor, name: str) -> dict[str, Any]:
    first, first_seconds = one_forward(model, value)
    second, second_seconds = one_forward(model, value)
    equal = {key: bool(torch.equal(first[key], second[key])) for key in first}
    if not all(equal.values()):
        raise ValueError(f"Repeated {name} forward was not bitwise deterministic: {equal}")
    feature = first["Mixed_5c"]
    in_t, in_h, in_w = value.shape[-3:]
    out_t, out_h, out_w = feature.shape[-3:]
    if in_t % out_t or in_h % out_h or in_w % out_w:
        raise ValueError(f"Non-integral endpoint stride for {name}")
    return {
        "name": name,
        "input": tensor_summary(value),
        "early_endpoint": {"name": "Conv3d_1a_7x7", **tensor_summary(first["Conv3d_1a_7x7"])},
        "feature_endpoint": {"name": "Mixed_5c", **tensor_summary(feature)},
        "pooled": tensor_summary(first["GAP"]),
        "temporal_stride": in_t // out_t,
        "spatial_stride": [in_h // out_h, in_w // out_w],
        "channels": int(feature.shape[1]),
        "repeated_forward_bitwise_equal": equal,
        "forward_seconds": [first_seconds, second_seconds],
    }


def real_smoke(model, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    pipeline_config = load_gate_config()
    train_pool, validation_pool = smoke_ids(pipeline_config)
    train_ids = train_pool[:config["smoke"]["train_samples"]]
    validation_ids = validation_pool[:config["smoke"]["validation_samples"]]
    ledger = AccessLedger()
    datasets = (
        BasketballClipDataset(pipeline_config, split="train", video_ids=train_ids, ledger=ledger),
        BasketballClipDataset(pipeline_config, split="validation", video_ids=validation_ids, ledger=ledger),
    )
    records = []
    for dataset in datasets:
        for index in range(len(dataset)):
            sample = dataset[index]
            source = sample["tensor"]
            before = tensor_digest(source)
            normalized, adapter = normalize_i3d_rgb(source)
            if tensor_digest(source) != before:
                raise ValueError("Real sample changed during normalization")
            with torch.inference_mode():
                feature = model.forward_features(normalized.unsqueeze(0))
                pooled = gap(feature)
            records.append({
                "video_id": sample["video_id"],
                "split": sample["split"],
                "source_shape": list(source.shape),
                "model_input_shape": list(normalized.unsqueeze(0).shape),
                "source_range": [float(source.min()), float(source.max())],
                "model_input_range": [float(normalized.min()), float(normalized.max())],
                "normalization": adapter,
                "feature_endpoint": {"name": "Mixed_5c", **tensor_summary(feature)},
                "pooled": tensor_summary(pooled),
                "sampling": {
                    "start_frame": sample["start_frame"],
                    "end_frame_exclusive": sample["end_frame_exclusive"],
                    "sampler_policy": sample["sampler_policy"],
                    "fallback": sample["fallback"],
                },
                "transform": {
                    "policy": sample["transform_policy"], "version": sample["transform_version"],
                    "pipeline_normalization": sample["normalization"],
                },
            })
    return records, ledger.statistics()


def runtime() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/i3d/i3d_gate_1c_forward_smoke_summary.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate artifacts/model/split construction without opening videos or forwarding")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite: {args.output}")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    config = load_config(args.config)
    config_identity = {
        "path": str(args.config.resolve().relative_to(ROOT)),
        "sha256": sha256_file(args.config.resolve()),
        "version": config["version"],
    }
    acquisition = load_acquisition_manifest(config)
    model, strict = load_model_strict(config)
    if model.training or any(module.training for module in model.modules() if isinstance(module, torch.nn.Dropout)):
        raise ValueError("eval/dropout contract failed")
    bn_before = {key: value.clone() for key, value in model.named_buffers()}

    pipeline_config = load_gate_config()
    train_pool, validation_pool = smoke_ids(pipeline_config)
    dry_ledger = AccessLedger()
    BasketballClipDataset(pipeline_config, split="train", video_ids=train_pool[:1], ledger=dry_ledger)
    BasketballClipDataset(pipeline_config, split="validation", video_ids=validation_pool[:1], ledger=dry_ledger)
    if dry_ledger.statistics()["test"] != {"opens": 0, "decodes": 0}:
        raise ValueError("Dry-run test access occurred")
    if args.dry_run:
        result = {
            "schema_version": "basketball.i3d-gate-1c.dry-run.v1", "status": "DRY_RUN_PASS",
            "strict_load": strict, "artifact_count": len(acquisition["files"]),
            "config_identity": config_identity,
            "split_access_statistics": dry_ledger.statistics(), "runtime": runtime(),
        }
    else:
        generator = torch.Generator(device="cpu").manual_seed(config["smoke"]["seed"])
        official_input = torch.rand((1, 3, 64, 224, 224), generator=generator, dtype=torch.float32).mul(2.0).sub(1.0).contiguous()
        official = check_path(model, official_input, "official_reference")
        del official_input
        project_source = torch.rand((1, 3, 64, 192, 256), generator=generator, dtype=torch.float32).contiguous()
        source_digest = tensor_digest(project_source)
        project_input, adapter = normalize_i3d_rgb(project_source)
        if tensor_digest(project_source) != source_digest:
            raise ValueError("Synthetic project source changed during normalization")
        project = check_path(model, project_input, "basketball51_project")
        project["source_before_adapter"] = tensor_summary(project_source)
        project["normalization"] = adapter
        del project_source, project_input

        real_records, access = real_smoke(model, config)
        if access["test"] != {"opens": 0, "decodes": 0}:
            raise ValueError("Test access occurred")
        bn_after = dict(model.named_buffers())
        bn_unchanged = all(torch.equal(before, bn_after[key]) for key, before in bn_before.items())
        if not bn_unchanged:
            raise ValueError("BatchNorm running statistics changed")
        result = {
            "schema_version": "basketball.i3d-gate-1c.forward-smoke.v1",
            "status": "PASS",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_identity": config_identity,
            "candidate": {
                "gate_1b_id": "C03", "upstream_gate_1b_id": "A01",
                "architecture": config["candidate"]["architecture"],
                "implementation_commit": config["candidate"]["implementation"]["commit"],
                "upstream_commit": config["candidate"]["official_upstream"]["commit"],
            },
            "artifact_sha256": {record["filename"]: record["sha256"] for record in acquisition["files"]},
            "strict_load": strict,
            "synthetic": {"official_reference": official, "basketball51_project": project},
            "real_smoke": real_records,
            "normalization_contract": adapter,
            "execution_contract": {
                "model_eval": not model.training, "inference_mode": True,
                "dropout_disabled": all(not module.training for module in model.modules() if isinstance(module, torch.nn.Dropout)),
                "batch_norm_running_statistics_unchanged": bn_unchanged,
                "optimizer_created": False, "optimizer_steps": 0, "backward_calls": 0,
                "gradient_accumulation_steps": 0, "training_samples": 0,
            },
            "split_access_statistics": access,
            "test_video_open_count": access["test"]["opens"],
            "test_video_decode_count": access["test"]["decodes"],
            "cross_framework_numerical_parity": {
                "status": "NOT_RUN",
                "reason": "Pinned official Sonnet 1.x runtime is unavailable; installed TensorFlow 2.19 alone is not the historical reference runtime.",
                "absolute_tolerance_predeclared": config["numerical_parity"]["absolute_tolerance"],
                "relative_tolerance_predeclared": config["numerical_parity"]["relative_tolerance"],
                "endpoints_predeclared": config["numerical_parity"]["endpoints"],
                "conversion_tensor_value_mapping": "PASS_EXACT",
            },
            "runtime": runtime(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
