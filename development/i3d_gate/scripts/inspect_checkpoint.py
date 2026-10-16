#!/usr/bin/env python3
"""Inventory and compare the official TF checkpoint and C03 conversion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from development.i3d_gate.gate import load_acquisition_manifest, load_config, load_model_strict


MIXED = {
    "branch_0": "Branch_0/Conv3d_0a_1x1",
    "branch_1.0": "Branch_1/Conv3d_0a_1x1",
    "branch_1.1": "Branch_1/Conv3d_0b_3x3",
    "branch_2.0": "Branch_2/Conv3d_0a_1x1",
    "branch_2.1": "Branch_2/Conv3d_0b_3x3",
    "branch_3.1": "Branch_3/Conv3d_0b_1x1",
}


def tf_base(pt_base: str) -> str:
    simple = {
        "conv3d_1a_7x7": "Conv3d_1a_7x7",
        "conv3d_2b_1x1": "Conv3d_2b_1x1",
        "conv3d_2c_3x3": "Conv3d_2c_3x3",
        "conv3d_0c_1x1": "Logits/Conv3d_0c_1x1",
    }
    if pt_base in simple:
        return simple[pt_base]
    mixed_name, branch = pt_base.split(".", 1)
    branch_name = MIXED[branch]
    if mixed_name == "mixed_5b" and branch == "branch_2.1":
        branch_name = "Branch_2/Conv3d_0a_3x3"
    return f"{mixed_name.replace('mixed_', 'Mixed_')}/{branch_name}"


def source_mapping(pt_key: str) -> tuple[str | None, str]:
    if ".conv3d." in pt_key:
        base, suffix = pt_key.split(".conv3d.")
        return f"RGB/inception_i3d/{tf_base(base)}/conv_3d/{'w' if suffix == 'weight' else 'b'}", ("transpose_4_3_0_1_2" if suffix == "weight" else "identity")
    base, suffix = pt_key.split(".batch3d.")
    if suffix == "weight":
        return None, "derived_constant_ones_sonnet_scale_false"
    tf_suffix = {"bias": "beta", "running_mean": "moving_mean", "running_var": "moving_variance"}[suffix]
    return f"RGB/inception_i3d/{tf_base(base)}/batch_norm/{tf_suffix}", "identity"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / "docs/i3d/checkpoint_tensor_inventory.json")
    parser.add_argument("--mapping", type=Path, default=ROOT / "docs/i3d/checkpoint_key_mapping.json")
    args = parser.parse_args(argv)
    for output in (args.inventory, args.mapping):
        if output.exists():
            raise ValueError(f"Refusing to overwrite: {output}")

    config = load_config()
    acquisition = load_acquisition_manifest(config)
    model, strict = load_model_strict(config)
    artifact_root = ROOT / config["artifact_directory"]
    converted = torch.load(artifact_root / "converted/model_rgb.pth", map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    inventory = []
    mapping = []
    checkpoint_prefix = str(artifact_root / "official_raw/model.ckpt")
    official_shapes = dict(tf.train.list_variables(checkpoint_prefix))
    reader = tf.train.load_checkpoint(checkpoint_prefix)
    used_official: list[str] = []
    mismatches = []
    for key, value in converted.items():
        kind = "parameter" if key in parameters else "buffer" if key in buffers else "unknown"
        inventory.append({
            "key": key, "shape": list(value.shape), "dtype": str(value.dtype), "kind": kind,
            "finite": bool(torch.isfinite(value).all()),
        })
        source, rule = source_mapping(key)
        if source is None:
            equal = bool(torch.equal(value, torch.ones_like(value)))
            source_shape = None
        else:
            raw = reader.get_tensor(source)
            source_shape = list(raw.shape)
            expected = np.transpose(raw, (4, 3, 0, 1, 2)) if rule.startswith("transpose") else raw
            equal = np.array_equal(value.cpu().numpy(), expected)
            used_official.append(source)
        if not equal:
            mismatches.append(key)
        mapping.append({
            "target_key": key, "target_shape": list(value.shape), "source_key": source,
            "source_shape": source_shape, "rule": rule, "exact_value_match": equal,
        })
    unused = sorted(set(official_shapes) - set(used_official))
    duplicate_uses = sorted({name for name in used_official if used_official.count(name) > 1})
    now = datetime.now(timezone.utc).isoformat()
    inventory_doc = {
        "schema_version": "basketball.i3d-gate-1c.tensor-inventory.v1",
        "generated_at_utc": now,
        "converted_checkpoint_sha256": next(x["sha256"] for x in acquisition["files"] if x["role"] == "converted_checkpoint"),
        "tensor_count": len(inventory),
        "parameter_count": sum(x["kind"] == "parameter" for x in inventory),
        "buffer_count": sum(x["kind"] == "buffer" for x in inventory),
        "input_channels": int(converted["conv3d_1a_7x7.conv3d.weight"].shape[1]),
        "first_kernel_shape": list(converted["conv3d_1a_7x7.conv3d.weight"].shape),
        "mixed_5c_output_channels": 1024,
        "classifier_present": True,
        "classifier_weight_shape": list(converted["conv3d_0c_1x1.conv3d.weight"].shape),
        "kinetics_class_count": int(converted["conv3d_0c_1x1.conv3d.weight"].shape[0]),
        "batch_norm_layer_count": sum(key.endswith(".batch3d.running_mean") for key in converted),
        "all_finite": all(item["finite"] for item in inventory),
        "strict_load": strict,
        "tensors": inventory,
    }
    mapping_doc = {
        "schema_version": "basketball.i3d-gate-1c.key-mapping.v1",
        "generated_at_utc": now,
        "official_source_tensor_count": len(official_shapes),
        "converted_target_tensor_count": len(converted),
        "official_tensors_used_once": len(used_official),
        "derived_constant_gamma_count": sum(item["source_key"] is None for item in mapping),
        "unused_official_tensors": unused,
        "duplicate_official_tensor_uses": duplicate_uses,
        "value_mismatches": mismatches,
        "all_exact": not unused and not duplicate_uses and not mismatches,
        "mapping": mapping,
    }
    args.inventory.write_text(json.dumps(inventory_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.mapping.write_text(json.dumps(mapping_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"inventory": str(args.inventory), "mapping": str(args.mapping), "strict": strict, "all_exact": mapping_doc["all_exact"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
