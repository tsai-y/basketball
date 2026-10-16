"""Core contracts for I3D Checkpoint Compatibility Gate 1C."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from development.i3d_gate.models import InceptionI3d


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "development/i3d_gate/configs/i3d_gate_1c.yaml"


class GateError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().to(torch.float32).numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().to(torch.float64)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all()),
        "min": float(detached.min()),
        "max": float(detached.max()),
        "mean": float(detached.mean()),
        "std_population": float(detached.std(unbiased=False)),
        "l1_norm": float(detached.abs().sum()),
        "l2_norm": float(torch.linalg.vector_norm(detached)),
        "sha256_float32_le": tensor_digest(value),
    }


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    need(config.get("version") == "i3d_checkpoint_compatibility_gate_1c", "Unsupported Gate config")
    need(config["candidate"]["gate_1b_id"] == "C03", "Candidate changed")
    need(config["candidate"]["upstream_gate_1b_id"] == "A01", "Upstream candidate changed")
    need(config["feature_endpoint"] == "Mixed_5c", "Feature endpoint changed")
    need(config["access_policy"] == {
        "allowed_splits": ["train", "validation"], "test": "reject_before_path_access"
    }, "Split policy changed")
    return config


def load_acquisition_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    root = ROOT / config["artifact_directory"]
    manifest = json.loads((root / "acquisition_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = root / record["filename"]
        need(path.is_file() and not path.is_symlink(), f"Missing/symlinked artifact: {path}")
        need(path.stat().st_size == record["bytes"], f"Artifact byte count changed: {path}")
        need(sha256_file(path) == record["sha256"], f"Artifact SHA-256 changed: {path}")
    return manifest


def converted_checkpoint_path(config: Mapping[str, Any]) -> Path:
    return ROOT / config["artifact_directory"] / "converted" / config["candidate"]["converted_checkpoint"]["filename"]


def load_model_strict(config: Mapping[str, Any]) -> tuple[InceptionI3d, dict[str, Any]]:
    checkpoint = converted_checkpoint_path(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    need(isinstance(state, Mapping) and all(isinstance(key, str) for key in state), "Checkpoint is not a state_dict")
    need(all(isinstance(value, torch.Tensor) for value in state.values()), "Checkpoint has non-tensor values")
    need(all(torch.isfinite(value).all() for value in state.values()), "Checkpoint has non-finite tensor values")
    model = InceptionI3d(num_classes=400, modality="rgb", dropout_prob=0.0)
    model_slots = model.state_dict()
    source_keys, target_keys = set(state), set(model_slots)
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    shape_mismatches = [
        {"key": key, "source": list(state[key].shape), "target": list(model_slots[key].shape)}
        for key in sorted(source_keys & target_keys) if state[key].shape != model_slots[key].shape
    ]
    need(not missing and not unexpected and not shape_mismatches, "Pre-strict state coverage mismatch")
    incompatible = model.load_state_dict(state, strict=True)
    need(not incompatible.missing_keys and not incompatible.unexpected_keys, "strict=True reported incompatible keys")
    model.eval()
    return model, {
        "strict": True,
        "success": True,
        "source_tensor_count": len(state),
        "model_slot_count": len(model_slots),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "unused_source_tensors": [],
        "shape_mismatches": shape_mismatches,
    }


def normalize_i3d_rgb(source: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    need(source.dtype == torch.float32, "I3D adapter requires float32")
    need(source.ndim in {4, 5}, "I3D adapter requires CTHW or NCTHW")
    need(source.shape[-4] == 3, "I3D adapter requires RGB channels")
    need(source.is_contiguous(), "I3D adapter requires contiguous input")
    need(bool(torch.isfinite(source).all()), "I3D adapter input must be finite")
    before_digest = tensor_digest(source)
    minimum, maximum = float(source.min()), float(source.max())
    need(minimum >= 0.0 and maximum <= 1.0, "I3D adapter source must be in [0,1]")
    output = source * 2.0 - 1.0
    need(output.data_ptr() != source.data_ptr(), "I3D adapter must be out-of-place")
    need(output.shape == source.shape and output.dtype == source.dtype and output.is_contiguous(), "I3D adapter changed layout")
    need(tensor_digest(source) == before_digest, "I3D adapter modified source")
    need(float(output.min()) >= -1.0 and float(output.max()) <= 1.0, "I3D adapter output outside [-1,1]")
    return output, {
        "version": "i3d_rgb_minus1_plus1_v1",
        "operation": "affine",
        "scale": 2.0,
        "bias": -1.0,
        "in_place": False,
        "source_range_contract": [0.0, 1.0],
        "output_range_contract": [-1.0, 1.0],
        "shape_preserved": True,
        "dtype_preserved": True,
        "source_unmodified": True,
    }


def gap(features: torch.Tensor) -> torch.Tensor:
    need(features.ndim == 5, "GAP requires NCTHW")
    pooled = features.mean(dim=(2, 3, 4))
    need(pooled.shape == features.shape[:2] and bool(torch.isfinite(pooled).all()), "Invalid GAP output")
    return pooled
