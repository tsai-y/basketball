"""Validated, ID-aligned train-only video loss coefficients; no sampling."""

import hashlib
import json

import torch

from development.data_quality.scripts import audit_three_window_quality_v1 as audit
from development.data_quality.scripts.simulate_quality_weights_v1 import (
    ALPHA, build_report, validate_alpha,
)


def aligned_weights(rows, video_ids, labels, *, split="train", alpha=ALPHA):
    if split != "train":
        raise ValueError("Quality loss weighting is train-only")
    validate_alpha(alpha)
    audit.require(len(video_ids) > 0 and len(set(video_ids)) == len(video_ids), "Invalid train IDs")
    audit.require(tuple(labels.shape) == (len(video_ids),) and labels.dtype == torch.float32,
                  "Invalid train labels")
    audit.require(set(labels.tolist()) <= {0., 1.}, "Invalid binary train labels")
    audit.require(len(rows) == len(video_ids), "Quality/train count mismatch")
    report, weighted = build_report(rows, alpha=alpha)
    by_id = {row["video_id"]: row for row in weighted}
    audit.require(set(by_id) == set(video_ids), "Quality/train IDs differ")
    ordered = [by_id[video_id] for video_id in video_ids]
    for row, label in zip(ordered, labels.tolist()):
        audit.require(row["label"] == ("Make" if label else "Miss"), "Quality label mismatch")
    weights = torch.tensor([row["global_mean_normalized_weight"] for row in ordered],
                           dtype=torch.float32)
    audit.require(torch.isfinite(weights).all().item() and (weights > 0).all().item(),
                  "Invalid normalized weights")
    audit.require(abs(float(weights.double().mean()) - 1) < 1e-6, "Weight mean must equal one")
    info = {
        "alpha": alpha, "raw_formula": f"1 - {alpha:g} * sum(overlap_frames) / 192",
        "normalization": "divide by ONE full-train raw-weight mean, NOT batch/class/shot means",
        "raw_train_mean": report["overall"]["raw_weight_mean"],
        "normalized_min": float(weights.min()), "normalized_max": float(weights.max()),
        "normalized_mean": float(weights.double().mean()),
        "scope": "train video-level loss only", "sampling": "unchanged shuffle without replacement",
        "window_feature_scaling": False, "validation_weighting": False,
        "label_balancing": False, "candidate_flags_are_verified_blur": False,
        "aligned_weights_sha256": hashlib.sha256(weights.numpy().tobytes()).hexdigest(),
        "aligned_ids_sha256": hashlib.sha256(json.dumps(video_ids).encode()).hexdigest(),
    }
    return weights, info, report


def load_train_weights(video_ids, labels, *, cache_root, split="train", alpha=ALPHA):
    # Reject held-out access BEFORE attempting to read metadata/cache paths.
    if split != "train":
        raise ValueError("Quality loss weighting is train-only")
    validate_alpha(alpha)
    source_report, rows = audit.run_audit(cache_root)
    weights, info, report = aligned_weights(rows, video_ids, labels, alpha=alpha)
    info["cache_run_id"] = source_report["cache_run_id"]
    info["input_identities"] = source_report["input_identities"]
    info["gate_config_sha256"] = source_report["gate_config_sha256"]
    return weights, info, report
