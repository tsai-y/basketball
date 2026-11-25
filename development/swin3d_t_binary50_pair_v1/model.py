"""Seed20261025 formal Swin3D-T initializer (locked Kinetics-400 + head_seed=20264025)."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import patch

import torch
from torchvision.models.video import swin_transformer as tv_swin

from development.swin3d_t_binary50_formal_v1.model import (
    NORM_POLICY,
    VARIANT,
    WEIGHT_SHA,
    WEIGHTS,
    need,
    sha,
    training_mode,
    validate_architecture,
)
from development.swin3d_t_binary50_v1.model import (
    MODEL_VARIANT_NAME,
    OFFICIAL_KINETICS400_URL,
    Swin3DTBinary50DynamicPos,
    build_swin3d_t_binary50_dynamic_pos,
)

SEED = 20261025
HEAD_SEED = 20264025  # SEED + 3000


def initialized(variant: str = VARIANT, seed: int = SEED) -> tuple[Swin3DTBinary50DynamicPos, dict[str, Any]]:
    """Build fresh pretrained Swin3D-T binary model for seed=20261025 (head_seed=20264025)."""
    need(variant == VARIANT, f"Only {VARIANT} is supported, got {variant!r}")
    need(int(seed) == SEED, f"Only seed {SEED} is supported here, got {seed!r}")
    need(WEIGHTS.is_file() and not WEIGHTS.is_symlink(), f"Locked weight file missing: {WEIGHTS}")
    actual_sha = sha(WEIGHTS)
    need(actual_sha == WEIGHT_SHA, f"Locked weight SHA-256 mismatch: {actual_sha} != {WEIGHT_SHA}")

    head_seed = int(seed) + 3000
    need(head_seed == HEAD_SEED, f"Unexpected head_seed {head_seed}")

    with patch.object(
        tv_swin.Swin3D_T_Weights.KINETICS400_V1,
        "get_state_dict",
        side_effect=AssertionError("Network weight download forbidden during formal initialization"),
    ):
        raw_sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        model = build_swin3d_t_binary50_dynamic_pos(
            strict_contract_shape=True,
            pretrained_400_state_dict=raw_sd,
            head_seed=head_seed,
        )

    sd = model.state_dict()
    non_head_keys = [k for k in raw_sd.keys() if k not in ("head.weight", "head.bias")]
    need(len(non_head_keys) == 183, f"Expected 183 non-head keys, got {len(non_head_keys)}")
    for k in non_head_keys:
        need(torch.equal(sd[k], raw_sd[k]), f"Non-head tensor diverged from checkpoint: {k}")

    arch = validate_architecture(model)
    head_weight_sha = hashlib.sha256(model.head.weight.detach().numpy().tobytes()).hexdigest()
    head_bias_zero = bool(torch.equal(model.head.bias.detach(), torch.zeros_like(model.head.bias)))
    need(head_bias_zero, "Expected zero-initialized binary head bias")

    metadata = {
        "model_variant": MODEL_VARIANT_NAME,
        "variant": variant,
        "train_seed": int(seed),
        "head_initialization_seed": head_seed,
        "head_weight_sha256": head_weight_sha,
        "head_bias_all_zero": head_bias_zero,
        "pretrained_weights": {
            "enum": "torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1",
            "official_url": OFFICIAL_KINETICS400_URL,
            "local_path": str(WEIGHTS),
            "full_sha256": actual_sha,
            "strict_400class_loaded_before_head_swap": True,
            "non_head_tensors_bitwise_verified": len(non_head_keys),
            "trained_weights_from_engineering_used": False,
        },
        "normalization_policy": NORM_POLICY,
        "architecture": arch,
    }
    return model, metadata
