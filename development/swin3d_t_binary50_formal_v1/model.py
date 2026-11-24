"""Formal Swin3D-T initializer (locked Kinetics-400 weight + isolated head_seed=20264024)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models.video import swin_transformer as tv_swin
from torchvision.ops import StochasticDepth

from development.swin3d_t_binary50_v1.model import (
    DEFAULT_LOCAL_WEIGHT_PATH,
    MODEL_VARIANT_NAME,
    OFFICIAL_KINETICS400_URL,
    SWIN3D_T_NUM_BLOCKS,
    DynamicPosShiftedWindowAttention3d,
    Swin3DTBinary50DynamicPos,
    build_swin3d_t_binary50_dynamic_pos,
)

VARIANT = "swin3d_t_dynamic_pos"
SEED = 20261024
HEAD_SEED = 20264024  # SEED + 3000; same seed does not imply identical weights across different head dims (768 vs 2048).
WEIGHTS = DEFAULT_LOCAL_WEIGHT_PATH
WEIGHT_SHA = "7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130"
NORM_POLICY = "swin3d_t_native_29_layernorm_0_batchnorm_12_stochastic_depth_0_to_0p1"
EXPECTED_TOTAL_PARAMS = 27852008
EXPECTED_LAYERNORM_COUNT = 29
EXPECTED_STOCHASTIC_DEPTH_COUNT = 12


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_architecture(model: nn.Module) -> dict[str, Any]:
    need(isinstance(model, Swin3DTBinary50DynamicPos), "Expected Swin3DTBinary50DynamicPos")
    bn_count = sum(
        1 for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    )
    ln_count = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    dyn_attn_count = sum(1 for m in model.modules() if isinstance(m, DynamicPosShiftedWindowAttention3d))
    sd_layers = [m for m in model.modules() if isinstance(m, StochasticDepth)]
    need(bn_count == 0, f"Expected 0 BatchNorm layers, found {bn_count}")
    need(ln_count == EXPECTED_LAYERNORM_COUNT, f"Expected {EXPECTED_LAYERNORM_COUNT} LayerNorm layers, found {ln_count}")
    need(dyn_attn_count == SWIN3D_T_NUM_BLOCKS, f"Expected {SWIN3D_T_NUM_BLOCKS} dynamic attn blocks, found {dyn_attn_count}")
    need(len(sd_layers) == EXPECTED_STOCHASTIC_DEPTH_COUNT, f"Expected {EXPECTED_STOCHASTIC_DEPTH_COUNT} StochasticDepth layers")
    need(isinstance(model.head, nn.Linear) and model.head.in_features == 768 and model.head.out_features == 2,
         "Expected Linear(768, 2) binary head")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    need(total_params == EXPECTED_TOTAL_PARAMS and trainable_params == EXPECTED_TOTAL_PARAMS,
         f"Unexpected param counts: total={total_params}, trainable={trainable_params}")
    return {
        "batchnorm_count": bn_count,
        "layernorm_count": ln_count,
        "dynamic_attn_blocks": dyn_attn_count,
        "stochastic_depth_count": len(sd_layers),
        "stochastic_depth_p_range": [float(sd_layers[0].p), float(sd_layers[-1].p)],
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def training_mode(model: nn.Module) -> None:
    """Native Swin3D-T training mode (0 BatchNorm; LayerNorm + StochasticDepth active)."""
    model.train()
    validate_architecture(model)


def initialized(variant: str = VARIANT, seed: int = SEED) -> tuple[Swin3DTBinary50DynamicPos, dict[str, Any]]:
    """Build fresh pretrained Swin3D-T binary model with locked Kinetics-400 weights + head_seed=20264024."""
    need(variant == VARIANT, f"Only {VARIANT} is supported, got {variant!r}")
    need(int(seed) == SEED, f"Only seed {SEED} is supported, got {seed!r}")
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

    # Confirm all 183 non-head tensors remain bitwise equal to locked Kinetics-400 checkpoint
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
        "head_seed_cross_architecture_note": (
            "Same numeric head_seed (20264024) does not produce identical initial weights "
            "across different feature dimensions (768 vs 2048/1536)."
        ),
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
