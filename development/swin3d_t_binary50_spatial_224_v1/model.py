"""Isolated 224x224 Swin3D-T dynamic-pos model initializer (same K400 weights + head_seed=20264024)."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor
from torchvision.models.video import swin_transformer as tv_swin

from development.swin3d_t_binary50_formal_v1.model import (
    EXPECTED_LAYERNORM_COUNT,
    EXPECTED_STOCHASTIC_DEPTH_COUNT,
    EXPECTED_TOTAL_PARAMS,
    HEAD_SEED,
    NORM_POLICY,
    SEED,
    VARIANT,
    WEIGHT_SHA,
    WEIGHTS,
    need,
    sha,
    validate_architecture as _validate_base_arch,
)
from development.swin3d_t_binary50_v1.model import (
    CONTRACT_CHANNELS,
    CONTRACT_FRAMES,
    MODEL_VARIANT_NAME,
    OFFICIAL_KINETICS400_URL,
    Swin3DTBinary50DynamicPos,
    replace_head_for_binary,
)

CONTRACT_HEIGHT_224 = 224
CONTRACT_WIDTH_224 = 224
EXPECTED_HEAD_WEIGHT_SHA = "8fbdade9834e77b610d7b4dfeb6c0ca3dae41055c16dec00d96036afdff0dfe9"


class Swin3DTBinary50DynamicPos224(Swin3DTBinary50DynamicPos):
    """Same dynamic-pos Swin3D-T architecture with strict [B, 3, 50, 224, 224] contract guard."""

    def validate_input(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x).__name__}")
        if x.ndim != 5:
            raise ValueError(f"Expected 5D [B, C, T, H, W] input, got shape {tuple(x.shape)}")
        if x.dtype != torch.float32:
            raise ValueError(f"Expected float32 input, got dtype {x.dtype}")
        b, c, t, h, w = x.shape
        if b < 1 or c != CONTRACT_CHANNELS:
            raise ValueError(f"Expected B>=1 and C={CONTRACT_CHANNELS}, got shape {tuple(x.shape)}")
        if self.strict_contract_shape:
            expected_cthw = (CONTRACT_CHANNELS, CONTRACT_FRAMES, CONTRACT_HEIGHT_224, CONTRACT_WIDTH_224)
            if (c, t, h, w) != expected_cthw:
                raise ValueError(
                    f"Strict 224x224 contract requires [B, 3, 50, 224, 224], got {tuple(x.shape)}"
                )
        if not torch.isfinite(x).all():
            raise ValueError("Input tensor contains non-finite values")


def validate_architecture(model: torch.nn.Module) -> dict[str, Any]:
    need(isinstance(model, Swin3DTBinary50DynamicPos224), "Expected Swin3DTBinary50DynamicPos224")
    need(model.strict_contract_shape is True, "strict_contract_shape must be True")
    return _validate_base_arch(model)


def training_mode(model: torch.nn.Module) -> None:
    model.train()
    validate_architecture(model)


def initialized(variant: str = VARIANT, seed: int = SEED) -> tuple[Swin3DTBinary50DynamicPos224, dict[str, Any]]:
    """Build fresh pretrained 224x224-guarded Swin3D-T with locked Kinetics-400 weights + head_seed=20264024."""
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
        model = Swin3DTBinary50DynamicPos224(num_classes=400, strict_contract_shape=True)
        incompatible = model.load_state_dict(raw_sd, strict=True)
        need(
            len(incompatible.missing_keys) == 0 and len(incompatible.unexpected_keys) == 0,
            f"Unexpected 400-class load mismatch: {incompatible}",
        )
        replace_head_for_binary(model, num_classes=2, head_seed=head_seed)

    sd = model.state_dict()
    non_head_keys = [k for k in raw_sd.keys() if k not in ("head.weight", "head.bias")]
    need(len(non_head_keys) == 183, f"Expected 183 non-head keys, got {len(non_head_keys)}")
    for k in non_head_keys:
        need(torch.equal(sd[k], raw_sd[k]), f"Non-head tensor diverged from checkpoint: {k}")

    arch = validate_architecture(model)
    head_weight_sha = hashlib.sha256(model.head.weight.detach().numpy().tobytes()).hexdigest()
    need(
        head_weight_sha == EXPECTED_HEAD_WEIGHT_SHA,
        f"Head weight SHA mismatch vs 120x160 parent: {head_weight_sha} != {EXPECTED_HEAD_WEIGHT_SHA}",
    )
    head_bias_zero = bool(torch.equal(model.head.bias.detach(), torch.zeros_like(model.head.bias)))
    need(head_bias_zero, "Expected zero-initialized binary head bias")

    metadata = {
        "model_variant": MODEL_VARIANT_NAME,
        "variant": variant,
        "input_shape_guard": [3, CONTRACT_FRAMES, CONTRACT_HEIGHT_224, CONTRACT_WIDTH_224],
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
