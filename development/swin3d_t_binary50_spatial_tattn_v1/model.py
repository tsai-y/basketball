"""Swin3D-T with Gated Temporal Attention Pooling (Ilse et al. 2018 form, Option A init).

Replaces uniform 25-step temporal average in SwinTransformer3d.forward with:
  h = x.mean(dim=(2, 3))                                    # [B, 25, 768]
  scores = w(tanh(V(h)) * sigmoid(U(h))).squeeze(-1)        # [B, 25]
  alpha = softmax(scores, dim=1)                            # [B, 25], temperature=1
  z = (alpha.unsqueeze(-1) * h).sum(dim=1)                  # [B, 768]
  logits = head(z)                                          # [B, 2]

Preserves all 183 Kinetics-400 backbone tensors and 2 head tensors verbatim by key name.
Adds 5 parameter tensors (196,992 params; total 28,049,000 params, 190 state_dict keys).
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor, nn
from torchvision.models.video import swin_transformer as tv_swin

from development.swin3d_t_binary50_formal_v1.model import (
    EXPECTED_LAYERNORM_COUNT,
    EXPECTED_STOCHASTIC_DEPTH_COUNT,
    HEAD_SEED,
    NORM_POLICY,
    SEED,
    WEIGHT_SHA,
    WEIGHTS,
    need,
    sha,
    validate_architecture as _validate_base_arch_helper,
)
from development.swin3d_t_binary50_v1.model import (
    MODEL_VARIANT_NAME,
    OFFICIAL_KINETICS400_URL,
    SWIN3D_T_NUM_BLOCKS,
    DynamicPosShiftedWindowAttention3d,
    Swin3DTBinary50DynamicPos,
    replace_head_for_binary,
)
from torchvision.ops import StochasticDepth

VARIANT = "swin3d_t_dynamic_pos_tattn"
ATTN_SEED = 20265024  # Isolated from train seed (20261024) and head_seed (20264024)
ATTN_HIDDEN_DIM = 128
ATTN_TRUNC_STD = 0.02
ATTN_TRUNC_BOUNDS = (-0.04, 0.04)  # Explicit fixed [-2*std, +2*std]
EXPECTED_NEW_ATTN_PARAMS = 768 * 128 + 128 + 768 * 128 + 128 + 128  # 196,992
EXPECTED_TOTAL_PARAMS_TATTN = 27852008 + EXPECTED_NEW_ATTN_PARAMS  # 28,049,000
EXPECTED_HEAD_WEIGHT_SHA = "8fbdade9834e77b610d7b4dfeb6c0ca3dae41055c16dec00d96036afdff0dfe9"
ATTN_PARAM_KEYS = (
    "temporal_attn.V.weight",
    "temporal_attn.V.bias",
    "temporal_attn.U.weight",
    "temporal_attn.U.bias",
    "temporal_attn.w.weight",
)


class GatedTemporalAttentionPool(nn.Module):
    """Ilse et al. (2018) gated attention over temporal tokens [B, T, C] -> [B, C].

    Option A init: V, U ~ trunc_normal_(std=0.02, a=-0.04, b=0.04), bias=0; w.weight = 0.
    Softmax strictly along temporal dim (dim=1), temperature=1.0, no dropout/norm/prior.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = ATTN_HIDDEN_DIM, *, attn_seed: int = ATTN_SEED) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        # Isolate BOTH nn.Linear default construction AND custom init from global CPU RNG
        with torch.random.fork_rng(devices=[]):
            self.V = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
            self.U = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
            self.w = nn.Linear(self.hidden_dim, 1, bias=False)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(attn_seed))
            nn.init.trunc_normal_(
                self.V.weight,
                mean=0.0,
                std=ATTN_TRUNC_STD,
                a=ATTN_TRUNC_BOUNDS[0],
                b=ATTN_TRUNC_BOUNDS[1],
                generator=gen,
            )
            nn.init.zeros_(self.V.bias)
            nn.init.trunc_normal_(
                self.U.weight,
                mean=0.0,
                std=ATTN_TRUNC_STD,
                a=ATTN_TRUNC_BOUNDS[0],
                b=ATTN_TRUNC_BOUNDS[1],
                generator=gen,
            )
            nn.init.zeros_(self.U.bias)
            nn.init.zeros_(self.w.weight)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        need(h.ndim == 3 and h.shape[-1] == self.embed_dim, f"Expected [B, T, {self.embed_dim}], got {tuple(h.shape)}")
        gated = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        scores = self.w(gated).squeeze(-1)  # [B, T]
        alpha = torch.softmax(scores, dim=1)  # [B, T], temperature=1.0
        pooled = (alpha.unsqueeze(-1) * h).sum(dim=1)  # [B, C]
        return pooled, alpha


class Swin3DTBinary50DynamicPosTAttn(Swin3DTBinary50DynamicPos):
    """Dynamic-pos Swin3D-T replacing uniform temporal average with GatedTemporalAttentionPool.

    Formal forward() returns raw [B, 2] logits verbatim. Use forward_with_attention()
    only for read-only diagnostics (detached alpha, zero side effects).
    """

    def __init__(
        self,
        *,
        num_classes: int = 2,
        strict_contract_shape: bool = True,
        head_seed: int | None = None,
        attn_seed: int = ATTN_SEED,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            strict_contract_shape=strict_contract_shape,
            head_seed=head_seed,
        )
        self.temporal_attn = GatedTemporalAttentionPool(
            embed_dim=self.num_features,
            hidden_dim=ATTN_HIDDEN_DIM,
            attn_seed=attn_seed,
        )

    def _forward_features_and_pool(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.validate_input(x)
        x = self.patch_embed(x)  # B, _T, _H, _W, C
        x = self.pos_drop(x)
        x = self.features(x)  # B, 25, 4, 5, 768
        x = self.norm(x)
        h = x.mean(dim=(2, 3))  # B, 25, 768 (spatial mean)
        pooled, alpha = self.temporal_attn(h)
        logits = self.head(pooled)
        return logits, pooled, alpha

    def forward(self, x: Tensor) -> Tensor:
        logits, _, _ = self._forward_features_and_pool(x)
        return logits

    def forward_with_attention(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Diagnostic helper: returns (raw logits [B, 2], detached alpha [B, 25])."""
        logits, _, alpha = self._forward_features_and_pool(x)
        return logits, alpha.detach()


def validate_architecture(model: nn.Module) -> dict[str, Any]:
    need(isinstance(model, Swin3DTBinary50DynamicPosTAttn), "Expected Swin3DTBinary50DynamicPosTAttn")
    need(model.strict_contract_shape is True, "strict_contract_shape must be True")
    need(isinstance(model.temporal_attn, GatedTemporalAttentionPool), "Missing GatedTemporalAttentionPool")
    bn_count = sum(1 for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)))
    ln_count = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    dyn_attn_count = sum(1 for m in model.modules() if isinstance(m, DynamicPosShiftedWindowAttention3d))
    sd_layers = [m for m in model.modules() if isinstance(m, StochasticDepth)]
    need(bn_count == 0, f"Expected 0 BatchNorm layers, found {bn_count}")
    need(ln_count == EXPECTED_LAYERNORM_COUNT, f"Expected {EXPECTED_LAYERNORM_COUNT} LayerNorm layers, found {ln_count}")
    need(dyn_attn_count == SWIN3D_T_NUM_BLOCKS, f"Expected {SWIN3D_T_NUM_BLOCKS} dynamic attn blocks, found {dyn_attn_count}")
    need(len(sd_layers) == EXPECTED_STOCHASTIC_DEPTH_COUNT, f"Expected {EXPECTED_STOCHASTIC_DEPTH_COUNT} StochasticDepth layers")
    need(
        isinstance(model.head, nn.Linear) and model.head.in_features == 768 and model.head.out_features == 2,
        "Expected Linear(768, 2) binary head",
    )
    need(
        model.temporal_attn.V.in_features == 768
        and model.temporal_attn.V.out_features == ATTN_HIDDEN_DIM
        and model.temporal_attn.U.in_features == 768
        and model.temporal_attn.U.out_features == ATTN_HIDDEN_DIM
        and model.temporal_attn.w.in_features == ATTN_HIDDEN_DIM
        and model.temporal_attn.w.out_features == 1
        and model.temporal_attn.w.bias is None,
        "Unexpected GatedTemporalAttentionPool shape/bias",
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attn_params = sum(p.numel() for p in model.temporal_attn.parameters())
    need(
        attn_params == EXPECTED_NEW_ATTN_PARAMS
        and total_params == EXPECTED_TOTAL_PARAMS_TATTN
        and trainable_params == EXPECTED_TOTAL_PARAMS_TATTN,
        f"Unexpected param counts: total={total_params}, trainable={trainable_params}, attn={attn_params}",
    )
    return {
        "batchnorm_count": bn_count,
        "layernorm_count": ln_count,
        "dynamic_attn_blocks": dyn_attn_count,
        "stochastic_depth_count": len(sd_layers),
        "stochastic_depth_p_range": [float(sd_layers[0].p), float(sd_layers[-1].p)],
        "temporal_attn_params": attn_params,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "state_dict_keys_count": len(model.state_dict()),
    }


def training_mode(model: nn.Module) -> None:
    model.train()
    validate_architecture(model)


def initialized(variant: str = VARIANT, seed: int = SEED) -> tuple[Swin3DTBinary50DynamicPosTAttn, dict[str, Any]]:
    """Build fresh pretrained Swin3D-T + GatedTemporalAttentionPool (Option A init)."""
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
        # Construct 400-class model first, load 185 K400 tensors strictly (excluding 5 new temporal_attn keys)
        model = Swin3DTBinary50DynamicPosTAttn(num_classes=400, strict_contract_shape=True, attn_seed=ATTN_SEED)
        incompatible = model.load_state_dict(raw_sd, strict=False)
        need(
            sorted(incompatible.missing_keys) == sorted(ATTN_PARAM_KEYS)
            and len(incompatible.unexpected_keys) == 0,
            f"Unexpected 400-class load mismatch: {incompatible}",
        )
        replace_head_for_binary(model, num_classes=2, head_seed=head_seed)

    sd = model.state_dict()
    non_head_keys = [k for k in raw_sd.keys() if k not in ("head.weight", "head.bias")]
    need(len(non_head_keys) == 183, f"Expected 183 non-head keys, got {len(non_head_keys)}")
    for k in non_head_keys:
        need(torch.equal(sd[k], raw_sd[k]), f"Non-head tensor diverged from checkpoint: {k}")

    need(len(sd) == 190, f"Expected 190 state_dict keys, got {len(sd)}")
    arch = validate_architecture(model)

    head_weight_sha = hashlib.sha256(model.head.weight.detach().numpy().tobytes()).hexdigest()
    need(
        head_weight_sha == EXPECTED_HEAD_WEIGHT_SHA,
        f"Head weight SHA mismatch vs 120x160 parent: {head_weight_sha} != {EXPECTED_HEAD_WEIGHT_SHA}",
    )
    head_bias_zero = bool(torch.equal(model.head.bias.detach(), torch.zeros_like(model.head.bias)))
    need(head_bias_zero, "Expected zero-initialized binary head bias")

    # Verify Option A init invariants on temporal_attn
    w_zero = bool(torch.equal(model.temporal_attn.w.weight.detach(), torch.zeros_like(model.temporal_attn.w.weight)))
    v_bias_zero = bool(torch.equal(model.temporal_attn.V.bias.detach(), torch.zeros_like(model.temporal_attn.V.bias)))
    u_bias_zero = bool(torch.equal(model.temporal_attn.U.bias.detach(), torch.zeros_like(model.temporal_attn.U.bias)))
    need(w_zero and v_bias_zero and u_bias_zero, "Option A zero-init invariant violated")
    tattn_sha = hashlib.sha256(
        b"".join(sd[k].detach().numpy().tobytes() for k in ATTN_PARAM_KEYS)
    ).hexdigest()

    metadata = {
        "model_variant": f"{MODEL_VARIANT_NAME}_gated_temporal_attn_v1",
        "variant": variant,
        "train_seed": int(seed),
        "head_initialization_seed": head_seed,
        "head_weight_sha256": head_weight_sha,
        "head_bias_all_zero": head_bias_zero,
        "temporal_attention": {
            "form": "Ilse_et_al_2018_gated_attention_tanh_V_mul_sigmoid_U_softmax_dim1_temp1",
            "project_specific_config": "hidden_dim=128, w.weight=0 (Option A), no bias on w, no dropout/norm/prior",
            "attn_initialization_seed": ATTN_SEED,
            "v_u_init": "trunc_normal_(mean=0.0, std=0.02, a=-0.04, b=0.04), bias=0",
            "w_init": "zeros_(w.weight), bias=None",
            "rng_isolated_via_fork_rng": True,
            "param_keys": list(ATTN_PARAM_KEYS),
            "param_count": EXPECTED_NEW_ATTN_PARAMS,
            "init_tensors_sha256": tattn_sha,
            "grad_clip_side_effect_disclosed": "Added 196,992 parameters participate in global clip_grad_norm_(1.0) and Adam state",
        },
        "pretrained_weights": {
            "enum": "torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1",
            "official_url": OFFICIAL_KINETICS400_URL,
            "local_path": str(WEIGHTS),
            "full_sha256": actual_sha,
            "strict_backbone_400class_loaded_before_head_swap": True,
            "non_head_tensors_bitwise_verified": len(non_head_keys),
            "trained_weights_from_engineering_used": False,
        },
        "normalization_policy": NORM_POLICY,
        "architecture": arch,
    }
    return model, metadata
