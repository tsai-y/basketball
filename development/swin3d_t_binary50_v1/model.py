"""Swin3D-T 50-frame binary model with dynamic window relative position index fix.

Behavioral disclosure vs upstream torchvision 0.26.0 swin3d_t:
- Upstream torchvision _get_relative_position_bias slices
  relative_position_index[:window_vol, :window_vol] whenever a stage feature map
  is smaller than the configured (8, 7, 7) window. Note that shrinking ONLY the
  outermost temporal dimension (e.g. (4, 7, 7)) preserves the row-major prefix
  ordering and does NOT mismatch; however, whenever spatial dimensions shrink
  below (7, 7) (for example, Stage 3 ceil(H/32) < 7 or ceil(W/32) < 7, i.e.
  H <= 192 or W <= 192), the inner strides (49, 7, 1) break and prefix slicing
  mismatches the true relative coordinates (for (8, 4, 5), 24,980 of 25,600
  entries differ).
- This isolated package replaces only the 12 ShiftedWindowAttention3d modules
  with DynamicPosShiftedWindowAttention3d, which uniformly selects rows/columns
  of the original [392, 392] relative_position_index buffer using the actual
  clipped (t, h, w) coordinates inside the original (8, 7, 7) grid.
- All parameter names, buffer names, shapes, relative_position_bias_table
  [2535, num_heads], relative_position_index [392, 392], LayerNorm layers, and
  StochasticDepth schedule (0 -> 0.1) remain 100% identical to upstream swin3d_t.
- Under a full (8, 7, 7) window across all 12 blocks (e.g. T >= 16, H >= 224,
  W >= 224), outputs and gradients match upstream swin3d_t identically.
- No pretrained weights are downloaded here; weights=None is enforced unless an
  explicit local state_dict is provided by an approved downstream caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torchvision.models.video.swin_transformer import (
    ShiftedWindowAttention3d,
    SwinTransformer3d,
    _get_window_and_shift_size,
    shifted_window_attention_3d,
)

MODEL_VARIANT_NAME = "swin3d_t_binary50_dynamic_pos_v1"
BINARY_CLASSES = ("Miss", "Make")
CONTRACT_CHANNELS = 3
CONTRACT_FRAMES = 50
CONTRACT_HEIGHT = 120
CONTRACT_WIDTH = 160
CONTRACT_MEAN = (0.485, 0.456, 0.406)
CONTRACT_STD = (0.229, 0.224, 0.225)
SWIN3D_T_PATCH_SIZE = (2, 4, 4)
SWIN3D_T_EMBED_DIM = 96
SWIN3D_T_DEPTHS = (2, 2, 6, 2)
SWIN3D_T_NUM_HEADS = (3, 6, 12, 24)
SWIN3D_T_WINDOW_SIZE = (8, 7, 7)
SWIN3D_T_STOCHASTIC_DEPTH_PROB = 0.1
SWIN3D_T_NUM_CLASSES_PRETRAINED = 400
SWIN3D_T_NUM_BLOCKS = sum(SWIN3D_T_DEPTHS)  # 12
OFFICIAL_KINETICS400_URL = "https://download.pytorch.org/models/swin3d_t-7615ae03.pth"
OFFICIAL_WEIGHT_FILENAME = "swin3d_t-7615ae03.pth"
OFFICIAL_HASH_PREFIX = "7615ae03"
DEFAULT_LOCAL_WEIGHT_PATH = Path(__file__).resolve().parent / "artifacts" / "weights" / OFFICIAL_WEIGHT_FILENAME


def select_dynamic_relative_position_index(
    relative_position_index: Tensor,
    base_window_size: Sequence[int],
    actual_window_size: Sequence[int],
) -> Tensor:
    """Select the exact [N, N] submatrix from the full [392, 392] index buffer.

    Each token (t, h, w) in the clipped window (wd, wh, ww) occupies flat index
    t * (Wh * Ww) + h * Ww + w inside the original (Wd, Wh, Ww) window grid.
    Indexing the original relative_position_index buffer along both axes by these
    actual token coordinates yields exact indices into the original pretrained
    relative_position_bias_table without altering table layout or buffer shapes.
    """
    if len(base_window_size) != 3 or len(actual_window_size) != 3:
        raise ValueError("base_window_size and actual_window_size must have length 3")
    base_d, base_h, base_w = (int(base_window_size[0]), int(base_window_size[1]), int(base_window_size[2]))
    wd, wh, ww = (int(actual_window_size[0]), int(actual_window_size[1]), int(actual_window_size[2]))
    if not (1 <= wd <= base_d and 1 <= wh <= base_h and 1 <= ww <= base_w):
        raise ValueError(
            f"actual_window_size={(wd, wh, ww)} must satisfy 1 <= actual <= base={(base_d, base_h, base_w)}"
        )
    expected_full_vol = base_d * base_h * base_w
    if tuple(relative_position_index.shape) != (expected_full_vol, expected_full_vol):
        raise ValueError(
            f"Expected relative_position_index shape {(expected_full_vol, expected_full_vol)}, "
            f"got {tuple(relative_position_index.shape)}"
        )
    if (wd, wh, ww) == (base_d, base_h, base_w):
        return relative_position_index

    device = relative_position_index.device
    t_coords = torch.arange(wd, device=device, dtype=torch.long)
    h_coords = torch.arange(wh, device=device, dtype=torch.long)
    w_coords = torch.arange(ww, device=device, dtype=torch.long)
    grid_t, grid_h, grid_w = torch.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    flat_coords = (grid_t * (base_h * base_w) + grid_h * base_w + grid_w).reshape(-1)
    return relative_position_index.index_select(0, flat_coords).index_select(1, flat_coords)


def independent_relative_position_index(
    base_window_size: Sequence[int],
    actual_window_size: Sequence[int],
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Independent reference formula computing relative indices into base bias table.

    Computes pairwise relative coordinates (dt, dh, dw) directly from the actual
    (wd, wh, ww) token grid and encodes them using the base (Wd, Wh, Ww) table
    offsets/strides:
      (dt + Wd - 1) * (2*Wh - 1) * (2*Ww - 1) + (dh + Wh - 1) * (2*Ww - 1) + (dw + Ww - 1)
    """
    base_d, base_h, base_w = (int(base_window_size[0]), int(base_window_size[1]), int(base_window_size[2]))
    wd, wh, ww = (int(actual_window_size[0]), int(actual_window_size[1]), int(actual_window_size[2]))
    if not (1 <= wd <= base_d and 1 <= wh <= base_h and 1 <= ww <= base_w):
        raise ValueError(
            f"actual_window_size={(wd, wh, ww)} must satisfy 1 <= actual <= base={(base_d, base_h, base_w)}"
        )
    t_coords = torch.arange(wd, device=device, dtype=torch.long)
    h_coords = torch.arange(wh, device=device, dtype=torch.long)
    w_coords = torch.arange(ww, device=device, dtype=torch.long)
    grid_t, grid_h, grid_w = torch.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
    flat_t = grid_t.reshape(-1)
    flat_h = grid_h.reshape(-1)
    flat_w = grid_w.reshape(-1)

    dt = flat_t[:, None] - flat_t[None, :] + (base_d - 1)
    dh = flat_h[:, None] - flat_h[None, :] + (base_h - 1)
    dw = flat_w[:, None] - flat_w[None, :] + (base_w - 1)

    stride_t = (2 * base_h - 1) * (2 * base_w - 1)
    stride_h = 2 * base_w - 1
    return dt * stride_t + dh * stride_h + dw


class DynamicPosShiftedWindowAttention3d(ShiftedWindowAttention3d):
    """ShiftedWindowAttention3d with geometrically exact dynamic window indexing.

    Preserves all upstream parameters (qkv, proj, relative_position_bias_table)
    and buffers (relative_position_index) verbatim.
    """

    @classmethod
    def from_upstream(cls, upstream: ShiftedWindowAttention3d) -> "DynamicPosShiftedWindowAttention3d":
        if not isinstance(upstream, ShiftedWindowAttention3d):
            raise TypeError(f"Expected ShiftedWindowAttention3d, got {type(upstream).__name__}")
        module = nn.Module.__new__(cls)
        nn.Module.__init__(module)
        module.window_size = list(upstream.window_size)
        module.shift_size = list(upstream.shift_size)
        module.num_heads = int(upstream.num_heads)
        module.attention_dropout = float(upstream.attention_dropout)
        module.dropout = float(upstream.dropout)
        module.qkv = upstream.qkv
        module.proj = upstream.proj
        module.relative_position_bias_table = upstream.relative_position_bias_table
        module.register_buffer("relative_position_index", upstream.relative_position_index)
        module._last_runtime_geometry: dict[str, Any] | None = None
        return module

    def get_relative_position_bias(self, window_size: list[int]) -> Tensor:
        sub_index = select_dynamic_relative_position_index(
            self.relative_position_index,
            self.window_size,
            window_size,
        )
        window_vol = int(window_size[0]) * int(window_size[1]) * int(window_size[2])
        relative_position_bias = self.relative_position_bias_table[sub_index.reshape(-1)]
        relative_position_bias = relative_position_bias.view(window_vol, window_vol, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        return relative_position_bias

    def forward(self, x: Tensor) -> Tensor:
        _, t, h, w, _ = x.shape
        size_dhw = [t, h, w]
        window_size, shift_size = self.window_size.copy(), self.shift_size.copy()
        window_size, shift_size = _get_window_and_shift_size(shift_size, size_dhw, window_size)
        self._last_runtime_geometry = {
            "input_dhw": [int(t), int(h), int(w)],
            "base_window_size": [int(v) for v in self.window_size],
            "actual_window_size": [int(v) for v in window_size],
            "configured_shift_size": [int(v) for v in self.shift_size],
            "actual_shift_size": [int(v) for v in shift_size],
            "window_clipped": tuple(window_size) != tuple(self.window_size),
        }
        relative_position_bias = self.get_relative_position_bias(window_size)
        return shifted_window_attention_3d(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            window_size,
            self.num_heads,
            shift_size=shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


def _patch_attention_modules_inplace(model: SwinTransformer3d) -> int:
    """Replace all ShiftedWindowAttention3d blocks on a single instance in place."""
    replaced = 0
    for stage_module in model.features:
        if isinstance(stage_module, nn.Sequential):
            for block in stage_module:
                if hasattr(block, "attn") and isinstance(block.attn, ShiftedWindowAttention3d):
                    block.attn = DynamicPosShiftedWindowAttention3d.from_upstream(block.attn)
                    replaced += 1
    if replaced != SWIN3D_T_NUM_BLOCKS:
        raise RuntimeError(
            f"Expected to patch {SWIN3D_T_NUM_BLOCKS} attention blocks, patched {replaced}"
        )
    return replaced


def replace_head_for_binary(
    model: SwinTransformer3d,
    *,
    num_classes: int = 2,
    head_seed: int | None = None,
) -> nn.Linear:
    """Replace classification head with Linear(768, num_classes) using Swin init.

    When head_seed is provided, head weights are determined solely by an
    isolated torch.Generator(head_seed) while torch.random.fork_rng(devices=[])
    reads, saves, and restores global CPU RNG so its state before and after
    head initialization is unchanged. Note that constructing the full Swin3D-T
    module still consumes global RNG; on resume, checkpoint CPU/CUDA RNG must
    be restored AFTER constructing and loading the model and optimizer.
    """
    if num_classes != 2:
        raise ValueError(f"Expected binary num_classes=2, got {num_classes}")
    if head_seed is not None:
        with torch.random.fork_rng(devices=[]):
            new_head = nn.Linear(model.num_features, num_classes)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(head_seed))
        nn.init.trunc_normal_(new_head.weight, std=0.02, generator=gen)
    else:
        new_head = nn.Linear(model.num_features, num_classes)
        nn.init.trunc_normal_(new_head.weight, std=0.02)
    if new_head.bias is not None:
        nn.init.zeros_(new_head.bias)
    model.head = new_head
    model.num_classes = num_classes
    return new_head


class Swin3DTBinary50DynamicPos(SwinTransformer3d):
    """Swin3D-T with dynamic window relative position index fix and 2-class head.

    Preserves upstream SwinTransformer3d state_dict keys verbatim (no wrapper
    prefix), allowing direct strict=True loading of a 400-class checkpoint before
    calling replace_head_for_binary(), or direct strict=True comparison with
    upstream swin3d_t.
    """

    def __init__(
        self,
        *,
        num_classes: int = 2,
        strict_contract_shape: bool = True,
        head_seed: int | None = None,
    ) -> None:
        super().__init__(
            patch_size=list(SWIN3D_T_PATCH_SIZE),
            embed_dim=SWIN3D_T_EMBED_DIM,
            depths=list(SWIN3D_T_DEPTHS),
            num_heads=list(SWIN3D_T_NUM_HEADS),
            window_size=list(SWIN3D_T_WINDOW_SIZE),
            stochastic_depth_prob=SWIN3D_T_STOCHASTIC_DEPTH_PROB,
            num_classes=SWIN3D_T_NUM_CLASSES_PRETRAINED,
        )
        self.model_variant = MODEL_VARIANT_NAME
        self.strict_contract_shape = bool(strict_contract_shape)
        self.patched_attention_blocks = _patch_attention_modules_inplace(self)
        if num_classes == 2:
            replace_head_for_binary(self, num_classes=2, head_seed=head_seed)
        elif num_classes != SWIN3D_T_NUM_CLASSES_PRETRAINED:
            raise ValueError(
                f"Supported num_classes are 2 or {SWIN3D_T_NUM_CLASSES_PRETRAINED}, got {num_classes}"
            )

    def collect_last_runtime_geometries(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for name, module in self.named_modules():
            if isinstance(module, DynamicPosShiftedWindowAttention3d):
                records.append({"module": name, "geometry": module._last_runtime_geometry})
        return records

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
            expected_cthw = (CONTRACT_CHANNELS, CONTRACT_FRAMES, CONTRACT_HEIGHT, CONTRACT_WIDTH)
            if (c, t, h, w) != expected_cthw:
                raise ValueError(
                    f"Strict contract requires [B, 3, 50, 120, 160], got {tuple(x.shape)}"
                )
        if not torch.isfinite(x).all():
            raise ValueError("Input tensor contains non-finite values")

    def forward(self, x: Tensor) -> Tensor:
        self.validate_input(x)
        return super().forward(x)


def build_swin3d_t_400_dynamic_pos(
    *,
    strict_contract_shape: bool = False,
) -> Swin3DTBinary50DynamicPos:
    """Build 400-class dynamic-position Swin3D-T (weights=None only)."""
    return Swin3DTBinary50DynamicPos(
        num_classes=SWIN3D_T_NUM_CLASSES_PRETRAINED,
        strict_contract_shape=strict_contract_shape,
    )


def build_swin3d_t_binary50_dynamic_pos(
    *,
    strict_contract_shape: bool = True,
    pretrained_400_state_dict: Mapping[str, Tensor] | None = None,
    head_seed: int | None = None,
) -> Swin3DTBinary50DynamicPos:
    """Build binary Swin3D-T with dynamic position index fix.

    If pretrained_400_state_dict is supplied (only after explicit local weight
    approval), verifies strict=True loading on the 400-class architecture BEFORE
    replacing model.head with Linear(768, 2) using head_seed. Otherwise
    initializes with weights=None and zero network/disk weight access.
    """
    if pretrained_400_state_dict is not None:
        model = build_swin3d_t_400_dynamic_pos(strict_contract_shape=strict_contract_shape)
        incompatible = model.load_state_dict(pretrained_400_state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"400-class strict load failed: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        replace_head_for_binary(model, num_classes=2, head_seed=head_seed)
        return model
    return Swin3DTBinary50DynamicPos(
        num_classes=2,
        strict_contract_shape=strict_contract_shape,
        head_seed=head_seed,
    )
