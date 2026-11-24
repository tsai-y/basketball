"""Isolated Swin3D-T 50-frame binary package with dynamic relative position index fix.

Model identity:
- "Swin3D-T with dynamic window relative position index fix"
  (含動態位置索引修正的 Swin3D-T).
- Strictly isolated under development/swin3d_t_binary50_v1; does NOT modify
  installed torchvision or any protected baseline files.
"""

from .model import (
    BINARY_CLASSES,
    CONTRACT_FRAMES,
    CONTRACT_HEIGHT,
    CONTRACT_MEAN,
    CONTRACT_STD,
    CONTRACT_WIDTH,
    MODEL_VARIANT_NAME,
    DynamicPosShiftedWindowAttention3d,
    Swin3DTBinary50DynamicPos,
    build_swin3d_t_400_dynamic_pos,
    build_swin3d_t_binary50_dynamic_pos,
    independent_relative_position_index,
    replace_head_for_binary,
    select_dynamic_relative_position_index,
)

__all__ = [
    "BINARY_CLASSES",
    "CONTRACT_FRAMES",
    "CONTRACT_HEIGHT",
    "CONTRACT_MEAN",
    "CONTRACT_STD",
    "CONTRACT_WIDTH",
    "MODEL_VARIANT_NAME",
    "DynamicPosShiftedWindowAttention3d",
    "Swin3DTBinary50DynamicPos",
    "build_swin3d_t_400_dynamic_pos",
    "build_swin3d_t_binary50_dynamic_pos",
    "independent_relative_position_index",
    "replace_head_for_binary",
    "select_dynamic_relative_position_index",
]
