"""Isolated single-clip 50-frame binary classifier wrapper over MixNet-I3D.

No training entry point, no LSTA/TSCI modules, no pretrained weight download.
"""

from development.mixnet_binary50_v1.model import (
    CLASS_ORDER,
    EXPECTED_CHANNELS,
    EXPECTED_FRAMES,
    EXPECTED_HEIGHT,
    EXPECTED_INPUT_SHAPE,
    EXPECTED_WIDTH,
    MODEL_NAME,
    UPSTREAM_NORMALIZATION,
    MixNetBinary50Model,
    build_mixnet_binary50,
    parameter_summary,
    snapshot_bn3d_buffers,
    summarize_bn3d,
    trace_stage_shapes,
)

__all__ = [
    "CLASS_ORDER",
    "EXPECTED_CHANNELS",
    "EXPECTED_FRAMES",
    "EXPECTED_HEIGHT",
    "EXPECTED_INPUT_SHAPE",
    "EXPECTED_WIDTH",
    "MODEL_NAME",
    "UPSTREAM_NORMALIZATION",
    "MixNetBinary50Model",
    "build_mixnet_binary50",
    "parameter_summary",
    "snapshot_bn3d_buffers",
    "summarize_bn3d",
    "trace_stage_shapes",
]
