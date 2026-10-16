"""Gate-only reproduction of hassony2 C03 Inception-v1 I3D topology.

The module names and parameter/buffer slots intentionally match the 287-entry
PyTorch 0.3 state_dict at commit 786fbae.  The pinned source has an unusable
duplicate ``mixed_4b`` call in ``forward``; this implementation follows the
declared layer topology (one call per layer) and exposes the pre-pooling
``Mixed_5c`` endpoint.  LegacyBatchNorm3d deliberately has no modern
``num_batches_tracked`` buffer, so strict loading means exact one-to-one state
coverage rather than PyTorch silently synthesizing 57 compatibility buffers.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ENDPOINTS = (
    "Conv3d_1a_7x7", "MaxPool3d_2a_3x3", "Conv3d_2b_1x1",
    "Conv3d_2c_3x3", "MaxPool3d_3a_3x3", "Mixed_3b", "Mixed_3c",
    "MaxPool3d_4a_3x3", "Mixed_4b", "Mixed_4c", "Mixed_4d", "Mixed_4e",
    "Mixed_4f", "MaxPool3d_5a_2x2", "Mixed_5b", "Mixed_5c",
)


def _padding_shape(kernel: tuple[int, int, int], stride: tuple[int, int, int], temporal_mod: int = 0) -> tuple[int, ...]:
    values: list[int] = []
    for index, (filter_dim, stride_dim) in enumerate(zip(kernel, stride)):
        mod = temporal_mod if index == 0 else 0
        along = max(filter_dim - mod, 0) if mod else max(filter_dim - stride_dim, 0)
        before = along // 2
        values.extend((before, along - before))
    depth_before, depth_after = values[:2]
    return tuple(values[2:] + [depth_before, depth_after])


class LegacyBatchNorm3d(nn.Module):
    """PyTorch-0.3-compatible BN state with deterministic eval-only forward."""

    def __init__(self, channels: int, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, value: Tensor) -> Tensor:
        if self.training:
            raise RuntimeError("Gate 1C legacy BatchNorm is eval-only")
        return F.batch_norm(
            value, self.running_mean, self.running_var, self.weight, self.bias,
            training=False, momentum=0.1, eps=self.eps,
        )


class Unit3Dpy(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (1, 1, 1),
        stride: tuple[int, int, int] = (1, 1, 1),
        activation: str | None = "relu",
        padding: str = "SAME",
        use_bias: bool = False,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if padding not in {"SAME", "VALID"}:
            raise ValueError(f"Unknown padding: {padding}")
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self.use_bn = use_bn
        base = _padding_shape(kernel_size, stride)
        symmetric = len(set(base)) == 1
        self.simplify_pad = symmetric
        if padding == "SAME" and not symmetric:
            count = stride[0] if stride[0] > 1 else 1
            self.pads = nn.ModuleList(
                nn.ConstantPad3d(_padding_shape(kernel_size, stride, mod), 0.0)
                for mod in range(count)
            )
            conv_padding: int | tuple[int, int, int] = 0
        elif padding == "SAME":
            conv_padding = base[0]
        else:
            conv_padding = 0
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=conv_padding, bias=use_bias,
        )
        if use_bn:
            self.batch3d = LegacyBatchNorm3d(out_channels, eps=1e-3)

    def forward(self, value: Tensor) -> Tensor:
        if self.padding == "SAME" and not self.simplify_pad:
            value = self.pads[value.shape[2] % self.stride[0]](value)
        value = self.conv3d(value)
        if self.use_bn:
            value = self.batch3d(value)
        if self.activation == "relu":
            value = F.relu(value)
        return value


class MaxPool3dTFPadding(nn.Module):
    def __init__(self, kernel_size: tuple[int, int, int], stride: tuple[int, int, int]) -> None:
        super().__init__()
        self.stride = stride
        count = stride[0] if stride[0] > 1 else 1
        self.pads = nn.ModuleList(
            nn.ConstantPad3d(_padding_shape(kernel_size, stride, mod), 0.0)
            for mod in range(count)
        )
        self.pool = nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, value: Tensor) -> Tensor:
        value = self.pads[value.shape[2] % self.stride[0]](value)
        return self.pool(value)


class Mixed(nn.Module):
    def __init__(self, in_channels: int, out_channels: Iterable[int]) -> None:
        super().__init__()
        output = list(out_channels)
        self.branch_0 = Unit3Dpy(in_channels, output[0])
        self.branch_1 = nn.Sequential(
            Unit3Dpy(in_channels, output[1]),
            Unit3Dpy(output[1], output[2], kernel_size=(3, 3, 3)),
        )
        self.branch_2 = nn.Sequential(
            Unit3Dpy(in_channels, output[3]),
            Unit3Dpy(output[3], output[4], kernel_size=(3, 3, 3)),
        )
        self.branch_3 = nn.Sequential(
            MaxPool3dTFPadding((3, 3, 3), (1, 1, 1)),
            Unit3Dpy(in_channels, output[5]),
        )

    def forward(self, value: Tensor) -> Tensor:
        return torch.cat((self.branch_0(value), self.branch_1(value), self.branch_2(value), self.branch_3(value)), 1)


class InceptionI3d(nn.Module):
    """RGB Inception-v1 I3D with the C03 checkpoint's exact state slots."""

    def __init__(self, num_classes: int = 400, modality: str = "rgb", dropout_prob: float = 0.0) -> None:
        super().__init__()
        if modality != "rgb":
            raise ValueError("Gate 1C is frozen to RGB")
        self.num_classes = num_classes
        self.modality = modality
        self.conv3d_1a_7x7 = Unit3Dpy(3, 64, (7, 7, 7), (2, 2, 2))
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding((1, 3, 3), (1, 2, 2))
        self.conv3d_2b_1x1 = Unit3Dpy(64, 64)
        self.conv3d_2c_3x3 = Unit3Dpy(64, 192, (3, 3, 3))
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding((1, 3, 3), (1, 2, 2))
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])
        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding((3, 3, 3), (2, 2, 2))
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])
        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding((2, 2, 2), (2, 2, 2))
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])
        self.avg_pool = nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.conv3d_0c_1x1 = Unit3Dpy(1024, num_classes, activation=None, use_bias=True, use_bn=False)
        self.softmax = nn.Softmax(1)

    def _endpoint_modules(self) -> OrderedDict[str, nn.Module]:
        return OrderedDict([
            ("Conv3d_1a_7x7", self.conv3d_1a_7x7),
            ("MaxPool3d_2a_3x3", self.maxPool3d_2a_3x3),
            ("Conv3d_2b_1x1", self.conv3d_2b_1x1),
            ("Conv3d_2c_3x3", self.conv3d_2c_3x3),
            ("MaxPool3d_3a_3x3", self.maxPool3d_3a_3x3),
            ("Mixed_3b", self.mixed_3b), ("Mixed_3c", self.mixed_3c),
            ("MaxPool3d_4a_3x3", self.maxPool3d_4a_3x3),
            ("Mixed_4b", self.mixed_4b), ("Mixed_4c", self.mixed_4c),
            ("Mixed_4d", self.mixed_4d), ("Mixed_4e", self.mixed_4e),
            ("Mixed_4f", self.mixed_4f), ("MaxPool3d_5a_2x2", self.maxPool3d_5a_2x2),
            ("Mixed_5b", self.mixed_5b), ("Mixed_5c", self.mixed_5c),
        ])

    def extract_endpoints(self, value: Tensor, requested: Iterable[str] = ("Mixed_5c",)) -> OrderedDict[str, Tensor]:
        wanted = tuple(requested)
        unknown = set(wanted) - set(ENDPOINTS)
        if unknown:
            raise ValueError(f"Unknown endpoints: {sorted(unknown)}")
        outputs: OrderedDict[str, Tensor] = OrderedDict()
        for name, module in self._endpoint_modules().items():
            value = module(value)
            if name in wanted:
                outputs[name] = value
            if len(outputs) == len(wanted):
                break
        return outputs

    def forward_features(self, value: Tensor) -> Tensor:
        return self.extract_endpoints(value, ("Mixed_5c",))["Mixed_5c"]

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        value = self.forward_features(value)
        value = self.avg_pool(value)
        value = self.dropout(value)
        value = self.conv3d_0c_1x1(value).squeeze(3).squeeze(3).mean(2)
        return self.softmax(value), value
