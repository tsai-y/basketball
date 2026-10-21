import torch
from torch import nn
from timm.layers.mixed_conv2d import MixedConv2d

from development.mixnet_i3d.inflation import (
    MixedConv3d,
    SqueezeExcite3d,
    inflate_bn_act2d,
    inflate_conv2d,
)

def _inflate_conv(old: nn.Module, *, temporal_stride: int = 1) -> nn.Module:
    if isinstance(old, MixedConv2d):
        return MixedConv3d(old, temporal_stride=temporal_stride)
    if isinstance(old, nn.Conv2d):
        return inflate_conv2d(old, temporal_stride=temporal_stride)
    raise TypeError(f"Unsupported convolution: {type(old).__name__}")

class InvertedResidual3d(nn.Module):
    def __init__(self, old: nn.Module, *, temporal_stride: int = 1):
        super().__init__()
        if old.conv_s2d is not None:
            raise NotImplementedError("conv_s2d is not handled")
        if not isinstance(old.aa, nn.Identity):
            raise NotImplementedError("Non-identity aa is not handled")
        if not isinstance(old.drop_path, nn.Identity):
            raise NotImplementedError("Non-identity drop_path is not handled")
        if old.has_skip and temporal_stride != 1:
            raise ValueError("A skip block cannot downsample time")

        self.conv_pw = _inflate_conv(old.conv_pw)
        self.bn1 = inflate_bn_act2d(old.bn1)
        self.conv_dw = _inflate_conv(old.conv_dw, temporal_stride=temporal_stride)
        self.bn2 = inflate_bn_act2d(old.bn2)

        self.se = (
            nn.Identity()
            if isinstance(old.se, nn.Identity)
            else SqueezeExcite3d(old.se)
        )

        self.conv_pwl = _inflate_conv(old.conv_pwl)
        self.bn3 = inflate_bn_act2d(old.bn3)
        self.has_skip = old.has_skip
        self.train(old.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.bn1(self.conv_pw(x))
        x = self.bn2(self.conv_dw(x))
        x = self.se(x)
        x = self.bn3(self.conv_pwl(x))
        if self.has_skip:
            x = x + shortcut
        return x
class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, old: nn.Module, *, temporal_stride: int = 1):
        super().__init__()
        if old.conv_s2d is not None:
            raise NotImplementedError("conv_s2d is not handled")
        if not all(
            isinstance(part, nn.Identity)
            for part in (old.aa, old.se, old.drop_path)
        ):
            raise NotImplementedError("Expected identity aa, se and drop_path")
        if old.has_skip and temporal_stride != 1:
            raise ValueError("A skip block cannot downsample time")

        self.conv_dw = inflate_conv2d(
            old.conv_dw, temporal_stride=temporal_stride
        )
        self.bn1 = inflate_bn_act2d(old.bn1)
        self.conv_pw = inflate_conv2d(old.conv_pw)
        self.bn2 = inflate_bn_act2d(old.bn2)
        self.has_skip = old.has_skip
        self.train(old.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.bn1(self.conv_dw(x))
        x = self.bn2(self.conv_pw(x))
        if self.has_skip:
            x = x + shortcut
        return x
