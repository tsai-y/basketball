import torch
from torch import nn

from development.mixnet_i3d.blocks import (
    DepthwiseSeparableConv3d,
    InvertedResidual3d,
)
from development.mixnet_i3d.inflation import (
    inflate_bn_act2d,
    inflate_conv2d,
)


class MixNetI3DBackbone(nn.Module):
    def __init__(self, old: nn.Module):
        super().__init__()
        stage_sizes = tuple(len(stage) for stage in old.blocks)
        if stage_sizes != (1, 2, 4, 3, 3, 3):
            raise ValueError(f"Unexpected MixNet-S stages: {stage_sizes}")

        self.conv_stem = inflate_conv2d(old.conv_stem, temporal_stride=2)
        self.bn1 = inflate_bn_act2d(old.bn1)

        stages = []
        for stage_id, stage in enumerate(old.blocks):
            converted = []
            for block_id, block in enumerate(stage):
                if stage_id == 0:
                    converted.append(DepthwiseSeparableConv3d(block))
                else:
                    time_stride = (
                        2 if stage_id in (3, 5) and block_id == 0 else 1
                    )
                    converted.append(
                        InvertedResidual3d(block, temporal_stride=time_stride)
                    )
            stages.append(nn.Sequential(*converted))
        self.blocks = nn.Sequential(*stages)

        self.conv_head = inflate_conv2d(old.conv_head)
        self.bn2 = inflate_bn_act2d(old.bn2)
        self.num_features = old.num_features
        self.train(old.training)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected [B, C, T, H, W]")
        x = self.bn1(self.conv_stem(x))
        x = self.blocks(x)
        x = self.bn2(self.conv_head(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
