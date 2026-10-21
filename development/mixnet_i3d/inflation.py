import torch
from torch import nn
from copy import deepcopy

def inflate_conv2d(
    old: nn.Conv2d, *, temporal_stride: int = 1
) -> nn.Conv3d:
    """將 timm MixNet-S 的 Conv2d 膨脹為 Conv3d。"""
    if not isinstance(old, nn.Conv2d):
        raise TypeError("old must be Conv2d")
    if temporal_stride not in (1, 2):
        raise ValueError("temporal_stride must be 1 or 2")

    kt = 1 if old.kernel_size == (1, 1) else 3

    new = nn.Conv3d(
        in_channels=old.in_channels,
        out_channels=old.out_channels,
        kernel_size=(kt, *old.kernel_size),
        stride=(temporal_stride, *old.stride),
        padding=(kt // 2, *old.padding),
        dilation=(1, *old.dilation),
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )

    with torch.no_grad():
        # 權重形狀：[out, in/groups, H, W] → [out, in/groups, T, H, W]
        new.weight.copy_(
            old.weight.unsqueeze(2).repeat(1, 1, kt, 1, 1) / kt
        )
        if old.bias is not None:
            new.bias.copy_(old.bias)

    return new

def inflate_bn_act2d(old: nn.BatchNorm2d) -> nn.Sequential:
    """轉換 timm MixNet-S 的 BatchNormAct2d。"""
    if not isinstance(old, nn.BatchNorm2d):
        raise TypeError("old must be BatchNorm2d")
    if not isinstance(old.drop, nn.Identity):
        raise NotImplementedError("Non-identity drop needs a separate 3D rule")

    bn3d = nn.BatchNorm3d(
        old.num_features,
        eps=old.eps,
        momentum=old.momentum,
        affine=old.affine,
        track_running_stats=old.track_running_stats,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    bn3d.load_state_dict(old.state_dict(), strict=True)

    new = nn.Sequential(bn3d, deepcopy(old.drop), deepcopy(old.act))
    new.train(old.training)
    return new
class SqueezeExcite3d(nn.Module):
    def __init__(self, old: nn.Module):
        super().__init__()
        self.conv_reduce = inflate_conv2d(old.conv_reduce)
        self.act1 = deepcopy(old.act1)
        self.conv_expand = inflate_conv2d(old.conv_expand)
        self.gate = deepcopy(old.gate)
        self.train(old.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected [B, C, T, H, W]")

        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class MixedConv3d(nn.Module):
    def __init__(self, old: nn.Module, *, temporal_stride: int = 1):
        super().__init__()
        self.branches = nn.ModuleList(
            inflate_conv2d(branch, temporal_stride=temporal_stride)
            for branch in old.children()
        )
        if not self.branches:
            raise ValueError("MixedConv must have at least one branch")
        self.split_sizes = tuple(branch.in_channels for branch in self.branches)
        self.train(old.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != sum(self.split_sizes):
            raise ValueError("Expected [B, matching_channels, T, H, W]")

        parts = torch.split(x, self.split_sizes, dim=1)
        outputs = [
            branch(part) for branch, part in zip(self.branches, parts)
        ]
        return torch.cat(outputs, dim=1)
