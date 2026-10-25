"""ACA components; unresolved paper details are explicit in ReconstructionConfig.

Tensor convention: [batch, channels, time, height, width]. Figure 1 combines
LSTA and TSCI in parallel. This is a reconstruction, not author-supplied code.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReconstructionConfig:
    frames: int = 50
    reduction: int = 16
    short_kernel: int = 3
    dynamic_kernel: int = 3
    temporal_hidden_ratio: float = 0.25
    gate_sigmoid: bool = True
    # Literal dense Conv3D interpretation; never silently substitute depthwise.
    gate_groups: str = "dense"

    def __post_init__(self):
        for name in ("frames", "reduction", "short_kernel", "dynamic_kernel"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.short_kernel % 2 != 1 or self.dynamic_kernel % 2 != 1:
            raise ValueError("Temporal kernels must be odd for symmetric padding")
        if not 0 < self.temporal_hidden_ratio <= 1:
            raise ValueError("temporal_hidden_ratio must lie in (0, 1]")
        if type(self.gate_sigmoid) is not bool:
            raise ValueError("gate_sigmoid must be bool")
        if self.gate_groups not in {"dense", "depthwise"}:
            raise ValueError("gate_groups must be dense or depthwise")


def dynamic_temporal_conv(x, kernels):
    """Per-video, per-channel SAME temporal cross-correlation (paper Eq. 6).

    kernels: [N,C,K]. Spatial sites share a channel's kernel; independent
    videos never share a generated kernel. unfold avoids an H*W*K expansion.
    """
    if x.ndim != 5 or kernels.ndim != 3 or kernels.shape[:2] != x.shape[:2]:
        raise ValueError("Expected x[N,C,T,H,W] and kernels[N,C,K]")
    k = kernels.shape[-1]
    if k < 1 or k % 2 != 1:
        raise ValueError("Kernel length must be positive and odd")
    windows = F.pad(x, (0, 0, 0, 0, k // 2, k // 2)).unfold(2, k, 1)
    return torch.einsum("ncthwk,nck->ncthw", windows, kernels)


class LSTA(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.config = config
        groups = channels if config.gate_groups == "depthwise" else 1

        def conv_block(kernel):
            return nn.Sequential(
                nn.Conv3d(channels, channels, kernel, padding=kernel // 2,
                          groups=groups, bias=False),
                nn.InstanceNorm3d(channels, affine=False, track_running_stats=False),
                nn.ReLU(),
            )

        self.gate_value = conv_block(1)
        self.gate_control = conv_block(3)
        self.gate_output = conv_block(3)
        hidden = max(1, channels // config.reduction)
        self.short_term = nn.Sequential(
            nn.Conv1d(channels, hidden, config.short_kernel,
                      padding=config.short_kernel // 2),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, config.short_kernel,
                      padding=config.short_kernel // 2),
            nn.Sigmoid(),
        )
        temporal_hidden = max(1, int(config.frames * config.temporal_hidden_ratio))
        # Shared MLP along T, applied independently to every channel (Fig. 2).
        self.kernel_generator = nn.Sequential(
            nn.Linear(config.frames, temporal_hidden), nn.ReLU(),
            nn.Linear(temporal_hidden, config.dynamic_kernel),
        )

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != self.config.frames:
            raise ValueError("LSTA requires its configured temporal length")
        control = self.gate_control(x)
        if self.config.gate_sigmoid:
            control = control.sigmoid()  # Fig. 2; absent from printed Eq. 1.
        g = x + self.gate_output(self.gate_value(x) * control)
        pooled = g.mean(dim=(-2, -1))
        weights = self.short_term(pooled)
        # Eq. 4 writes P*V, but P is 3D after pooling. Use the full G feature
        # map to preserve the spatial shape required by Fig. 2 and Eq. 6.
        z = g * weights[..., None, None]
        kernels = self.kernel_generator(pooled).softmax(dim=-1)
        return dynamic_temporal_conv(z, kernels)


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        pooled = torch.cat((x.mean(dim=1, keepdim=True),
                            x.amax(dim=1, keepdim=True)), dim=1)
        return x * self.bn(self.conv(pooled)).sigmoid()


class TSCI(nn.Module):
    """Figure 3 / Eqs. 7-8: three independent 7x7 Conv2D attention gates."""
    def __init__(self):
        super().__init__()
        self.channel = SpatialGate()
        self.height = SpatialGate()
        self.width = SpatialGate()

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("Expected [N,C,T,H,W]")
        n, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
        a = self.channel(frames)
        b = self.height(frames.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        d = self.width(frames.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = (a + b + d) / 3
        return out.reshape(n, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class ACAModule(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.lsta = LSTA(channels, config)
        self.tsci = TSCI()

    def forward(self, x):
        return self.lsta(x) + self.tsci(x)
