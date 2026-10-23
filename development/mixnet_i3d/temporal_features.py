import torch
from torch import nn


class ThreeWindowFeatureExtractor(nn.Module):
    """Frozen backbone producing [B, windows, time, channels]."""

    def __init__(self, backbone: nn.Module, *, mean, std):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or (std <= 0).any():
            raise ValueError("Expected three RGB means and positive stds")

        self.register_buffer("rgb_mean", mean.view(1, 1, 3, 1, 1, 1))
        self.register_buffer("rgb_std", std.view(1, 1, 3, 1, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen BatchNorm must keep its running statistics unchanged.
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 6 or clips.shape[1:3] != (3, 3):
            raise ValueError("Expected [B, 3, 3, T, H, W]")

        clips = (clips - self.rgb_mean) / self.rgb_std
        batch, windows, channels, frames, height, width = clips.shape

        maps = self.backbone(
            clips.reshape(batch * windows, channels, frames, height, width)
        )
        # [B*3, C, T, H, W] -> [B*3, T, C]
        features = maps.mean(dim=(-2, -1)).transpose(1, 2)

        return features.reshape(
            batch, windows, maps.shape[2], maps.shape[1]
        ).float()
