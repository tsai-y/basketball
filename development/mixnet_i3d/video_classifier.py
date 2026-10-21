import torch
from torch import nn

from development.mixnet_i3d.backbone import MixNetI3DBackbone


class ThreeWindowMakeMissModel(nn.Module):
    def __init__(self, backbone: MixNetI3DBackbone, *, mean, std):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.num_features, 1)

        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or (std <= 0).any():
            raise ValueError("Expected three RGB means and positive stds")
        self.register_buffer("rgb_mean", mean.view(1, 1, 3, 1, 1, 1))
        self.register_buffer("rgb_std", std.view(1, 1, 3, 1, 1, 1))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # Dataset supplies RGB float32 in [0, 1].
        if clips.ndim != 6 or clips.shape[1:3] != (3, 3):
            raise ValueError("Expected [B, 3, 3, T, H, W]")

        clips = (clips - self.rgb_mean) / self.rgb_std
        batch, windows, channels, frames, height, width = clips.shape
        maps = self.backbone(
            clips.reshape(batch * windows, channels, frames, height, width)
        )
        window_features = maps.mean(dim=(2, 3, 4))
        video_feature = window_features.reshape(
            batch, windows, -1
        ).mean(dim=1)
        return self.classifier(video_feature).squeeze(-1)
