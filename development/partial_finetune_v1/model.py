"""RGB three-window model with only the last MixNet stage unfrozen."""

import torch
from torch import nn


class PartialFineTuneModel(nn.Module):
    """Train blocks[5], conv_head, and the temporal head; freeze every BatchNorm."""

    def __init__(self, backbone, head, *, mean, std):
        super().__init__()
        if len(backbone.blocks) != 6:
            raise ValueError("Expected six MixNet stages")
        self.backbone = backbone
        self.head = head
        backbone.requires_grad_(False)
        backbone.blocks[5].requires_grad_(True)
        backbone.conv_head.requires_grad_(True)
        head.requires_grad_(True)
        # Keep both affine parameters AND running statistics frozen, including in the tail.
        for module in backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.requires_grad_(False)
        mean, std = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        if (mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Expected finite RGB means and positive finite stds")
        self.register_buffer("rgb_mean", mean.view(1, 1, 3, 1, 1, 1))
        self.register_buffer("rgb_std", std.view(1, 1, 3, 1, 1, 1))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.backbone.blocks[5].train(mode)
        self.backbone.conv_head.train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward_features(self, clips):
        if clips.ndim != 6 or tuple(clips.shape[1:4]) != (3, 3, 64):
            raise ValueError("Expected [B, 3, 3, 64, H, W]")
        if clips.dtype != torch.float32 or len(clips) == 0 or min(clips.shape[-2:]) < 2:
            raise ValueError("Expected nonempty float32 RGB clips with positive spatial size")
        batch, windows, channels, frames, height, width = clips.shape
        # Prefix has no trainable weights. Detach here, NOT around the trainable tail.
        with torch.no_grad():
            x = (clips - self.rgb_mean) / self.rgb_std
            x = x.reshape(batch * windows, channels, frames, height, width)
            x = self.backbone.bn1(self.backbone.conv_stem(x))
            for index in range(5):
                x = self.backbone.blocks[index](x)
        x = self.backbone.blocks[5](x)
        maps = self.backbone.bn2(self.backbone.conv_head(x))
        features = maps.mean(dim=(-2, -1)).transpose(1, 2)
        return features.reshape(batch, windows, maps.shape[2], maps.shape[1]).float()

    def forward(self, clips):
        return self.head(self.forward_features(clips))

    def optimizer_groups(self, *, backbone_lr=1e-5, head_lr=1e-4):
        if backbone_lr <= 0 or head_lr <= 0:
            raise ValueError("Learning rates must be positive")
        return [
            {"name": "backbone_tail", "params": [p for p in self.backbone.parameters() if p.requires_grad],
             "lr": backbone_lr},
            {"name": "temporal_head", "params": [p for p in self.head.parameters() if p.requires_grad],
             "lr": head_lr},
        ]
