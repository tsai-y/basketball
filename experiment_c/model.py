"""Reuse A1/A2's feature stack; replace their per-video head for C.

The original C3DStyle.forward and head remain unchanged. C imports its
features and GAP modules, then applies a separate video-level classifier.
"""

import torch
from torch import nn

from baseline.model import C3DStyle
from experiment_c.aggregate import mean_l2_clip_features


class ContinuousModifiedC3D(nn.Module):
    def __init__(self):
        super().__init__()
        original = C3DStyle(num_classes=2)
        self.features = original.features
        self.pool = original.pool
        self.classifier = nn.Linear(256, 2)

    def extract_clip_features(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (3, 16, 112, 112):
            raise ValueError(f"Expected [B,3,16,112,112], got {tuple(clips.shape)}")
        return torch.flatten(self.pool(self.features(clips)), start_dim=1)

    def forward_video(self, clips: torch.Tensor, clip_batch_size: int):
        if clip_batch_size < 1 or clips.shape[0] < 1:
            raise ValueError("A video needs at least one clip and positive chunk size")
        pieces = [self.extract_clip_features(chunk)
                  for chunk in clips.split(clip_batch_size)]
        descriptor = mean_l2_clip_features(torch.cat(pieces, dim=0))
        return self.classifier(descriptor), descriptor
