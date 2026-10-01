"""Video descriptor for Experiment C (Basketball-51 project adaptation)."""

import torch
from torch.nn import functional as F


def mean_l2_clip_features(features: torch.Tensor) -> torch.Tensor:
    """[clips, 256] -> [256]; retain gradients for video-level training."""
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != 256:
        raise ValueError(f"Expected [clips, 256], got {tuple(features.shape)}")
    return F.normalize(features.mean(dim=0), p=2, dim=0)
