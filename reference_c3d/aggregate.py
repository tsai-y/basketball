"""Paper-confirmed video descriptor: mean clip fc6, then L2-normalize."""

import torch
import torch.nn.functional as F


def aggregate_fc6(clip_features: torch.Tensor) -> torch.Tensor:
    """[clips, 4096] -> [4096]; callers handle one video at a time."""
    if clip_features.ndim != 2 or clip_features.shape[0] < 1 or clip_features.shape[1] != 4096:
        raise ValueError(f"Expected nonempty [clips,4096], got {tuple(clip_features.shape)}")
    if not torch.isfinite(clip_features).all():
        raise ValueError("Nonfinite clip features")
    return F.normalize(clip_features.mean(dim=0), p=2, dim=0)
