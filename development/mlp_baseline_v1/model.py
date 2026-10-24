"""Capacity-matched, order-invariant classifier for pooled frozen features."""

from torch import nn


class PooledFeatureMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 352),
            nn.LayerNorm(352),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(352, 1),
        )

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 1536:
            raise ValueError("Expected pooled features [B, 1536]")
        return self.layers(features).squeeze(-1)
