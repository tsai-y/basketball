import torch
from torch import nn


class WindowTemporalTransformer(nn.Module):
    """Frozen MixNet-I3D features -> one video-level logit."""

    def __init__(self):
        super().__init__()

        self.projection = nn.Linear(1536, 256)

        # 八個窗口內時間位置，各有一個可學習的位置向量。
        self.position = nn.Parameter(torch.empty(1, 8, 256))
        nn.init.normal_(self.position, std=0.02)

        # 各層獨立初始化；同一組網路共享於三個窗口。
        self.encoder = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        ])

        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, 1)

    def forward(self, features):
        if features.ndim != 4 or tuple(features.shape[1:]) != (3, 8, 1536):
            raise ValueError("Expected [B, 3, 8, 1536]")

        batch = features.shape[0]

        # 三個窗口暫時當成三個獨立樣本，而不是串成長影片。
        x = features.reshape(batch * 3, 8, 1536)
        x = self.projection(x)
        x = x + self.position
        x = self.encoder(x)

        # 每窗口得到一個 256 維向量。
        window_features = self.norm(x).mean(dim=1)

        # 還原影片分組，再平均三窗口。
        video_features = window_features.reshape(batch, 3, 256).mean(dim=1)

        # 回傳 logits；訓練時交給 BCEWithLogitsLoss。
        return self.classifier(video_features).squeeze(-1)
