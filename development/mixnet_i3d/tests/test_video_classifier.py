import unittest

import torch
from torch import nn

from development.mixnet_i3d.video_classifier import ThreeWindowMakeMissModel


class TinyBackbone(nn.Module):
    num_features = 4

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class VideoClassifierTests(unittest.TestCase):
    def test_video_logits_and_backward(self):
        model = ThreeWindowMakeMissModel(TinyBackbone(), mean=(0, 0, 0), std=(1, 1, 1) )
        clips = torch.randn(2, 3, 3, 4, 8, 8)
        labels = torch.tensor([0.0, 1.0])

        logits = model(clips)
        self.assertEqual(tuple(logits.shape), (2,))

        loss = nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertIsNotNone(model.backbone.proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
