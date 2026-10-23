import unittest

import torch
from torch import nn

from development.mixnet_i3d.temporal_features import ThreeWindowFeatureExtractor


class TemporalFeatureTests(unittest.TestCase):
    def test_time_windows_and_normalization(self):
        # 每個窗口、時間位置與通道都有不同數值。
        values = torch.arange(72, dtype=torch.float32).reshape(2, 3, 4, 3) / 100
        clips = values.permute(0, 1, 3, 2)
        clips = clips.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 2, 2)

        mean = (0.1, 0.2, 0.3)
        std = (0.5, 0.25, 1.0)
        extractor = ThreeWindowFeatureExtractor(
            nn.Identity(), mean=mean, std=std
        )
        features = extractor(clips)
        expected = (values - torch.tensor(mean)) / torch.tensor(std)

        self.assertEqual(tuple(features.shape), (2, 3, 4, 3))
        torch.testing.assert_close(features, expected)

    def test_backbone_and_batchnorm_stay_frozen(self):
        backbone = nn.BatchNorm3d(3)
        extractor = ThreeWindowFeatureExtractor(
            backbone, mean=(0, 0, 0), std=(1, 1, 1)
        )
        before = {
            key: value.clone() for key, value in backbone.state_dict().items()
        }

        extractor.train()
        features = extractor(
            torch.randn(2, 3, 3, 4, 2, 2, requires_grad=True)
        )

        self.assertFalse(backbone.training)
        self.assertFalse(features.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))
        for key, value in backbone.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_downstream_head_can_learn(self):
        torch.manual_seed(42)
        backbone = nn.Conv3d(3, 4, kernel_size=1)
        extractor = ThreeWindowFeatureExtractor(
            backbone, mean=(0, 0, 0), std=(1, 1, 1)
        )
        features = extractor(torch.randn(2, 3, 3, 4, 2, 2))

        head = nn.Linear(4, 1)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
        before = head.weight.detach().clone()

        # 測試用分類頭，暫時平均窗口和時間。
        logits = head(features.mean(dim=(1, 2))).squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(logits, torch.tensor([0.0, 1.0]))
        loss.backward()

        self.assertTrue(torch.isfinite(head.weight.grad).all().item())
        self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))

        optimizer.step()
        self.assertFalse(torch.equal(before, head.weight))


if __name__ == "__main__":
    unittest.main()
