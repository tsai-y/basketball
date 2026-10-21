import unittest

import timm
import torch

from development.mixnet_i3d.blocks import InvertedResidual3d, DepthwiseSeparableConv3d


class BlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = timm.create_model("mixnet_s.ft_in1k", pretrained=False)

    def test_downsampling_block(self):
        old = self.model.blocks[2][0]  # 24 → 40 通道，空間減半
        new = InvertedResidual3d(old).eval()

        with torch.inference_mode():
            y = new(torch.randn(1, 24, 4, 8, 8))

        self.assertEqual(tuple(y.shape), (1, 40, 4, 4, 4))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_skip_connection(self):
        old = self.model.blocks[2][1]  # 輸入、輸出形狀相同
        new = InvertedResidual3d(old).eval()
        x = torch.randn(1, 40, 4, 4, 4)

        # 將主分支最後的 BN 輸出設為零；若殘差正確，輸出應等於 x。
        with torch.no_grad():
            new.bn3[0].weight.zero_()
            new.bn3[0].bias.zero_()
        with torch.inference_mode():
            y = new(x)

        torch.testing.assert_close(y, x, rtol=0, atol=0)

    def test_first_depthwise_block_matches_2d_at_middle_frame(self):
        old = self.model.blocks[0][0].eval()
        new = DepthwiseSeparableConv3d(old).eval()
        self.assertTrue(old.has_skip)

        x2d = torch.randn(1, 16, 8, 8)
        x3d = x2d.unsqueeze(2).repeat(1, 1, 3, 1, 1)

        with torch.inference_mode():
            expected = old(x2d)
            output = new(x3d)

        self.assertEqual(tuple(output.shape), (1, 16, 3, 8, 8))
        torch.testing.assert_close(
            output[:, :, 1], expected, rtol=1e-4, atol=1e-5
        )

if __name__ == "__main__":
    unittest.main()
