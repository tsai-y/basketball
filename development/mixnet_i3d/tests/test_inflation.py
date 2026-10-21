import unittest

import timm
import torch
from development.mixnet_i3d.inflation import inflate_bn_act2d, inflate_conv2d, SqueezeExcite3d, MixedConv3d
from development.mixnet_i3d.inflation import inflate_conv2d


class InflationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = timm.create_model("mixnet_s.ft_in1k", pretrained=False)

    def test_stem(self):
        old = self.model.conv_stem
        new = inflate_conv2d(old, temporal_stride=2)
        self.assertEqual(tuple(new.weight.shape), (16, 3, 3, 3, 3))
        self.assertEqual(new.stride, (2, 2, 2))
        self.assertTrue(torch.allclose(new.weight.sum(dim=2), old.weight))

    def test_pointwise(self):
        old = self.model.blocks[2][0].conv_pw
        new = inflate_conv2d(old)
        self.assertEqual(new.kernel_size, (1, 1, 1))
        self.assertTrue(torch.equal(new.weight.squeeze(2), old.weight))

    def test_mixed_depthwise_5x5(self):
        old = list(self.model.blocks[2][0].conv_dw.children())[1]
        new = inflate_conv2d(old)
        self.assertEqual(new.kernel_size, (3, 5, 5))
        self.assertEqual(new.groups, old.groups)
        self.assertTrue(torch.allclose(new.weight.sum(dim=2), old.weight))

    def test_bn_act_single_frame_matches_2d(self):
        old = self.model.blocks[2][0].bn1
        new = inflate_bn_act2d(old)

        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new[0].state_dict()[key]))
        self.assertEqual(type(new[2]), type(old.act))

        old.eval()
        new.eval()
        x = torch.randn(2, old.num_features, 5, 7)
        with torch.no_grad():
            expected = old(x)
            actual = new(x.unsqueeze(2)).squeeze(2)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_se_single_and_multiple_frames(self):
        old = self.model.blocks[2][0].se
        new = SqueezeExcite3d(old)
        old.eval()
        new.eval()

        x2d = torch.randn(2, 144, 5, 7)
        with torch.no_grad():
            expected = old(x2d)
            actual = new(x2d.unsqueeze(2)).squeeze(2)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        x3d = torch.randn(2, 144, 4, 5, 7)
        self.assertEqual(tuple(new(x3d).shape), tuple(x3d.shape))

    def test_mixed_conv_middle_frame_matches_2d(self):
        old = self.model.blocks[2][0].conv_dw
        new = MixedConv3d(old)
        old.eval()
        new.eval()

        x2d = torch.randn(1, 144, 8, 8)
        x3d = x2d.unsqueeze(2).repeat(1, 1, 3, 1, 1)

        with torch.no_grad():
            expected = old(x2d)
            output = new(x3d)

        self.assertEqual(tuple(output.shape), (1, 144, 3, 4, 4))
        torch.testing.assert_close(
            output[:, :, 1], expected, rtol=1e-4, atol=1e-5
        )

if __name__ == "__main__":
    unittest.main()
