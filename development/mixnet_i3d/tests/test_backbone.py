import unittest

import timm
import torch

from development.mixnet_i3d.backbone import MixNetI3DBackbone


class BackboneTests(unittest.TestCase):
    def test_small_forward(self):
        model2d = timm.create_model(
            "mixnet_s.ft_in1k", pretrained=False
        ).eval()
        model3d = MixNetI3DBackbone(model2d).eval()

        x = torch.randn(1, 3, 8, 64, 64)
        with torch.inference_mode():
            y = model3d(x)

        self.assertEqual(tuple(y.shape), (1, 1536, 1, 2, 2))
        self.assertTrue(torch.isfinite(y).all().item())


if __name__ == "__main__":
    unittest.main()
