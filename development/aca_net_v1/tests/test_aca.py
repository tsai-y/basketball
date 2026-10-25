import unittest
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.modules import (
    ACAModule, LSTA, ReconstructionConfig, TSCI, dynamic_temporal_conv,
)


class ACATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)

    def test_dynamic_matches_independent_conv3d(self):
        x = torch.randn(2, 3, 5, 4, 2, dtype=torch.float64)
        k = torch.randn(2, 3, 3, dtype=torch.float64).softmax(-1)
        expected = torch.cat([
            F.conv3d(x[n:n + 1], k[n].reshape(3, 1, 3, 1, 1),
                     padding=(1, 0, 0), groups=3) for n in range(2)
        ])
        torch.testing.assert_close(dynamic_temporal_conv(x, k), expected)

    def test_dynamic_video_isolation_and_spatial_preservation(self):
        x = torch.randn(2, 2, 5, 3, 4)
        kernels = torch.zeros(2, 2, 3)
        kernels[0, :, 1] = 1
        kernels[1, :, 0] = 1
        out = dynamic_temporal_conv(x, kernels)
        torch.testing.assert_close(out[0], x[0])
        torch.testing.assert_close(out[1, :, 1:], x[1, :, :-1])
        torch.testing.assert_close(out[1, :, 0], torch.zeros_like(x[1, :, 0]))

    def test_dynamic_autograd(self):
        x = torch.randn(1, 2, 3, 2, 2, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(dynamic_temporal_conv, (x, k)))

    def test_tsci_half_when_all_attention_logits_zero(self):
        module = TSCI().eval()
        for p in module.parameters():
            nn.init.zeros_(p)
        x = torch.randn(2, 3, 4, 5, 7)
        torch.testing.assert_close(module(x), x * 0.5)

    def test_tsci_framewise_semantics_and_batch_isolation(self):
        module = TSCI().eval()
        x = torch.randn(2, 3, 4, 5, 7)
        out = module(x)
        for n in range(2):
            for t in range(4):
                single = module(x[n:n + 1, :, t:t + 1])
                torch.testing.assert_close(single, out[n:n + 1, :, t:t + 1])

    def test_aca_parallel_sum_not_composition(self):
        class Scale(nn.Module):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor

            def forward(self, x):
                return x * self.factor

        module = ACAModule(4, ReconstructionConfig(frames=3))
        module.lsta, module.tsci = Scale(2), Scale(3)
        x = torch.randn(1, 4, 3, 2, 5)
        torch.testing.assert_close(module(x), x * 5)

    def test_lsta_gradients_reach_gate_short_and_dynamic_paths(self):
        module = LSTA(4, ReconstructionConfig(frames=4, reduction=2))
        x = torch.randn(2, 4, 4, 3, 5, requires_grad=True)
        out = module(x)
        self.assertEqual(out.shape, x.shape)
        out.square().mean().backward()
        for name, p in module.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(module.kernel_generator[2].weight.grad.abs().sum().item(), 0)

    def test_config_and_shape_rejections(self):
        for kwargs in ({"frames": 0}, {"dynamic_kernel": 4}, {"short_kernel": 2},
                       {"gate_groups": "silent_auto"}, {"reduction": 0}):
            with self.assertRaises(ValueError):
                ReconstructionConfig(**kwargs)
        module = LSTA(4, ReconstructionConfig(frames=4))
        with self.assertRaises(ValueError):
            module(torch.randn(1, 4, 3, 2, 2))

    def test_baseline_matches_framewise_torchvision_after_transfer(self):
        source = resnet50(weights=None).eval()
        model = VideoResNet50(config=ReconstructionConfig(frames=2)).eval()
        info = model.load_imagenet_state(source.state_dict())
        self.assertGreater(info["backbone_tensors_loaded"], 300)
        # Synthetic source here verifies transfer mechanics, NOT ImageNet accuracy.
        source.fc = model.fc
        x = torch.randn(1, 3, 2, 32, 40)
        with torch.no_grad():
            flat = x.permute(0, 2, 1, 3, 4).reshape(2, 3, 32, 40)
            expected = source(flat).mean(dim=0, keepdim=True)
            torch.testing.assert_close(model(x), expected, rtol=1e-4, atol=1e-5)

    def test_incomplete_transfer_rejected_before_state_changes(self):
        model = VideoResNet50(config=ReconstructionConfig(frames=2))
        before = model.conv1.weight.detach().clone()
        with self.assertRaises(ValueError):
            model.load_imagenet_state({"conv1.weight": torch.zeros_like(before)})
        torch.testing.assert_close(model.conv1.weight, before, rtol=0, atol=0)

    def test_transfer_preserves_new_attention_and_task_head(self):
        source = resnet50(weights=None)
        model = VideoResNet50(variant="tsci", config=ReconstructionConfig(frames=2))
        keep = {k: v.clone() for k, v in model.state_dict().items()
                if ".aca." in k or k.startswith("fc.")}
        model.load_imagenet_state(source.state_dict())
        actual = model.state_dict()
        for key, value in keep.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        for key, value in source.state_dict().items():
            if not key.startswith("fc."):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)

    def test_bad_shape_transfer_rejected_before_mutation(self):
        source = resnet50(weights=None).state_dict()
        model = VideoResNet50()
        before = model.conv1.weight.detach().clone()
        source["layer4.2.conv3.weight"] = torch.empty(1)
        with self.assertRaises(ValueError):
            model.load_imagenet_state(source)
        torch.testing.assert_close(model.conv1.weight, before, rtol=0, atol=0)

    def test_full_shapes_and_module_placement_on_meta(self):
        for variant in ("baseline", "lsta", "tsci", "aca"):
            with torch.device("meta"):
                model = VideoResNet50(variant=variant)
                self.assertEqual(model(torch.empty(1, 3, 50, 120, 160)).shape, (1, 8))
            blocks = [m for m in model.modules() if type(m).__name__ == "VideoBottleneck"]
            self.assertEqual(len(blocks), 16)
        # Literal dense interpretation has a material mismatch with reported size.
        self.assertGreater(sum(p.numel() for p in model.parameters()), 30_000_000)


if __name__ == "__main__":
    unittest.main()
