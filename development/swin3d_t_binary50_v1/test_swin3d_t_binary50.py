"""Focused unit tests for development.swin3d_t_binary50_v1 (CPU only, weights=None)."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

import torch
from torchvision.models.video import swin_transformer as tv_swin

from .model import (
    SWIN3D_T_WINDOW_SIZE,
    build_swin3d_t_binary50_dynamic_pos,
    independent_relative_position_index,
    replace_head_for_binary,
    select_dynamic_relative_position_index,
)


class TestSwin3DTBinary50DynamicPos(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)

    def test_dynamic_index_matches_independent_formula_across_clipped_windows(self) -> None:
        ref = tv_swin.swin3d_t(weights=None)
        base_idx = ref.features[0][0].attn.relative_position_index
        for win in [(8, 7, 7), (8, 4, 5), (4, 7, 7), (4, 4, 4), (4, 2, 2), (1, 1, 1)]:
            indep = independent_relative_position_index(SWIN3D_T_WINDOW_SIZE, win)
            dyn = select_dynamic_relative_position_index(base_idx, SWIN3D_T_WINDOW_SIZE, win)
            self.assertTrue(torch.equal(dyn, indep), f"Mismatch for window {win}")
        indep_845 = independent_relative_position_index(SWIN3D_T_WINDOW_SIZE, (8, 4, 5))
        naive_mismatches = int((base_idx[:160, :160] != indep_845).sum().item())
        self.assertEqual(naive_mismatches, 24980)

    def test_state_dict_compatibility_and_strict_contract_validation(self) -> None:
        up = tv_swin.swin3d_t(weights=None)
        dyn = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=True)
        self.assertEqual(list(up.state_dict().keys()), list(dyn.state_dict().keys()))
        for k, v in up.state_dict().items():
            if k == "head.weight":
                self.assertEqual(tuple(dyn.state_dict()[k].shape), (2, 768))
            elif k == "head.bias":
                self.assertEqual(tuple(dyn.state_dict()[k].shape), (2,))
            else:
                self.assertEqual(tuple(dyn.state_dict()[k].shape), tuple(v.shape))
        with self.assertRaises(ValueError):
            dyn(torch.randn(1, 3, 8, 64, 64))

    def test_full_877_window_matches_upstream_forward_and_gradients(self) -> None:
        torch.manual_seed(7)
        up = tv_swin.swin3d_t(weights=None)
        replace_head_for_binary(up, num_classes=2)
        dyn = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=False)
        dyn.load_state_dict(up.state_dict(), strict=True)

        x = torch.randn(1, 3, 16, 224, 224)
        up.eval()
        dyn.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(up(x), dyn(x)))

        up.train()
        dyn.train()
        torch.manual_seed(99)
        up(x).sum().backward()
        torch.manual_seed(99)
        dyn(x).sum().backward()
        up_params = dict(up.named_parameters())
        for name, p in dyn.named_parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.equal(p.grad, up_params[name].grad), f"Grad mismatch in {name}")

    def test_isolated_head_seed_does_not_touch_global_rng_or_backbone(self) -> None:
        up = tv_swin.swin3d_t(weights=None)
        sd_400 = up.state_dict()
        torch.manual_seed(12345)
        rng_before = torch.get_rng_state().clone()
        m1 = build_swin3d_t_binary50_dynamic_pos(
            strict_contract_shape=True,
            pretrained_400_state_dict=sd_400,
            head_seed=20261124,
        )
        m2 = build_swin3d_t_binary50_dynamic_pos(
            strict_contract_shape=True,
            pretrained_400_state_dict=sd_400,
            head_seed=20261124,
        )
        m3 = build_swin3d_t_binary50_dynamic_pos(
            strict_contract_shape=True,
            pretrained_400_state_dict=sd_400,
            head_seed=20261125,
        )
        self.assertTrue(torch.equal(m1.head.weight, m2.head.weight))
        self.assertTrue(torch.equal(m1.head.bias, m2.head.bias))
        self.assertFalse(torch.equal(m1.head.weight, m3.head.weight))
        for k, v in sd_400.items():
            if k not in ("head.weight", "head.bias"):
                self.assertTrue(torch.equal(m1.state_dict()[k], v), f"Non-head altered: {k}")

        # Verify replace_head_for_binary itself leaves global RNG completely unchanged
        torch.manual_seed(777)
        rng_snap = torch.get_rng_state().clone()
        replace_head_for_binary(m1, num_classes=2, head_seed=20261124)
        self.assertTrue(torch.equal(rng_snap, torch.get_rng_state()))
        self.assertTrue(torch.equal(m1.head.weight, m2.head.weight))
        _ = rng_before


if __name__ == "__main__":
    unittest.main()
