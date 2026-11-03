import unittest

import torch
from torch import nn

from development.aca_dw3d_v1.model import DW3DConfig, DW3DVideoResNet50, convert_lsta_gating, gating_audit
from development.aca_net_v1.audit_architecture import parameter_breakdown
from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.modules import LSTA, ReconstructionConfig


class DW3DModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_parameter_counts_groups_and_shape_on_meta(self):
        with torch.device("meta"):
            eight = DW3DVideoResNet50(num_classes=8)
            binary = DW3DVideoResNet50(num_classes=2)
            logits = binary(torch.empty(1, 3, 50, 120, 160))
        self.assertEqual(sum(p.numel() for p in eight.parameters()), 25476580)
        self.assertEqual(sum(p.numel() for p in binary.parameters()), 25464286)
        self.assertEqual(parameter_breakdown(eight)["categories"]["lsta_gated_3d_convolutions"], 1461376)
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertEqual(binary.config.gate_groups, "depthwise_3x3")
        audit = gating_audit(binary)
        self.assertEqual(audit["depthwise_3x3x3_convolutions"], 32)
        for item in audit["layers"]:
            c = item["channels"]
            self.assertEqual(item["value_weight_shape"], [c, c, 1, 1, 1])
            self.assertEqual(item["control_weight_shape"], [c, 1, 3, 3, 3])

    def test_conversion_preserves_value_normalization_and_other_branches(self):
        lsta = LSTA(4, DW3DConfig(frames=3))
        value, norm, relu = lsta.gate_value, lsta.gate_control[1], lsta.gate_output[2]
        short, generator = lsta.short_term, lsta.kernel_generator
        convert_lsta_gating(lsta)
        self.assertIs(value, lsta.gate_value)
        self.assertIs(norm, lsta.gate_control[1])
        self.assertIs(relu, lsta.gate_output[2])
        self.assertIs(short, lsta.short_term)
        self.assertIs(generator, lsta.kernel_generator)
        with self.assertRaises(ValueError):
            convert_lsta_gating(lsta)

    def test_all_unreplaced_initial_weights_match_dense_same_seed(self):
        torch.manual_seed(42)
        dense = VideoResNet50(variant="aca", num_classes=2)
        torch.manual_seed(42)
        lite = DW3DVideoResNet50(num_classes=2)
        source = dense.state_dict()
        changed, identical = [], 0
        for name, value in lite.state_dict().items():
            if ".gate_control.0.weight" in name or ".gate_output.0.weight" in name:
                changed.append(name)
                self.assertNotEqual(value.shape, source[name].shape)
            else:
                self.assertTrue(torch.equal(value, source[name]), name)
                identical += 1
        self.assertEqual(len(changed), 32)
        self.assertGreater(identical, 300)

    def test_small_LSTA_forward_backward_is_finite_in_all_branches(self):
        torch.manual_seed(6)
        lsta = LSTA(4, DW3DConfig(frames=3))
        convert_lsta_gating(lsta)
        x = torch.randn(2, 4, 3, 4, 4, requires_grad=True)
        y = lsta(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, p in lsta.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        for branch in (lsta.gate_value, lsta.gate_control, lsta.gate_output, lsta.short_term, lsta.kernel_generator):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in branch.parameters()), 0)

    def test_original_dense_and_all_depthwise_semantics_remain_unchanged(self):
        dense = LSTA(4, ReconstructionConfig(frames=3))
        all_dw = LSTA(4, ReconstructionConfig(frames=3, gate_groups="depthwise"))
        self.assertEqual([b[0].groups for b in (dense.gate_value, dense.gate_control, dense.gate_output)], [1, 1, 1])
        self.assertEqual([b[0].groups for b in (all_dw.gate_value, all_dw.gate_control, all_dw.gate_output)], [4, 4, 4])
        with self.assertRaises(ValueError):
            DW3DConfig(gate_groups="dense")
        with self.assertRaises(ValueError):
            DW3DVideoResNet50(config=ReconstructionConfig())


if __name__ == "__main__":
    unittest.main()
