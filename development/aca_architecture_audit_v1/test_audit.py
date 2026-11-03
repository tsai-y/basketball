import unittest

import torch
from torch.nn import functional as F

from development.aca_architecture_audit_v1.audit import (
    ProposalLSTA, grouped_temporal_conv, parameter_audit,
)
from development.aca_net_v1.modules import dynamic_temporal_conv


class ArchitectureAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.report = parameter_audit()

    def test_component_counts_reconcile_all_models_and_heads(self):
        expected = {"baseline": 23_512_130, "lsta": 93_159_070,
                    "tsci": 23_516_930, "aca": 93_163_870,
                    "dw3d_existing": 25_464_286,
                    "all_gates_depthwise_diagnostic": 24_210_590,
                    "proposal_default_audit_only": 26_330_018}
        for row in self.report["rows"]:
            self.assertEqual(row["total"], expected[row["variant"]] + (12_294 if row["classes"] == 8 else 0))
            self.assertEqual(row["total"], sum(row["components"].values()))
            self.assertEqual(row["meta_logits_shape"], [1, row["classes"]])

    def test_gating_difference_exactly_explains_dense_to_dw3d(self):
        rows = {r["variant"]: r for r in self.report["rows"] if r["classes"] == 8}
        f = self.report["gate_formulas"]
        self.assertEqual(f["dense_55_C_squared"], 69_160_960)
        self.assertEqual(f["mixed_C_squared_plus_54_C"], 1_461_376)
        self.assertEqual(f["all_depthwise_55_C"], 207_680)
        self.assertEqual(rows["aca"]["total"] - rows["dw3d_existing"]["total"],
                         f["dense_55_C_squared"] - f["mixed_C_squared_plus_54_C"])
        self.assertLess(self.report["class_head_8_minus_2"], 13_000)

    def test_grouped_conv_matches_independent_per_video_conv3d(self):
        generator = torch.Generator().manual_seed(444)
        x = torch.randn(2, 3, 5, 2, 4, generator=generator, dtype=torch.float64)
        k = torch.randn(2, 3, 3, generator=generator, dtype=torch.float64).softmax(-1)
        expected = torch.cat([F.conv3d(x[n:n+1], k[n].reshape(3, 1, 3, 1, 1),
                                      padding=(1, 0, 0), groups=3) for n in range(2)])
        torch.testing.assert_close(grouped_temporal_conv(x, k), expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(dynamic_temporal_conv(x, k), expected, rtol=1e-12, atol=1e-12)

    def test_grouped_conv_gradient_has_no_cross_video_dependency(self):
        x = torch.randn(2, 2, 3, 2, 2, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
        grouped_temporal_conv(x, k)[0].sum().backward()
        self.assertEqual(x.grad[1].abs().sum().item(), 0)
        self.assertEqual(k.grad[1].abs().sum().item(), 0)

    def test_proposal_preserves_dense_pointwise_and_channel_BN(self):
        p = ProposalLSTA(64)
        self.assertEqual(p.gate_branch_1x1[0].groups, 1)
        self.assertEqual(p.gate_branch_3x3[0].groups, 64)
        self.assertEqual(p.gate_proj[0].groups, 64)
        self.assertEqual(p.long_bn.num_features, 64)
        self.assertEqual(p.long_fc1.out_features, 100)
        self.assertEqual(p.short_branch[0].out_channels, 16)


if __name__ == "__main__":
    unittest.main()
