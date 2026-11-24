"""Unit tests for development.mixnet_binary50_v1 (CPU-only, no video decoding)."""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from development.mixnet_binary50_v1.data_adapter import (
    binary_cross_entropy_from_logits,
    collate_saved_binary_fixtures,
    load_saved_rgb_fixture_item,
    verify_sample15_metadata,
)
from development.mixnet_binary50_v1.model import (
    CLASS_ORDER,
    EXPECTED_INPUT_SHAPE,
    MixNetBinary50Model,
    build_mixnet_binary50,
    compare_bn3d_snapshots,
    parameter_summary,
    snapshot_bn3d_buffers,
    summarize_bn3d,
    trace_stage_shapes,
)
from development.mixnet_i3d.video_classifier import ThreeWindowMakeMissModel


class MixNetBinary50Tests(unittest.TestCase):
    def test_build_contract_and_no_internal_normalization(self) -> None:
        model = build_mixnet_binary50(pretrained=False, seed=20261123, eval_mode=True)
        self.assertIsInstance(model, MixNetBinary50Model)
        self.assertEqual(model.num_features, 1536)
        self.assertEqual(model.backbone.num_features, 1536)
        self.assertEqual(tuple(model.classifier.weight.shape), (2, 1536))
        self.assertEqual(tuple(model.classifier.bias.shape), (2,))
        self.assertEqual(model.class_order, CLASS_ORDER)
        self.assertFalse(hasattr(model, "rgb_mean"))
        self.assertFalse(hasattr(model, "rgb_std"))

        params = parameter_summary(model)
        self.assertEqual(params["backbone_total_params"], 3044630)
        self.assertEqual(params["classifier_total_params"], 3074)
        self.assertEqual(params["total_params"], 3047704)

        bn_info = summarize_bn3d(model)
        self.assertEqual(bn_info["count"], 49)
        self.assertEqual(bn_info["training_modes"], [False])
        self.assertEqual(bn_info["affine_values"], [True])
        self.assertEqual(bn_info["track_running_stats_values"], [True])
        self.assertFalse(bn_info["formal_train_bn_policy_approved"])

        with self.assertRaises(ValueError):
            build_mixnet_binary50(pretrained=True)

    def test_input_shape_and_dtype_guards(self) -> None:
        model = build_mixnet_binary50(pretrained=False, seed=20261123, eval_mode=True)
        bad_inputs = [
            torch.randn(1, 3, 3, 50, 120, 160),  # 6D three-window
            torch.randn(1, 50, 3, 120, 160),  # swapped C and T
            torch.randn(1, 3, 64, 120, 160),  # wrong T=64
            torch.randn(1, 3, 50, 240, 320),  # wrong spatial 320x240
            torch.randint(0, 256, (1, 3, 50, 120, 160), dtype=torch.uint8),
        ]
        nan_input = torch.zeros(1, *EXPECTED_INPUT_SHAPE, dtype=torch.float32)
        nan_input[0, 0, 0, 0, 0] = float("nan")
        bad_inputs.append(nan_input)

        for bad in bad_inputs:
            with self.assertRaises(ValueError):
                model(bad)

    def test_full_50frame_forward_passthrough_stage_shapes_and_bn_invariance(self) -> None:
        model = build_mixnet_binary50(pretrained=False, seed=20261123, eval_mode=True)
        g = torch.Generator().manual_seed(12345)
        x = torch.randn((1, 3, 50, 120, 160), generator=g, dtype=torch.float32)

        before_bn = snapshot_bn3d_buffers(model)
        with torch.inference_mode():
            logits, trace = trace_stage_shapes(model, x)
        after_bn = snapshot_bn3d_buffers(model)

        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(trace["backbone_input_same_object"])
        self.assertTrue(trace["backbone_input_bitwise_equal"])
        self.assertEqual(trace["backbone_input_max_abs_diff"], 0.0)
        self.assertEqual(
            trace["temporal_progression"],
            {
                "input": 50,
                "stem_bn1": 25,
                "stage_0": 25,
                "stage_1": 25,
                "stage_2": 25,
                "stage_3": 13,
                "stage_4": 13,
                "stage_5": 7,
                "head_bn2": 7,
            },
        )
        self.assertEqual(trace["stage_shapes"]["head_bn2"], [1, 1536, 7, 4, 5])
        self.assertEqual(trace["stage_shapes"]["global_avg_pool_thw"], [1, 1536])
        self.assertEqual(trace["stage_shapes"]["logits"], [1, 2])

        bn_cmp = compare_bn3d_snapshots(before_bn, after_bn)
        self.assertTrue(bn_cmp["all_buffers_bitwise_unchanged"])
        self.assertEqual(bn_cmp["bn3d_layer_count"], 49)

    def test_small_synthetic_backward_finite_gradients_no_optimizer_step(self) -> None:
        model = build_mixnet_binary50(
            pretrained=False,
            seed=20261123,
            strict_clip_shape=False,
            eval_mode=True,
        )
        g = torch.Generator().manual_seed(54321)
        x = torch.randn((2, 3, 8, 64, 64), generator=g, dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)

        before_bn = snapshot_bn3d_buffers(model)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = binary_cross_entropy_from_logits(logits, targets)
        loss.backward()
        after_bn = snapshot_bn3d_buffers(model)

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(loss).item())
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
            self.assertTrue(
                torch.isfinite(param.grad).all().item(),
                f"Non-finite gradient for {name}",
            )
        self.assertTrue(
            compare_bn3d_snapshots(before_bn, after_bn)["all_buffers_bitwise_unchanged"]
        )

    def test_same_seed_reproducibility(self) -> None:
        m1 = build_mixnet_binary50(pretrained=False, seed=20261123, eval_mode=True)
        m2 = build_mixnet_binary50(pretrained=False, seed=20261123, eval_mode=True)

        sd1, sd2 = m1.state_dict(), m2.state_dict()
        self.assertEqual(list(sd1.keys()), list(sd2.keys()))
        for key in sd1:
            self.assertTrue(torch.equal(sd1[key], sd2[key]), f"Mismatch in {key}")

        g = torch.Generator().manual_seed(777)
        x = torch.randn((1, 3, 50, 120, 160), generator=g, dtype=torch.float32)
        with torch.inference_mode():
            y1 = m1(x)
            y2 = m2(x)
        self.assertTrue(torch.equal(y1, y2))

    def test_raw_logit_ce_and_target_semantics(self) -> None:
        logits = torch.tensor([[1.2, -0.4], [-0.8, 0.9]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        ce = binary_cross_entropy_from_logits(logits, targets)
        manual = -F.log_softmax(logits, dim=1)[torch.arange(2), targets].mean()
        torch.testing.assert_close(ce, manual, rtol=1e-6, atol=1e-6)

        wrong_softmax_ce = F.cross_entropy(F.softmax(logits, dim=1), targets)
        self.assertGreater(abs(ce.item() - wrong_softmax_ce.item()), 1e-2)

        with self.assertRaises(ValueError):
            binary_cross_entropy_from_logits(logits, targets.float())
        with self.assertRaises(ValueError):
            binary_cross_entropy_from_logits(logits, torch.tensor([0, 2], dtype=torch.long))

    def test_sample15_metadata_and_saved_fixtures_04_22(self) -> None:
        info15 = verify_sample15_metadata()
        self.assertEqual(info15["original_class"], "3p1")
        self.assertEqual(info15["original_class_id"], 3)
        self.assertEqual(info15["effective_class"], "2p1")
        self.assertEqual(info15["effective_class_id"], 1)
        self.assertEqual(info15["binary_label"], 1)
        self.assertFalse(info15["overwritten_by_filename"])
        self.assertFalse(info15["video_opened"])

        with self.assertRaises(ValueError):
            load_saved_rgb_fixture_item("15")

        batch, provs = collate_saved_binary_fixtures(("04", "22"))
        self.assertEqual(batch["task"], "basketball51_direct_binary")
        self.assertEqual(tuple(batch["tensor"].shape), (2, 3, 50, 120, 160))
        self.assertEqual(batch["targets"].dtype, torch.long)
        self.assertEqual(batch["targets"].tolist(), [0, 1])
        self.assertEqual(batch["binary_targets"].tolist(), [0, 1])
        self.assertEqual(batch["eight_class_targets"].tolist(), [0, 5])
        self.assertTrue(all(p["video_open_count"] == 0 for p in provs))
        self.assertTrue(all(p["tensor_matches_manifest_sha"] for p in provs))

    def test_legacy_three_window_classifier_preserved(self) -> None:
        self.assertTrue(hasattr(ThreeWindowMakeMissModel, "forward"))
        model = build_mixnet_binary50(pretrained=False, seed=20261123, eval_mode=True)
        legacy = ThreeWindowMakeMissModel(
            model.backbone,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        with self.assertRaises(ValueError):
            legacy(torch.randn(1, 3, 50, 120, 160))


if __name__ == "__main__":
    unittest.main()
