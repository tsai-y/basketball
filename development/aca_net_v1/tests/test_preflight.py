import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import resnet50

from development.aca_net_v1 import baseline_preflight as probe
from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.weights import pin, verify_payload


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_legacy_only_bn_counters_may_be_missing(self):
        state = resnet50(weights=None).state_dict()
        state = {k: v for k, v in state.items() if not k.endswith(".num_batches_tracked")}
        model = VideoResNet50()
        result = model.load_imagenet_state(state)
        self.assertEqual(result["backbone_tensors_loaded"], 265)
        self.assertEqual(len(result["legacy_bn_counters_initialized_to_zero"]), 53)
        self.assertTrue(all(model.state_dict()[k].item() == 0
                            for k in result["legacy_bn_counters_initialized_to_zero"]))
        del state["layer4.2.bn3.running_var"]
        with self.assertRaises(ValueError):
            model.load_imagenet_state(state)

    def test_wrong_dtype_is_rejected_before_mutation(self):
        state = resnet50(weights=None).state_dict()
        state["conv1.weight"] = state["conv1.weight"].double()
        model = VideoResNet50()
        before = model.conv1.weight.detach().clone()
        with self.assertRaises(ValueError):
            model.load_imagenet_state(state)
        torch.testing.assert_close(model.conv1.weight, before, rtol=0, atol=0)

    def test_microbatch_gradients_match_actual_batch_mean(self):
        torch.manual_seed(3)
        a = nn.Linear(3, 2).double()
        b = copy.deepcopy(a)
        x = torch.randn(5, 3, dtype=torch.double)
        y = torch.tensor([0, 1, 0, 1, 1])
        expected = nn.functional.cross_entropy(a(x), y)
        expected.backward()
        actual = probe.accumulated_backward(b, x, y, 2)
        torch.testing.assert_close(actual, expected.detach())
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_training_mode_preserves_bn_state_but_updates_conv_gradients(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        probe.training_mode(model)
        before = probe.frozen_snapshot(model)
        model(torch.randn(2, 3, 4, 4)).square().mean().backward()
        probe.check_state(model, before)
        self.assertFalse(model[1].training)
        self.assertIsNone(model[1].weight.grad)
        self.assertIsNotNone(model[0].weight.grad)

    def test_gpu_unavailable_rejected_before_weights_and_video_access(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(probe, "initialized_baseline", side_effect=AssertionError("Must not load")), \
             patch.object(probe, "VideoScope", side_effect=AssertionError("Must not open")):
            with self.assertRaises(ValueError):
                probe.run(Path("/nonexistent"), dry_run=True)

    def test_weight_pin_and_payload_integrity_rejection(self):
        identity = pin()
        self.assertEqual(identity["source_tensor_count"], 267)
        self.assertEqual(len(identity["sha256"]), 64)
        with self.assertRaises(ValueError):
            verify_payload(Path(__file__), identity)


if __name__ == "__main__":
    unittest.main()
