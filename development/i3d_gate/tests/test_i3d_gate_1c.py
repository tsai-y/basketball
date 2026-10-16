from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from development.data_pipeline.pipeline import AccessLedger, BasketballClipDataset, GateError as PipelineGateError, load_gate_config
from development.i3d_gate.gate import (
    ROOT, GateError, gap, load_acquisition_manifest, load_config,
    load_model_strict, normalize_i3d_rgb, tensor_digest,
)
from development.i3d_gate.models import InceptionI3d


class NormalizationTests(unittest.TestCase):
    def test_affine_is_exact_and_out_of_place(self):
        source = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32).reshape(1, 1, 1, 1, 5).expand(1, 3, 1, 1, 5).contiguous()
        before = tensor_digest(source)
        result, metadata = normalize_i3d_rgb(source)
        self.assertTrue(torch.equal(result[0, 0, 0, 0], torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])))
        self.assertNotEqual(source.data_ptr(), result.data_ptr())
        self.assertEqual(before, tensor_digest(source))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(metadata["version"], "i3d_rgb_minus1_plus1_v1")
        self.assertFalse(metadata["in_place"])

    def test_adapter_fails_closed(self):
        with self.assertRaises(GateError):
            normalize_i3d_rgb(torch.zeros((1, 3, 1, 1, 1), dtype=torch.float64))
        with self.assertRaises(GateError):
            normalize_i3d_rgb(torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32))
        with self.assertRaises(GateError):
            normalize_i3d_rgb(torch.full((1, 3, 1, 1, 1), 1.1, dtype=torch.float32))


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        load_acquisition_manifest(cls.config)

    def test_strict_load_exact_slots(self):
        model, result = load_model_strict(self.config)
        self.assertTrue(result["strict"] and result["success"])
        self.assertEqual(result["source_tensor_count"], 287)
        self.assertEqual(result["model_slot_count"], 287)
        self.assertEqual(result["missing_keys"], [])
        self.assertEqual(result["unexpected_keys"], [])
        self.assertEqual(result["unused_source_tensors"], [])
        self.assertEqual(result["shape_mismatches"], [])
        self.assertEqual(len(model.state_dict()), 287)

    def test_checkpoint_inventory_and_mapping(self):
        inventory = json.loads((ROOT / "docs/i3d/checkpoint_tensor_inventory.json").read_text())
        mapping = json.loads((ROOT / "docs/i3d/checkpoint_key_mapping.json").read_text())
        self.assertEqual(inventory["input_channels"], 3)
        self.assertEqual(inventory["first_kernel_shape"], [64, 3, 7, 7, 7])
        self.assertEqual(inventory["mixed_5c_output_channels"], 1024)
        self.assertEqual(inventory["kinetics_class_count"], 400)
        self.assertTrue(inventory["all_finite"])
        self.assertEqual(mapping["official_source_tensor_count"], 230)
        self.assertEqual(mapping["official_tensors_used_once"], 230)
        self.assertEqual(mapping["derived_constant_gamma_count"], 57)
        self.assertTrue(mapping["all_exact"])

    def test_feature_endpoint_and_gap_do_not_use_fixed_pool(self):
        torch.set_num_threads(1)
        model = InceptionI3d().eval()
        value = torch.zeros((1, 3, 8, 32, 32), dtype=torch.float32)
        before = {key: item.clone() for key, item in model.named_buffers()}
        with torch.inference_mode():
            feature = model.forward_features(value)
            pooled = gap(feature)
        self.assertEqual(feature.shape[:2], (1, 1024))
        self.assertEqual(pooled.shape, (1, 1024))
        self.assertTrue(torch.isfinite(feature).all())
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(all(torch.equal(item, dict(model.named_buffers())[key]) for key, item in before.items()))

    def test_strict_loading_rejects_missing_and_unexpected(self):
        model = InceptionI3d()
        state = model.state_dict()
        state.pop(next(iter(state)))
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state, strict=True)
        state = model.state_dict()
        state["invented.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state, strict=True)

    def test_test_split_rejected_with_zero_access(self):
        ledger = AccessLedger()
        with self.assertRaises(PipelineGateError):
            BasketballClipDataset(load_gate_config(), split="test", ledger=ledger)
        self.assertEqual(ledger.statistics()["test"], {"opens": 0, "decodes": 0})


if __name__ == "__main__":
    unittest.main()
