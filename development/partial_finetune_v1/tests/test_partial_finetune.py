from contextlib import redirect_stderr
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import timm
import torch
from torch import nn

from development.mixnet_i3d.temporal_features import ThreeWindowFeatureExtractor
from development.partial_finetune_v1.initialization import EXTRACTOR, HEAD, load_initialized_model
from development.partial_finetune_v1.model import PartialFineTuneModel
from development.partial_finetune_v1 import preflight


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Conv3d(3, 4, 1, stride=(8, 1, 1))
        self.bn1 = nn.BatchNorm3d(4)
        self.blocks = nn.Sequential(*[
            nn.Sequential(nn.Conv3d(4, 4, 1), nn.BatchNorm3d(4), nn.SiLU())
            for _ in range(6)
        ])
        self.conv_head = nn.Conv3d(4, 4, 1)
        self.bn2 = nn.BatchNorm3d(4)

    def forward(self, x):
        return self.bn2(self.conv_head(self.blocks(self.bn1(self.conv_stem(x)))))


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x.mean((1, 2))).squeeze(-1)


class PartialFineTuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(20261024)
        self.model = PartialFineTuneModel(TinyBackbone(), TinyHead(),
                                          mean=(.1, .2, .3), std=(.2, .3, .4))
        self.clips = torch.rand(2, 3, 3, 64, 4, 4)
        self.labels = torch.tensor([0., 1.])

    def test_same_features_as_original_frozen_extractor_before_updates(self):
        reference = ThreeWindowFeatureExtractor(deepcopy(self.model.backbone),
                                                 mean=(.1, .2, .3), std=(.2, .3, .4))
        self.model.eval()
        with torch.no_grad():
            actual = self.model.forward_features(self.clips)
            expected = reference(self.clips)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), (2, 3, 8, 4))
        self.assertEqual(tuple(self.model(self.clips).shape), (2,))

    def test_exact_trainable_scope_and_all_batchnorm_frozen(self):
        bn_ids = {id(p) for module in self.model.backbone.modules()
                  if isinstance(module, nn.modules.batchnorm._BatchNorm)
                  for p in module.parameters()}
        for name, parameter in self.model.named_parameters():
            expected = (name.startswith(("head.", "backbone.blocks.5.", "backbone.conv_head."))
                        and id(parameter) not in bn_ids)
            self.assertEqual(parameter.requires_grad, expected, name)
        for _ in range(2):
            self.model.eval().train()
            self.assertTrue(self.model.head.training)
            self.assertTrue(self.model.backbone.blocks[5].training)
            self.assertFalse(self.model.backbone.blocks[4].training)
            for module in self.model.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    self.assertFalse(module.training)
                    self.assertTrue(all(not p.requires_grad for p in module.parameters()))

    def test_backward_and_updates_leave_prefix_and_bn_bitwise_unchanged(self):
        frozen = preflight.frozen_snapshot(self.model)
        before = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW(self.model.optimizer_groups(), weight_decay=.0001)
        self.clips.requires_grad_(True)
        nn.functional.binary_cross_entropy_with_logits(self.model(self.clips), self.labels).backward()
        gradients = preflight.check_gradients(self.model)
        self.assertGreater(gradients["backbone"], 0.)
        self.assertGreater(gradients["head"], 0.)
        self.assertIsNone(self.clips.grad)
        optimizer.step()
        preflight.check_frozen_state(self.model, frozen)
        changed = [n for n, p in self.model.named_parameters()
                   if p.requires_grad and not torch.equal(p, before[n])]
        for prefix in ("backbone.blocks.5.", "backbone.conv_head.", "head."):
            self.assertTrue(any(n.startswith(prefix) for n in changed))

    def test_optimizer_groups_exactly_cover_trainable_parameters(self):
        groups = self.model.optimizer_groups()
        self.assertEqual([g["lr"] for g in groups], [1e-5, 1e-4])
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in self.model.parameters() if p.requires_grad})
        with self.assertRaises(ValueError):
            self.model.optimizer_groups(backbone_lr=0)

    def test_invalid_inputs_rejected(self):
        for clips in (self.clips[:, :2], self.clips[:, :, :, :63], self.clips.double(),
                      self.clips[:0], self.clips[..., :1, :1]):
            with self.subTest(shape=tuple(clips.shape)), self.assertRaises(ValueError):
                self.model(clips)

    def test_invalid_normalization_rejected(self):
        for mean, std in (((0, 0), (1, 1, 1)), ((0, 0, 0), (1, 0, 1)),
                          ((0, float("nan"), 0), (1, 1, 1))):
            with self.assertRaises(ValueError):
                PartialFineTuneModel(TinyBackbone(), TinyHead(), mean=mean, std=std)

    def test_missing_gradient_and_frozen_mutation_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Missing trainable gradient"):
            preflight.check_gradients(self.model)
        snapshot = preflight.frozen_snapshot(self.model)
        with torch.no_grad():
            self.model.backbone.bn1.running_mean.add_(1)
        with self.assertRaisesRegex(RuntimeError, "Frozen state changed"):
            preflight.check_frozen_state(self.model, snapshot)

    def test_no_gpu_fails_before_loading_or_creating_outputs(self):
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(preflight, "load_initialized_model", side_effect=AssertionError("Read")), \
             patch.object(preflight, "mkdtemp", side_effect=AssertionError("Write")):
            with self.assertRaisesRegex(RuntimeError, "GPU unavailable"):
                preflight.main(["--dry-run"])

    def test_invalid_arguments_rejected_before_gpu_or_file_access(self):
        for args in (("--steps", "0"), ("--seed", "-1"), ("--threads", "0")):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), \
                 patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU")):
                with self.assertRaises(SystemExit):
                    preflight.main(list(args))

    @unittest.skipUnless(EXTRACTOR.is_file() and HEAD.is_file(), "Requires the local pinned checkpoints")
    def test_offline_real_initialization_and_small_rgb_forward(self):
        with patch.object(timm, "create_model", wraps=timm.create_model) as create:
            model, info = load_initialized_model()
        self.assertFalse(create.call_args.kwargs["pretrained"])
        self.assertEqual(info["head_seed"], 20261024)
        model.eval()
        clips = torch.rand(1, 3, 3, 64, 32, 32)
        with torch.no_grad():
            actual = model.forward_features(clips)
            x = ((clips - model.rgb_mean) / model.rgb_std).reshape(3, 3, 64, 32, 32)
            maps = model.backbone(x)
            expected = maps.mean((-2, -1)).transpose(1, 2).reshape(1, 3, 8, 1536)
        self.assertEqual(tuple(actual.shape), (1, 3, 8, 1536))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
