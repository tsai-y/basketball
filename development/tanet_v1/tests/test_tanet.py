import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from development.tanet_v1.config import COMMIT, InputConfig, load_config, verify_upstream
from development.tanet_v1.data import FrameInterval, VideoScope, decode, normalize_rgb, uniform_indices
from development.tanet_v1.model import TANetBinary
from development.tanet_v1.upstream.temporal_module import TAM, TemporalBottleneck


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_upstream_hash_and_config(self):
        self.assertEqual(verify_upstream()["commit"], COMMIT)
        self.assertFalse(load_config()["formal_training_authorized"])

    def test_uniform_and_interval(self):
        ids = uniform_indices(150)
        self.assertEqual((ids[0], ids[-1], len(set(ids))), (0, 149, 50))
        self.assertEqual(ids, tuple(i * 149 // 49 for i in range(50)))
        selected = uniform_indices(150, interval=FrameInterval(30, 100))
        self.assertEqual((selected[0], selected[-1], len(set(selected))), (30, 99, 50))
        for count, interval in ((49, None), (150, FrameInterval(80, 100)),
                                (150, FrameInterval(-1, 100)), (150, FrameInterval(10, 151))):
            with self.assertRaises(ValueError):
                uniform_indices(count, interval=interval)

    def test_normalization(self):
        x = normalize_rgb(np.full((50, 192, 256, 3), 255, dtype=np.uint8))
        self.assertEqual(tuple(x.shape), (3, 50, 192, 256))
        expected = (torch.ones(3) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(x[:, 0, 0, 0], expected)

    def test_test_rejected_before_metadata_or_path(self):
        with patch("development.tanet_v1.data.gate1a.load_scope", side_effect=AssertionError("metadata touched")):
            with self.assertRaisesRegex(ValueError, "before metadata"):
                VideoScope("test")

    def test_cross_split_id_rejected_without_open(self):
        train = VideoScope("train")
        val = VideoScope("validation")
        with patch("development.tanet_v1.data.gate1a.open_guarded", side_effect=AssertionError("video opened")):
            with self.assertRaises(ValueError):
                VideoScope("train", [val.rows[0]["video_id"]])
            with self.assertRaises(ValueError):
                train.authorize(val.rows[0]["video_id"])

    def test_meta_full_resolution_and_16_author_blocks(self):
        with torch.device("meta"):
            model = TANetBinary().eval()
            self.assertEqual(sum(isinstance(m, TAM) for m in model.modules()), 16)
            self.assertEqual(sum(isinstance(m, TemporalBottleneck) for m in model.modules()), 16)
            out = model(torch.empty(2, 3, 50, 192, 256))
            self.assertEqual(tuple(out.shape), (2, 2))
            for t in (8, 16, 100):
                with self.assertRaisesRegex(ValueError, "configured_T"):
                    model(torch.empty(1, 3, t, 32, 32))
            with self.assertRaises(ValueError):
                model(torch.empty(1, 50, 3, 32, 32))
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_download_and_head_consensus(self):
        torch.manual_seed(42)
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("network attempted")):
            model = TANetBinary(frames=8).eval()
        x = torch.randn(2, 3, 8, 32, 32)
        with torch.inference_mode():
            expected = model.classifier(model.backbone(x.permute(0, 2, 1, 3, 4).reshape(16, 3, 32, 32)))
            expected = expected.reshape(2, 8, 2).mean(1)
            torch.testing.assert_close(model(x), expected)
            torch.testing.assert_close(model(x[:1]), model(x)[:1], rtol=1e-4, atol=2e-6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_author_tam_reference_math_and_gradients(self):
        # Tiny synthetic backward only; never constructs/steps an optimizer.
        torch.manual_seed(5)
        tam = TAM(8, 5).eval()
        reference = copy.deepcopy(tam)
        x = torch.randn(10, 8, 3, 4, requires_grad=True)
        other = x.detach().clone().requires_grad_(True)
        actual = tam(x)
        video = other.reshape(2, 5, 8, 3, 4).permute(0, 2, 1, 3, 4)
        pooled = video.mean((-2, -1))
        kernels = reference.G(pooled.reshape(16, 5)).reshape(2, 8, 3)
        local = reference.L(pooled)
        z = video * local[..., None, None]
        windows = F.pad(z, (0, 0, 0, 0, 1, 1)).unfold(2, 3, 1)
        expected = torch.einsum("ncthwk,nck->ncthw", windows, kernels)
        expected = expected.permute(0, 2, 1, 3, 4).reshape(10, 8, 3, 4)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, other.grad, rtol=1e-4, atol=1e-6)
        for p, q in zip(tam.parameters(), reference.parameters()):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)

    def test_even_bad_input_config_rejected(self):
        for kwargs in ({"frames": 1}, {"frames": True}, {"width": 0}):
            with self.assertRaises(ValueError):
                InputConfig(**kwargs)

    def test_decoder_exact_indices_and_rgb_with_fake_authorized_source(self):
        # Synthetic decoder only; this test opens no filesystem/video path.
        from types import SimpleNamespace
        from unittest.mock import Mock
        scope = Mock()
        scope.authorize.return_value = ({"binary_label": "1", "source_group": "fake_train_group",
                                        "split": "train", "original_class": "2p1"}, "fake_allowed_path")
        info = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=5)
        native = [np.full((240, 320, 3), [i, 20, 40], dtype=np.uint8) for i in range(60)]
        class FakeCapture:
            def __init__(self):
                self.index = 0
                self.released = False
            def isOpened(self):
                return True
            def get(self, key):
                return 60 if key == 7 else 30  # CAP_PROP_FRAME_COUNT=7
            def read(self):
                if self.index == 60:
                    return False, None
                value = native[self.index]
                self.index += 1
                return True, value
            def release(self):
                self.released = True
        cap = FakeCapture()
        with patch("development.tanet_v1.data.gate1a.open_guarded", return_value=123) as opened, \
             patch("development.tanet_v1.data.os.fstat", return_value=info), \
             patch("development.tanet_v1.data.os.close") as closed, \
             patch("development.tanet_v1.data.cv2.VideoCapture", return_value=cap):
            result = decode(scope, "fake_train_id", InputConfig(height=8, width=8))
        self.assertEqual(result["metadata"]["selected_indices"], list(uniform_indices(60)))
        pixel = result["tensor"][:, -1, 0, 0]
        # BGR [59,20,40] must become RGB [40,20,59], normalized exactly once.
        rgb = torch.tensor([40., 20., 59.]) / 255
        expected = (rgb - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(pixel, expected)
        opened.assert_called_once_with("fake_allowed_path")
        closed.assert_called_once_with(123)
        self.assertTrue(cap.released)

    def test_foreign_or_missing_checkpoint_refused_without_download(self):
        with self.assertRaisesRegex(ValueError, "identity required"):
            TANetBinary(checkpoint="not_an_imagenet_checkpoint.pth")
        with self.assertRaisesRegex(ValueError, "Identity without checkpoint"):
            TANetBinary(checkpoint_sha256="wrong")

    def test_cross_entropy_gradient_interface(self):
        logits = torch.tensor([[1., -1.], [-1., 1.]], requires_grad=True)
        loss = F.cross_entropy(logits, torch.tensor([0, 1], dtype=torch.int64))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(loss.item(), .2)


if __name__ == "__main__":
    unittest.main()
