from contextlib import redirect_stdout, redirect_stderr
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn

from development.mlp_baseline_v1.model import PooledFeatureMLP
from development.mlp_baseline_v1 import train_mlp


class PooledMLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        cls.old_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        torch.use_deterministic_algorithms(cls.old_deterministic)

    def setUp(self):
        torch.manual_seed(20261024)
        self.model = PooledFeatureMLP()
        self.x = torch.randn(2, 1536)
        self.y = torch.tensor([0., 1.])

    def test_forward_and_all_parameter_gradients(self):
        logits = self.model(self.x)
        self.assertEqual(tuple(logits.shape), (2,))
        loss = nn.BCEWithLogitsLoss()(logits, self.y)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_parameter_count_matches_capacity_target(self):
        count = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(count, 1453697)
        self.assertLess(abs(count - 1450497) / 1450497, 0.005)

    def test_unpooled_or_wrong_width_inputs_rejected(self):
        for shape in ((2, 3, 8, 1536), (2, 1535), (1536,)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    self.model(torch.zeros(shape))

    def test_pooling_does_not_expose_token_order(self):
        sequence = torch.randn(2, 3, 8, 1536)
        self.model.eval()
        with torch.no_grad():
            original = self.model(sequence.mean(dim=(1, 2)))
            reordered = self.model(sequence[:, [2, 0, 1]].flip(2).mean(dim=(1, 2)))
        torch.testing.assert_close(original, reordered, rtol=1e-5, atol=1e-6)

    def test_evaluation_does_not_change_weights(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        metrics, logits = train_mlp.evaluate(self.model, self.x, self.y, batch_size=1)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 2)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_training_recipe_matches_predeclared_temporal_recipe(self):
        args = train_mlp.parse_args([])
        config = train_mlp.build_config(args, self.model, train_mlp.CACHE_RUN_ID, 7193, 1547)
        expected = {
            "device": "cpu", "cpu_threads": 4, "epochs": 50, "seed": 20261024,
            "batch_size": 64, "eval_batch_size": 64, "optimizer": "AdamW",
            "lr": 0.0001, "weight_decay": 0.0001, "max_grad_norm": 1.0,
            "selection": "minimum_validation_loss", "threshold": 0.5,
            "early_stopping": {"monitor": "validation_loss", "patience": 10, "min_delta": 0.0001},
        }
        for key, value in expected.items():
            with self.subTest(setting=key):
                self.assertEqual(config[key], value)

    def test_invalid_arguments_rejected_before_data_access(self):
        for argv in (("--epochs", "0"), ("--seed", "-1"), ("--threads", "0")):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with patch.object(train_mlp, "load_vectors", side_effect=AssertionError("Read")):
                    with self.assertRaises(SystemExit):
                        train_mlp.main(list(argv))

    def test_dry_run_does_not_create_optimizer_or_outputs(self):
        def fake_load(cache, *, split):
            return self.x, self.y, [f"{split}_0", f"{split}_1"], train_mlp.CACHE_RUN_ID

        with patch.object(train_mlp, "load_vectors", side_effect=fake_load), \
             patch.object(torch.optim, "AdamW", side_effect=AssertionError("Optimizer created")), \
             patch.object(train_mlp, "mkdtemp", side_effect=AssertionError("Run directory created")), \
             patch.object(train_mlp, "train_one_epoch", side_effect=AssertionError("Training called")), \
             redirect_stdout(io.StringIO()) as captured:
            train_mlp.main(["--dry-run", "--threads", "1"])
        self.assertIn('"optimizer_steps": 0', captured.getvalue())
        self.assertIn('"output_files_written": false', captured.getvalue())


if __name__ == "__main__":
    unittest.main()
