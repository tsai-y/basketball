from contextlib import redirect_stdout, redirect_stderr
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from development.linear_baseline_v1.train_linear import evaluate as linear_evaluate
from development.temporal_baseline_v1 import train_temporal


class TinyTemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x.mean(dim=(1, 2))).squeeze(-1)


class TrainingHelperTests(unittest.TestCase):
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
        self.model = TinyTemporalHead()
        self.x = torch.randn(5, 3, 8, 4)
        self.y = torch.tensor([0., 1., 0., 1., 1.])

    def test_batched_metrics_match_linear_reference(self):
        self.model.eval()
        with torch.no_grad():
            expected_logits = self.model(self.x)
        expected_metrics, _ = linear_evaluate(
            nn.Identity(), expected_logits.unsqueeze(-1), self.y
        )
        actual_metrics, actual_logits = train_temporal.evaluate(
            self.model, self.x, self.y, batch_size=2
        )
        torch.testing.assert_close(actual_logits, expected_logits)
        self.assertEqual(actual_metrics["confusion_matrix"], expected_metrics["confusion_matrix"])
        self.assertEqual(actual_metrics["per_class"], expected_metrics["per_class"])
        self.assertAlmostEqual(actual_metrics["loss"], expected_metrics["loss"], places=6)
        self.assertEqual(actual_metrics["macro_f1"], expected_metrics["macro_f1"])

    def test_evaluation_does_not_update_weights_or_create_gradients(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        train_temporal.evaluate(self.model, self.x, self.y, batch_size=2)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertFalse(self.model.training)

    def test_optimizer_updates_only_the_toy_training_model(self):
        before = self.model.linear.weight.detach().clone()
        loader = DataLoader(TensorDataset(self.x, self.y), batch_size=2)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        loss = train_temporal.train_one_epoch(self.model, loader, optimizer)
        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertFalse(torch.equal(before, self.model.linear.weight))

    def test_nonfinite_training_loss_rejected_before_update(self):
        before = self.model.linear.weight.detach().clone()
        loader = DataLoader(TensorDataset(torch.full_like(self.x, float("nan")), self.y))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        with self.assertRaisesRegex(RuntimeError, "Non-finite training loss"):
            train_temporal.train_one_epoch(self.model, loader, optimizer)
        torch.testing.assert_close(before, self.model.linear.weight, rtol=0, atol=0)

    def test_invalid_arguments_fail_before_loading_inputs(self):
        for argv in (("--epochs", "0"), ("--seed", "-1"), ("--threads", "0")):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with patch.object(train_temporal, "load_sequences", side_effect=AssertionError("Read")):
                    with self.assertRaises(SystemExit):
                        train_temporal.main(list(argv))

    def test_dry_run_never_creates_optimizer_or_run_directory(self):
        x = torch.zeros(2, 3, 8, 1536)
        y = torch.tensor([0., 1.])

        def fake_load(cache, *, split):
            return x, y, [f"{split}_0", f"{split}_1"], train_temporal.CACHE_RUN_ID

        with patch.object(train_temporal, "load_sequences", side_effect=fake_load), \
             patch.object(torch.optim, "AdamW", side_effect=AssertionError("Optimizer created")), \
             patch.object(train_temporal, "mkdtemp", side_effect=AssertionError("Run directory created")), \
             patch.object(train_temporal, "train_one_epoch", side_effect=AssertionError("Training called")), \
             redirect_stdout(io.StringIO()) as captured:
            train_temporal.main(["--dry-run", "--threads", "1"])
        self.assertIn('"optimizer_steps": 0', captured.getvalue())
        self.assertIn('"output_files_written": false', captured.getvalue())


if __name__ == "__main__":
    unittest.main()
