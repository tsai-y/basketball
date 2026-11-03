import unittest
import torch
from .checkpoint import csv_probabilities, json_equivalent


class ConfigSerializationTests(unittest.TestCase):
    def test_json_serialized_label_keys(self):
        self.assertTrue(json_equivalent({"counts": {0: 740, 1: 807}}, {"counts": {"0": 740, "1": 807}}))

    def test_actual_value_or_missing_key_not_ignored(self):
        self.assertFalse(json_equivalent({"counts": {0: 740, 1: 807}}, {"counts": {"0": 741, "1": 807}}))
        self.assertFalse(json_equivalent({"seed": 20261024, "epoch": 0}, {"seed": 20261024}))

    def test_probabilities_reproduce_original_scalar_csv_writer(self):
        logits = torch.tensor([-2.7443442, 2.476909, 0., 100., -100.], dtype=torch.float32)
        self.assertEqual(csv_probabilities(logits), [float(logit.sigmoid()) for logit in logits])


if __name__ == "__main__":
    unittest.main()
