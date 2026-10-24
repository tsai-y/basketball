import csv
import json
from pathlib import Path
import tempfile
import unittest

from development.temporal_balanced_v1.compare import inspect_run


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.groups = {"miss": "ft0", "make": "ft1"}
        self.summary = {"best_epoch": 1, "validation": {
            "loss": .4, "accuracy": 1., "macro_f1": 1.,
            "confusion_matrix": [[1, 0], [0, 1]],
        }}
        self.predictions = [["miss", 0, .2, 0], ["make", 1, .8, 1]]
        self.write_fixture()

    def write_fixture(self):
        (self.path / "summary.json").write_text(json.dumps(self.summary))
        with (self.path / "history.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "val_loss"])
            writer.writerows([[1, .4], [2, .5]])
        with (self.path / "validation_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["video_id", "true_label", "p_make", "predicted_label"])
            writer.writerows(self.predictions)

    def test_valid_metrics_and_free_throw_recall(self):
        _, _, result = inspect_run(self.path, self.groups)
        self.assertEqual(result["ft_accuracy"], 1.)
        self.assertEqual(result["ft_miss_recall"], 1.)
        self.assertEqual(result["ft_make_recall"], 1.)

    def test_wrong_selected_epoch_rejected(self):
        self.summary["best_epoch"] = 2
        self.write_fixture()
        with self.assertRaisesRegex(RuntimeError, "minimum-loss"):
            inspect_run(self.path, self.groups)

    def test_wrong_threshold_rejected(self):
        self.predictions[0][2] = .8
        self.write_fixture()
        with self.assertRaisesRegex(RuntimeError, "threshold"):
            inspect_run(self.path, self.groups)

    def test_confusion_matrix_mismatch_rejected(self):
        self.summary["validation"]["confusion_matrix"] = [[0, 1], [0, 1]]
        self.write_fixture()
        with self.assertRaisesRegex(RuntimeError, "Confusion matrix"):
            inspect_run(self.path, self.groups)


if __name__ == "__main__":
    unittest.main()
