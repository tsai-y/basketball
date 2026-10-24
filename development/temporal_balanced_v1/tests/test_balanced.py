from collections import Counter
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from development.temporal_balanced_v1.sampling import CLASSES, make_balanced_sampler
from development.temporal_balanced_v1 import train_balanced


class BalancedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "manifest.csv"
        self.rows = []
        for split in ("train", "validation", "test"):
            for i, group in enumerate(CLASSES):
                for j in range(i + 1 if split == "train" else 1):
                    self.rows.append({"video_id": f"{split}/{group}/{j}", "split": split,
                                      "excluded": "0", "original_class": group,
                                      "binary_label": group[-1]})
        self.write_manifest()
        train = [r for r in self.rows if r["split"] == "train"]
        self.ids = [r["video_id"] for r in train]
        self.labels = torch.tensor([float(r["binary_label"]) for r in train])

    def write_manifest(self):
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def build(self, **kwargs):
        options = dict(manifest_path=self.path, manifest_sha256=self.sha, seed=123)
        options.update(kwargs)
        return make_balanced_sampler(self.ids, self.labels, **options)

    def test_equal_expected_class_mass_and_unchanged_epoch_length(self):
        sampler, info = self.build()
        self.assertTrue(sampler.replacement)
        self.assertEqual(len(sampler), len(self.ids))
        self.assertEqual(sampler.weights.dtype, torch.float64)
        for group in CLASSES:
            mass = sampler.weights[torch.tensor([g == group for g in sampler.groups])].sum()
            self.assertAlmostEqual(float(mass / sampler.weights.sum()), .125)
        self.assertEqual(info["class_counts"]["ft0"], 5)
        self.assertFalse(info["loss_weighting_used"])

    def test_reproducibility_and_audit(self):
        a, _ = self.build()
        b, _ = self.build()
        for _ in range(2):
            indices = list(a)
            self.assertEqual(indices, list(b))
            self.assertEqual(sum(a.last_counts.values()), len(self.ids))
            self.assertEqual(a.last_unique_count, len(set(indices)))
            self.assertEqual(a.last_counts, dict(Counter(a.groups[i] for i in indices)))

    def test_preview_does_not_consume_rng(self):
        a, _ = self.build()
        b, _ = self.build()
        preview = a.preview()
        self.assertEqual(a.last_counts, {})
        self.assertEqual(list(a), list(b))
        self.assertEqual(a.last_counts, preview["draw_counts"])

    def test_reordered_ids_are_aligned(self):
        a, _ = self.build()
        self.ids.reverse()
        self.labels = self.labels.flip(0)
        b, _ = self.build()
        torch.testing.assert_close(a.weights.flip(0), b.weights)
        self.assertEqual(a.groups[::-1], b.groups)

    def test_other_split_frequencies_do_not_affect_sampler(self):
        a, info_a = self.build()
        self.rows += [dict(r, video_id="extra/" + r["video_id"])
                      for r in self.rows if r["split"] != "train"]
        self.write_manifest()
        b, info_b = self.build()
        self.assertEqual(info_a["class_counts"], info_b["class_counts"])
        torch.testing.assert_close(a.weights, b.weights)
        self.assertEqual(list(a), list(b))

    def test_nontrain_rejected_before_path_access(self):
        for split in ("validation", "test"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.build(split=split, manifest_path=Path("/does/not/exist"))

    def test_identity_labels_missing_class_and_crosssplit_ids_fail(self):
        with self.assertRaisesRegex(RuntimeError, "identity"):
            self.build(manifest_sha256="wrong")
        old = self.labels.clone()
        self.labels[0] = 1 - self.labels[0]
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            self.build()
        self.labels = old
        self.ids[0] = "test/ft0/0"
        with self.assertRaisesRegex(RuntimeError, "full requested split"):
            self.build()
        self.ids[0] = "train/2p0/0"
        self.rows = [r for r in self.rows if not (r["split"] == "train" and r["original_class"] == "2p0")]
        self.ids, self.labels = self.ids[1:], self.labels[1:]
        self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, "eight"):
            self.build()

    def test_shot_diagnostics(self):
        result = train_balanced.per_shot_metrics(
            torch.tensor([-2., 2., 2., 2.]), torch.tensor([0., 1., 0., 1.]),
            ["2p0", "2p1", "ft0", "ft1"],
        )
        self.assertEqual(result["2p"]["accuracy"], 1.)
        self.assertEqual(result["ft"]["per_class"]["Miss"]["recall"], 0.)
        self.assertEqual(result["ft"]["confusion_matrix"], [[0, 1], [0, 1]])

    def test_dry_run_has_no_optimizer_or_output_and_preserves_sampler(self):
        sampler, info = self.build()
        protocol = {"input_identities": {"manifest_sha256": self.sha}}
        run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        train_x = torch.zeros(len(self.ids), 3, 8, 1536)
        val_x = torch.zeros(8, 3, 8, 1536)
        val_y = torch.tensor([float(g[-1]) for g in CLASSES])

        def load(cache, *, split):
            if split == "train":
                return train_x, self.labels, self.ids, run_id
            return val_x, val_y, [f"validation/{g}/0" for g in CLASSES], run_id

        old_threads = torch.get_num_threads()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            with patch.object(train_balanced, "load_sequences", side_effect=load), \
                 patch.object(train_balanced, "CACHE_RUN_ID", run_id), \
                 patch.object(train_balanced, "MANIFEST", self.path), \
                 patch.object(torch, "load", return_value=protocol), \
                 patch.object(train_balanced, "make_balanced_sampler", return_value=(sampler, info)), \
                 patch.object(torch.optim, "AdamW", side_effect=AssertionError("Optimizer")), \
                 patch.object(train_balanced, "mkdtemp", side_effect=AssertionError("Output")), \
                 patch.object(train_balanced, "train_one_epoch", side_effect=AssertionError("Train")), \
                 redirect_stdout(io.StringIO()) as output:
                train_balanced.main(["--dry-run", "--threads", "1"])
            self.assertIn('"optimizer_steps": 0', output.getvalue())
            self.assertIn('"output_files_written": false', output.getvalue())
            self.assertEqual(sampler.last_counts, {})
        finally:
            torch.set_num_threads(old_threads)
            torch.use_deterministic_algorithms(old_deterministic)


if __name__ == "__main__":
    unittest.main()
