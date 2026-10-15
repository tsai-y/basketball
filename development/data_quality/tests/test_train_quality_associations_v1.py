"""Tests for the derived-only train quality association audit."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


DQ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DQ / "scripts"))
import audit_train_quality_associations_v1 as audit  # noqa: E402


class TrainQualityAssociationsV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = audit.parser().parse_args([])
        cls.data = audit.load_and_validate(cls.args)

    def test_candidate_statistics_use_one_row_per_video(self):
        rows = [
            {"video_id": "make-a", "label": "Make", "source_group": "s1", "candidate_ratio": 0.5},
            {"video_id": "make-b", "label": "Make", "source_group": "s1", "candidate_ratio": 0.0},
            {"video_id": "miss-a", "label": "Miss", "source_group": "s1", "candidate_ratio": 0.1},
        ]
        result = audit.summarize_candidate_by_label(rows)
        self.assertEqual(result["Make"]["n"], 2)
        self.assertEqual(result["Make"]["mean"], 0.25)
        self.assertEqual(result["Make"]["affected_count"], 1)
        self.assertEqual(result["Make"]["affected_rate"], 0.5)
        with self.assertRaisesRegex(audit.AuditError, "Duplicate video_id"):
            audit.summarize_candidate_by_label(rows + [dict(rows[0])])

    def test_normalized_candidate_positions(self):
        center, length = audit.normalize_segment(20, 40, 100)
        self.assertEqual(center, 0.3)
        self.assertEqual(length, 0.2)
        self.assertEqual(audit.position_band(center), "front")
        self.assertEqual(audit.position_band(0.5), "middle")
        self.assertEqual(audit.position_band(0.8), "rear")
        with self.assertRaisesRegex(audit.AuditError, "Invalid half-open"):
            audit.normalize_segment(40, 20, 100)

    def test_normalized_sampler_positions(self):
        normalized_start, center = audit.normalize_selection(18, 82, 100, 64)
        self.assertEqual(normalized_start, 0.5)
        self.assertEqual(center, 0.5)
        with self.assertRaisesRegex(audit.AuditError, "requires frame_count"):
            audit.normalize_selection(0, 64, 64, 64)

    def test_stratified_counts_preserve_videos(self):
        groups = audit.grouped_candidate(self.data["video_rows"], "shot_type")
        self.assertEqual({row["group"] for row in groups}, {"2p", "3p", "ft", "mp"})
        self.assertEqual(sum(row["videos"] for row in groups), audit.EXPECTED["train_videos"])
        for row in groups:
            self.assertEqual(row["videos"], row["Make"]["n"] + row["Miss"]["n"])

    def test_missing_and_duplicate_ids_are_rejected(self):
        audit.validate_exact_id_set(["a", "b"], ["a", "b"], "fixture")
        with self.assertRaisesRegex(audit.AuditError, "Duplicate video_id"):
            audit.validate_exact_id_set(["a", "a"], ["a"], "fixture")
        with self.assertRaisesRegex(audit.AuditError, "ID mismatch"):
            audit.validate_exact_id_set(["a"], ["a", "b"], "fixture")
        with self.assertRaisesRegex(audit.AuditError, "ID mismatch"):
            audit.validate_exact_id_set(["a", "b", "c"], ["a", "b"], "fixture")

    def test_existing_frozen_totals(self):
        self.assertEqual(self.data["totals"], audit.EXPECTED)
        self.assertEqual(len(self.data["video_rows"]), 7193)
        self.assertEqual(len(self.data["segment_rows"]), 1798)
        self.assertEqual(len(self.data["selection_rows"]), 7193)
        self.assertEqual(self.data["selection_origin"], "existing_derived_csv")

    def test_missing_ignored_selections_are_reconstructed_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            args = audit.parser().parse_args(["--selections", str(Path(directory) / "missing.csv")])
            rebuilt = audit.load_and_validate(args)
        self.assertEqual(rebuilt["selection_origin"], "metadata_only_reconstruction")
        self.assertEqual(rebuilt["totals"], audit.EXPECTED)
        keys = ("video_id", "start_frame", "end_frame_exclusive", "candidate_overlap", "fallback")
        expected = [tuple(row[key] for key in keys) for row in self.data["selection_rows"]]
        actual = [tuple(row[key] for key in keys) for row in rebuilt["selection_rows"]]
        self.assertEqual(actual, expected)

    def test_split_is_source_group_holdout_only(self):
        check = self.data["split_check"]
        self.assertEqual(check["source_groups"], 50)
        self.assertEqual(check["cross_split_source_groups"], {})
        self.assertTrue(check["v160_v166_same_source_group"])
        self.assertIn("not confirmed game-level", check["holdout_description"])
        self.assertFalse(check["near_duplicate_video_scan_performed"])

    def test_cluster_bootstrap_is_deterministic(self):
        rows = [
            {"video_id": "a", "label": "Make", "source_group": "s1", "candidate_ratio": 0.2},
            {"video_id": "b", "label": "Miss", "source_group": "s1", "candidate_ratio": 0.0},
            {"video_id": "c", "label": "Make", "source_group": "s2", "candidate_ratio": 0.0},
            {"video_id": "d", "label": "Miss", "source_group": "s2", "candidate_ratio": 0.1},
        ]
        first = audit.cluster_bootstrap_two_part(rows, "candidate_ratio", replicates=50, seed=7)
        second = audit.cluster_bootstrap_two_part(rows, "candidate_ratio", replicates=50, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["mean_difference"]["estimate"], 0.05)

    def test_multi_epoch_counterfactual_is_deterministic(self):
        videos = [
            {"video_id": "make-a", "label": "Make", "frame_count": 100},
            {"video_id": "miss-a", "label": "Miss", "frame_count": 100},
        ]
        segments = [
            {"video_id": "make-a", "start_frame": 30, "end_frame_exclusive": 50},
        ]
        config = {
            "clip_length": 20, "num_clips_per_video": 1, "global_seed": 7,
            "train_selection": "seeded_random", "candidate_margin_frames": 0,
        }
        first = audit.multi_epoch_sampler_counterfactual(videos, segments, config, epochs=3)
        second = audit.multi_epoch_sampler_counterfactual(videos, segments, config, epochs=3)
        self.assertEqual(first, second)
        self.assertEqual(first["epochs"], 3)
        self.assertEqual(len(first["per_epoch"]), 3)
        self.assertIn(
            "quality_induced_difference_in_differences",
            first["summary_across_epochs"],
        )

    def test_check_allows_missing_ignored_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracked = root / "tracked.json"
            tracked.write_text("current\n", encoding="utf-8")
            required = audit.verify_text_outputs({tracked: "current\n"}, allow_missing=False)
            optional = audit.verify_text_outputs(
                {root / "ignored.csv": "derived\n"}, allow_missing=True)
        self.assertEqual(required, {"verified": 1, "missing_optional": 0})
        self.assertEqual(optional, {"verified": 0, "missing_optional": 1})

    def test_refresh_writes_and_check_rejects_stale_generated_text(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.csv"
            audit.write_text_outputs({output: "fresh\n"})
            self.assertEqual(output.read_text(encoding="utf-8"), "fresh\n")
            output.write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(audit.AuditError, "stale"):
                audit.verify_text_outputs({output: "fresh\n"}, allow_missing=True)


if __name__ == "__main__":
    unittest.main()
