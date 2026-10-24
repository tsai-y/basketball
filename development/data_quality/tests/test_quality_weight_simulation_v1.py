"""Arithmetic-only checks: no models, feature loading, or training."""

import math
import unittest

from development.data_quality.scripts import simulate_quality_weights_v1 as simulation


def fixture(video_id, label, shot, counts):
    return {"video_id": video_id, "label": label, "shot_type": shot,
            "original_class": shot + ("1" if label == "Make" else "0"),
            "source_group": "s1", "fps_rational": "24/1",
            "candidate_overlap_frames": counts, "window_exposure_ratio": sum(counts) / 192}


class QualityWeightSimulationTests(unittest.TestCase):
    def test_zero_full_and_known_ratio(self):
        self.assertEqual(simulation.video_weight([0, 0, 0]), 1)
        self.assertEqual(simulation.video_weight([64, 64, 64]), .75)
        self.assertAlmostEqual(simulation.video_weight([8, 0, 5]), 1 - .25 * 13 / 192)

    def test_alpha05_formula_and_legacy_default_are_distinct(self):
        self.assertEqual(simulation.video_weight([64, 64, 64], alpha=.5), .5)
        self.assertEqual(simulation.video_weight([0, 0, 0], alpha=.5), 1)
        self.assertEqual(simulation.video_weight([24, 16, 11], alpha=.5), .8671875)
        self.assertEqual(simulation.video_weight([24, 16, 11]), .93359375)

    def test_unsupported_alpha_rejected(self):
        for alpha in (-.1, 0, .75, 1, True, float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                simulation.video_weight([0, 0, 0], alpha=alpha)

    def test_more_overlap_never_has_higher_weight(self):
        values = [simulation.video_weight([x, 0, 0]) for x in range(65)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_bad_counts_fail_closed(self):
        for counts in ([-1, 0, 0], [65, 0, 0], [True, 0, 0], [1., 0, 0],
                       [math.nan, 0, 0], [0, 0]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                simulation.video_weight(counts)

    def test_group_total_and_global_scale_invariance(self):
        rows = [fixture("a", "Make", "2p", [64, 64, 64]),
                fixture("b", "Miss", "2p", [0, 0, 0]),
                fixture("c", "Make", "ft", [0, 0, 0])]
        report, weights = simulation.build_report(rows)
        self.assertEqual(report["overall"]["raw_weight_sum"], 2.75)
        self.assertEqual(report["by_label"]["Make"]["raw_weight_sum"], 1.75)
        self.assertAlmostEqual(report["by_label"]["Make"]["population_share_after"], 1.75 / 2.75)
        self.assertAlmostEqual(sum(row["global_mean_normalized_weight"] for row in weights), 3)
        for field in ("by_label", "by_shot_type", "by_original_class"):
            self.assertAlmostEqual(sum(group["raw_weight_sum"] for group in report[field].values()), 2.75)
            self.assertAlmostEqual(sum(group["population_share_change_percentage_points"]
                                       for group in report[field].values()), 0)
        make = report["by_label"]["Make"]
        self.assertAlmostEqual(make["global_mean_normalized_weight_sum"] / 3,
                               make["population_share_after"])
        self.assertFalse(report["training_started"])

    def test_empty_duplicate_or_inconsistent_ratio_rejected(self):
        row = fixture("a", "Make", "2p", [0, 0, 0])
        for rows in ([], [row, row], [{**row, "window_exposure_ratio": .1}],
                     [{**row, "window_exposure_ratio": math.nan}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                simulation.make_weights(rows)

    def test_original_rows_are_not_mutated(self):
        rows = [fixture("a", "Miss", "ft", [8, 0, 0])]
        simulation.make_weights(rows)
        self.assertNotIn("raw_weight", rows[0])
        self.assertNotIn("global_mean_normalized_weight", rows[0])


if __name__ == "__main__":
    unittest.main()
