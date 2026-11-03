import copy
from html.parser import HTMLParser
from pathlib import Path
import unittest
from unittest.mock import patch

from .build import QUESTIONS, STRATA, rank_id, render, review_fields, select, source_path, stratum


def fixtures():
    rows = []
    for label, pred in ((0, 1), (1, 0), (0, 0), (1, 1)):
        for i in range(16):
            rows.append({"video_id": f"ft{label}/f{label}{pred}_{i}.mp4", "split": "validation", "label": label,
                         "prediction": pred, "shot_type": "ft", "original_class": f"ft{label}",
                         "source_group": f"v{i%8}", "p_make": .6,
                         "models": {name: {"prediction": pred, "p_make": .6} for name in ("dense", "dw3d", "mainline")}})
    for shot in ("2p", "3p", "mp"):
        for i in range(8):
            rows.append({"video_id": f"{shot}0/{shot}_{i}.mp4", "split": "validation", "label": 0,
                         "prediction": 1, "shot_type": shot, "original_class": shot+"0", "source_group": f"v{i}",
                         "p_make": .7, "models": {name: {"prediction": 1, "p_make": .7} for name in ("dense", "dw3d", "mainline")}})
    return rows


class ChecklistTests(unittest.TestCase):
    def test_fixed_24_unique_and_strata(self):
        chosen, _ = select(fixtures())
        self.assertEqual(len(chosen), 24)
        self.assertEqual(len({r["video_id"] for r in chosen}), 24)
        for key, _, q in STRATA:
            self.assertEqual(sum(r["stratum"] == key for r in chosen), q)
        self.assertEqual(sorted(r["shot_type"] for r in chosen if r["stratum"] == "other_joint_error"), ["2p", "2p", "3p", "mp"])

    def test_order_independent_and_confidence_independent(self):
        a = fixtures()
        b = copy.deepcopy(a[::-1])
        for row in b:
            row["p_make"] = .999
        self.assertEqual([r["video_id"] for r in select(a)[0]], [r["video_id"] for r in select(b)[0]])

    def test_source_diversity_and_no_input_mutation(self):
        a = fixtures(); before = copy.deepcopy(a)
        result, _ = select(a)
        self.assertEqual(a, before)
        wrong_miss = [r for r in result if r["stratum"] == "ft_false_make"]
        self.assertEqual(len({r["source_group"] for r in wrong_miss}), 8)

    def test_insufficient_duplicate_and_test_rows_refused(self):
        for rows in (fixtures()[:4], fixtures()+fixtures()[:1], [{**r, "split": "test"} for r in fixtures()]):
            with self.assertRaises(ValueError):
                select(rows)

    def test_other_joint_error_requires_all_three_wrong(self):
        row = fixtures()[-1]
        self.assertEqual(stratum(row), "other_joint_error")
        row["models"]["mainline"]["prediction"] = 0
        self.assertIsNone(stratum(row))

    def test_source_authorization_before_filesystem_access(self):
        row = fixtures()[0]
        with patch.object(Path, "is_symlink", side_effect=AssertionError("filesystem accessed")):
            with self.assertRaises(ValueError):
                source_path(row, {})

    def test_human_fields_not_invented(self):
        fields = review_fields()
        self.assertFalse(fields["reviewed"])
        self.assertIsNone(fields["human_outcome"])
        self.assertIsNone(fields["key_result_time_seconds"])
        self.assertFalse(fields["aca_sampled_frames_checked"])

    def test_html_escaped_and_no_video_auto_loading(self):
        cases, _ = select(fixtures())
        for case in cases:
            case["source_url"] = "file:///safe.mp4"
        cases[0]["video_id"] = "<script>bad</script>"
        page = render(cases, {"selected_source_group_count": 8})
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("<video", page)
        self.assertNotIn("<script", page)
        self.assertEqual(page.count("<article "), 24)
        self.assertEqual(page.count("開啟原始影片"), 24)
        self.assertEqual(page.count("<li>□"), 24*len(QUESTIONS))


if __name__ == "__main__":
    unittest.main()
