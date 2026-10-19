import unittest

from development.data_pipeline.pipeline import GateError, load_gate_config
from development.data_pipeline.multiwindow_v1 import (
    BasketballThreeWindowDataset,
    candidate_overlap_counts,
    select_covering_windows,
)

class MultiWindowTests(unittest.TestCase):
    def test_covers_every_frame(self):
        for n in (145, 146, 180):
            with self.subTest(frame_count=n):
                windows = select_covering_windows(n)
                self.assertEqual(len(windows), 3)
                self.assertEqual(
                    {i for start, end in windows for i in range(start, end)},
                    set(range(n)),
                )

    def test_rejects_uncovered_video(self):
        with self.assertRaises(ValueError):
            select_covering_windows(193)

    def test_dataset_contract(self):
        config = load_gate_config()
        for split in ("train", "validation"):
            with self.subTest(split=split):
                dataset = BasketballThreeWindowDataset(config, split=split)
                item = dataset[0]
                self.assertEqual(tuple(item["tensor"].shape), (3, 3, 64, 192, 256))
                self.assertEqual(
                    {f for start, end in item["windows"] for f in range(start, end)},
                    set(range(item["frame_count"])),
                )
                self.assertIn(item["binary_label"], (0, 1))
                overlaps = tuple(item["candidate_overlap_frames"].tolist())
                self.assertFalse(item["quality_aware_selection_used"])

                if split == "train":
                    self.assertTrue(item["quality_metadata_available"])
                    segments = dataset.base.quality_index.segments.get(
                        item["video_id"], []
                    )
                    self.assertEqual(
                        overlaps,
                        candidate_overlap_counts(item["windows"], segments),
                    )
                    self.assertTrue(all(0 <= count <= 64 for count in overlaps))
                else:
                    self.assertFalse(item["quality_metadata_available"])
                    self.assertIsNone(dataset.base.quality_index)
                    self.assertEqual(overlaps, (-1, -1, -1))
    def test_rejects_test_split(self):
        with self.assertRaises(GateError):
            BasketballThreeWindowDataset(load_gate_config(), split="test")

    def test_candidate_overlap_counts(self):
        windows = select_covering_windows(146)
        self.assertEqual(candidate_overlap_counts(windows, [(90, 98)]), (0, 8, 8))
        self.assertEqual(candidate_overlap_counts(windows, [(64, 65)]), (0, 1, 0))

if __name__ == "__main__":
    unittest.main()
