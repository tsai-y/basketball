import unittest

from development.data_pipeline.pipeline import GateError, load_gate_config
from development.data_pipeline.multiwindow_v1 import BasketballThreeWindowDataset

from development.data_pipeline.multiwindow_v1 import select_covering_windows


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
                item = BasketballThreeWindowDataset(config, split=split)[0]
                self.assertEqual(tuple(item["tensor"].shape), (3, 3, 64, 192, 256))
                self.assertEqual(
                    {f for start, end in item["windows"] for f in range(start, end)},
                    set(range(item["frame_count"])),
                )
                self.assertIn(item["binary_label"], (0, 1))

    def test_rejects_test_split(self):
        with self.assertRaises(GateError):
            BasketballThreeWindowDataset(load_gate_config(), split="test")


if __name__ == "__main__":
    unittest.main()
