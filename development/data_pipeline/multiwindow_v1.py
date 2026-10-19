"""Select three contiguous 64-frame windows that cover one video."""

import torch

from development.data_pipeline.pipeline import BasketballClipDataset
from development.data_quality.transforms.video_clip_transforms import (
    ClipTransform, derived_seed, normalize_tensor,
)

def select_covering_windows(frame_count: int) -> tuple[tuple[int, int], ...]:
    if type(frame_count) is not int or frame_count < 66:
        raise ValueError("Need enough frames for three distinct windows")

    last_start = frame_count - 64
    starts = (0, last_start // 2, last_start)
    windows = tuple((start, start + 64) for start in starts)

    if any(left[1] < right[0] for left, right in zip(windows, windows[1:])):
        raise ValueError("Three 64-frame windows cannot cover this video")
    return windows

class BasketballThreeWindowDataset(torch.utils.data.Dataset):
    """Engineering prototype: three windows, one video-level label."""

    def __init__(self, config, *, split, video_ids=None, epoch=0):
        self.base = BasketballClipDataset(
            config, split=split, video_ids=video_ids, epoch=epoch
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        base = self.base
        row = base.rows[index]
        middle, _, selected, n, fps = base.decoder(
            row, base.guard, selection=None,
            expected_frame_count=None, expected_fps_rational=None,
        )
        windows = select_covering_windows(n)
        if (selected["start_frame"], selected["end_frame_exclusive"]) != windows[1]:
            raise RuntimeError("Middle window mismatch")

        config = {k: v for k, v in base.config.raw["transform"].items()
                  if k != "policy"}
        transform = ClipTransform(config, "reference")
        tensors = []
        for i, (start, end) in enumerate(windows):
            if i == 1:
                clip = middle
            else:
                clip, _, _, _, _ = base.decoder(
                    row, base.guard,
                    selection={"start_frame": start, "end_frame_exclusive": end},
                    expected_frame_count=n, expected_fps_rational=None,
                )
            array, _ = transform(
                clip,
                seed=derived_seed(base.config.raw["seed"], row["video_id"],
                                  base.epoch, i, "reference"),
                split=base.split,
                sample_metadata={"start_frame": start,
                                 "end_frame_exclusive": end},
            )
            array, _ = normalize_tensor(array, mode="identity")
            tensors.append(torch.from_numpy(array))

        return {
            "tensor": torch.stack(tensors),
            "binary_label": int(row["binary_label"]),
            "video_id": row["video_id"],
            "split": base.split,
            "frame_count": n,
            "fps": fps,
            "windows": windows,
            "policy_version": "three_window_coverage_v1",
            "quality_aware_selection_used": False,
        }
