"""Continuous RGB clips from the fixed, cleaned Basketball-51 manifest."""

import csv
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ContinuousClipSampler:
    def __init__(self, length=16, stride=8, tail_policy="discard_incomplete"):
        if length < 1 or stride < 1 or tail_policy != "discard_incomplete":
            raise ValueError("Invalid clip settings")
        self.length, self.stride, self.tail_policy = length, stride, tail_policy

    def indices(self, total_frames):
        if total_frames < self.length:
            raise ValueError(f"Video has {total_frames} frames, needs {self.length}")
        return [tuple(range(start, start + self.length))
                for start in range(0, total_frames - self.length + 1, self.stride)]


def preprocess_frames(frames, *, resize_height, resize_width, crop_size, crop_mode,
                      flip_probability, pixel_preprocessing):
    if pixel_preprocessing != "rgb_float_minus_one_to_one":
        raise ValueError(f"Unsupported pixel preprocessing: {pixel_preprocessing}")
    if crop_size > min(resize_height, resize_width):
        raise ValueError("Crop exceeds resized frame")
    max_y, max_x = resize_height - crop_size, resize_width - crop_size
    if crop_mode == "random":
        top, left = random.randint(0, max_y), random.randint(0, max_x)
    elif crop_mode == "center":
        top, left = max_y // 2, max_x // 2
    else:
        raise ValueError(f"Unknown crop mode: {crop_mode}")
    flip = flip_probability > 0 and random.random() < flip_probability
    result = []
    for frame in frames:
        resized = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(resized[top:top + crop_size, left:left + crop_size], cv2.COLOR_BGR2RGB)
        result.append(crop[:, ::-1, :] if flip else crop)
    array = np.ascontiguousarray(np.stack(result))
    return (array.astype(np.float32) / 255.0 - 0.5) / 0.5


class C3DVideoDataset(Dataset):
    """One item is one video and all complete overlapping clips in that video."""

    def __init__(self, manifest_path, dataset_path, split, sampler, preprocessing, video_ids=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError(split)
        self.dataset_path = Path(dataset_path)
        self.sampler = sampler
        self.preprocessing = dict(preprocessing)
        with Path(manifest_path).open(newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle)
                         if row["split"] == split and row["excluded"] == "0"
                         and (video_ids is None or row["video_id"] in video_ids)]
        if not self.rows:
            raise ValueError(f"No usable {split} videos selected")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        capture = cv2.VideoCapture(str(self.dataset_path / row["video_path"]))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {row['video_id']}")
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        indices = self.sampler.indices(len(frames))
        transformed = preprocess_frames(frames, **self.preprocessing)
        clips = np.stack([transformed[list(clip)] for clip in indices])
        # [clips, time, height, width, RGB] -> [clips, RGB, time, height, width]
        tensor = torch.from_numpy(np.ascontiguousarray(clips.transpose(0, 4, 1, 2, 3)))
        return {"clips": tensor, "indices": indices, "total_frames": len(frames),
                "label": int(row["binary_label"]), "video_id": row["video_id"]}
