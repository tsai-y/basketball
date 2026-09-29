"""RGB video loading with replaceable temporal sampling."""

import csv
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UniformBinSampler:
    """Sample one frame in each temporal bin of the entire decoded video."""

    def __init__(self, num_frames, jitter=False):
        self.num_frames = num_frames
        self.jitter = jitter

    def indices(self, frame_count):
        if frame_count < self.num_frames:
            raise ValueError(f"Only {frame_count} frames for {self.num_frames}-frame input")
        selected = []
        for i in range(self.num_frames):
            lo = i * frame_count // self.num_frames
            hi = (i + 1) * frame_count // self.num_frames - 1
            selected.append(random.randint(lo, hi) if self.jitter else (lo + hi) // 2)
        return selected


def letterbox_rgb(frame, image_size):
    height, width = frame.shape[:2]
    scale = min(image_size / height, image_size / width)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    x, y = (image_size - new_width) // 2, (image_size - new_height) // 2
    canvas[y:y + new_height, x:x + new_width] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return canvas


class BasketballVideoDataset(Dataset):
    def __init__(self, manifest_path, dataset_path, split, sampler, image_size, flip_probability=0.0,
                 video_ids=None):
        self.dataset_path = Path(dataset_path).resolve()
        self.sampler = sampler
        self.image_size = image_size
        self.flip_probability = flip_probability
        with Path(manifest_path).open(newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle)
                         if row["split"] == split and row["excluded"] == "0"
                         and (video_ids is None or row["video_id"] in video_ids)]
        if not self.rows:
            raise ValueError(f"Empty dataset for split {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.dataset_path / row["video_path"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {row['video_id']}")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            wanted = self.sampler.indices(frame_count)
            needed = set(wanted)
            frames = {}
            for frame_index in range(wanted[-1] + 1):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Decode ended at frame {frame_index}: {row['video_id']}")
                if frame_index in needed:
                    frames[frame_index] = letterbox_rgb(frame, self.image_size)
            array = np.stack([frames[i] for i in wanted], axis=0)
        finally:
            capture.release()
        if self.flip_probability and random.random() < self.flip_probability:
            array = array[:, :, ::-1, :].copy()
        tensor = torch.from_numpy(array.copy()).permute(3, 0, 1, 2).float()
        tensor = (tensor / 255.0 - 0.5) / 0.5
        return tensor, int(row["binary_label"]), row["video_id"]


def seed_worker(worker_id):
    cv2.setNumThreads(0)
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
