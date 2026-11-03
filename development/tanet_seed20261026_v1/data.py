"""Guarded fixed-split data adapter; the temporal selector is NOT integrated.

Default: full-video endpoint-uniform50, full-frame 256x192, no augmentation.
The pure index/decoder interval argument is a future integration seam, not an
automatic endorsement of any unreviewed temporal-selection proposal.
"""
from dataclasses import dataclass
import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from development.data_pipeline.pipeline import gate1a
from .config import InputConfig, need

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FrameInterval:
    start: int
    end_exclusive: int


def uniform_indices(frame_count, frames=50, interval=None):
    need(type(frame_count) is int and frame_count >= 2, "Invalid frame count")
    need(type(frames) is int and frames >= 2, "Invalid sampling count")
    interval = FrameInterval(0, frame_count) if interval is None else interval
    need(isinstance(interval, FrameInterval) and type(interval.start) is int and
         type(interval.end_exclusive) is int and
         0 <= interval.start < interval.end_exclusive <= frame_count, "Invalid [start,end) interval")
    length = interval.end_exclusive - interval.start
    need(length >= frames, "Short interval rejected; no repetition/black padding")
    return tuple(interval.start + i * (length - 1) // (frames - 1) for i in range(frames))


class VideoScope:
    def __init__(self, split, video_ids=None):
        need(split in ("train", "validation"), "Test/unknown split rejected before metadata/path access")
        rows, self.denied, self.identity = gate1a.load_scope([split])
        self.allowed = {r["video_id"]: dict(r) for r in rows}
        ids = sorted(self.allowed) if video_ids is None else list(video_ids)
        need(len(ids) == len(set(ids)) and all(v in self.allowed for v in ids),
             "Unknown/cross-split/test/excluded ID rejected before video path access")
        self.rows = tuple(dict(self.allowed[v]) for v in ids)
        self.split = split

    def authorize(self, video_id):
        need(video_id in self.allowed, "Forbidden video ID")
        row = dict(self.allowed[video_id])
        return row, gate1a.authorize(row, self.allowed, self.denied)


def normalize_rgb(rgb, config=InputConfig()):
    need(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and
         rgb.shape == (config.frames, config.height, config.width, 3), "Unexpected RGB clip")
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(3, 0, 1, 2).float() / 255
    return ((x - x.new_tensor(MEAN)[:, None, None, None]) /
            x.new_tensor(STD)[:, None, None, None]).contiguous()


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def decode(scope, video_id, config=InputConfig(), *, interval=None):
    row, path = scope.authorize(video_id)
    fd = gate1a.open_guarded(path)
    cap = None
    try:
        before = os.fstat(fd)
        cap = cv2.VideoCapture(f"/proc/self/fd/{fd}", cv2.CAP_FFMPEG)
        need(cap.isOpened(), "Cannot decode authorized video")
        raw_n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        need(math.isfinite(raw_n) and raw_n >= config.frames and abs(raw_n - round(raw_n)) < 1e-6,
             "Invalid/short video frame count")
        need(math.isfinite(fps) and fps > 0, "Invalid FPS")
        count = int(round(raw_n))
        indices = uniform_indices(count, config.frames, interval)
        wanted, retained = set(indices), []
        for index in range(count):
            ok, bgr = cap.read()
            need(ok, f"Truncated video at frame {index}")
            need(isinstance(bgr, np.ndarray) and bgr.dtype == np.uint8 and bgr.shape == (240, 320, 3),
                 "Native 320x240 BGR source required")
            if index in wanted:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                retained.append(cv2.resize(rgb, (config.width, config.height), interpolation=cv2.INTER_AREA))
        need(not cap.read()[0], "Metadata/decoded frame count mismatch")
        need(len(retained) == config.frames, "Incomplete sample")
        need(_identity(os.fstat(fd)) == _identity(before), "Source changed during decode")
        tensor = normalize_rgb(np.stack(retained), config)
        need(torch.isfinite(tensor).all().item(), "Nonfinite clip")
        chosen = FrameInterval(0, count) if interval is None else interval
        metadata = {"frame_count": count, "fps": fps, "selected_indices": list(indices),
                    "selected_times_seconds": [i / fps for i in indices],
                    "interval": {"start": chosen.start, "end_exclusive": chosen.end_exclusive},
                    "temporal_selection_used": interval is not None,
                    "video_open_count": 1, "decoded_frames": count,
                    "source_unchanged_during_decode": True, "source_size_bytes": before.st_size}
        return {"tensor": tensor, "binary_label": int(row["binary_label"]),
                "video_id": video_id, "source_group": row["source_group"],
                "split": row["split"], "original_class": row["original_class"], "metadata": metadata}
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)


class BasketballTANetDataset(Dataset):
    def __init__(self, *, split, video_ids=None, config=InputConfig()):
        self.scope = VideoScope(split, video_ids)
        self.config = config

    def __len__(self):
        return len(self.scope.rows)

    def __getitem__(self, index):
        return decode(self.scope, self.scope.rows[index]["video_id"], self.config)
