"""Fixed-split, one-pass, 50-frame full-video input for the reconstruction.

Uniform endpoint sampling and full-frame resize are PROJECT choices. The paper
only specifies T=50 and 120x160, not its exact sampling/crop/normalization.
"""
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from development.data_pipeline.pipeline import gate1a


ROOT = Path(__file__).resolve().parents[2]
FRAMES, HEIGHT, WIDTH = 50, 120, 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASSES = ("2p0", "2p1", "3p0", "3p1", "ft0", "ft1", "mp0", "mp1")
POLICY = {
    "version": "aca_full_video_uniform50_v1",
    "sampling": "floor(i*(frame_count-1)/49), i=0..49",
    "short_video": "reject_if_fewer_than_50_frames",
    "resize": "full_frame_320x240_to_160x120_INTER_AREA_no_crop",
    "color": "RGB", "dtype": "float32", "layout": "C,T,H,W",
    "normalization": {"scale": 255.0, "mean": MEAN, "std": STD},
    "normalization_reference": "torchvision.ResNet50_Weights.IMAGENET1K_V1",
    "weight_checkpoint_loaded_by_pipeline": False,
    "train_augmentation": "none", "train_quality_filter": "none",
    "test": "rejected_before_video_path_access",
    "paper_exact_preprocessing_reproduced": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    return gate1a.digest(Path(path))


def uniform_indices(frame_count):
    require(type(frame_count) is int and frame_count >= FRAMES,
            "Need at least 50 decoded frames; no looping/padding")
    return tuple(i * (frame_count - 1) // (FRAMES - 1) for i in range(FRAMES))


def normalize_rgb(rgb):
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and
            rgb.shape == (FRAMES, HEIGHT, WIDTH, 3), "Invalid resized RGB clip")
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(3, 0, 1, 2).float() / 255.0
    mean, std = x.new_tensor(MEAN)[:, None, None, None], x.new_tensor(STD)[:, None, None, None]
    return ((x - mean) / std).contiguous()


def preview_rgb(tensor):
    require(tuple(tensor.shape) == (3, FRAMES, HEIGHT, WIDTH), "Wrong tensor shape")
    x = tensor.detach().cpu()
    x = x * x.new_tensor(STD)[:, None, None, None] + x.new_tensor(MEAN)[:, None, None, None]
    return x.mul(255).round().clamp(0, 255).byte().permute(1, 2, 3, 0).numpy()


class VideoScope:
    """Pinned manifest metadata; sources remain in the shared read-only dataset."""
    def __init__(self, *, source_root, split, video_ids=None):
        # Must precede even metadata reads or source-root path construction.
        require(split in ("train", "validation"), "Only train/validation accepted")
        manifest = ROOT / "data/processed/dataset_manifest.csv"
        summary_path = ROOT / "data/processed/split_summary.json"
        self.hashes = {"manifest": file_sha256(manifest), "split_summary": file_sha256(summary_path)}
        require(self.hashes == gate1a.HASHES, "Frozen manifest/split identity changed")
        with manifest.open(newline="", encoding="utf-8") as stream:
            all_rows = list(csv.DictReader(stream))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(Counter(r["split"] for r in all_rows) ==
                {"train": 7193, "validation": 1547, "test": 1539, "excluded": 32},
                "Unexpected fixed-split counts")
        require(len({r["video_id"] for r in all_rows}) == len(all_rows) and
                len({r["video_path"] for r in all_rows}) == len(all_rows), "Duplicate IDs/paths")
        for row in all_rows:
            gate1a.safe_relative(row["video_path"])
            require(row["video_id"] == row["video_path"] and row["original_class"] in CLASSES and
                    row["binary_label"] == row["original_class"][-1], "Invalid label or ID")
            if row["split"] != "excluded":
                require(row["excluded"] == "0" and
                        summary["group_assignment"].get(row["source_group"]) == row["split"],
                        "Group isolation or exclusion mismatch")
        for name in ("train", "validation", "test"):
            require(dict(Counter(r["binary_label"] for r in all_rows if r["split"] == name)) ==
                    summary["label_counts"][name], "Class counts changed")
        self.source_root = Path(source_root).absolute()
        self.split = split
        self._allowed = {r["video_id"]: dict(r) for r in all_rows if r["split"] == split}
        requested = sorted(self._allowed) if video_ids is None else list(video_ids)
        require(len(requested) == len(set(requested)) and
                all(i in self._allowed for i in requested), "Duplicate or forbidden requested ID")
        self.rows = tuple(dict(self._allowed[i]) for i in requested)
        self.counts = dict(Counter(r["split"] for r in all_rows))

    def authorize(self, video_id):
        require(video_id in self._allowed, "Unknown/cross-split/test/excluded ID rejected before path access")
        row = dict(self._allowed[video_id])
        relative = gate1a.safe_relative(row["video_path"])
        return row, self.source_root / "archive/Basketball_51 dataset" / relative


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def guarded_source_sha256(scope, video_id):
    _, path = scope.authorize(video_id)
    fd = gate1a.open_guarded(path)
    digest = hashlib.sha256()
    with os.fdopen(fd, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_uniform50(scope, video_id):
    row, path = scope.authorize(video_id)
    fd = gate1a.open_guarded(path)
    cap = None
    try:
        before = os.fstat(fd)
        cap = cv2.VideoCapture(f"/proc/self/fd/{fd}", cv2.CAP_FFMPEG)
        require(cap.isOpened(), "Cannot decode authorized source")
        raw_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        require(math.isfinite(raw_count) and raw_count >= FRAMES and
                abs(raw_count - round(raw_count)) < 1e-6, "Invalid/short source frame count")
        require(math.isfinite(fps) and fps > 0, "Invalid source FPS")
        n = int(round(raw_count))
        indices = uniform_indices(n)
        wanted = set(indices)
        retained = []
        for index in range(n):
            ok, bgr = cap.read()
            require(ok, f"Truncated source at frame {index}")
            require(isinstance(bgr, np.ndarray) and bgr.dtype == np.uint8 and
                    bgr.shape == (240, 320, 3), "Expected original 320x240 BGR frame")
            if index in wanted:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                retained.append(cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
        require(not cap.read()[0], "Metadata frame count is smaller than decoded count")
        require(len(retained) == FRAMES, "Incomplete 50-frame selection")
        require(_identity(os.fstat(fd)) == _identity(before), "Source changed while decoding")
        metadata = {
            "frame_count": n, "fps": fps, "selected_indices": indices,
            "selected_times_seconds": [i / fps for i in indices],
            "decoder": "OpenCV_CAP_FFMPEG_sequential_authorized_fd",
            "video_open_count": 1, "decoded_frames": n,
            "eof_check_reads": 1, "source_unchanged_during_decode": True,
            "source_size_bytes": before.st_size,
        }
        return np.stack(retained), metadata, row
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)


class Basketball50Dataset(Dataset):
    def __init__(self, *, source_root, split, video_ids=None):
        self.scope = VideoScope(source_root=source_root, split=split, video_ids=video_ids)

    def __len__(self):
        return len(self.scope.rows)

    def __getitem__(self, index):
        video_id = self.scope.rows[index]["video_id"]
        rgb, metadata, row = decode_uniform50(self.scope, video_id)
        tensor = normalize_rgb(rgb)
        require(torch.isfinite(tensor).all().item(), "Nonfinite normalized input")
        return {"tensor": tensor, "binary_label": int(row["binary_label"]),
                "video_id": video_id, "original_class": row["original_class"],
                "source_group": row["source_group"], "split": row["split"],
                "metadata": metadata}


def select_smoke_ids(scope):
    """Four metadata-selected examples, different groups, before opening videos."""
    selected, groups = [], set()
    for cls in ("2p0", "2p1", "ft0", "ft1"):
        candidates = [r for r in scope.rows if r["original_class"] == cls and r["source_group"] not in groups]
        require(bool(candidates), "Insufficient source-disjoint smoke strata")
        row = candidates[0]
        selected.append(row["video_id"])
        groups.add(row["source_group"])
    return selected
