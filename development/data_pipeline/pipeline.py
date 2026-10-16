"""Reproducible PyTorch data path for Data Pipeline Integration Gate v1.

The module performs no work at import time and never accepts the test split.
It reuses the frozen quality-aware sampler and reference transform rather than
copying their algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from development.data_quality.samplers.quality_aware_clip_sampler import (
    POLICY_VERSION as QUALITY_POLICY_VERSION,
    CandidateSegment,
    select_clips,
)
from development.data_quality.transforms.video_clip_transforms import (
    VERSION as TRANSFORM_VERSION,
    ClipTransform,
    derived_seed,
    normalize_tensor,
    validate_config as validate_transform_config,
)
from development.data_quality.scripts import audit_quality_aware_sampler_v1 as quality_io


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "development/data_pipeline/configs/data_pipeline_gate_v1.yaml"
ALLOWED_SPLITS = ("train", "validation")
VALIDATION_POLICY_VERSION = "validation_temporal_center_contiguous_v1"


class GateError(ValueError):
    """A fail-closed contract violation."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _load_gate_module():
    path = ROOT / "scripts/audit_i3d_temporal_feasibility.py"
    spec = importlib.util.spec_from_file_location("data_pipeline_gate1a", path)
    need(spec is not None and spec.loader is not None, "Cannot load guarded video access module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate1a = _load_gate_module()


@dataclass(frozen=True)
class GateConfig:
    path: Path
    sha256: str
    raw: dict[str, Any]
    paths: dict[str, Path]


def load_gate_config(path: Path = DEFAULT_CONFIG) -> GateConfig:
    """Validate the complete Gate config and all immutable input identities."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "version", "seed", "allowed_splits", "paths", "identities", "clip",
        "train_sampling", "validation_sampling", "transform", "normalization", "access_policy",
    }
    need(isinstance(raw, dict) and set(raw) == required, "Unknown/missing Gate config keys")
    need(raw["version"] == "data_pipeline_integration_gate_v1", "Unsupported Gate version")
    need(raw["allowed_splits"] == ["train", "validation"], "Only train/validation are allowed")
    need(type(raw["seed"]) is int and 0 <= raw["seed"] < 2**64, "Invalid Gate seed")
    need(raw["clip"] == {"length": 64, "clips_per_video": 1}, "Gate requires one 64-frame clip")
    need(raw["train_sampling"] == {
        "policy_version": QUALITY_POLICY_VERSION,
        "selection_mode": "seeded_random",
        "global_seed": 20261009,
        "candidate_margin_frames": 0,
        "fallback_rule": "minimum_candidate_overlap",
    }, "Train sampler contract mismatch")
    need(raw["validation_sampling"] == {
        "policy_version": VALIDATION_POLICY_VERSION,
        "rule": "floor((frame_count-clip_length)/2)",
        "quality_aware_mode": "disabled",
    }, "Validation sampling contract mismatch")
    need(raw["transform"].get("policy") == "reference", "Only reference preprocessing is allowed")
    transform_cfg = {key: value for key, value in raw["transform"].items() if key != "policy"}
    validate_transform_config(transform_cfg)
    need(transform_cfg["version"] == TRANSFORM_VERSION and
         (transform_cfg["clip_length"], transform_cfg["output_height"], transform_cfg["output_width"])
         == (64, 192, 256), "Reference transform shape contract mismatch")
    need(raw["normalization"] == {"mode": "identity"}, "Only explicit identity normalization is allowed")
    need(raw["access_policy"] == {
        "test": "reject_before_path_access",
        "decoder": "sequential_read_no_seek",
        "source": "read_only_guarded_descriptor",
    }, "Access policy mismatch")

    expected_paths = {
        "manifest", "split_summary", "sampler_config", "sidecar_metadata",
        "sidecar_segments", "sidecar_video_summary",
    }
    need(set(raw["paths"]) == expected_paths, "Unexpected input paths")
    paths = {key: ROOT / value for key, value in raw["paths"].items()}
    need(all(path_.is_file() and not path_.is_symlink() for path_ in paths.values()),
         "Missing or symlinked immutable input")
    identities = raw["identities"]
    expected_identity_keys = {f"{name}_sha256" for name in expected_paths} | {"sidecar_run_identity"}
    need(set(identities) == expected_identity_keys, "Unexpected identity fields")
    for name, input_path in paths.items():
        need(sha256(input_path) == identities[f"{name}_sha256"], f"Input identity mismatch: {name}")
    sidecar = _read_json(paths["sidecar_metadata"])
    need(sidecar.get("run_identity") == identities["sidecar_run_identity"], "Sidecar run identity mismatch")
    need(sidecar.get("manifest", {}).get("sha256") == identities["manifest_sha256"],
         "Sidecar manifest identity mismatch")
    return GateConfig(path=path, sha256=sha256(path), raw=raw, paths=paths)


def select_validation_clip(frame_count: int, clip_length: int = 64) -> dict[str, Any]:
    """Select the frozen deterministic validation center window."""
    need(type(frame_count) is int and frame_count > 0, "frame_count must be a positive integer")
    need(type(clip_length) is int and clip_length == 64, "Validation requires 64 frames")
    need(frame_count >= clip_length, "Validation video is shorter than 64 frames")
    start = (frame_count - clip_length) // 2
    return {
        "clip_index": 0,
        "start_frame": start,
        "end_frame_exclusive": start + clip_length,
        "candidate_overlap": None,
        "fallback": False,
        "fallback_reason": None,
        "policy_version": VALIDATION_POLICY_VERSION,
        "selection_mode": "temporal_center",
    }


class TrainQualityIndex:
    """Strict adapter around the existing v3.1 sidecar loader."""

    def __init__(self, config: GateConfig, expected_run_identity: str | None = None):
        expected = expected_run_identity or config.raw["identities"]["sidecar_run_identity"]
        summaries, segments, metadata = quality_io.load_sidecars(
            config.paths["manifest"],
            config.paths["sidecar_segments"],
            config.paths["sidecar_video_summary"],
        )
        need(metadata.get("run_identity") == expected, "Sidecar identity mismatch")
        need(metadata.get("policy_version") == "blur_screening_v3_1", "Sidecar policy mismatch")
        self.summaries = summaries
        # The historical loader imports the sampler under its legacy top-level
        # module name.  Normalize at this adapter boundary so the same sampler
        # loaded through the package namespace receives its documented tuple
        # input rather than a class object with a different module identity.
        self.segments = {
            video_id: [(segment.start_frame, segment.end_frame_exclusive) for segment in values]
            for video_id, values in segments.items()
        }
        self.metadata = metadata

    def selection(self, video_id: str, *, seed: int, epoch: int, clip_index: int) -> tuple[dict[str, Any], dict[str, str]]:
        need(video_id in self.summaries, f"Missing train sidecar row: {video_id}")
        summary = self.summaries[video_id]
        need(clip_index == 0, "Gate v1 allows one clip per video")
        record = select_clips(
            video_id=video_id,
            frame_count=int(summary["frame_count"]),
            candidate_segments=self.segments.get(video_id, []),
            clip_length=64,
            num_clips=1,
            seed=seed,
            epoch=epoch,
            selection_mode="seeded_random",
            candidate_margin_frames=0,
        )[0].to_dict()
        return record, summary


class AccessLedger:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, *, split: str, video_id: str, operation: str) -> None:
        self.events.append({"split": split, "video_id": video_id, "operation": operation, "read_only": True})

    def statistics(self) -> dict[str, dict[str, int]]:
        result = {name: {"opens": 0, "decodes": 0} for name in ("train", "validation", "test")}
        for event in self.events:
            split = event["split"]
            result.setdefault(split, {"opens": 0, "decodes": 0})
            result[split]["opens"] += 1
            if event["operation"] == "decode_contiguous_rgb_clip":
                result[split]["decodes"] += 1
        return result


class SplitGuard:
    """Exact split allowlist; test is rejected before any path operation."""

    def __init__(self, split: str, ledger: AccessLedger):
        need(split in ALLOWED_SPLITS, "Test/unknown split rejected before path access")
        rows, denied, scope = gate1a.load_scope([split])
        self.split = split
        self.allowed = {row["video_id"]: row for row in rows}
        self.denied = denied
        self.scope = scope
        self.ledger = ledger

    def authorize(self, video_id: str, video_path: str | None = None) -> Path:
        need(video_id in self.allowed, "ID outside split allowlist rejected before path access")
        row = self.allowed[video_id]
        need(video_path is None or video_path == row["video_path"],
             "Path/ID mismatch rejected before path access")
        need(row["split"] == self.split, "Cross-split row rejected before path access")
        return gate1a.authorize(row, self.allowed, self.denied)

    def identity(self, video_id: str) -> dict[str, Any]:
        path = self.authorize(video_id)
        # gate1a.open_guarded repeats component/symlink and link-count checks.
        info = path.stat()
        need(info.st_nlink == 1, "Hardlinked source rejected")
        return {
            "video_id": video_id,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
            "device": info.st_dev,
            "inode": info.st_ino,
            "manifest_sha256": self.scope["input_hashes"]["manifest"],
        }

    def open(self, video_id: str, video_path: str, operation: str) -> int:
        path = self.authorize(video_id, video_path)
        before = self.identity(video_id)
        fd = gate1a.open_guarded(path)
        try:
            after = os.fstat(fd)
            need((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
                 (before["device"], before["inode"], before["size"], before["mtime_ns"]),
                 "Source identity changed during open")
        except BaseException:
            os.close(fd)
            raise
        self.ledger.record(split=self.split, video_id=video_id, operation=operation)
        return fd


def validate_decoded_clip(clip: np.ndarray, decoded_indices: Sequence[int], selection: Mapping[str, Any]) -> None:
    need(isinstance(clip, np.ndarray) and clip.dtype == np.uint8,
         "Decoder must return uint8 RGB ndarray")
    need(clip.ndim == 4 and clip.shape[0] == 64 and clip.shape[-1] == 3 and min(clip.shape[1:3]) >= 2,
         "Decoder returned wrong frame count or shape")
    expected = list(range(int(selection["start_frame"]), int(selection["end_frame_exclusive"])))
    need(list(decoded_indices) == expected and len(expected) == 64,
         "Decoded frame indices differ from sampler half-open interval")
    need(clip.flags.c_contiguous, "Decoder output must be contiguous")


class SequentialRGBDecoder:
    """One-open sequential decoder; no compressed-video seek is used."""

    def __call__(
        self,
        row: Mapping[str, str],
        guard: SplitGuard,
        *,
        selection: dict[str, Any] | None,
        expected_frame_count: int | None,
        expected_fps_rational: str | None,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], int, float]:
        video_id = row["video_id"]
        before = guard.identity(video_id)
        fd = guard.open(video_id, row["video_path"], "decode_contiguous_rgb_clip")
        cap = None
        frames: list[np.ndarray] = []
        decoded_indices: list[int] = []
        timestamps: list[float] = []
        try:
            cap = cv2.VideoCapture(f"/proc/self/fd/{fd}", cv2.CAP_FFMPEG)
            need(cap.isOpened(), "Decoder failed to open authorized descriptor")
            raw_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            need(math.isfinite(raw_count) and raw_count > 0 and abs(raw_count - round(raw_count)) < 1e-6,
                 "Invalid decoder frame count")
            frame_count = int(round(raw_count))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            need(math.isfinite(fps) and fps > 0, "Invalid decoder FPS")
            if expected_frame_count is not None:
                need(frame_count == expected_frame_count, "Decoder/sidecar frame count mismatch")
            if expected_fps_rational is not None:
                expected_fps = float(Fraction(expected_fps_rational))
                need(abs(fps - expected_fps) <= max(1e-6, expected_fps * 1e-6),
                     "Decoder/sidecar FPS mismatch")
            if selection is None:
                selection = select_validation_clip(frame_count)
            start, end = int(selection["start_frame"]), int(selection["end_frame_exclusive"])
            need(0 <= start < end <= frame_count and end - start == 64, "Invalid selected interval")
            for index in range(end):
                ok, bgr = cap.read()
                need(ok, f"Truncated video at frame {index}")
                need(isinstance(bgr, np.ndarray) and bgr.dtype == np.uint8 and
                     bgr.ndim == 3 and bgr.shape[-1] == 3, "Unexpected decoded frame")
                if index >= start:
                    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    decoded_indices.append(index)
                    timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
        finally:
            if cap is not None:
                cap.release()
            os.close(fd)
        need(guard.identity(video_id) == before, "Source identity changed during decode")
        need(len(frames) == 64, "Decoder returned a short clip")
        clip = np.ascontiguousarray(np.stack(frames), dtype=np.uint8)
        validate_decoded_clip(clip, decoded_indices, selection)
        expected_timestamps = np.arange(selection["start_frame"], selection["end_frame_exclusive"]) * 1000.0 / fps
        need(np.isfinite(timestamps).all() and
             float(np.max(np.abs(np.asarray(timestamps) - expected_timestamps))) < 500.0 / fps,
             "Frame timestamps disagree with retained indices")
        metadata = {
            "decoder": "opencv_CAP_FFMPEG_sequential_v1",
            "retained_frames": 64,
            "decoder_read_frames": int(selection["end_frame_exclusive"]),
            "prefix_frames": int(selection["start_frame"]),
            "decoded_frame_indices": decoded_indices,
            "timestamps_match_indices": True,
            "source_unchanged": True,
            "source_identity": before,
            "input_shape": list(clip.shape),
        }
        return clip, metadata, selection, frame_count, fps


Decoder = Callable[..., tuple[np.ndarray, dict[str, Any], dict[str, Any], int, float]]


class BasketballClipDataset(Dataset):
    """One-split PyTorch Dataset implementing the frozen Gate v1 contract."""

    def __init__(
        self,
        config: GateConfig,
        *,
        split: str,
        video_ids: Iterable[str] | None = None,
        epoch: int = 0,
        clip_index: int = 0,
        decoder: Decoder | None = None,
        ledger: AccessLedger | None = None,
    ) -> None:
        # This is deliberately the first operation: test is rejected before even
        # manifest/config path use by the dataset constructor.
        need(split in ALLOWED_SPLITS, "Test/unknown split rejected before path access")
        need(type(epoch) is int and epoch >= 0, "epoch must be a nonnegative integer")
        need(clip_index == 0, "Gate v1 supports only clip_index=0")
        self.config = config
        self.split = split
        self.epoch = epoch
        self.clip_index = clip_index
        self.ledger = ledger or AccessLedger()
        self.decoder = decoder or SequentialRGBDecoder()

        manifest_rows = _read_csv(config.paths["manifest"])
        need(len({row["video_id"] for row in manifest_rows}) == len(manifest_rows), "Duplicate manifest video_id")
        split_summary = _read_json(config.paths["split_summary"])
        rows = [row for row in manifest_rows if row["split"] == split and row["excluded"] == "0"]
        by_id = {row["video_id"]: row for row in rows}
        requested = list(by_id) if video_ids is None else list(video_ids)
        need(len(requested) == len(set(requested)), "Duplicate requested video_id")
        need(all(video_id in by_id for video_id in requested),
             "Unknown/cross-split/test video ID rejected before path access")
        self.rows = [by_id[video_id] for video_id in requested]
        for row in self.rows:
            need(split_summary["group_assignment"].get(row["source_group"]) == split,
                 "Manifest/split-summary mismatch")
            need(row["video_id"] == row["video_path"] and row["binary_label"] in {"0", "1"},
                 "Invalid manifest row")

        self.guard = SplitGuard(split, self.ledger)
        self.quality_index = TrainQualityIndex(config) if split == "train" else None
        self._transform_cfg = {key: value for key, value in config.raw["transform"].items() if key != "policy"}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sidecar_summary: dict[str, str] | None = None
        if self.split == "train":
            need(self.quality_index is not None, "Missing train quality index")
            selection, sidecar_summary = self.quality_index.selection(
                row["video_id"],
                seed=self.config.raw["train_sampling"]["global_seed"],
                epoch=self.epoch,
                clip_index=self.clip_index,
            )
            expected_frame_count = int(sidecar_summary["frame_count"])
            expected_fps_rational = sidecar_summary["fps_rational"]
        else:
            need(self.quality_index is None, "Validation must not load train quality sidecar")
            selection = None
            expected_frame_count = None
            expected_fps_rational = None

        clip, decoding, selection, frame_count, fps = self.decoder(
            row,
            self.guard,
            selection=selection,
            expected_frame_count=expected_frame_count,
            expected_fps_rational=expected_fps_rational,
        )
        need(type(frame_count) is int and frame_count >= 64, "Decoder returned invalid frame count")
        if expected_frame_count is not None:
            need(frame_count == expected_frame_count, "Decoder/sidecar frame count mismatch")
        need(0 <= int(selection["start_frame"]) < int(selection["end_frame_exclusive"]) <= frame_count,
             "Selection exceeds decoder frame count")
        validate_decoded_clip(clip, decoding["decoded_frame_indices"], selection)
        seed = derived_seed(
            self.config.raw["seed"], row["video_id"], self.epoch, self.clip_index, "reference"
        )
        array, transform_metadata = ClipTransform(self._transform_cfg, "reference")(
            clip,
            seed=seed,
            split=self.split,
            sample_metadata=selection,
        )
        array, normalization = normalize_tensor(array, mode="identity")
        need(array.shape == (3, 64, 192, 256) and array.dtype == np.float32 and
             array.flags.c_contiguous and np.isfinite(array).all() and
             float(array.min()) >= 0.0 and float(array.max()) <= 1.0,
             "Reference tensor contract violation")
        tensor = torch.from_numpy(array)
        need(tensor.dtype == torch.float32 and tensor.is_contiguous() and
             tuple(tensor.shape) == (3, 64, 192, 256), "PyTorch tensor contract violation")

        denominator = frame_count - 64
        normalized_start = 0.0 if denominator == 0 else selection["start_frame"] / denominator
        sidecar_identity = ({
            "used": True,
            "path": str(self.config.paths["sidecar_metadata"].relative_to(ROOT)),
            "sha256": self.config.raw["identities"]["sidecar_metadata_sha256"],
            "run_identity": self.config.raw["identities"]["sidecar_run_identity"],
        } if self.split == "train" else {
            "used": False, "path": None, "sha256": None, "run_identity": None,
        })
        result = {
            "tensor": tensor,
            "binary_label": int(row["binary_label"]),
            "video_id": row["video_id"],
            "split": self.split,
            "source_group": row["source_group"],
            "shot_type": row["original_class"][:-1],
            "fps": fps,
            "start_frame": int(selection["start_frame"]),
            "end_frame_exclusive": int(selection["end_frame_exclusive"]),
            "normalized_start": float(normalized_start),
            "candidate_overlap": selection["candidate_overlap"],
            "fallback": bool(selection["fallback"]),
            "fallback_reason": selection["fallback_reason"],
            "sampler_policy": selection["policy_version"],
            "sampler_version": selection["policy_version"],
            "transform_policy": "reference",
            "transform_version": TRANSFORM_VERSION,
            "normalization": normalization,
            "config_identity": {
                "version": self.config.raw["version"],
                "path": str(self.config.path.relative_to(ROOT)),
                "sha256": self.config.sha256,
            },
            "manifest_identity": {
                "path": str(self.config.paths["manifest"].relative_to(ROOT)),
                "sha256": self.config.raw["identities"]["manifest_sha256"],
            },
            "split_identity": {
                "path": str(self.config.paths["split_summary"].relative_to(ROOT)),
                "sha256": self.config.raw["identities"]["split_summary_sha256"],
            },
            "sidecar_identity": sidecar_identity,
            "selection": selection,
            "decoding": decoding,
            "transform_metadata": transform_metadata["transform"],
        }
        return result

