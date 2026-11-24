"""Read-only data/label contract helpers for MixNetBinary50Model CPU acceptance.

Rules enforced here:
- Targets are torch.long with Miss=0, Make=1.
- Loss is unweighted CrossEntropyLoss on raw [B, 2] logits (via aca_binary_v1).
- Reuses adjudicated revision metadata (diagnostic32_label_revision_v1);
  effective_class / class_id / binary_label are NEVER derived from filenames.
- Pixel-level verification uses ONLY saved RGB fixtures 04 (Miss) and 22 (Make)
  under /home/tsai/projects/basketball/outputs/01_current/input_diagnostic32_20261119_v1;
  NO video files are opened or decoded.
- Sample 15 (3p1/3p1_v150_003223_x264.mp4) is verified strictly from metadata
  without opening or visually re-adjudicating any video.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from development.aca_binary_v1.data import BinaryCollator
from development.aca_binary_v1.metrics import CLASSES as BINARY_CLASSES
from development.aca_binary_v1.metrics import binary_ce
from development.aca_eight_class_data_v1 import foundation as f
from development.aca_net_v1.data import (
    FRAMES,
    HEIGHT,
    WIDTH,
    normalize_rgb,
    uniform_indices,
)
from development.adjudicated_input_v1 import adapter as labels

DIAGNOSTIC32_DIR = labels.CURRENT / "input_diagnostic32_20261119_v1"
ALLOWED_PIXEL_FIXTURE_IDS = ("04", "22")
SAMPLE15_VIDEO_ID = labels.TARGET


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def binary_cross_entropy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Delegate raw-logit direct binary CE to development.aca_binary_v1.metrics."""
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.long:
        raise ValueError(
            f"Binary targets must be torch.long (Miss=0, Make=1), got {type(targets).__name__} "
            f"dtype={getattr(targets, 'dtype', None)}"
        )
    return binary_ce(logits, targets)


def verify_sample15_metadata(
    revision: labels.LabelRevision | None = None,
) -> dict[str, Any]:
    """Verify sample 15 effective label comes from adjudicated metadata, not filename."""
    rev = labels.LabelRevision() if revision is None else revision
    rec = rev.record(SAMPLE15_VIDEO_ID, "train")

    filename_prefix_class = SAMPLE15_VIDEO_ID.split("/", 1)[0]
    if filename_prefix_class != "3p1":
        raise ValueError(f"Unexpected sample 15 filename prefix: {filename_prefix_class}")
    if rec["original_class"] != "3p1" or rec["original_class_id"] != 3:
        raise ValueError(f"Unexpected sample 15 original provenance: {rec}")
    if rec["effective_class"] != "2p1" or rec["class_id"] != 1:
        raise ValueError(
            "Sample 15 effective class was overwritten or mismatched: "
            f"effective_class={rec['effective_class']} class_id={rec['class_id']}"
        )
    if rec["binary_label"] != 1:
        raise ValueError(
            f"Sample 15 binary_label must remain 1 (Make), got {rec['binary_label']}"
        )

    return {
        "video_id": rec["video_id"],
        "split": rec["split"],
        "source_group": rec["source_group"],
        "filename_prefix_class": filename_prefix_class,
        "original_class": rec["original_class"],
        "original_class_id": rec["original_class_id"],
        "effective_class": rec["effective_class"],
        "effective_class_id": rec["class_id"],
        "binary_label": rec["binary_label"],
        "binary_class_name": BINARY_CLASSES[rec["binary_label"]],
        "data_version": labels.VERSION,
        "manifest_sha256": labels.MANIFEST_SHA,
        "label_patch_sha256": labels.PATCH_SHA,
        "decision_sha256": labels.DECISION_SHA,
        "overwritten_by_filename": rec["effective_class"] == filename_prefix_class,
        "video_opened": False,
        "visual_adjudication_added": False,
    }


def load_saved_rgb_fixture_item(
    fixture_id: str,
    *,
    revision: labels.LabelRevision | None = None,
    diagnostic_dir: Path = DIAGNOSTIC32_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an AdjudicatedCollator/BinaryCollator item from saved RGB .npy only.

    Strictly restricted to fixture_id in ('04', '22'). Never opens or decodes any
    .mp4 video file.
    """
    if fixture_id not in ALLOWED_PIXEL_FIXTURE_IDS:
        raise ValueError(
            f"Fixture {fixture_id!r} forbidden; only {ALLOWED_PIXEL_FIXTURE_IDS} allowed"
        )
    rev = labels.LabelRevision() if revision is None else revision
    fixture_dir = Path(diagnostic_dir) / fixture_id
    evidence_path = fixture_dir / "evidence.json"
    rgb_path = fixture_dir / "source_frames_rgb.npy"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    actual_npy_sha = _file_sha256(rgb_path)
    if actual_npy_sha != evidence["frames_array_sha256"]:
        raise ValueError(
            f"Saved RGB fixture {fixture_id} SHA mismatch: "
            f"{actual_npy_sha} != {evidence['frames_array_sha256']}"
        )

    video_id = evidence["video_id"]
    ref = rev.record(video_id, "train")
    frames_rgb = np.load(rgb_path, mmap_mode="r")
    frame_count = int(frames_rgb.shape[0])
    if tuple(frames_rgb.shape) != tuple(evidence["source_shape"]):
        raise ValueError(f"Saved RGB shape mismatch for fixture {fixture_id}")
    if frames_rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB fixture, got {frames_rgb.dtype}")

    indices = uniform_indices(frame_count)
    if list(indices) != evidence["selected_indices"]:
        raise ValueError(f"Uniform indices mismatch for fixture {fixture_id}")

    resized = np.stack(
        [
            cv2.resize(
                np.asarray(frames_rgb[idx]),
                (WIDTH, HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            for idx in indices
        ]
    )
    if resized.shape != (FRAMES, HEIGHT, WIDTH, 3) or resized.dtype != np.uint8:
        raise ValueError(f"Unexpected resized RGB shape: {resized.shape}")

    tensor = normalize_rgb(resized)
    actual_tensor_sha = labels.tensor_sha(tensor)
    if actual_tensor_sha != evidence["tensor_sha256"]:
        raise ValueError(
            f"Normalized tensor SHA mismatch vs evidence.json for fixture {fixture_id}"
        )
    if actual_tensor_sha != ref["tensor_sha256"]:
        raise ValueError(
            f"Normalized tensor SHA mismatch vs revised manifest for fixture {fixture_id}"
        )

    metadata = {
        "frame_count": frame_count,
        "fps": evidence["fps"],
        "selected_indices": indices,
        "input_contract_sha256": f.CONTRACT_SHA256,
        "data_version": labels.VERSION,
        "label_revision_sha256": labels.MANIFEST_SHA,
        "label_patch_sha256": labels.PATCH_SHA,
        "source_sha256": ref["source_sha256"],
        "tensor_sha256": actual_tensor_sha,
        "source_size_bytes": ref["source_size_bytes"],
        "loaded_from_saved_rgb_fixture": str(rgb_path),
        "video_open_count": 0,
        "decoded_frames": 0,
    }
    item = {
        "tensor": tensor,
        "video_id": video_id,
        "split": ref["split"],
        "source_group": ref["source_group"],
        "original_class": ref["original_class"],
        "original_class_id": ref["original_class_id"],
        "effective_class": ref["effective_class"],
        "class_id": ref["class_id"],
        "binary_label": ref["binary_label"],
        "metadata": metadata,
    }
    provenance = {
        "fixture_id": fixture_id,
        "video_id": video_id,
        "split": ref["split"],
        "original_class": ref["original_class"],
        "effective_class": ref["effective_class"],
        "class_id": ref["class_id"],
        "binary_label": ref["binary_label"],
        "binary_class_name": BINARY_CLASSES[ref["binary_label"]],
        "rgb_npy_path": str(rgb_path),
        "rgb_npy_sha256": actual_npy_sha,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_sha256": actual_tensor_sha,
        "tensor_matches_evidence_sha": True,
        "tensor_matches_manifest_sha": True,
        "video_open_count": 0,
        "video_decode_count": 0,
    }
    return item, provenance


def collate_saved_binary_fixtures(
    fixture_ids: tuple[str, ...] = ALLOWED_PIXEL_FIXTURE_IDS,
    *,
    revision: labels.LabelRevision | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collate fixed saved-RGB fixtures (04=Miss, 22=Make) via real BinaryCollator."""
    rev = labels.LabelRevision() if revision is None else revision
    items = []
    provenances = []
    for fid in fixture_ids:
        item, prov = load_saved_rgb_fixture_item(fid, revision=rev)
        items.append(item)
        provenances.append(prov)
    collator = BinaryCollator()
    batch = collator(items)
    return batch, provenances
