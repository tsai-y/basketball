"""Run the original mainline RGB dataset on CPU, without loading a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

SOURCE = Path("/home/tsai/projects/basketball")
FOLDER = Path(__file__).resolve().parent
REVIEW = FOLDER.parent / "aca_review_v1/artifacts/review24_ckaqp3hd/review_template.json"
# Original dataset's guarded source root must be the original checkout, not
# the reproduction worktree (whose archive directory is deliberately absent).
sys.path.insert(0, str(SOURCE))

import cv2
import numpy as np
import torch
from development.data_pipeline.multiwindow_v1 import BasketballThreeWindowDataset
from development.data_pipeline.pipeline import load_gate_config


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def export(review_id, output):
    require(not torch.cuda.is_initialized(), "CPU only")
    cases = json.loads(REVIEW.read_text())["cases"]
    found = [row for row in cases if row["review_id"] == review_id]
    require(len(found) == 1 and found[0]["split"] == "validation", "Unknown/non-validation review ID")
    case = found[0]
    require(output.is_dir() and output.parent.parent == FOLDER / "artifacts" and output.name == review_id,
            "Output must be a new per-case frame-review folder")
    config_path = SOURCE / "development/partial_finetune_v1/artifacts/rgb_seed20261024_bkegdbsu/config.json"
    training = json.loads(config_path.read_text())
    gate = load_gate_config()
    require(gate.sha256 == training["gate_config_sha256"], "Mainline Gate config changed")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    dataset = BasketballThreeWindowDataset(gate, split="validation", video_ids=[case["video_id"]], epoch=0)
    item = dataset[0]
    require(item["video_id"] == case["video_id"] and item["split"] == "validation" and
            not item["quality_aware_selection_used"] and item["binary_label"] == case["label"], "Input identity mismatch")
    unit = item["tensor"].numpy().transpose(0, 2, 3, 4, 1)
    require(unit.shape == (3, 64, 192, 256, 3) and unit.dtype == np.float32 and np.isfinite(unit).all(),
            "Unexpected original mainline tensor")
    rgb = np.rint(unit * 255).astype(np.uint8)
    require(np.array_equal(unit, rgb.astype(np.float32) / 255), "Mainline display RGB is not losslessly reversible")
    with (output / "mainline_rgb.npz").open("xb") as stream:
        np.savez_compressed(stream, rgb=rgb)
    access = dataset.base.ledger.statistics()
    require(access["validation"] == {"opens": 3, "decodes": 3} and access["train"]["opens"] == 0 and
            access["test"]["opens"] == 0 and not torch.cuda.is_initialized(), "Unexpected access/GPU usage")
    write_new(output / "mainline_dataset.json", {"video_id": item["video_id"], "split": "validation",
              "frame_count": item["frame_count"], "fps": item["fps"], "windows": item["windows"],
              "gate_config_sha256": gate.sha256, "actual_unit_tensor_sha256": hashlib.sha256(unit.tobytes()).hexdigest(),
              "display_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
              "display_rgb_reconstructs_actual_unit_tensor_exactly": True, "access": access,
              "model_instantiated": False, "optimizer_updates": 0, "test_videos_opened": 0,
              "model_normalization": {"mean": training["initialization"]["mean"], "std": training["initialization"]["std"]}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.review_id, args.output.absolute())
