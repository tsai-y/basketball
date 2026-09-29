#!/usr/bin/env python3
"""Run clip verification, C3D smoke, and tiny train-only descriptor gate.

This script has no full-train or test-evaluation stage and refuses Experiment A's run path.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference_c3d.aggregate import aggregate_fc6
from reference_c3d.data import C3DVideoDataset, ContinuousClipSampler
from reference_c3d.model import ReferenceC3D


def path(value):
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def save_json(destination, obj):
    destination.write_text(json.dumps(obj, indent=2) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_train_ids(config):
    dataset = C3DVideoDataset(path(config["manifest_path"]), path(config["dataset_path"]),
                              "train", ContinuousClipSampler(config["clip_length"],
                              config["clip_stride"], config["tail_policy"]),
                              preprocessing(config, "validation"))
    amount = config["small_subset_per_class"]
    selected = []
    for label in (0, 1):
        selected += [row["video_id"] for row in dataset.rows
                     if int(row["binary_label"]) == label][:amount]
    if len(selected) != 2 * amount:
        raise ValueError("Too few training videos of either label")
    return selected


def preprocessing(config, split):
    training = split == "train"
    return {"resize_height": config["resize_height"], "resize_width": config["resize_width"],
            "crop_size": config["crop_size"],
            "crop_mode": config["train_crop"] if training else config["eval_crop"],
            "flip_probability": config["train_horizontal_flip_probability"] if training else 0.0,
            "pixel_preprocessing": config["pixel_preprocessing"]}


def dataset_for(config, ids, split="validation"):
    # All gates use TRAIN manifest rows; validation-style preprocessing makes reruns stable.
    return C3DVideoDataset(path(config["manifest_path"]), path(config["dataset_path"]),
                           "train", ContinuousClipSampler(config["clip_length"],
                           config["clip_stride"], config["tail_policy"]),
                           preprocessing(config, split), set(ids))


def check_indices(sample, length, stride):
    indices = sample["indices"]
    expected = ContinuousClipSampler(length, stride).indices(sample["total_frames"])
    if indices != expected or not indices:
        raise AssertionError("Clip indices differ from deterministic sampler")
    for clip in indices:
        if len(clip) != length or any(b - a != 1 for a, b in zip(clip, clip[1:])):
            raise AssertionError("Nonconsecutive clip frames")
    if any(right[0] - left[0] != stride for left, right in zip(indices, indices[1:])):
        raise AssertionError("Unexpected clip overlap")
    return True


def verify(config, run_dir, selected):
    ds = dataset_for(config, selected)
    records = []
    for i in range(len(ds)):
        sample = ds[i]
        check_indices(sample, config["clip_length"], config["clip_stride"])
        clips = sample["clips"]
        if tuple(clips.shape[1:]) != (3, config["clip_length"], config["crop_size"], config["crop_size"]):
            raise AssertionError(f"Bad clip tensor shape {tuple(clips.shape)}")
        records.append({"video_id": sample["video_id"], "label": sample["label"],
                        "total_frames": sample["total_frames"], "clip_count": len(sample["indices"]),
                        "clip_shape": list(clips.shape),
                        "clip_bounds_zero_based_inclusive": [[clip[0], clip[-1]] for clip in sample["indices"]],
                        "first_two_full_indices_zero_based": [list(x) for x in sample["indices"][:2]],
                        "all_clips_contiguous": True, "overlap": config["clip_length"] - config["clip_stride"]})
    result = {"status": "PASS", "protocol_sha256": config["_protocol_sha256"],
              "tail_policy": config["tail_policy"], "samples": records}
    save_json(run_dir / "clip_verification.json", result)
    return result


def features_for_video(model, clips, device, batch_size):
    pieces = []
    for batch in clips.split(batch_size):
        pieces.append(model.extract_fc6(batch.to(device)))
    features = torch.cat(pieces)
    return features, aggregate_fc6(features)


def smoke(config, run_dir, selected, device):
    verification = run_dir / "clip_verification.json"
    if not verification.exists():
        raise RuntimeError("Run clip verification first")
    prior = json.loads(verification.read_text())
    if prior.get("status") != "PASS" or prior.get("protocol_sha256") != config["_protocol_sha256"]:
        raise RuntimeError("Clip verification does not match current protocol")
    sample = dataset_for(config, selected[:1])[0]
    model = ReferenceC3D(config["num_classes"], config["pool_ceil_mode"]).to(device)
    model.eval()
    clips = sample["clips"][:config["clip_batch_size"]].to(device)
    shapes = {}
    fc6 = model.extract_fc6(clips, shapes)
    descriptor = aggregate_fc6(fc6)
    if tuple(fc6.shape) != (len(clips), 4096) or tuple(descriptor.shape) != (4096,):
        raise AssertionError("fc6/descriptor shape mismatch")
    norm = float(torch.linalg.vector_norm(descriptor).item())
    if not 0.999 <= norm <= 1.001 or not torch.isfinite(fc6).all().item():
        raise AssertionError("Nonfinite or unnormalized feature")
    loss = fc6.mean()
    loss.backward()
    gradient_ok = all(p.grad is not None and torch.isfinite(p.grad).all().item()
                      for p in (model.conv1a.weight, model.fc6.weight))
    gradient_nonzero = bool(model.conv1a.weight.grad.abs().sum().item() > 0 and
                            model.fc6.weight.grad.abs().sum().item() > 0)
    if not gradient_ok or not gradient_nonzero:
        raise AssertionError("Missing, zero or nonfinite gradient")
    result = {"status": "PASS", "protocol_sha256": config["_protocol_sha256"],
              "device": str(device), "input_shape": list(clips.shape),
              "layer_shapes": shapes, "fc6_shape": list(fc6.shape),
              "descriptor_shape": list(descriptor.shape), "descriptor_l2_norm": norm,
              "loss": float(loss.item()), "gradients_finite_nonzero": True,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None}
    save_json(run_dir / "smoke.json", result)
    return result


def small_subset(config, run_dir, selected, device):
    smoke_record = run_dir / "smoke.json"
    if not smoke_record.exists():
        raise RuntimeError("Run smoke test first")
    prior = json.loads(smoke_record.read_text())
    if prior.get("status") != "PASS" or prior.get("protocol_sha256") != config["_protocol_sha256"]:
        raise RuntimeError("Smoke test does not match current protocol")
    ds = dataset_for(config, selected)
    model = ReferenceC3D(config["num_classes"], config["pool_ceil_mode"]).to(device).eval()
    descriptors, labels, records = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            check_indices(sample, config["clip_length"], config["clip_stride"])
            fc6, descriptor = features_for_video(model, sample["clips"], device,
                                                 config["clip_batch_size"])
            descriptors.append(descriptor)
            labels.append(sample["label"])
            records.append({"video_id": sample["video_id"], "label": sample["label"],
                            "clip_count": len(sample["indices"]), "fc6_shape": list(fc6.shape),
                            "descriptor_l2_norm": float(descriptor.norm().item())})
    x = torch.stack(descriptors).detach()
    y = torch.tensor(labels, device=device, dtype=torch.long)
    distance = float((x[0] - x[1]).norm().item())
    if not torch.isfinite(x).all().item() or distance <= 0:
        raise AssertionError("Descriptors are invalid or identical")
    if config["small_subset_feature_scaling"] != "train_subset_center_and_global_std":
        raise ValueError("Unknown small-subset diagnostic scaling")
    # Diagnostic only: train-set-only centering and one global scale make tiny
    # differences in random, L2-normalized features visible to the classifier.
    center = x.mean(dim=0, keepdim=True)
    scale = x.std(dim=0, unbiased=False).norm().clamp_min(1e-6)
    diagnostic_x = (x - center) / scale
    # PROJECT_ADAPTATION diagnostic only: linear binary head on FROZEN random C3D features.
    # It verifies the complete video-to-descriptor path and label/loss wiring, not C3D learning.
    head = nn.Linear(4096, config["num_classes"]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=config["small_subset_linear_lr"])
    history = []
    passed = False
    for step in range(1, config["small_subset_max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(diagnostic_x), y)
        if not torch.isfinite(loss).item():
            break
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0:
            with torch.no_grad():
                scores = head(diagnostic_x)
                accuracy = float((scores.argmax(1) == y).float().mean().item())
                eval_loss = float(F.cross_entropy(scores, y).item())
            history.append({"step": step, "accuracy": accuracy, "loss": eval_loss})
            if accuracy >= config["small_subset_target_accuracy"] and eval_loss < 0.05:
                passed = True
                break
    result = {"status": "PASS" if passed else "FAIL",
              "protocol_sha256": config["_protocol_sha256"], "train_only": True,
              "model_training": "frozen_random_c3d_plus_tiny_linear_head_diagnostic",
              "sample_count": len(ds), "samples": records, "descriptor_shape": list(x.shape),
              "first_two_descriptor_distance": distance,
              "diagnostic_feature_scaling": config["small_subset_feature_scaling"],
              "diagnostic_scale": float(scale.item()), "history": history,
              "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None}
    save_json(run_dir / "small_subset.json", result)
    if not passed:
        raise RuntimeError("Small-subset diagnostic failed; no full training is allowed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c3d_reference.yaml")
    parser.add_argument("--stage", choices=("verify", "smoke", "subset"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    fingerprint = hashlib.sha256()
    for source in (args.config, ROOT / "reference_c3d/data.py", ROOT / "reference_c3d/model.py",
                   ROOT / "reference_c3d/aggregate.py", Path(__file__)):
        fingerprint.update(source.read_bytes())
    config["_protocol_sha256"] = fingerprint.hexdigest()
    run_dir = path(config["run_dir"]).resolve()
    if run_dir == (ROOT / "runs/full_video_c3d_style_v1").resolve():
        raise RuntimeError("Experiment A run directory is protected")
    if not str(run_dir).startswith(str((ROOT / "runs").resolve()) + "/"):
        raise RuntimeError("Run directory must be under this project's runs/")
    set_seed(config["seed"])
    device = torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available()
                          else "cpu" if config["device"] == "auto" else config["device"])
    if args.stage == "smoke" and device.type != "cuda":
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    selected = select_train_ids(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_bytes(args.config.read_bytes())
    metadata = {"manifest_sha256": hashlib.sha256(path(config["manifest_path"]).read_bytes()).hexdigest(),
                "split_summary_sha256": hashlib.sha256(path(config["split_summary_path"]).read_bytes()).hexdigest(),
                "seed": config["seed"], "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "device_used": str(device)}
    save_json(run_dir / "environment.json", metadata)
    result = {"verify": lambda: verify(config, run_dir, selected),
              "smoke": lambda: smoke(config, run_dir, selected, device),
              "subset": lambda: small_subset(config, run_dir, selected, device)}[args.stage]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
