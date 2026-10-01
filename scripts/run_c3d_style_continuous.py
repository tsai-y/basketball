#!/usr/bin/env python3
"""Experiment C checks and future train/validation-only full training.

No test Dataset or test evaluation path exists here. Check, shape, smoke and
dry-run stages write no run artifacts. Training accepts an absent run directory
or an exact, verified metadata-only directory; started runs are protected.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_c.model import ContinuousModifiedC3D
from reference_c3d.data import C3DVideoDataset, ContinuousClipSampler


def path(value):
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def sha256(file_path):
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_worker_id):
    cv2.setNumThreads(0)
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def identity_collate(items):
    if len(items) != 1:
        raise ValueError("Experiment C uses one video per DataLoader item")
    return items[0]


def verify(config, config_path):
    if sha256(path(config["manifest_path"])) != config["expected_manifest_sha256"]:
        raise RuntimeError("Cleaned manifest hash differs from fixed split")
    if sha256(path(config["split_summary_path"])) != config["expected_split_summary_sha256"]:
        raise RuntimeError("Split summary hash differs from fixed split")
    reference = yaml.safe_load(path(config["reference_clip_config_path"]).read_text())
    keys = ("clip_length", "clip_stride", "tail_policy", "resize_height", "resize_width",
            "crop_size", "train_crop", "eval_crop", "train_horizontal_flip_probability",
            "pixel_preprocessing")
    differences = [key for key in keys if config[key] != reference[key]]
    if differences:
        raise RuntimeError(f"Experiment C preprocessing differs from B: {differences}")
    expected = {"clip_length": 16, "clip_stride": 8, "tail_policy": "discard_incomplete",
                "crop_size": 112, "clip_feature_dim": 256, "num_classes": 2,
                "aggregation": "mean_clip_features_then_l2",
                "objective": "video_level_cross_entropy", "video_batch_size": 1,
                "optimizer": "adamw", "lr_scheduler": "reduce_on_plateau",
                "mixed_precision": False, "checkpoint_criterion": "lowest_validation_loss"}
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"Unsupported {key}: {config[key]}")
    run_dir = path(config["run_dir"])
    if run_dir.is_symlink():
        raise RuntimeError("Experiment C run directory must not be a symlink")
    run_dir = run_dir.resolve()
    if run_dir != (ROOT / "runs/full_video_c3d_style_continuous_v1").resolve():
        raise ValueError("Experiment C must use its independent run directory")
    if config["early_stopping_min_delta"] != 0.0:
        raise ValueError("Strict minimum validation loss requires min_delta=0")
    return run_dir


METADATA_FILES = frozenset({"config.yaml", "dataset_manifest.csv",
                            "split_summary.json", "environment.json"})
TRAINING_FILES = frozenset({"best.pt", "last.pt", "training_log.json",
                            "training_status.json"})


def environment_metadata(device):
    """Reproducibility snapshot; source and package fields also validate migration."""
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "opencv": cv2.__version__, "numpy": np.__version__, "pyyaml": yaml.__version__,
            "model_source_sha256": sha256(ROOT / "baseline/model.py"),
            "data_source_sha256": sha256(ROOT / "reference_c3d/data.py"),
            "experiment_c_model_sha256": sha256(ROOT / "experiment_c/model.py"),
            "experiment_c_aggregate_sha256": sha256(ROOT / "experiment_c/aggregate.py")}


def inspect_run_directory(run_dir, config, config_path, device=None):
    """Read-only preflight. Never interpret a partial/started run as new."""
    if not run_dir.exists():
        return "absent"
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise RuntimeError(f"Run path is not a regular directory: {run_dir}")
    entries = {entry.name: entry for entry in run_dir.iterdir()}
    present_training = sorted(TRAINING_FILES.intersection(entries))
    if present_training:
        raise FileExistsError(f"Training run already started; refusing overwrite: {present_training}")
    if set(entries) != METADATA_FILES:
        raise FileExistsError(
            f"Run directory is neither empty/new nor exact metadata-only: {sorted(entries)}")
    if any(not entry.is_file() or entry.is_symlink() for entry in entries.values()):
        raise RuntimeError("Metadata-only run contains a non-regular file or symlink")
    expected_files = {"config.yaml": config_path,
                      "dataset_manifest.csv": path(config["manifest_path"]),
                      "split_summary.json": path(config["split_summary_path"])}
    for name, source in expected_files.items():
        if sha256(entries[name]) != sha256(source):
            raise RuntimeError(f"Metadata-only run differs from current {name}; refusing reuse")
    saved_environment = json.loads(entries["environment.json"].read_text())
    current_environment = environment_metadata(torch.device("cpu"))
    for name in current_environment.keys() - {"device", "gpu"}:
        if saved_environment.get(name) != current_environment[name]:
            raise RuntimeError(f"Metadata-only run environment/source differs: {name}")
    if set(saved_environment) != set(current_environment):
        raise RuntimeError("Metadata-only run environment fields differ")
    # The read-only check can run without GPU access. Formal training must
    # use the same device and GPU recorded by the pre-created metadata.
    if device is not None:
        expected_environment = environment_metadata(device)
        if saved_environment != expected_environment:
            raise RuntimeError("Metadata-only run device/GPU differs from training environment")
    return "metadata_only"


def preprocessing(config, split):
    training = split == "train"
    return {"resize_height": config["resize_height"],
            "resize_width": config["resize_width"], "crop_size": config["crop_size"],
            "crop_mode": config["train_crop"] if training else config["eval_crop"],
            "flip_probability": config["train_horizontal_flip_probability"] if training else 0.0,
            "pixel_preprocessing": config["pixel_preprocessing"]}


def dataset(config, split, video_ids=None):
    if split not in {"train", "validation"}:
        raise ValueError("Only train and validation are available in this script")
    return C3DVideoDataset(path(config["manifest_path"]), path(config["dataset_path"]),
                           split, ContinuousClipSampler(config["clip_length"],
                                                        config["clip_stride"],
                                                        config["tail_policy"]),
                           preprocessing(config, split), video_ids=video_ids)


def loader(config, split, epoch):
    workers = config["num_workers"]
    generator = torch.Generator().manual_seed(config["seed"] + 2 * epoch +
                                               (0 if split == "train" else 1))
    options = {"dataset": dataset(config, split), "batch_size": 1,
               "shuffle": split == "train", "num_workers": workers,
               "pin_memory": torch.cuda.is_available(), "collate_fn": identity_collate,
               "worker_init_fn": seed_worker, "generator": generator,
               "persistent_workers": False}
    if workers:
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def choose_device(config, stage):
    requested = config["device"]
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(requested)
    if selected.type == "cuda" and not torch.cuda.is_available():
        if stage == "train":
            raise RuntimeError("CUDA requested for full training but unavailable")
        return torch.device("cpu")
    return selected


def counts(model):
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters()
                                        if p.requires_grad)}


def clip_index_check():
    clips = ContinuousClipSampler(16, 8, "discard_incomplete").indices(40)
    assert len(clips) == 4
    assert [clip[0] for clip in clips] == [0, 8, 16, 24]
    assert [clip[-1] for clip in clips] == [15, 23, 31, 39]
    assert all(list(clip) == list(range(clip[0], clip[0] + 16)) for clip in clips)
    assert all(len(set(clips[i]) & set(clips[i + 1])) == 8 for i in range(3))
    return {"total_frames": 40, "clip_count": 4,
            "clip_ranges": [[clip[0], clip[-1]] for clip in clips], "overlap": 8}


def inspect_shape(model, device):
    model.eval()
    with torch.no_grad():
        clips = torch.zeros((2, 3, 16, 112, 112), device=device)
        feature_map = model.features(clips)
        pooled = model.pool(feature_map)
        features = model.extract_clip_features(clips)
        logits, descriptor = model.forward_video(clips, 2)
    shapes = {"input": list(clips.shape), "feature_map": list(feature_map.shape),
              "gap": list(pooled.shape), "clip_descriptor": list(features.shape),
              "video_descriptor": list(descriptor.shape), "logits": list(logits.shape)}
    assert shapes["clip_descriptor"] == [2, 256]
    assert shapes["video_descriptor"] == [256] and shapes["logits"] == [2]
    return shapes


def one_video(model, sample, device, chunk_size):
    clips = sample["clips"]
    if clips.ndim != 5 or tuple(clips.shape[1:]) != (3, 16, 112, 112):
        raise RuntimeError("Decoded clip tensor shape differs from C config")
    logits, descriptor = model.forward_video(clips.to(device), chunk_size)
    label = torch.tensor([sample["label"]], dtype=torch.long, device=device)
    loss = F.cross_entropy(logits.unsqueeze(0), label)
    if not (torch.isfinite(loss) and torch.isfinite(descriptor).all()):
        raise RuntimeError("Non-finite video-level loss or descriptor")
    return logits, descriptor, loss


def smoke(model, config, device):
    model.train()
    model.zero_grad(set_to_none=True)
    clips = torch.randn((2, 3, 16, 112, 112), device=device)
    logits, descriptor = model.forward_video(clips, config["clip_batch_size"])
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([1], device=device))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    valid = bool(torch.isfinite(loss).item() and
                 all(g is not None and torch.isfinite(g).all().item() for g in grads) and
                 any(g.abs().sum().item() > 0 for g in grads))
    if not valid:
        raise RuntimeError("Smoke loss/gradient check failed")
    return {"status": "PASS", "device": str(device), "index_check": clip_index_check(),
            "input_shape": list(clips.shape), "descriptor_shape": list(descriptor.shape),
            "logits_shape": list(logits.shape), "finite_loss": True,
            "finite_nonzero_gradient": True, **counts(model)}


def dry_run(model, config, device):
    # Only the first train and validation rows are decoded; no test Dataset exists.
    train_set, validation_set = dataset(config, "train"), dataset(config, "validation")
    train_item, validation_item = train_set[0], validation_set[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _, train_descriptor, train_loss = one_video(model, train_item, device,
                                                config["clip_batch_size"])
    train_loss.backward()
    grad = model.features[0].weight.grad
    if grad is None or not torch.isfinite(grad).all() or grad.abs().sum().item() == 0:
        raise RuntimeError("Dry-run backbone gradient missing/non-finite/zero")
    optimizer.step()
    model.eval()
    with torch.no_grad():
        validation_logits, _, validation_loss = one_video(
            model, validation_item, device, config["clip_batch_size"])
    return {"status": "PASS", "device": str(device),
            "train_video_id": train_item["video_id"],
            "train_total_frames": train_item["total_frames"],
            "train_clip_count": len(train_item["indices"]),
            "train_first_three_ranges": [[x[0], x[-1]] for x in train_item["indices"][:3]],
            "validation_video_id": validation_item["video_id"],
            "validation_total_frames": validation_item["total_frames"],
            "validation_clip_count": len(validation_item["indices"]),
            "validation_logits_shape": list(validation_logits.shape),
            "train_descriptor_shape": list(train_descriptor.shape),
            "train_loss_finite": bool(torch.isfinite(train_loss).item()),
            "validation_loss_finite": bool(torch.isfinite(validation_loss).item()),
            "optimizer_step": True, "test_accessed": False}


def epoch(model, data_loader, config, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    confusion = [[0, 0], [0, 0]]
    accumulation = config["gradient_accumulation_videos"]
    for index, sample in enumerate(data_loader):
        with torch.set_grad_enabled(training):
            logits, _, loss = one_video(model, sample, device, config["clip_batch_size"])
        loss_sum += float(loss.item())
        confusion[sample["label"]][int(logits.detach().argmax().item())] += 1
        if training:
            group_size = min(accumulation,
                             len(data_loader) - (index // accumulation) * accumulation)
            (loss / group_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(data_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config["gradient_clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    count = len(data_loader)
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"loss": loss_sum / count, "accuracy": (tn + tp) / count,
            "precision_make": precision, "recall_make": recall,
            "f1_make": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
            "confusion_matrix_rows_true_0_1": confusion, "count": count}


def atomic_json(file_path, value):
    temporary = file_path.with_name(file_path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, file_path)


def atomic_checkpoint(file_path, value):
    temporary = file_path.with_name(file_path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, file_path)


def train(model, config, config_path, run_dir, device):
    # This function is intentionally unreachable from check/smoke/dry-run.
    run_state = inspect_run_directory(run_dir, config, config_path, device=device)
    if run_state == "absent":
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.yaml").write_bytes(config_path.read_bytes())
        (run_dir / "dataset_manifest.csv").write_bytes(path(config["manifest_path"]).read_bytes())
        (run_dir / "split_summary.json").write_bytes(path(config["split_summary_path"]).read_bytes())
        atomic_json(run_dir / "environment.json", environment_metadata(device))
    # metadata_only: preserve all four existing snapshots byte-for-byte.
    # Exclusive creation also prevents two processes from starting the same
    # metadata-only run between preflight and the first checkpoint.
    with os.fdopen(os.open(run_dir / ".training_started", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                          0o644), "w") as marker:
        marker.write(f"pid={os.getpid()}\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config["lr_factor"], patience=config["lr_patience"])
    best_loss, best_epoch, stale, history = float("inf"), None, 0, []
    for number in range(1, config["max_epochs"] + 1):
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = epoch(model, loader(config, "train", number), config, device, optimizer)
        with torch.no_grad():
            validation_metrics = epoch(model, loader(config, "validation", number),
                                       config, device)
        scheduler.step(validation_metrics["loss"])
        improved = validation_metrics["loss"] < best_loss
        if improved:
            best_loss, best_epoch, stale = validation_metrics["loss"], number, 0
        else:
            stale += 1
        record = {"epoch": number, "train": train_metrics,
                  "validation": validation_metrics,
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "epoch_seconds": time.perf_counter() - start,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device)
                  if device.type == "cuda" else None}
        history.append(record)
        checkpoint = {"epoch": number, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "best_epoch": best_epoch, "best_validation_loss": best_loss,
                      "early_stopping_counter": stale, "history": history,
                      "random_state": {"python": random.getstate(),
                                       "numpy": np.random.get_state(),
                                       "torch": torch.get_rng_state(),
                                       "cuda": torch.cuda.get_rng_state_all()},
                      "config_sha256": sha256(config_path),
                      "manifest_sha256": config["expected_manifest_sha256"]}
        atomic_checkpoint(run_dir / "last.pt", checkpoint)
        if improved:
            atomic_checkpoint(run_dir / "best.pt", checkpoint)
        atomic_json(run_dir / "training_log.json", history)
        status = {"status": "EARLY_STOPPED" if stale >= config["early_stopping_patience"]
                  else "COMPLETE" if number == config["max_epochs"] else "IN_PROGRESS",
                  "epochs_completed": number, "best_epoch": best_epoch,
                  "best_validation_loss": best_loss,
                  "early_stopping_counter": stale, "test_evaluation_performed": False}
        atomic_json(run_dir / "training_status.json", status)
        print(json.dumps(record), flush=True)
        if stale >= config["early_stopping_patience"]:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/c3d_style_continuous.yaml")
    parser.add_argument("--stage", choices=("check", "shape", "smoke", "dry-run", "train"),
                        default="check")
    args = parser.parse_args()
    config_path = path(args.config)
    config = yaml.safe_load(config_path.read_text())
    run_dir = verify(config, config_path)
    run_state = inspect_run_directory(run_dir, config, config_path)
    if args.stage == "check":
        print(json.dumps({"status": "PASS", "manifest_and_split": "fixed SHA-256 matched",
                          "temporal_and_spatial_preprocessing": "matches Experiment B",
                          "run_directory_state": run_state,
                          "new_training_allowed": True, "test_accessed": False}, indent=2))
        return
    seed_all(config["seed"])
    device = choose_device(config, args.stage)
    model = ContinuousModifiedC3D().to(device)
    if args.stage == "shape":
        result = {"status": "PASS", "device": str(device), **counts(model),
                  "shapes": inspect_shape(model, device), "test_accessed": False}
    elif args.stage == "smoke":
        result = smoke(model, config, device)
        result["test_accessed"] = False
    elif args.stage == "dry-run":
        result = dry_run(model, config, device)
    else:
        train(model, config, config_path, run_dir, device)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
