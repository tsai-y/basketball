#!/usr/bin/env python3
"""Proposed full Reference C3D adaptation training. Run only after protocol approval.

Uses the existing cleaned train/validation manifest; never reads the test split.
Video-level loss is applied after mean-fc6 and L2 normalization. Checkpoints
and logs go to a new run directory, with epoch-boundary resume support.
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference_c3d.aggregate import aggregate_fc6
from reference_c3d.data import C3DVideoDataset, ContinuousClipSampler
from reference_c3d.model import ReferenceC3D


# One-time compatibility for the already completed 10-epoch run. The original
# protocol hash included this script, so fixing resume changes that hash.
LEGACY_TEN_EPOCH_PROTOCOL_SHA256 = "793ce3bf97ff8cdddbff74ed8081ea7d8cb6a2dfecddf33455765b3af1b55b10"
# Hash of the completed 13-epoch run before the resume-only patience override.
THIRTEEN_EPOCH_PROTOCOL_SHA256 = "d327941d35d9a9832af8ecf90ee5f6fa9ba21899bc9a8e61e71de56b501e039a"
LEGACY_SOURCE_SHA256 = {
    "reference_c3d/data.py": "108993d2e14680649733eb59796f8d18b9d19fe7774bf7042aa9b53cb7529db6",
    "reference_c3d/model.py": "efe0585d8ffa8693cef575b7a5d426abbb25edfe9545aa3a2b01f892b9ddd3d9",
    "reference_c3d/aggregate.py": "ebbfe7cb1548c5e60842d5502fa3dd05e01682f49634e5321cccf8e9278c0640",
    "configs/c3d_reference.yaml": "8ac4000eb5f9a2b82bd1680affe66779912b0a9bd95e5cc6a33be01bddf93e82",
}


def root_path(value):
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    cv2.setNumThreads(0)
    value = torch.initial_seed() % (2 ** 32)
    random.seed(value)
    np.random.seed(value)


def identity_collate(batch):
    if len(batch) != 1:
        raise ValueError("Video batch size must be one")
    return batch[0]


def verify_config(config_path, config):
    if config["initialization"] != "scratch_official_fillers":
        raise ValueError("This protocol permits only documented from-scratch initialization")
    if config["trainable_scope"] != "full_fc6_path":
        raise ValueError("Unsupported trainable scope")
    if config["objective"] != "video_level_fc6_mean_l2_cross_entropy":
        raise ValueError("Unsupported objective")
    if config["classifier"] != "linear_4096_to_2" or config["video_batch_size"] != 1:
        raise ValueError("Only one-video linear-head training is implemented")
    if config["optimizer"] != "adamw" or config["lr_scheduler"] != "reduce_on_plateau":
        raise ValueError("Unsupported optimizer or scheduler")
    if config["mixed_precision"] is not False:
        raise ValueError("Mixed precision is not validated by this feasibility study")
    manifest = root_path(config["manifest_path"])
    split = root_path(config["split_summary_path"])
    if sha256(manifest) != config["expected_manifest_sha256"]:
        raise RuntimeError("Fixed cleaned manifest hash changed")
    if sha256(split) != config["expected_split_summary_sha256"]:
        raise RuntimeError("Fixed split hash changed")
    reference_path = root_path(config["reference_config_path"])
    reference = yaml.safe_load(reference_path.read_text())
    if (reference["clip_length"], reference["clip_stride"], reference["crop_size"],
        reference["video_aggregation"]) != (16, 8, 112, "mean_fc6_then_l2"):
        raise RuntimeError("Reference clip/aggregation settings changed")
    if (reference["pool_ceil_mode"] is not True or reference["tail_policy"] != "discard_incomplete"
            or reference["fc6_feature_point"] != "post_relu_pre_dropout"
            or reference["num_classes"] != 2):
        raise RuntimeError("Reference model or adaptation settings changed")
    run_dir = root_path(config["run_dir"]).resolve()
    if run_dir != (ROOT / "runs/full_video_c3d_reference_train_v1").resolve():
        raise RuntimeError("Training must use the separate approved run directory")
    fingerprint = hashlib.sha256()
    for source in (config_path, reference_path, ROOT / "reference_c3d/data.py",
                   ROOT / "reference_c3d/model.py", ROOT / "reference_c3d/aggregate.py",
                   Path(__file__)):
        fingerprint.update(source.read_bytes())
    return reference, run_dir, fingerprint.hexdigest()


def preprocessing(reference, split):
    training = split == "train"
    return {"resize_height": reference["resize_height"],
            "resize_width": reference["resize_width"], "crop_size": reference["crop_size"],
            "crop_mode": reference["train_crop"] if training else reference["eval_crop"],
            "flip_probability": reference["train_horizontal_flip_probability"] if training else 0.0,
            "pixel_preprocessing": reference["pixel_preprocessing"]}


def make_loader(config, reference, split, epoch):
    dataset = C3DVideoDataset(root_path(config["manifest_path"]),
                              root_path(config["dataset_path"]), split,
                              ContinuousClipSampler(reference["clip_length"],
                                                    reference["clip_stride"], reference["tail_policy"]),
                              preprocessing(reference, split))
    workers = config["num_workers"]
    generator = torch.Generator().manual_seed(config["seed"] + 2 * epoch + (0 if split == "train" else 1))
    options = {"dataset": dataset, "batch_size": 1, "shuffle": split == "train",
               "num_workers": workers, "pin_memory": True,
               "collate_fn": identity_collate, "worker_init_fn": seed_worker,
               "generator": generator, "persistent_workers": False}
    if workers:
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def video_logits(model, head, clips, clip_batch_size):
    pieces = []
    for chunk in clips.split(clip_batch_size):
        pieces.append(model.extract_fc6(chunk.cuda(non_blocking=True)))
    return head(aggregate_fc6(torch.cat(pieces)).unsqueeze(0))


def epoch_metrics(confusion, total_loss, count):
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    divide = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    return {"loss": total_loss / count, "accuracy": divide(tn + tp, count),
            "precision_make": precision, "recall_make": recall,
            "f1_make": divide(2 * precision * recall, precision + recall),
            "confusion_matrix_rows_true_0_1": confusion, "count": count}


def run_epoch(model, head, loader, optimizer, config):
    training = optimizer is not None
    model.train(training)
    head.train(training)
    confusion = [[0, 0], [0, 0]]
    total_loss = 0.0
    accumulation = config["gradient_accumulation_videos"]
    if training:
        optimizer.zero_grad(set_to_none=True)
    for index, sample in enumerate(loader):
        label = torch.tensor([sample["label"]], dtype=torch.long, device="cuda")
        with torch.set_grad_enabled(training):
            logits = video_logits(model, head, sample["clips"], config["clip_batch_size"])
            loss = F.cross_entropy(logits, label)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite loss for train/validation video index {index}")
        total_loss += float(loss.item())
        prediction = int(logits.detach().argmax(1).item())
        confusion[sample["label"]][prediction] += 1
        if training:
            group_size = min(accumulation, len(loader) - (index // accumulation) * accumulation)
            (loss / group_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad],
                    config["gradient_clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    return epoch_metrics(confusion, total_loss, len(loader))


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def load_resume_checkpoint(run_dir, config_path, config, protocol_hash, max_epochs,
                           early_stopping_patience):
    """Read-only preflight; never modify the existing run on failure."""
    if not run_dir.is_dir():
        raise RuntimeError("Resume requested but run directory does not exist")
    last_path = run_dir / "last.pt"
    if not last_path.is_file():
        raise RuntimeError("Resume requested but no last.pt checkpoint exists")
    if (run_dir / "config.yaml").read_bytes() != config_path.read_bytes():
        raise RuntimeError("Resume config differs from original run; use --max-epochs for extension")
    status = json.loads((run_dir / "training_status.json").read_text())
    if status.get("test_evaluation_performed") is not False:
        raise RuntimeError("Resume requires an untouched test set")
    for name, expected in (("dataset_manifest.csv", config["expected_manifest_sha256"]),
                           ("split_summary.json", config["expected_split_summary_sha256"])):
        if sha256(run_dir / name) != expected:
            raise RuntimeError(f"Run snapshot {name} differs from the fixed split")
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False, mmap=True)
    recorded_patience = checkpoint.get("effective_early_stopping_patience")
    if recorded_patience is not None and recorded_patience != early_stopping_patience:
        raise RuntimeError("Resume early-stopping patience differs from the checkpoint")
    saved_hash = checkpoint["protocol_sha256"]
    environment = json.loads((run_dir / "environment.json").read_text())
    legacy_compatible = (
        environment["protocol_sha256"] == LEGACY_TEN_EPOCH_PROTOCOL_SHA256
        and config["max_epochs"] == 10
        and all(sha256(ROOT / name) == expected
                for name, expected in LEGACY_SOURCE_SHA256.items())
    )
    if saved_hash != protocol_hash:
        known_run_hash = (saved_hash == LEGACY_TEN_EPOCH_PROTOCOL_SHA256
                          or (saved_hash == THIRTEEN_EPOCH_PROTOCOL_SHA256
                              and checkpoint["epoch"] == 13
                              and checkpoint.get("target_max_epochs") == 20))
        if not (known_run_hash and legacy_compatible):
            raise RuntimeError("Resume code/config protocol hash differs")
    epoch = checkpoint["epoch"]
    history = checkpoint["history"]
    if (not isinstance(epoch, int) or epoch < 1 or len(history) != epoch
            or [record["epoch"] for record in history] != list(range(1, epoch + 1))):
        raise RuntimeError("Checkpoint epoch and history disagree")
    if json.loads((run_dir / "training_log.json").read_text()) != history:
        raise RuntimeError("Existing training log differs from checkpoint history")
    if max_epochs <= epoch:
        raise RuntimeError("Checkpoint already reached max_epochs")
    if checkpoint["stale_epochs"] >= early_stopping_patience:
        raise RuntimeError("Checkpoint already met the early-stopping condition")
    if (checkpoint["best_epoch"] is None or checkpoint["best_epoch"] > epoch
            or checkpoint["best_epoch"] < 1):
        raise RuntimeError("Checkpoint best epoch is invalid")
    best_path = run_dir / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("Existing best.pt is missing")
    best = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
    if (best["epoch"] != checkpoint["best_epoch"]
            or best["best_loss"] != checkpoint["best_loss"]
            or (best["protocol_sha256"] != saved_hash
                and not (best["protocol_sha256"] == LEGACY_TEN_EPOCH_PROTOCOL_SHA256
                         and legacy_compatible))):
        raise RuntimeError("Existing best.pt differs from checkpoint best metadata")
    for key in ("model_state", "head_state", "optimizer_state", "scheduler_state",
                "rng_state", "stale_epochs", "best_loss", "best_epoch"):
        if key not in checkpoint:
            raise RuntimeError(f"Checkpoint lacks {key}")
    if set(checkpoint["rng_state"]) != {"python", "numpy", "torch", "cuda"}:
        raise RuntimeError("Checkpoint lacks a complete random state")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c3d_reference_train.yaml")
    parser.add_argument("--check", action="store_true", help="Validate protocol without starting training")
    parser.add_argument("--dry-run", action="store_true", help="One train and one validation video; no run files")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed epoch")
    parser.add_argument("--max-epochs", type=int, help="Resume-only extension of the original epoch limit")
    parser.add_argument("--early-stopping-patience", type=int,
                        help="Resume-only increase; keeps the checkpoint's existing stale-epoch counter")
    args = parser.parse_args()
    if (args.max_epochs is not None or args.early_stopping_patience is not None) and not args.resume:
        parser.error("Epoch-limit and early-stopping overrides require --resume")
    if args.resume and args.dry_run:
        parser.error("--resume cannot be combined with --dry-run")
    config = yaml.safe_load(args.config.read_text())
    reference, run_dir, protocol_hash = verify_config(args.config, config)
    max_epochs = args.max_epochs if args.max_epochs is not None else config["max_epochs"]
    early_stopping_patience = (args.early_stopping_patience
                               if args.early_stopping_patience is not None
                               else config["early_stopping_patience"])
    if args.max_epochs is not None and args.max_epochs <= config["max_epochs"]:
        parser.error("--max-epochs must exceed the original config max_epochs")
    if (args.early_stopping_patience is not None
            and args.early_stopping_patience <= config["early_stopping_patience"]):
        parser.error("--early-stopping-patience must exceed the original config value")
    checkpoint = None
    if args.resume:
        checkpoint = load_resume_checkpoint(run_dir, args.config, config, protocol_hash,
                                            max_epochs, early_stopping_patience)
    if args.check:
        print(json.dumps({"status": "RESUME_VALID" if args.resume else "CONFIG_VALID",
                          "run_dir": str(run_dir.relative_to(ROOT)),
                          "manifest_sha256": config["expected_manifest_sha256"],
                          "split_summary_sha256": config["expected_split_summary_sha256"],
                          "protocol_sha256": protocol_hash,
                          "current_epoch": checkpoint["epoch"] if checkpoint else None,
                          "next_epoch": checkpoint["epoch"] + 1 if checkpoint else None,
                          "max_epochs": max_epochs,
                          "original_early_stopping_patience": config["early_stopping_patience"],
                          "effective_early_stopping_patience": early_stopping_patience,
                          "current_learning_rate": checkpoint["optimizer_state"]["param_groups"][0]["lr"] if checkpoint else None,
                          "best_epoch": checkpoint["best_epoch"] if checkpoint else None,
                          "best_validation_loss": checkpoint["best_loss"] if checkpoint else None,
                          "early_stopping_counter": checkpoint["stale_epochs"] if checkpoint else None,
                          "run_files_written": False,
                          "test_split_accessed": False}, indent=2))
        return
    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This proposed full protocol requires CUDA")
    seed_all(config["seed"])
    if args.dry_run:
        model = ReferenceC3D(reference["num_classes"], reference["pool_ceil_mode"]).cuda()
        for layer in (model.fc7, model.fc8):
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
        head = nn.Linear(4096, 2).cuda()
        optimizer = torch.optim.AdamW(
            [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad],
            lr=config["learning_rate"], weight_decay=config["weight_decay"])
        train_sample = next(iter(make_loader(config, reference, "train", 1)))
        validation_sample = next(iter(make_loader(config, reference, "validation", 1)))
        train_metrics = run_epoch(model, head, [train_sample], optimizer, config)
        validation_metrics = run_epoch(model, head, [validation_sample], None, config)
        print(json.dumps({"status": "PASS", "train_video_clips": len(train_sample["indices"]),
                          "validation_video_clips": len(validation_sample["indices"]),
                          "train_loss_finite": bool(np.isfinite(train_metrics["loss"])),
                          "validation_loss_finite": bool(np.isfinite(validation_metrics["loss"])),
                          "test_split_accessed": False, "run_files_written": False}, indent=2))
        return
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if not args.resume:
        if run_dir.exists():
            raise RuntimeError("New training run directory already exists; use --resume or a new experiment")
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_bytes(args.config.read_bytes())
        (run_dir / "dataset_manifest.csv").write_bytes(root_path(config["manifest_path"]).read_bytes())
        (run_dir / "split_summary.json").write_bytes(root_path(config["split_summary_path"]).read_bytes())
        save_json(run_dir / "environment.json",
                  {"python": platform.python_version(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                   "numpy": np.__version__, "opencv": cv2.__version__,
                   "manifest_sha256": config["expected_manifest_sha256"],
                   "split_summary_sha256": config["expected_split_summary_sha256"],
                   "protocol_sha256": protocol_hash, "seed": config["seed"]})
    model = ReferenceC3D(reference["num_classes"], reference["pool_ceil_mode"]).cuda()
    for layer in (model.fc7, model.fc8):
        for parameter in layer.parameters():
            parameter.requires_grad_(False)  # fc6 video objective does not traverse them.
    head = nn.Linear(4096, 2).cuda()
    parameters = [p for p in list(model.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config["lr_factor"], patience=config["lr_patience"])
    best_loss = float("inf")
    best_epoch = None
    stale_epochs = 0
    history = []
    start_epoch = 1
    if args.resume:
        model.load_state_dict(checkpoint["model_state"])
        head.load_state_dict(checkpoint["head_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng(checkpoint["rng_state"])
        best_loss, best_epoch = checkpoint["best_loss"], checkpoint["best_epoch"]
        stale_epochs = checkpoint["stale_epochs"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
    for epoch in range(start_epoch, max_epochs + 1):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        train = run_epoch(model, head, make_loader(config, reference, "train", epoch), optimizer, config)
        validation = run_epoch(model, head, make_loader(config, reference, "validation", epoch),
                               None, config)
        scheduler.step(validation["loss"])
        improved = validation["loss"] < best_loss - config["early_stopping_min_delta"]
        if improved:
            best_loss = validation["loss"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        record = {"epoch": epoch, "train": train, "validation": validation,
                  "epoch_seconds": time.perf_counter() - start,
                  "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                  "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        state = {"epoch": epoch, "model_state": model.state_dict(),
                 "head_state": head.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(), "rng_state": rng_state(),
                 "best_loss": best_loss, "best_epoch": best_epoch,
                 "stale_epochs": stale_epochs,
                 "history": history, "protocol_sha256": protocol_hash,
                 "target_max_epochs": max_epochs,
                 "effective_early_stopping_patience": early_stopping_patience}
        if improved:
            save_checkpoint(best_path, state)
        save_checkpoint(last_path, state)
        save_json(run_dir / "training_log.json", history)
        print(json.dumps(record), flush=True)
        if stale_epochs >= early_stopping_patience:
            break
    save_json(run_dir / "training_status.json",
              {"status": "COMPLETE", "epochs_completed": len(history),
               "target_max_epochs": max_epochs,
               "effective_early_stopping_patience": early_stopping_patience,
               "best_validation_loss": best_loss,
               "best_epoch": best_epoch,
               "manifest_sha256": config["expected_manifest_sha256"],
               "protocol_sha256": protocol_hash, "test_evaluation_performed": False})


if __name__ == "__main__":
    main()
