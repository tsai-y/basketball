#!/usr/bin/env python3
"""Experiment A2: warm-start A1 weights in a separate, validation-only run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.data import BasketballVideoDataset, UniformBinSampler, seed_worker
from baseline.model import C3DStyle


# The completed 15-epoch A2 checkpoint hashes this script before resume support.
ORIGINAL_A2_PROTOCOL_SHA256 = "b9971e9a40c9b7dffc35b449ca5f2c1a38a034a7a0923ecd8aa32ae366b4fa07"


def path_from_root(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def validate(config_path, config, resume=False):
    run_dir = path_from_root(config["run_dir"]).resolve()
    if run_dir != (ROOT / "runs/full_video_c3d_style_extended_v1").resolve():
        raise RuntimeError("A2 must use its separate run directory")
    if resume:
        if not run_dir.is_dir():
            raise RuntimeError("A2 resume requires the existing run directory")
    elif run_dir.exists():
        raise RuntimeError("A2 run directory already exists; refusing to overwrite it")
    expected = {
        "num_classes": 2, "num_frames": 16, "image_size": 96,
        "temporal_sampler": "uniform_bins", "train_temporal_jitter": True,
        "train_horizontal_flip_probability": 0.5, "optimizer": "adamw",
        "lr_scheduler": "none", "max_total_epochs": 15,
        "early_stopping_patience": 5, "early_stopping_min_delta": 0.0,
        "checkpoint_criterion": "lowest_validation_loss",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise RuntimeError(f"A2 protocol mismatch: {key}")
    for field, expected_hash in (
        ("manifest_path", config["expected_manifest_sha256"]),
        ("split_summary_path", config["expected_split_summary_sha256"]),
        ("a1_checkpoint_path", config["expected_a1_checkpoint_sha256"]),
    ):
        if sha256(path_from_root(config[field])) != expected_hash:
            raise RuntimeError(f"Source hash mismatch: {field}")
    if sha256(ROOT / "baseline/model.py") != config["expected_model_sha256"]:
        raise RuntimeError("A1 model source changed")
    if sha256(ROOT / "baseline/data.py") != config["expected_data_sha256"]:
        raise RuntimeError("A1 data source changed")
    checkpoint_path = path_from_root(config["a1_checkpoint_path"])
    if checkpoint_path.parent != (ROOT / "runs/full_video_c3d_style_v1"):
        raise RuntimeError("A1 checkpoint source must be the completed A1 run")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    a1_config = yaml.safe_load((checkpoint_path.parent / "config.yaml").read_text())
    if (checkpoint["epoch"] != 5
            or checkpoint["manifest_sha256"] != config["expected_manifest_sha256"]
            or {k: v for k, v in checkpoint["config"].items() if not k.startswith("_")} != a1_config):
        raise RuntimeError("A1 checkpoint metadata differs from its frozen config")
    for key in ("seed", "dataset_path", "manifest_path", "split_summary_path",
                "num_classes", "num_frames", "image_size", "temporal_sampler",
                "train_temporal_jitter", "train_horizontal_flip_probability",
                "batch_size", "num_workers", "optimizer", "learning_rate",
                "weight_decay"):
        if config[key] != a1_config[key]:
            raise RuntimeError(f"A2 changed A1's model/data/optimizer setting: {key}")
    model = C3DStyle(config["num_classes"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    fingerprint = hashlib.sha256()
    for source in (config_path, ROOT / "baseline/model.py", ROOT / "baseline/data.py",
                   Path(__file__)):
        fingerprint.update(source.read_bytes())
    return run_dir, checkpoint, fingerprint.hexdigest()


def load_resume_checkpoint(run_dir, config_path, config, protocol_hash,
                           max_epochs, early_stopping_patience):
    """Read-only integrity check before any A2 resume state is loaded on device."""
    if not (run_dir / "last.pt").is_file():
        raise RuntimeError("A2 resume requires last.pt")
    if (run_dir / "config.yaml").read_bytes() != config_path.read_bytes():
        raise RuntimeError("A2 run config differs; use resume-only overrides")
    for name, expected in (("dataset_manifest.csv", config["expected_manifest_sha256"]),
                           ("split_summary.json", config["expected_split_summary_sha256"])):
        if sha256(run_dir / name) != expected:
            raise RuntimeError(f"A2 run snapshot changed: {name}")
    environment = json.loads((run_dir / "environment.json").read_text())
    status = json.loads((run_dir / "training_status.json").read_text())
    if status.get("test_evaluation_performed") is not False:
        raise RuntimeError("A2 resume requires an untouched test set")
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu",
                            weights_only=False, mmap=True)
    saved_hash = checkpoint["protocol_sha256"]
    legacy_ok = (saved_hash == ORIGINAL_A2_PROTOCOL_SHA256
                 and environment.get("protocol_sha256") == saved_hash
                 and checkpoint["epoch"] == 15
                 and config["max_total_epochs"] == 15)
    if saved_hash != protocol_hash and not legacy_ok:
        raise RuntimeError("A2 resume code/config protocol hash differs")
    if checkpoint["a1_checkpoint_sha256"] != config["expected_a1_checkpoint_sha256"]:
        raise RuntimeError("A2 checkpoint has a different A1 initialization")
    required = {"state_dict", "optimizer_state", "rng_state", "epoch", "history",
                "best_epoch", "best_validation_loss", "stale_epochs"}
    if not required.issubset(checkpoint):
        raise RuntimeError(f"A2 last.pt lacks resume fields: {sorted(required - set(checkpoint))}")
    if set(checkpoint["rng_state"]) != {"python", "numpy", "torch", "cuda"}:
        raise RuntimeError("A2 last.pt lacks a complete random state")
    history = checkpoint["history"]
    epoch = checkpoint["epoch"]
    if (not isinstance(epoch, int) or epoch < 6
            or [record["epoch"] for record in history] != list(range(6, epoch + 1))
            or [record["a2_epoch"] for record in history] != list(range(1, len(history) + 1))):
        raise RuntimeError("A2 checkpoint epoch/history disagree")
    if json.loads((run_dir / "training_log.json").read_text()) != history:
        raise RuntimeError("A2 training log differs from last.pt history")
    if max_epochs <= epoch:
        raise RuntimeError("A2 checkpoint already reached the requested max epoch")
    if checkpoint["stale_epochs"] >= early_stopping_patience:
        raise RuntimeError("A2 checkpoint already met the requested early-stopping condition")
    if (checkpoint.get("target_max_epochs", max_epochs) != max_epochs
            or checkpoint.get("effective_early_stopping_patience",
                              early_stopping_patience) != early_stopping_patience):
        raise RuntimeError("A2 resume overrides differ from the prior resumed checkpoint")
    current_lr = checkpoint["optimizer_state"]["param_groups"][0]["lr"]
    if current_lr != config["learning_rate"]:
        raise RuntimeError("A2 checkpoint learning rate differs from the fixed protocol")
    if not checkpoint["optimizer_state"]["state"]:
        raise RuntimeError("A2 optimizer state is empty; true resume is impossible")
    best = torch.load(run_dir / "best.pt", map_location="cpu",
                      weights_only=False, mmap=True)
    if (best["epoch"] != checkpoint["best_epoch"]
            or best["validation_loss"] != checkpoint["best_validation_loss"]
            or best["protocol_sha256"] not in {saved_hash, ORIGINAL_A2_PROTOCOL_SHA256}):
        raise RuntimeError("A2 best.pt disagrees with last.pt best metadata")
    return checkpoint


def dataset(config, split, augment=False, video_ids=None):
    return BasketballVideoDataset(
        path_from_root(config["manifest_path"]), path_from_root(config["dataset_path"]),
        split, UniformBinSampler(config["num_frames"],
                                 augment and config["train_temporal_jitter"]),
        config["image_size"],
        config["train_horizontal_flip_probability"] if augment else 0.0,
        video_ids)


def loader(config, split, epoch, device, augment=False, video_ids=None,
           batch_size=None, num_workers=None):
    workers = config["num_workers"] if num_workers is None else num_workers
    generator = torch.Generator().manual_seed(config["seed"] + epoch)
    return DataLoader(dataset(config, split, augment, video_ids),
                      batch_size=batch_size or config["batch_size"],
                      shuffle=augment, num_workers=workers,
                      pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=workers > 0)


def make_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                             weight_decay=config["weight_decay"])


def epoch_pass(model, data_loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    for x, y, _ in data_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite train/validation loss")
        if training:
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.item()) * len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        count += len(y)
    return {"loss": loss_sum / count, "accuracy": correct / count, "count": count}


def dry_run(config, checkpoint, device, resume=False):
    seed_all(config["seed"])
    train_rows = dataset(config, "train").rows
    selected_train = {next(row["video_id"] for row in train_rows
                           if int(row["binary_label"]) == label) for label in (0, 1)}
    validation_rows = dataset(config, "validation").rows
    selected_validation = {next(row["video_id"] for row in validation_rows
                                if int(row["binary_label"]) == label) for label in (0, 1)}
    model = C3DStyle(config["num_classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = make_optimizer(model, config)
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng(checkpoint["rng_state"])
    epoch = checkpoint["epoch"] + 1 if resume else 6
    train_batch = next(iter(loader(config, "train", epoch, device, True,
                                   selected_train, 2, 0)))
    validation_batch = next(iter(loader(config, "validation", epoch, device, False,
                                        selected_validation, 2, 0)))
    x, y, _ = train_batch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x.to(device))
    loss = torch.nn.functional.cross_entropy(logits, y.to(device))
    loss.backward()
    gradients_finite = all(p.grad is not None and torch.isfinite(p.grad).all().item()
                           for p in model.parameters() if p.requires_grad)
    if not gradients_finite or not torch.isfinite(loss).item():
        raise RuntimeError("A2 train dry-run gradients/loss are invalid")
    optimizer.step()  # Ephemeral model only; no run files or A1 files are written.
    model.eval()
    with torch.no_grad():
        val_x, val_y, _ = validation_batch
        val_logits = model(val_x.to(device))
        val_loss = torch.nn.functional.cross_entropy(val_logits, val_y.to(device))
    result = {"status": "PASS", "train_split": "train",
              "validation_split": "validation", "train_shape": list(x.shape),
              "resume": resume, "next_epoch": epoch,
              "optimizer_lr": optimizer.param_groups[0]["lr"],
              "validation_shape": list(val_x.shape),
              "output_shape": list(logits.shape), "train_labels": y.tolist(),
              "validation_labels": val_y.tolist(),
              "train_loss": float(loss.item()), "validation_loss": float(val_loss.item()),
              "gradients_finite": gradients_finite,
              "device": str(device), "test_accessed": False, "run_files_written": False}
    if not torch.isfinite(val_loss).item() or tuple(logits.shape) != (2, 2):
        raise RuntimeError("A2 validation dry-run logits/loss are invalid")
    print(json.dumps(result, indent=2))


def train(config_path, config, run_dir, source, checkpoint, protocol_hash, device,
          max_epochs, early_stopping_patience):
    if checkpoint is None:
        # Exclusive creation protects A1 and prevents accidentally restarting A2.
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_dir / "config.yaml")
        shutil.copy2(path_from_root(config["manifest_path"]), run_dir / "dataset_manifest.csv")
        shutil.copy2(path_from_root(config["split_summary_path"]), run_dir / "split_summary.json")
        save_json(run_dir / "environment.json",
                  {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "opencv": cv2.__version__,
                   "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "device": str(device), "seed": config["seed"],
                   "a1_checkpoint_sha256": config["expected_a1_checkpoint_sha256"],
                   "protocol_sha256": protocol_hash})
    seed_all(config["seed"])
    model = C3DStyle(config["num_classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"] if checkpoint else source["state_dict"])
    optimizer = make_optimizer(model, config)
    if checkpoint is None:
        best_loss, best_epoch = float(source["validation_loss"]), int(source["epoch"])
        best_state = {"state_dict": model.state_dict(), "epoch": best_epoch,
                      "validation_loss": best_loss, "origin": "A1_warm_start",
                      "a1_checkpoint_sha256": config["expected_a1_checkpoint_sha256"],
                      "manifest_sha256": config["expected_manifest_sha256"],
                      "protocol_sha256": protocol_hash}
        save_checkpoint(run_dir / "best.pt", best_state)
        records, stale_epochs, start_epoch = [], 0, best_epoch + 1
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng(checkpoint["rng_state"])
        best_loss = checkpoint["best_validation_loss"]
        best_epoch = checkpoint["best_epoch"]
        records = checkpoint["history"]
        stale_epochs = checkpoint["stale_epochs"]
        start_epoch = checkpoint["epoch"] + 1
    started = time.monotonic()
    for epoch in range(start_epoch, max_epochs + 1):
        train_metrics = epoch_pass(model, loader(config, "train", epoch, device, True),
                                   optimizer, device)
        val_metrics = epoch_pass(model, loader(config, "validation", epoch, device),
                                 None, device)
        improved = val_metrics["loss"] < best_loss - config["early_stopping_min_delta"]
        if improved:
            best_loss, best_epoch, stale_epochs = val_metrics["loss"], epoch, 0
        else:
            stale_epochs += 1
        record = {"epoch": epoch, "a2_epoch": epoch - source["epoch"],
                  "train": train_metrics, "validation": val_metrics,
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "elapsed_seconds": round(time.monotonic() - started, 2)}
        records.append(record)
        state = {"state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                 "rng_state": rng_state(), "epoch": epoch,
                 "best_epoch": best_epoch, "best_validation_loss": best_loss,
                 "stale_epochs": stale_epochs, "history": records,
                 "protocol_sha256": protocol_hash,
                 "target_max_epochs": max_epochs,
                 "effective_early_stopping_patience": early_stopping_patience,
                 "a1_checkpoint_sha256": config["expected_a1_checkpoint_sha256"]}
        if improved:
            save_checkpoint(run_dir / "best.pt",
                            {"state_dict": model.state_dict(), "epoch": epoch,
                             "validation_loss": best_loss, "origin": "A2_validation_best",
                             "a1_checkpoint_sha256": config["expected_a1_checkpoint_sha256"],
                             "manifest_sha256": config["expected_manifest_sha256"],
                             "protocol_sha256": protocol_hash})
        save_checkpoint(run_dir / "last.pt", state)
        save_json(run_dir / "training_log.json", records)
        print(json.dumps(record), flush=True)
        if stale_epochs >= early_stopping_patience:
            break
    save_json(run_dir / "training_status.json",
              {"status": "COMPLETE", "epochs_completed_in_a2": len(records),
               "last_epoch": records[-1]["epoch"], "best_epoch": best_epoch,
               "best_validation_loss": best_loss,
               "target_max_epochs": max_epochs,
               "effective_early_stopping_patience": early_stopping_patience,
               "a1_checkpoint_sha256": config["expected_a1_checkpoint_sha256"],
               "protocol_sha256": protocol_hash, "test_evaluation_performed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c3d_style_extended.yaml")
    parser.add_argument("--stage", choices=("check", "dry-run", "train"))
    parser.add_argument("--check", action="store_true", help="Read-only resume/config preflight")
    parser.add_argument("--resume", action="store_true", help="Resume A2 from last.pt")
    parser.add_argument("--max-epochs", type=int, help="Resume-only total epoch limit")
    parser.add_argument("--early-stopping-patience", type=int,
                        help="Resume-only patience increase; retain the saved stale counter")
    args = parser.parse_args()
    if args.check and args.stage is not None:
        parser.error("Use --check or --stage, not both")
    stage = "check" if args.check else args.stage
    if stage is None:
        parser.error("Choose --check or --stage")
    if (args.max_epochs is not None or args.early_stopping_patience is not None) and not args.resume:
        parser.error("Epoch/patience overrides require --resume")
    config = yaml.safe_load(args.config.read_text())
    max_epochs = args.max_epochs if args.max_epochs is not None else config["max_total_epochs"]
    early_stopping_patience = (args.early_stopping_patience
                               if args.early_stopping_patience is not None
                               else config["early_stopping_patience"])
    if args.max_epochs is not None and args.max_epochs <= config["max_total_epochs"]:
        parser.error("--max-epochs must exceed the original config limit")
    if (args.early_stopping_patience is not None
            and args.early_stopping_patience <= config["early_stopping_patience"]):
        parser.error("--early-stopping-patience must exceed the original config value")
    run_dir, source, protocol_hash = validate(args.config, config, args.resume)
    checkpoint = (load_resume_checkpoint(run_dir, args.config, config, protocol_hash,
                                         max_epochs, early_stopping_patience)
                  if args.resume else None)
    if stage == "check":
        print(json.dumps({"status": "PASS", "resume_classification":
                          "TRUE_RESUME" if args.resume else "WARM_START_ONLY",
                          "current_epoch": checkpoint["epoch"] if checkpoint else source["epoch"],
                          "next_epoch": checkpoint["epoch"] + 1 if checkpoint else source["epoch"] + 1,
                          "best_epoch": checkpoint["best_epoch"] if checkpoint else source["epoch"],
                          "best_validation_loss": (checkpoint["best_validation_loss"]
                                                   if checkpoint else source["validation_loss"]),
                          "current_learning_rate": (checkpoint["optimizer_state"]["param_groups"][0]["lr"]
                                                    if checkpoint else config["learning_rate"]),
                          "history_count": len(checkpoint["history"]) if checkpoint else 0,
                          "stale_epochs": checkpoint["stale_epochs"] if checkpoint else 0,
                          "max_epochs": max_epochs,
                          "early_stopping_patience": early_stopping_patience,
                          "a2_run_dir": str(run_dir.relative_to(ROOT)),
                          "protocol_sha256": protocol_hash,
                          "test_accessed": False, "run_files_written": False}, indent=2))
        return
    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    if stage == "dry-run":
        dry_run(config, checkpoint if checkpoint else source, device, args.resume)
    else:
        train(args.config, config, run_dir, source, checkpoint, protocol_hash,
              device, max_epochs, early_stopping_patience)


if __name__ == "__main__":
    main()
