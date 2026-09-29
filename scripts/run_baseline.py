#!/usr/bin/env python3
"""Run gated smoke, overfit, training, and evaluation for Experiment A."""

import argparse
import csv
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
import torchvision
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.data import BasketballVideoDataset, UniformBinSampler, seed_worker
from baseline.model import C3DStyle


ROOT = Path(__file__).resolve().parents[1]


def resolved(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # CUDA max_pool3d backward has no deterministic implementation in this torch build.
    # Keep C3D-style max pooling and surface a warning instead of silently changing the model.
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment(device):
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__, "pyyaml": yaml.__version__,
            "cuda_compiled": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_used": str(device), "platform": platform.platform()}


def make_dataset(config, split, augment=False, video_ids=None):
    return BasketballVideoDataset(
        resolved(ROOT, config["manifest_path"]), resolved(ROOT, config["dataset_path"]), split,
        UniformBinSampler(config["num_frames"], augment and config["train_temporal_jitter"]),
        config["image_size"], config["train_horizontal_flip_probability"] if augment else 0.0,
        video_ids)


def make_loader(dataset, config, shuffle, device, batch_size=None, workers=None):
    count = config["num_workers"] if workers is None else workers
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=batch_size or config["batch_size"], shuffle=shuffle,
                      num_workers=count, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=count > 0)


def make_optimizer(model, config):
    if config["optimizer"].lower() != "adamw":
        raise ValueError("Only the protocol's AdamW optimizer is implemented")
    return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                             weight_decay=config["weight_decay"])


def finite_gradients(model):
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    return bool(gradients) and all(g is not None and torch.isfinite(g).all().item() for g in gradients)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def gate(run_dir, stage, manifest_hash, protocol_hash):
    path = run_dir / f"{stage}.json"
    if not path.exists():
        raise RuntimeError(f"Required {stage} gate result missing")
    record = json.loads(path.read_text())
    if (record.get("status") != "PASS" or record.get("manifest_sha256") != manifest_hash
            or record.get("protocol_sha256") != protocol_hash):
        raise RuntimeError(f"Required {stage} gate did not pass for this manifest")


def smoke(config, run_dir, device, manifest_hash):
    setup_seed(config["seed"])
    train_rows = make_dataset(config, "train").rows
    selected = {next(row["video_id"] for row in train_rows if int(row["binary_label"]) == label)
                for label in (0, 1)}
    dataset = make_dataset(config, "train", video_ids=selected)
    loader = make_loader(dataset, config, False, device, config["smoke_batch_size"], workers=0)
    x, y, ids = next(iter(loader))
    expected = (config["smoke_batch_size"], 3, config["num_frames"],
                config["image_size"], config["image_size"])
    if tuple(x.shape) != expected or set(y.tolist()) != {0, 1}:
        raise AssertionError(f"Invalid input shape or labels: {x.shape}, {y.tolist()}")
    x, y = x.to(device), y.to(device)
    model = C3DStyle(config["num_classes"]).to(device)
    optimizer = make_optimizer(model, config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    if tuple(logits.shape) != (len(y), 2) or not torch.isfinite(loss).item() or not finite_gradients(model):
        raise AssertionError("Invalid logits, loss, or gradients")
    optimizer.step()
    result = {"status": "PASS", "manifest_sha256": manifest_hash,
              "protocol_sha256": config["_protocol_sha256"], "input_shape": list(x.shape),
              "output_shape": list(logits.shape), "labels": y.tolist(), "loss": float(loss.item()),
              "gradients_finite": True, "optimizer_step": True, "device": str(device),
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "video_ids": list(ids)}
    write_json(run_dir / "smoke.json", result)
    print(json.dumps(result, indent=2))


def overfit(config, run_dir, device, manifest_hash):
    gate(run_dir, "smoke", manifest_hash, config["_protocol_sha256"])
    setup_seed(config["seed"])
    all_train = make_dataset(config, "train")
    size = config["overfit_subset_size"]
    if size % 2:
        raise ValueError("overfit_subset_size must be even")
    ids = []
    for label in (0, 1):
        ids += [r["video_id"] for r in all_train.rows if int(r["binary_label"]) == label][:size // 2]
    if len(ids) != size:
        raise ValueError("Not enough train examples for overfit subset")
    dataset = make_dataset(config, "train", video_ids=set(ids))
    loader = make_loader(dataset, config, False, device, batch_size=size, workers=0)
    x, y, _ = next(iter(loader))
    x, y = x.to(device), y.to(device)
    model = C3DStyle(config["num_classes"]).to(device)
    optimizer = make_optimizer(model, config)
    history = []
    passed = False
    for step in range(1, config["overfit_steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite overfit loss")
        loss.backward()
        if not finite_gradients(model):
            raise RuntimeError("Nonfinite overfit gradients")
        optimizer.step()
        if step == 1 or step % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                accuracy = float((logits.argmax(1) == y).float().mean().item())
                eval_loss = float(torch.nn.functional.cross_entropy(logits, y).item())
            history.append({"step": step, "train_loss": float(loss.item()),
                            "subset_eval_loss": eval_loss, "subset_accuracy": accuracy})
            if accuracy >= config["overfit_target_accuracy"] and eval_loss < 0.05:
                passed = True
                break
    result = {"status": "PASS" if passed else "FAIL", "manifest_sha256": manifest_hash,
              "protocol_sha256": config["_protocol_sha256"],
              "subset_size": size, "video_ids": ids, "history": history,
              "device": str(device),
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    write_json(run_dir / "overfit.json", result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError("Small-subset overfit gate failed; full training is prohibited")


def epoch_pass(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    for x, y, _ in loader:
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


def train(config, run_dir, device, manifest_hash):
    gate(run_dir, "smoke", manifest_hash, config["_protocol_sha256"])
    gate(run_dir, "overfit", manifest_hash, config["_protocol_sha256"])
    setup_seed(config["seed"])
    train_loader = make_loader(make_dataset(config, "train", augment=True), config, True, device)
    val_loader = make_loader(make_dataset(config, "validation"), config, False, device)
    model = C3DStyle(config["num_classes"]).to(device)
    optimizer = make_optimizer(model, config)
    best = float("inf")
    records = []
    start = time.monotonic()
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = epoch_pass(model, train_loader, optimizer, device)
        val_metrics = epoch_pass(model, val_loader, None, device)
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                  "elapsed_seconds": round(time.monotonic() - start, 2)}
        records.append(record)
        write_json(run_dir / "training_log.json", records)
        print(json.dumps(record), flush=True)
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "validation_loss": best, "manifest_sha256": manifest_hash,
                        "seed": config["seed"], "config": config,
                        "protocol_sha256": config["_protocol_sha256"]},
                       resolved(ROOT, config["checkpoint_path"]))
    result = {"status": "COMPLETE", "best_epoch": min(records, key=lambda x: x["validation"]["loss"])["epoch"],
              "best_validation_loss": best, "manifest_sha256": manifest_hash,
              "epochs": len(records), "elapsed_seconds": round(time.monotonic() - start, 2),
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    write_json(run_dir / "training_status.json", result)
    print(json.dumps(result, indent=2))


def evaluate(config, run_dir, device, manifest_hash):
    status_path = run_dir / "training_status.json"
    if not status_path.exists() or json.loads(status_path.read_text()).get("status") != "COMPLETE":
        raise RuntimeError("Training is not complete; test evaluation is prohibited")
    checkpoint = torch.load(resolved(ROOT, config["checkpoint_path"]), map_location="cpu", weights_only=True)
    if (checkpoint["manifest_sha256"] != manifest_hash
            or checkpoint.get("protocol_sha256") != config["_protocol_sha256"]):
        raise RuntimeError("Checkpoint manifest or protocol hash mismatch")
    setup_seed(config["seed"])
    model = C3DStyle(config["num_classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = make_loader(make_dataset(config, "test"), config, False, device)
    rows = []
    confusion = [[0, 0], [0, 0]]
    with torch.no_grad():
        for x, y, ids in loader:
            probabilities = torch.softmax(model(x.to(device, non_blocking=True)), dim=1)[:, 1].cpu().tolist()
            for name, truth, probability in zip(ids, y.tolist(), probabilities):
                prediction = int(probability >= 0.5)
                confusion[truth][prediction] += 1
                rows.append({"video_id": name, "ground_truth": truth,
                             "prediction": prediction, "probability_make": probability})
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    safe = lambda a, b: a / b if b else 0.0
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    metrics = {"accuracy": safe(tp + tn, len(rows)), "precision_make": precision,
               "recall_make": recall, "f1_make": safe(2 * precision * recall, precision + recall),
               "confusion_matrix_rows_true_0_1": confusion, "positive_class": "Make=1",
               "n": len(rows), "checkpoint_epoch": checkpoint["epoch"],
               "manifest_sha256": manifest_hash}
    with (run_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(run_dir / "test_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--stage", required=True, choices=("smoke", "overfit", "train", "evaluate"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    fingerprint = hashlib.sha256()
    for path in (args.config, ROOT / "baseline" / "data.py", ROOT / "baseline" / "model.py",
                 Path(__file__)):
        fingerprint.update(path.read_bytes())
    config["_protocol_sha256"] = fingerprint.hexdigest()
    manifest = resolved(ROOT, config["manifest_path"])
    run_dir = resolved(ROOT, config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    shutil.copy2(args.config, run_dir / "config.yaml")
    shutil.copy2(manifest, run_dir / "dataset_manifest.csv")
    shutil.copy2(resolved(ROOT, config["split_summary_path"]), run_dir / "split_summary.json")
    write_json(run_dir / "environment.json", environment(device))
    manifest_hash = digest(manifest)
    {"smoke": smoke, "overfit": overfit, "train": train, "evaluate": evaluate}[args.stage](
        config, run_dir, device, manifest_hash)


if __name__ == "__main__":
    main()
