"""Train a temporal head on fixed features; CPU-only, with a no-update dry run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp
import time
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from development.linear_baseline_v1.train_linear import EarlyStopping
from development.temporal_baseline_v1.cached_features import ROOT, load_sequences
from development.temporal_baseline_v1.model import WindowTemporalTransformer


FOLDER = Path(__file__).resolve().parent
CACHE = ROOT / "development/mixnet_i3d/artifacts/feature_cache_v1/67b6400e0982bb6d"
CACHE_RUN_ID = "67b6400e0982bb6d4be8f659d5067add9a0ba790bb456e00442601b1e563c4f6"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Check full inputs and forward passes; no updates or output files")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.threads <= 0:
        parser.error("--epochs and --threads must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    return args


@torch.no_grad()
def evaluate(model, x, y, *, batch_size=64):
    """Evaluate in batches, preserving sample order and global metric definitions."""
    if batch_size <= 0 or len(x) == 0 or len(x) != len(y):
        raise ValueError("Expected positive batch size and matching nonempty inputs")
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        logits = model(batch)
        require(tuple(logits.shape) == (len(batch),), "Unexpected logit shape")
        require(torch.isfinite(logits).all().item(), "Non-finite validation output")
        outputs.append(logits)
    logits = torch.cat(outputs)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    require(torch.isfinite(loss).item(), "Non-finite validation loss")
    predictions = (logits >= 0).long()
    cm = torch.bincount(2 * y.long() + predictions, minlength=4).reshape(2, 2)
    per_class = {}
    for label, name in enumerate(("Miss", "Make")):
        tp = int(cm[label, label])
        fp = int(cm[:, label].sum()) - tp
        fn = int(cm[label, :].sum()) - tp
        per_class[name] = {
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "support": int(cm[label, :].sum()),
        }
    return {
        "loss": float(loss),
        "accuracy": float(cm.diag().sum()) / len(y),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / 2,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }, logits


def train_one_epoch(model, loader, optimizer, *, max_grad_norm=1.0):
    model.train()
    loss_sum = 0.0
    seen = 0
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        require(tuple(logits.shape) == tuple(y.shape), "Training logit/label shape mismatch")
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        require(torch.isfinite(loss).item(), "Non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        loss_sum += loss.item() * len(y)
        seen += len(y)
    require(seen > 0, "Empty training loader")
    return loss_sum / seen


def save_checkpoint(payload, path):
    """Atomically replace only this run's previous best checkpoint."""
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.pt")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    print("Loading full train/validation sequence caches...", flush=True)
    train_x, train_y, train_ids, run_id = load_sequences(CACHE, split="train")
    val_x, val_y, val_ids, val_run_id = load_sequences(CACHE, split="validation")
    require(run_id == val_run_id == CACHE_RUN_ID, "Unexpected cache run identity")
    require(set(train_ids).isdisjoint(val_ids), "Overlapping train/validation IDs")
    for name, x, y, ids in (
        ("train", train_x, train_y, train_ids),
        ("validation", val_x, val_y, val_ids),
    ):
        require(tuple(x.shape) == (len(ids), 3, 8, 1536), f"Unexpected {name} shape")
        require(tuple(y.shape) == (len(ids),), f"Unexpected {name} labels")
        require(set(y.tolist()) == {0.0, 1.0}, f"{name} must contain both classes")
        require(not x.requires_grad, f"{name} features must stay frozen")

    model = WindowTemporalTransformer().cpu()
    sources = [
        FOLDER / name for name in ("model.py", "cached_features.py", "train_temporal.py")
    ] + [ROOT / "development/linear_baseline_v1/train_linear.py"]
    config = {
        "experiment": "frozen_mixnet_i3d_window_transformer_v1",
        "model": "WindowTemporalTransformer",
        "architecture": {
            "input_shape": [3, 8, 1536], "d_model": 256, "layers": 2,
            "heads": 4, "feedforward_dim": 512, "dropout": 0.1,
            "positions": "learned_within_window", "window_weights": "equal",
        },
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": "cpu", "cpu_threads": args.threads,
        "seed": args.seed, "epochs": args.epochs,
        "batch_size": 64, "eval_batch_size": 64,
        "optimizer": "AdamW", "lr": 0.0001, "weight_decay": 0.0001,
        "max_grad_norm": 1.0, "loss": "BCEWithLogitsLoss",
        "selection": "minimum_validation_loss", "threshold": 0.5,
        "early_stopping": {"monitor": "validation_loss", "patience": 10, "min_delta": 0.0001},
        "cache_root": str(CACHE), "cache_run_id": run_id,
        "backbone_trainable": False, "quality_weighting_used": False,
        "train_count": len(train_y), "validation_count": len(val_y),
        "torch_version": str(torch.__version__), "deterministic_algorithms": True,
        "precision": "float32", "resume_supported": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    if args.dry_run:
        model.eval()
        with torch.no_grad():
            for x in (train_x[:2], val_x[:2]):
                logits = model(x)
                require(tuple(logits.shape) == (len(x),), "Dry-run output shape mismatch")
                require(torch.isfinite(logits).all().item(), "Non-finite dry-run output")
        print(json.dumps({
            "status": "PASS", "dry_run": True, "training_started": False,
            "optimizer_steps": 0, "output_files_written": False,
            "train_shape": list(train_x.shape), "validation_shape": list(val_x.shape),
            "config": config,
        }, indent=2), flush=True)
        return

    parent = FOLDER / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=f"temporal_seed{args.seed}_", dir=parent))
    with (output / "config.json").open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    print("Output:", output, flush=True)

    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=config["batch_size"],
        shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    early_stopper = EarlyStopping(
        patience=config["early_stopping"]["patience"],
        min_delta=config["early_stopping"]["min_delta"],
    )
    best_loss = float("inf")
    stop_reason = "max_epochs"
    with (output / "history.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "seconds"
        ])
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss = train_one_epoch(
                model, loader, optimizer, max_grad_norm=config["max_grad_norm"]
            )
            metrics, _ = evaluate(model, val_x, val_y, batch_size=config["eval_batch_size"])
            elapsed = time.perf_counter() - started
            writer.writerow([
                epoch, train_loss, metrics["loss"], metrics["accuracy"],
                metrics["macro_f1"], elapsed,
            ])
            stream.flush()
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save_checkpoint({
                    "model_state_dict": model.state_dict(), "epoch": epoch,
                    "validation_loss": best_loss, "config": config,
                }, output / "best.pt")
            print(
                f"epoch {epoch:02d}: train_loss={train_loss:.4f}, "
                f"val_loss={metrics['loss']:.4f}, val_macro_f1={metrics['macro_f1']:.4f}, "
                f"seconds={elapsed:.1f}", flush=True,
            )
            if early_stopper.update(metrics["loss"]):
                stop_reason = "early_stopping"
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    metrics, logits = evaluate(model, val_x, val_y, batch_size=config["eval_batch_size"])
    summary = {
        "experiment": config["experiment"], "seed": args.seed,
        "epochs_completed": epoch, "stop_reason": stop_reason,
        "best_epoch": best["epoch"], "cache_run_id": run_id,
        "validation": metrics,
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    with (output / "validation_predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "true_label", "p_make", "predicted_label"])
        for video_id, label, logit in zip(val_ids, val_y, logits):
            writer.writerow([video_id, int(label), float(logit.sigmoid()), int(logit >= 0)])
    print(json.dumps(summary, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
