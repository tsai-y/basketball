"""CPU MLP control: pooled features, matched training rules, no-update dry run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from development.linear_baseline_v1.cached_features import ROOT, load_vectors
from development.mlp_baseline_v1.model import PooledFeatureMLP
from development.temporal_baseline_v1.train_temporal import (
    CACHE, CACHE_RUN_ID, EarlyStopping, evaluate, require,
    save_checkpoint, train_one_epoch,
)


FOLDER = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Check full caches and forward passes without training or writing outputs")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.threads <= 0:
        parser.error("--epochs and --threads must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    return args


def build_config(args, model, run_id, train_count, validation_count):
    sources = [FOLDER / name for name in ("model.py", "train_mlp.py")]
    sources += [ROOT / path for path in (
        "development/linear_baseline_v1/cached_features.py",
        "development/linear_baseline_v1/train_linear.py",
        "development/temporal_baseline_v1/train_temporal.py",
    )]
    return {
        "experiment": "frozen_mixnet_i3d_pooled_mlp_v1",
        "model": "PooledFeatureMLP",
        "architecture": {
            "input_shape": [1536], "hidden_dims": [768, 352],
            "normalization": "LayerNorm", "activation": "GELU", "dropout": 0.1,
            "pooling": "equal_mean_over_windows_and_time_before_mlp",
        },
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "capacity_reference": {
            "model": "WindowTemporalTransformer", "parameters": 1450497,
            "interpretation": "approximately matched capacity, not identical computation or architecture",
        },
        "device": "cpu", "cpu_threads": args.threads,
        "seed": args.seed, "epochs": args.epochs,
        "batch_size": 64, "eval_batch_size": 64,
        "optimizer": "AdamW", "lr": 0.0001, "weight_decay": 0.0001,
        "max_grad_norm": 1.0, "loss": "BCEWithLogitsLoss",
        "selection": "minimum_validation_loss", "threshold": 0.5,
        "early_stopping": {"monitor": "validation_loss", "patience": 10, "min_delta": 0.0001},
        "cache_root": str(CACHE), "cache_run_id": run_id,
        "backbone_trainable": False, "quality_weighting_used": False,
        "train_count": train_count, "validation_count": validation_count,
        "torch_version": str(torch.__version__), "deterministic_algorithms": True,
        "precision": "float32", "resume_supported": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    print("Loading full pooled train/validation caches...", flush=True)
    train_x, train_y, train_ids, run_id = load_vectors(CACHE, split="train")
    val_x, val_y, val_ids, val_run_id = load_vectors(CACHE, split="validation")
    require(run_id == val_run_id == CACHE_RUN_ID, "Unexpected cache run identity")
    require(set(train_ids).isdisjoint(val_ids), "Overlapping train/validation IDs")
    for name, x, y, ids in (
        ("train", train_x, train_y, train_ids),
        ("validation", val_x, val_y, val_ids),
    ):
        require(tuple(x.shape) == (len(ids), 1536), f"Unexpected {name} shape")
        require(tuple(y.shape) == (len(ids),), f"Unexpected {name} labels")
        require(set(y.tolist()) == {0.0, 1.0}, f"{name} must contain both classes")
        require(not x.requires_grad, f"{name} features must stay frozen")

    model = PooledFeatureMLP().cpu()
    config = build_config(args, model, run_id, len(train_y), len(val_y))
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
    output = Path(mkdtemp(prefix=f"mlp_seed{args.seed}_", dir=parent))
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
