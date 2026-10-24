"""Controlled eight-class balanced-sampling experiment; unchanged temporal head."""

import csv
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from development.linear_baseline_v1.train_linear import evaluate as score_logits
from development.temporal_baseline_v1.cached_features import ROOT, load_sequences
from development.temporal_baseline_v1.model import WindowTemporalTransformer
from development.temporal_baseline_v1.train_temporal import (
    CACHE, CACHE_RUN_ID, EarlyStopping, evaluate, parse_args, require,
    save_checkpoint, train_one_epoch,
)
from development.temporal_balanced_v1.sampling import (
    CLASSES, aligned_groups, make_balanced_sampler,
)


FOLDER = Path(__file__).resolve().parent
MANIFEST = ROOT / "data/processed/dataset_manifest.csv"


def per_shot_metrics(logits, labels, groups):
    require(len(groups) == len(logits) == len(labels), "Diagnostic length mismatch")
    result = {}
    for shot in ("2p", "3p", "ft", "mp"):
        mask = torch.tensor([g[:-1] == shot for g in groups], dtype=torch.bool)
        if mask.any():
            result[shot], _ = score_logits(nn.Identity(), logits[mask, None], labels[mask])
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    print("Loading unchanged full train/validation sequence caches...", flush=True)
    train_x, train_y, train_ids, run_id = load_sequences(CACHE, split="train")
    val_x, val_y, val_ids, val_run_id = load_sequences(CACHE, split="validation")
    require(run_id == val_run_id == CACHE_RUN_ID, "Unexpected cache run identity")
    require(set(train_ids).isdisjoint(val_ids), "Overlapping split IDs")
    for name, x, y, ids in (("train", train_x, train_y, train_ids),
                            ("validation", val_x, val_y, val_ids)):
        require(tuple(x.shape) == (len(ids), 3, 8, 1536), f"Unexpected {name} shape")
        require(tuple(y.shape) == (len(ids),), f"Unexpected {name} labels")
        require(set(y.tolist()) == {0., 1.}, f"{name} must contain both labels")
        require(not x.requires_grad, "Features must stay frozen")

    protocol = torch.load(CACHE / "run.pt", map_location="cpu", weights_only=True)
    require(hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == run_id,
            "Protocol identity mismatch")
    manifest_sha = protocol["input_identities"]["manifest_sha256"]
    sampler, sampling_info = make_balanced_sampler(
        train_ids, train_y, manifest_path=MANIFEST, manifest_sha256=manifest_sha,
        seed=args.seed,
    )
    # Validation classes are used only for diagnostic reporting, never for sampling.
    val_groups = aligned_groups(val_ids, val_y, split="validation", manifest_path=MANIFEST,
                                manifest_sha256=manifest_sha)
    model = WindowTemporalTransformer().cpu()
    sources = [FOLDER / "sampling.py", FOLDER / "train_balanced.py"] + [
        ROOT / "development/temporal_baseline_v1" / n
        for n in ("model.py", "cached_features.py", "train_temporal.py")
    ] + [ROOT / "development/linear_baseline_v1/train_linear.py"]
    config = {
        "experiment": "frozen_mixnet_i3d_window_transformer_balanced_v1",
        "model": "WindowTemporalTransformer",
        "architecture": {
            "input_shape": [3, 8, 1536], "d_model": 256, "layers": 2,
            "heads": 4, "feedforward_dim": 512, "dropout": 0.1,
            "positions": "learned_within_window", "window_weights": "equal",
        },
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": "cpu", "cpu_threads": args.threads, "seed": args.seed,
        "epochs": args.epochs, "batch_size": 64, "eval_batch_size": 64,
        "optimizer": "AdamW", "lr": .0001, "weight_decay": .0001,
        "max_grad_norm": 1., "loss": "BCEWithLogitsLoss",
        "selection": "minimum_validation_loss", "threshold": .5,
        "early_stopping": {"monitor": "validation_loss", "patience": 10, "min_delta": .0001},
        "cache_root": str(CACHE), "cache_run_id": run_id,
        "backbone_trainable": False, "quality_weighting_used": False,
        "train_count": len(train_y), "validation_count": len(val_y),
        "torch_version": str(torch.__version__), "deterministic_algorithms": True,
        "precision": "float32", "resume_supported": False,
        "sampling": sampling_info,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    }
    if args.dry_run:
        model.eval()
        with torch.no_grad():
            for x in (train_x[:2], val_x[:2]):
                logits = model(x)
                require(tuple(logits.shape) == (len(x),), "Dry-run shape mismatch")
                require(torch.isfinite(logits).all().item(), "Non-finite dry-run output")
        print(json.dumps({"status": "PASS", "dry_run": True, "training_started": False,
                          "optimizer_steps": 0, "output_files_written": False,
                          "first_epoch_preview": sampler.preview(), "config": config}, indent=2),
              flush=True)
        return

    parent = FOLDER / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=f"balanced_seed{args.seed}_", dir=parent))
    with (output / "config.json").open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    print("Output:", output, flush=True)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64,
                        sampler=sampler, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    stopper = EarlyStopping(**{k: config["early_stopping"][k] for k in ("patience", "min_delta")})
    best_loss, stop_reason = float("inf"), "max_epochs"
    with (output / "history.csv").open("x", newline="") as history, \
         (output / "sampling_counts.csv").open("x", newline="") as audit:
        writer, draws = csv.writer(history), csv.writer(audit)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "seconds"])
        draws.writerow(["epoch", "total_draws", "unique_videos", *CLASSES])
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss = train_one_epoch(model, loader, optimizer, max_grad_norm=1.)
            require(sum(sampler.last_counts.values()) == len(train_ids), "Wrong epoch draw count")
            draws.writerow([epoch, len(train_ids), sampler.last_unique_count,
                            *(sampler.last_counts.get(g, 0) for g in CLASSES)])
            audit.flush()
            metrics, _ = evaluate(model, val_x, val_y, batch_size=64)
            seconds = time.perf_counter() - started
            writer.writerow([epoch, train_loss, metrics["loss"], metrics["accuracy"],
                             metrics["macro_f1"], seconds])
            history.flush()
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save_checkpoint({"model_state_dict": model.state_dict(), "epoch": epoch,
                                 "validation_loss": best_loss, "config": config}, output / "best.pt")
            print(f"epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={metrics['loss']:.4f}, "
                  f"val_macro_f1={metrics['macro_f1']:.4f}, seconds={seconds:.1f}", flush=True)
            if stopper.update(metrics["loss"]):
                stop_reason = "early_stopping"
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    metrics, logits = evaluate(model, val_x, val_y, batch_size=64)
    summary = {"experiment": config["experiment"], "seed": args.seed,
               "epochs_completed": epoch, "stop_reason": stop_reason,
               "best_epoch": best["epoch"], "cache_run_id": run_id,
               "validation": metrics,
               "validation_by_shot_type": per_shot_metrics(logits, val_y, val_groups)}
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    with (output / "validation_predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "true_label", "p_make", "predicted_label"])
        for video_id, label, logit in zip(val_ids, val_y, logits):
            writer.writerow([video_id, int(label), float(logit.sigmoid()), int(logit >= 0)])
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()
