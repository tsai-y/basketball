"""Alpha=.25/.5 with three paired seeds, full-cache temporal-head experiment.

An explicit --train is mandatory. --dry-run validates all inputs and performs
forward/backward checks with NO optimizer, parameter update, or output files.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from development.temporal_baseline_v1.cached_features import ROOT, load_sequences
from development.temporal_baseline_v1.model import WindowTemporalTransformer
from development.temporal_baseline_v1.train_temporal import (
    CACHE, CACHE_RUN_ID, EarlyStopping, evaluate, require, save_checkpoint,
)
from development.temporal_balanced_v1.sampling import aligned_groups
from development.temporal_balanced_v1.train_balanced import per_shot_metrics
from development.temporal_quality_v1.weights import load_train_weights
from development.data_quality.scripts.simulate_quality_weights_v1 import ALLOWED_ALPHAS


FOLDER = Path(__file__).resolve().parent
BASELINES = {
    20261024: ROOT / "development/temporal_baseline_v1/artifacts/temporal_seed20261024_y7wv4h0p",
    20261025: ROOT / "development/temporal_baseline_v1/artifacts/temporal_seed20261025_cwac4gwz",
    20261026: ROOT / "development/temporal_baseline_v1/artifacts/temporal_seed20261026_59bm89qx",
}
MANIFEST = ROOT / "data/processed/dataset_manifest.csv"


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def alpha_tag(alpha):
    require(type(alpha) in (int, float) and alpha in ALLOWED_ALPHAS, "Unsupported alpha")
    return {0.25: "025", 0.5: "050"}[alpha]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--train", action="store_true")
    # Replicate the same alpha across predeclared seeds, not an alpha sweep.
    parser.add_argument("--epochs", type=int, choices=(50,), default=50)
    parser.add_argument("--seed", type=int, choices=tuple(BASELINES), default=20261024)
    parser.add_argument("--alpha", type=float, choices=ALLOWED_ALPHAS, default=.25)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.train and args.threads != 4:
        parser.error("Training uses 4 CPU threads, matching the recorded baseline")
    return args


def load_baseline_reference(seed):
    require(seed in BASELINES, "No predeclared paired baseline for this seed")
    baseline_path = BASELINES[seed]
    config = json.loads((baseline_path / "config.json").read_text())
    summary = json.loads((baseline_path / "summary.json").read_text())
    expected = {"experiment": "frozen_mixnet_i3d_window_transformer_v1",
                "seed": seed, "epochs": 50, "cpu_threads": 4,
                "batch_size": 64, "eval_batch_size": 64, "optimizer": "AdamW",
                "lr": .0001, "weight_decay": .0001, "max_grad_norm": 1.,
                "selection": "minimum_validation_loss", "threshold": .5,
                "cache_run_id": CACHE_RUN_ID, "quality_weighting_used": False,
                "backbone_trainable": False, "device": "cpu", "precision": "float32",
                "train_count": 7193, "validation_count": 1547,
                "trainable_parameters": 1450497, "deterministic_algorithms": True,
                "early_stopping": {"monitor": "validation_loss", "patience": 10, "min_delta": .0001}}
    for key, value in expected.items():
        require(config.get(key) == value, f"Baseline setting changed: {key}")
    require(config["torch_version"] == str(torch.__version__), "Baseline PyTorch version differs")
    for name, digest in config["source_sha256"].items():
        require(file_sha(ROOT / name) == digest, f"Baseline source changed: {name}")
    require(summary["cache_run_id"] == CACHE_RUN_ID and summary["seed"] == config["seed"]
            and summary["experiment"] == expected["experiment"],
            "Baseline summary identity mismatch")
    return config, {"path": str(baseline_path), "paired_seed": seed, "summary": summary,
                    "config_sha256": file_sha(baseline_path / "config.json"),
                    "summary_sha256": file_sha(baseline_path / "summary.json"),
                    "head_checkpoint_loaded": False}


def weighted_bce(logits, labels, weights):
    """Mean over videos of FIXED train-mean-normalized coefficients * BCE."""
    require(logits.ndim == 1 and logits.shape == labels.shape == weights.shape
            and len(labels) > 0, "Weighted loss shape mismatch")
    require(all(t.device.type == "cpu" for t in (logits, labels, weights)), "Expected CPU loss")
    require(all(t.dtype == torch.float32 for t in (logits, labels, weights)), "Expected float32 loss")
    require(not weights.requires_grad, "Weights must not require gradients")
    require(torch.isfinite(weights).all().item() and (weights > 0).all().item(), "Invalid loss weights")
    require(set(labels.tolist()) <= {0., 1.}, "Invalid binary labels")
    losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = (weights * losses).mean()  # NOT divided by the current batch's weight sum.
    require(torch.isfinite(loss).item(), "Non-finite training loss")
    return loss, losses.mean()


def make_loader(x, y, weights, seed, batch_size=64):
    require(len(x) == len(y) == len(weights) > 0, "Training input count mismatch")
    require(not x.requires_grad, "Cached features must stay frozen")
    return DataLoader(TensorDataset(x, y, weights, torch.arange(len(y))),
                      batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def train_one_epoch(model, loader, optimizer, *, max_grad_norm=1.):
    model.train()
    total = len(loader.dataset)
    require(total > 0 and not loader.drop_last, "Full training coverage is required")
    seen, weighted_total, unweighted_total, updates = set(), 0., 0., 0
    for x, y, weights, indices in loader:
        batch_indices = indices.tolist()
        require(len(set(batch_indices)) == len(batch_indices)
                and set(batch_indices).isdisjoint(seen)
                and all(0 <= i < total for i in batch_indices), "Duplicate/invalid train indices")
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss, ordinary = weighted_bce(logits, y, weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        seen.update(batch_indices)
        weighted_total += float(loss.detach()) * len(y)
        unweighted_total += float(ordinary.detach()) * len(y)
        updates += 1
    require(seen == set(range(total)), "Incomplete epoch coverage")
    return {"weighted_loss": weighted_total / total, "unweighted_loss": unweighted_total / total,
            "videos_seen": len(seen), "optimizer_steps": updates}


def prepare(args):
    baseline, reference = load_baseline_reference(args.seed)
    print("Loading full frozen train/validation features; no video decoding...", flush=True)
    train_x, train_y, train_ids, train_run = load_sequences(CACHE, split="train")
    val_x, val_y, val_ids, val_run = load_sequences(CACHE, split="validation")
    require(train_run == val_run == CACHE_RUN_ID, "Unexpected cache run")
    require(set(train_ids).isdisjoint(val_ids), "Overlapping train/validation IDs")
    for name, x, y, ids, count in (("train", train_x, train_y, train_ids, 7193),
                                  ("validation", val_x, val_y, val_ids, 1547)):
        require(len(ids) == count and tuple(x.shape) == (count, 3, 8, 1536), f"Wrong full {name} shape")
        require(tuple(y.shape) == (count,) and set(y.tolist()) == {0., 1.}, f"Invalid {name} labels")
        require(x.dtype == y.dtype == torch.float32 and not x.requires_grad, f"Invalid {name} dtype/grad")

    print(f"Validating train-only candidate metadata and alpha={args.alpha:g} weights...", flush=True)
    weights, weight_info, weight_audit = load_train_weights(train_ids, train_y, cache_root=CACHE,
                                                          alpha=args.alpha)
    require(weight_info["cache_run_id"] == CACHE_RUN_ID, "Weight/cache run mismatch")
    require(weight_info["alpha"] == args.alpha, "Requested/actual weighting alpha mismatch")
    manifest_sha = weight_info["input_identities"]["manifest_sha256"]
    val_groups = aligned_groups(val_ids, val_y, split="validation", manifest_path=MANIFEST,
                                manifest_sha256=manifest_sha)
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = {row["video_id"]: row for row in csv.DictReader(stream)}
    require({rows[i]["source_group"] for i in train_ids}.isdisjoint(
            {rows[i]["source_group"] for i in val_ids}), "Overlapping source groups")

    # No trained temporal checkpoint is loaded. Metadata reading must not consume
    # RNG; reseeding here explicitly defines the baseline's fresh initialization.
    torch.manual_seed(args.seed)
    model = WindowTemporalTransformer().cpu()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        initial_sha = state_sha(WindowTemporalTransformer().cpu())
    require(state_sha(model) == initial_sha, "Fresh initialization mismatch")
    sources = [FOLDER / "weights.py", FOLDER / "train_quality.py",
               ROOT / "development/data_quality/scripts/simulate_quality_weights_v1.py",
               ROOT / "development/data_quality/scripts/audit_three_window_quality_v1.py",
               ROOT / "development/data_pipeline/pipeline.py",
               ROOT / "development/data_pipeline/multiwindow_v1.py",
               ROOT / "development/temporal_balanced_v1/sampling.py",
               ROOT / "development/temporal_balanced_v1/train_balanced.py"]
    config = copy.deepcopy(baseline)
    config.update({
        "experiment": f"frozen_mixnet_i3d_window_transformer_quality_alpha{alpha_tag(args.alpha)}_v1",
        "alpha": args.alpha,
        "analysis_role": ("exploratory_strength_comparison_after_alpha025"
                          if args.alpha == .5 else "initial_strength_comparison"),
        "cpu_threads": args.threads, "quality_weighting_used": True,
        "quality_weighting": weight_info, "baseline_reference": reference,
        "initialization": {"method": "fresh_random_same_seed_as_baseline",
                           "state_sha256": initial_sha, "trained_head_loaded": False},
        "loss": "mean(full_train_mean_normalized_video_weight * unreduced_BCE)",
        "validation_loss": "ordinary_unweighted_BCE_mean_over_all_1547_videos",
        "sampling": "uniform_shuffle_once_per_epoch_without_replacement",
        "early_stopping_initialization": "no_epoch0, matching frozen-head baseline",
        "test_used": False, "explicit_train_flag_required": True,
    })
    config["source_sha256"].update({str(path.relative_to(ROOT)): file_sha(path) for path in sources})
    require(sum(p.numel() for p in model.parameters()) == config["trainable_parameters"], "Parameter count changed")
    return {"model": model, "train_x": train_x, "train_y": train_y, "train_ids": train_ids,
            "val_x": val_x, "val_y": val_y, "val_ids": val_ids, "val_groups": val_groups,
            "weights": weights, "weight_audit": weight_audit, "config": config}


def dry_run(prepared):
    model = prepared["model"]
    before = state_sha(model)
    model.eval()
    with torch.no_grad():
        logits = model(prepared["val_x"][:2])
        require(tuple(logits.shape) == (2,) and torch.isfinite(logits).all().item(), "Validation forward failed")
    # Two real train videos, one from each class. Backward only; no optimizer.
    y = prepared["train_y"]
    indices = torch.tensor([int((y == label).nonzero(as_tuple=True)[0][0]) for label in (0, 1)])
    model.train()
    logits = model(prepared["train_x"][indices])
    loss, _ = weighted_bce(logits, y[indices], prepared["weights"][indices])
    loss.backward()
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                f"Missing/non-finite gradient: {name}")
    require(state_sha(model) == before, "Dry run changed model state")
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"status": "PASS", "dry_run": True, "training_started": False,
            "optimizer_created": False, "optimizer_steps": 0, "output_files_written": False,
            "full_train_shape": list(prepared["train_x"].shape),
            "full_validation_shape": list(prepared["val_x"].shape),
            "real_train_weighted_backward": "PASS", "real_validation_forward": "PASS",
            "model_state_unchanged": True, "source_groups_disjoint": True,
            "fresh_initialization_matches_seeded_baseline": True,
            "config": prepared["config"]}


def run_training(prepared, args):
    require(args.train and not args.dry_run, "Training requires explicit --train")
    model, config = prepared["model"], prepared["config"]
    tag = alpha_tag(args.alpha)
    require(config["alpha"] == config["quality_weighting"]["alpha"] == args.alpha,
            "Training config/weights/requested alpha mismatch")
    parent = FOLDER / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=f"quality_alpha{tag}_seed{args.seed}_", dir=parent))
    for name, value in (("config.json", config), ("weight_audit.json", prepared["weight_audit"])):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    save_checkpoint({"video_ids": prepared["train_ids"], "labels": prepared["train_y"],
                     "normalized_weights": prepared["weights"],
                     "quality_weighting": config["quality_weighting"]}, output / "train_weights.pt")
    print("Output:", output, flush=True)
    loader = make_loader(prepared["train_x"], prepared["train_y"], prepared["weights"],
                         args.seed, config["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    stopper = EarlyStopping(patience=config["early_stopping"]["patience"],
                            min_delta=config["early_stopping"]["min_delta"])
    best_loss, reason = float("inf"), "max_epochs"
    with (output / "history.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_weighted_loss", "train_unweighted_loss", "val_loss",
                         "val_accuracy", "val_macro_f1", "train_videos", "optimizer_steps", "seconds"])
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train = train_one_epoch(model, loader, optimizer, max_grad_norm=config["max_grad_norm"])
            metrics, _ = evaluate(model, prepared["val_x"], prepared["val_y"], batch_size=config["eval_batch_size"])
            seconds = time.perf_counter() - started
            writer.writerow([epoch, train["weighted_loss"], train["unweighted_loss"], metrics["loss"],
                             metrics["accuracy"], metrics["macro_f1"], train["videos_seen"],
                             train["optimizer_steps"], seconds])
            stream.flush()
            checkpoint = {"model_state_dict": model.state_dict(), "epoch": epoch,
                          "validation_loss": metrics["loss"], "config": config}
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save_checkpoint(checkpoint, output / "best.pt")
            save_checkpoint(checkpoint, output / "last.pt")
            print(f"epoch {epoch:02d}: train_weighted_loss={train['weighted_loss']:.4f}, "
                  f"val_loss={metrics['loss']:.4f}, val_accuracy={metrics['accuracy']:.4f}, "
                  f"val_macro_f1={metrics['macro_f1']:.4f}, seconds={seconds:.1f}", flush=True)
            if stopper.update(metrics["loss"]):
                reason = "early_stopping"
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    metrics, logits = evaluate(model, prepared["val_x"], prepared["val_y"], batch_size=config["eval_batch_size"])
    original = config["baseline_reference"]["summary"]["validation"]
    summary = {"experiment": config["experiment"], "seed": args.seed, "alpha": args.alpha,
               "epochs_completed": epoch, "stop_reason": reason, "best_epoch": best["epoch"],
               "cache_run_id": CACHE_RUN_ID, "validation": metrics,
               "validation_by_shot_type": per_shot_metrics(logits, prepared["val_y"], prepared["val_groups"]),
               "baseline_validation": original,
               "change_from_same_seed_baseline": {key: metrics[key] - original[key]
                                                  for key in ("loss", "accuracy", "macro_f1")},
               "test_used": False, "validation_weighting_used": False,
               "quality_weighting_used": True, "resume_supported": False}
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with (output / "validation_predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "true_label", "p_make", "predicted_label"])
        for video_id, label, logit in zip(prepared["val_ids"], prepared["val_y"], logits):
            writer.writerow([video_id, int(label), float(logit.sigmoid()), int(logit >= 0)])
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", output, flush=True)
    return output


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    prepared = prepare(args)
    if args.dry_run:
        print(json.dumps(dry_run(prepared), indent=2, allow_nan=False), flush=True)
        return
    run_training(prepared, args)


if __name__ == "__main__":
    main()
