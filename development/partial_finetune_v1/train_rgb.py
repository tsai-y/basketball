"""RGB partial fine-tuning. Requires explicit --train; --dry-run never updates weights."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from tempfile import mkdtemp
import time

import cv2
import torch
from torch.utils.data import DataLoader, Subset, default_collate

from development.data_pipeline.pipeline import load_gate_config
from development.linear_baseline_v1.train_linear import EarlyStopping
from development.partial_finetune_v1.data import RGBTrainingDataset, check_split_metadata
from development.partial_finetune_v1.engine import (
    evaluate_loader, metrics_from_logits, move_batch, train_one_epoch,
)
from development.partial_finetune_v1.initialization import ROOT, file_sha, load_initialized_model
from development.partial_finetune_v1.preflight import check_frozen_state, check_gradients, frozen_snapshot
from development.temporal_baseline_v1.train_temporal import require, save_checkpoint


FOLDER = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Small RGB forward/backward; no optimizer or files")
    mode.add_argument("--train", action="store_true", help="Explicitly start full train/validation fine-tuning")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum additional fine-tuning epochs")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, choices=(1, 2), default=2,
                        help="GPU video batch; effective optimizer batch stays 64")
    parser.add_argument("--log-every", type=int, default=50, help="Progress every N micro-batches")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.threads < 1 or args.log_every < 1 or not 0 <= args.seed < 2**32:
        parser.error("Expected positive epochs/threads/log-every and seed in [0,2**32)")
    return args


def configure_runtime(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8"), "Invalid CUBLAS workspace")
    require(torch.cuda.is_available(), "CUDA unavailable; use a local terminal with GPU access")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda")


def make_loader(dataset, args, device, *, shuffle):
    return DataLoader(dataset, batch_size=args.micro_batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(args.seed),
                      num_workers=0, pin_memory=device.type == "cuda", drop_last=False)


def make_config(args, initialization, split_info, gate, model, device):
    paths = list(FOLDER.glob("*.py")) + [
        ROOT / "development/data_pipeline/pipeline.py",
        ROOT / "development/data_pipeline/multiwindow_v1.py",
        ROOT / "development/data_quality/transforms/video_clip_transforms.py",
        ROOT / "development/temporal_baseline_v1/model.py",
        ROOT / "development/temporal_baseline_v1/train_temporal.py",
        ROOT / "development/linear_baseline_v1/train_linear.py",
    ]
    return {
        "experiment": "rgb_partial_mixnet_i3d_window_transformer_v1",
        "schema_version": "partial_rgb_training_v1", "initialization": initialization,
        "input": {"clips_per_video": 3, "clip_shape": [3, 64, 192, 256],
                  "window_policy": "three_window_coverage_v1", "transform": "reference",
                  "dataset_epoch": 0, "rgb_range": [0., 1.], "window_weights": "equal"},
        "trainable_backbone": ["blocks.5 (excluding BatchNorm)", "conv_head"],
        "all_batchnorm_frozen": True,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer": "AdamW", "backbone_lr": 1e-5, "head_lr": 1e-4,
        "weight_decay": 1e-4, "max_grad_norm": 1., "loss": "BCEWithLogitsLoss",
        "micro_batch_size": args.micro_batch, "effective_batch_size": 64,
        "accumulation_steps": 64 // args.micro_batch,
        "partial_group_scaling": "sum BCE divided by actual videos in the group",
        "optimizer_steps_per_epoch": math.ceil(split_info["train"]["videos"] / 64),
        "sampling": "shuffle all train videos once per epoch, without replacement",
        "epochs": args.epochs, "seed": args.seed, "threshold": .5,
        "early_stopping": {"monitor": "validation_loss", "patience": 10, "min_delta": 1e-4},
        "selection": "minimum_validation_loss_including_epoch_0",
        "validation": "all original validation videos once each per epoch; no weight updates",
        "quality_weighting_used": False, "balanced_sampling_used": False, "test_used": False,
        "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "precision": "float32_no_amp_no_tf32", "cpu_threads": args.threads,
        "opencv_threads": 1, "loader_workers": 0, "log_every": args.log_every,
        "torch_version": str(torch.__version__), "opencv_version": cv2.__version__,
        "cuda_version": str(torch.version.cuda), "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": True, "resume_supported": False,
        "split_info": split_info, "gate_config_sha256": gate.sha256,
        "input_identities": gate.raw["identities"],
        "source_sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(paths)},
    }


def dry_run(model, train, validation, args, device, config):
    """Decode only a small train/validation probe; metadata checks cover full lists."""
    before = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    train_indices = [next(i for i, r in enumerate(train.rows) if int(r["binary_label"]) == label)
                     for label in (0, 1)][:args.micro_batch]
    val_indices = [next(i for i, r in enumerate(validation.rows) if int(r["binary_label"]) == label)
                   for label in (0, 1)]
    model.train()
    x, y, train_ids = move_batch(default_collate([train[i] for i in train_indices]), device)
    logits = model(x)
    require(logits.shape == y.shape, "Dry-run logit shape mismatch")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    require(torch.isfinite(loss).item(), "Non-finite dry-run loss")
    loss.backward()
    gradients = check_gradients(model)
    model.zero_grad(set_to_none=True)
    del x, y, logits, loss
    val_loader = make_loader(Subset(validation, val_indices), args, device, shuffle=False)
    val_ids = [validation.rows[i]["video_id"] for i in val_indices]
    metrics, _, _, observed_ids = evaluate_loader(model, val_loader, device=device,
                                                 expected_ids=val_ids, progress=None)
    for name, value in model.state_dict().items():
        require(torch.equal(value.detach().cpu(), before[name]), f"Dry-run changed state: {name}")
    require(all(p.grad is None for p in model.parameters()), "Dry-run gradients not cleared")
    print(json.dumps({"status": "PASS", "dry_run": True, "training_started": False,
                      "optimizer_steps": 0, "output_files_written": False,
                      "full_split_metadata_checked": True, "all_videos_decoded": False,
                      "decoded_train_ids": train_ids, "decoded_validation_ids": observed_ids,
                      "probe_validation_loss_not_full_split": metrics["loss"],
                      "gradient_abs_sums": gradients, "all_model_state_unchanged": True,
                      "config": config}, indent=2), flush=True)


def checkpoint_payload(model, epoch, config, metrics, logits, labels, ids):
    return {"schema_version": "partial_rgb_training_v1", "epoch": epoch, "config": config,
            "model_state_dict": {n: v.detach().cpu() for n, v in model.state_dict().items()},
            "validation": metrics, "validation_logits": logits.cpu(),
            "validation_labels": labels.cpu(), "validation_video_ids": list(ids),
            "engineering_only": False, "resume_supported": False}


def selected_summary(best, validation, *, completed, stop_reason):
    ids = [r["video_id"] for r in validation.rows]
    labels = torch.tensor([int(r["binary_label"]) for r in validation.rows], dtype=torch.float32)
    require(best["validation_video_ids"] == ids, "Best-checkpoint validation IDs changed")
    require(torch.equal(best["validation_labels"], labels), "Best-checkpoint labels changed")
    logits = best["validation_logits"]
    metrics = metrics_from_logits(logits, labels)
    require(metrics == best["validation"], "Saved validation metrics mismatch")
    by_shot = {}
    for shot in ("2p", "3p", "ft", "mp"):
        mask = torch.tensor([r["original_class"][:-1] == shot for r in validation.rows])
        if mask.any():
            by_shot[shot] = metrics_from_logits(logits[mask], labels[mask])
    return {"experiment": best["config"]["experiment"], "seed": best["config"]["seed"],
            "epochs_completed": completed, "stop_reason": stop_reason,
            "best_epoch": best["epoch"], "selected_initial_weights": best["epoch"] == 0,
            "validation": metrics, "validation_by_shot_type": by_shot,
            "test_used": False, "frozen_parameters_and_buffers_unchanged": True}


def run_training(model, train, validation, args, device, config):
    """Only called after the explicit --train gate. Never called by dry-run."""
    require(args.train and not args.dry_run, "Explicit --train required")
    parent = FOLDER / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=f"rgb_seed{args.seed}_", dir=parent))
    with (output / "config.json").open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    print("Output:", output, flush=True)
    train_loader = make_loader(train, args, device, shuffle=True)
    val_loader = make_loader(validation, args, device, shuffle=False)
    ids = [r["video_id"] for r in validation.rows]
    frozen = frozen_snapshot(model)
    stopper = EarlyStopping(patience=10, min_delta=1e-4)
    progress = lambda text: print(text, flush=True)
    print("Epoch 00: full validation of original warm-start weights; no optimizer steps.", flush=True)
    started = time.perf_counter()
    metrics, logits, labels, observed = evaluate_loader(
        model, val_loader, device=device, expected_ids=ids, log_every=args.log_every, progress=progress)
    check_frozen_state(model, frozen)
    initial_metrics = metrics.copy()
    best_loss = metrics["loss"]
    stopper.update(best_loss)
    save_checkpoint(checkpoint_payload(model, 0, config, metrics, logits, labels, observed), output / "best.pt")
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4)
    stop_reason, completed = "max_epochs", 0
    with (output / "history.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1",
                         "optimizer_steps", "seconds"])
        writer.writerow([0, "", metrics["loss"], metrics["accuracy"], metrics["macro_f1"], 0,
                         time.perf_counter() - started])
        stream.flush()
        print(f"epoch 00: val_loss={best_loss:.6f}, val_accuracy={metrics['accuracy']:.4f}", flush=True)
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            log = lambda text: print(f"[epoch {epoch:02d}] {text}", flush=True)
            train_metrics = train_one_epoch(model, train_loader, optimizer, device=device,
                                             effective_batch_size=64, max_grad_norm=1.,
                                             log_every=args.log_every, progress=log)
            metrics, logits, labels, observed = evaluate_loader(
                model, val_loader, device=device, expected_ids=ids, log_every=args.log_every, progress=log)
            check_frozen_state(model, frozen)
            payload = checkpoint_payload(model, epoch, config, metrics, logits, labels, observed)
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                save_checkpoint(payload, output / "best.pt")
            save_checkpoint(payload, output / "last.pt")
            completed = epoch
            seconds = time.perf_counter() - started
            writer.writerow([epoch, train_metrics["loss"], metrics["loss"], metrics["accuracy"],
                             metrics["macro_f1"], train_metrics["optimizer_steps"], seconds])
            stream.flush()
            print(f"epoch {epoch:02d}: train_loss={train_metrics['loss']:.5f}, "
                  f"val_loss={metrics['loss']:.5f}, val_accuracy={metrics['accuracy']:.4f}, "
                  f"val_macro_f1={metrics['macro_f1']:.4f}, seconds={seconds:.1f}", flush=True)
            if stopper.update(metrics["loss"]):
                stop_reason = "early_stopping"
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    summary = selected_summary(best, validation, completed=completed, stop_reason=stop_reason)
    summary["initial_validation"] = initial_metrics
    summary["validation_loss_change_from_initial"] = summary["validation"]["loss"] - initial_metrics["loss"]
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    # These logits were recorded during the full validation of THIS saved checkpoint.
    # No second expensive decode of all validation videos is necessary.
    with (output / "validation_predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "true_label", "p_make", "predicted_label"])
        for video_id, label, logit in zip(ids, best["validation_labels"], best["validation_logits"]):
            writer.writerow([video_id, int(label), float(logit.sigmoid()), int(logit >= 0)])
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", output, flush=True)
    return output


def main(argv=None):
    args = parse_args(argv)
    device = configure_runtime(args)
    gate = load_gate_config()
    print("Checking complete train/validation split metadata...", flush=True)
    train, validation = (RGBTrainingDataset(gate, split=split) for split in ("train", "validation"))
    split_info = check_split_metadata(train, validation)
    model, initialization = load_initialized_model()
    model = model.to(device)
    config = make_config(args, initialization, split_info, gate, model, device)
    print(f"Train={len(train)}; validation={len(validation)}; "
          f"micro-batch={args.micro_batch}; effective batch=64; "
          f"accumulation={config['accumulation_steps']}; test excluded.", flush=True)
    if args.dry_run:
        dry_run(model, train, validation, args, device, config)
        return
    run_training(model, train, validation, args, device, config)


if __name__ == "__main__":
    main()
