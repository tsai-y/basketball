"""Candidate-specific checkpoint schema and loop; baseline code stays frozen.

Numeric helpers and the guarded loader are imported read-only from the existing
baseline. The loop is separate because its checkpoint loader deliberately rejects
ACA. Both real and synthetic tests exercise this exact candidate loop.
"""
import math
import os
from pathlib import Path
import signal
import tempfile
import time

import torch
from torch import nn

from development.aca_net_v1.baseline_preflight import training_mode
from development.aca_net_v1.data import require
from development.aca_net_v1.train import (Metrics, atomic_json, canonical_hash, capture_rng,
                                        epoch_order, initial_progress, make_loader, restore_rng,
                                        validation_pass)
from .experiment import HighResConfig as CandidateConfig, SCHEMA


def save_checkpoint(path, model, optimizer, progress, plan):
    validate_progress(progress, plan)
    check_adam(optimizer, progress["optimizer_steps"])
    engineering, kind = checkpoint_identity(plan)
    payload = {"schema": SCHEMA, "engineering_only": engineering,
               "kind": kind,
               "paper_exact_reproduction": False, "plan_sha256": canonical_hash(plan),
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "progress": progress, "rng": capture_rng()}
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_checkpoint(path, model, optimizer, plan):
    require(path.is_file() and not path.is_symlink(), "Resume needs a regular candidate checkpoint")
    value = torch.load(path, map_location="cpu", weights_only=True)
    engineering, kind = checkpoint_identity(plan)
    require(value.get("schema") == SCHEMA and value.get("engineering_only") is engineering and
            value.get("kind") == kind,
            "Baseline/engineering/synthetic checkpoints cannot initialize or resume formal ACA")
    require(value.get("plan_sha256") == canonical_hash(plan), "Resume config/source/data/runtime mismatch")
    cfg, progress = CandidateConfig(**plan["config"]), value["progress"]
    cursor, epoch = progress["cursor"], progress["epoch"]
    require(type(cursor) is int and type(epoch) is int and 0 <= cursor <= plan["train_count"] and
            (cursor % cfg.effective_batch == 0 or cursor == plan["train_count"]), "Invalid optimizer boundary")
    require(0 <= epoch <= cfg.epochs and progress["train_metrics"]["count"] == cursor and
            len(progress["history"]) == epoch and progress["optimizer_steps"] ==
            epoch * math.ceil(plan["train_count"] / cfg.effective_batch) + math.ceil(cursor / cfg.effective_batch),
            "Checkpoint progress/history mismatch")
    validate_progress(progress, plan)
    require(all(not t.is_floating_point() or (t.dtype == torch.float32 and torch.isfinite(t).all().item())
                for t in value["model"].values()), "Nonfinite/non-FP32 checkpoint state")
    model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    check_adam(optimizer, progress["optimizer_steps"])
    restore_rng(value["rng"])
    return progress


def fit(model, optimizer, train_data, validation_data, config, plan, output, progress, *, device):
    require(len(train_data) == plan["train_count"] and len(validation_data) == plan["validation_count"],
            "Dataset count mismatch")
    if progress["finished"]:
        # Recover JSON mirrors if interrupted after the final atomic checkpoint.
        atomic_json(output / "history.json", progress["history"])
        result = final_summary(output, progress, config)
        atomic_json(output / "summary.json", result)
        return result
    stop = {"requested": False}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def request_stop(signum, frame):
        stop["requested"] = True
        print("Stop requested: checkpoint after current effective batch; no partial gradients saved.", flush=True)

    for sig in previous:
        signal.signal(sig, request_stop)
    try:
        while progress["epoch"] < config.epochs and not progress["finished"]:
            epoch, cursor = progress["epoch"], progress["cursor"]
            order = epoch_order(len(train_data), config.seed, epoch)
            iterator = iter(checked_loader(make_loader(train_data, order[cursor:], config, epoch), plan, "train"))
            metrics = Metrics(progress["train_metrics"])
            training_mode(model)
            while cursor < len(order):
                actual = min(config.effective_batch, len(order) - cursor)
                optimizer.zero_grad(set_to_none=True)
                consumed = 0
                started = time.perf_counter()
                while consumed < actual:
                    batch = next(iterator)
                    x, y = batch["tensor"].to(device, non_blocking=True), batch["binary_label"].to(device)
                    expected = [train_data.scope.rows[i]["video_id"]
                                for i in order[cursor + consumed:cursor + consumed + len(y)]]
                    require(list(batch["video_id"]) == expected and consumed + len(y) <= actual,
                            "Loader order/batch boundary mismatch")
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y, reduction="sum")
                    require(torch.isfinite(loss).item(), "Nonfinite candidate loss")
                    (loss / actual).backward()
                    metrics.add(logits, y, loss.item(), batch["original_class"])
                    consumed += len(y)
                norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                config.clip_norm, error_if_nonfinite=True)
                optimizer.step()
                cursor += actual
                progress.update(cursor=cursor, optimizer_steps=progress["optimizer_steps"] + 1,
                                train_metrics=metrics.state)
                if progress["optimizer_steps"] % config.checkpoint_every == 0 or cursor == len(order) or stop["requested"]:
                    save_checkpoint(output / "last.pt", model, optimizer, progress, plan)
                    print(f"ACA256 candidate epoch={epoch+1} cursor={cursor}/{len(order)} "
                          f"steps={progress['optimizer_steps']} train_loss={metrics.result()['loss']:.6f} "
                          f"grad_norm={norm.item():.4f} effective_batch_seconds={time.perf_counter()-started:.3f}", flush=True)
                if stop["requested"]:
                    return {"status": "stopped_at_optimizer_boundary", "output": str(output),
                            "cursor": cursor, "optimizer_steps": progress["optimizer_steps"]}
            del iterator
            validation = validation_pass(model, checked_loader(make_loader(validation_data, list(range(len(validation_data))), config, epoch), plan, "validation"),
                                         device, stop)
            if validation is None:
                save_checkpoint(output / "last.pt", model, optimizer, progress, plan)
                return {"status": "stopped_validation_will_repeat", "output": str(output)}
            require(validation["count"] == len(validation_data), "Incomplete candidate validation")
            improved = progress["best_loss"] is None or validation["loss"] < progress["best_loss"]
            progress["bad_epochs"] = 0 if improved else progress["bad_epochs"] + 1
            if improved:
                progress["best_loss"] = validation["loss"]
            progress["history"].append({"epoch": epoch + 1, "train": metrics.result(),
                                        "validation": validation, "selected_as_best": improved})
            progress.update(epoch=epoch + 1, cursor=0, train_metrics=Metrics().state)
            progress["finished"] = progress["epoch"] >= config.epochs or progress["bad_epochs"] >= config.patience
            if improved:
                save_checkpoint(output / "best.pt", model, optimizer, progress, plan)
            save_checkpoint(output / "last.pt", model, optimizer, progress, plan)
            atomic_json(output / "history.json", progress["history"])
            print(f"epoch={epoch+1} val_loss={validation['loss']:.6f} val_accuracy={validation.get('accuracy',0):.4%} "
                  f"best={improved} bad_epochs={progress['bad_epochs']}", flush=True)
        result = final_summary(output, progress, config)
        atomic_json(output / "summary.json", result)
        return result
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def final_summary(output, progress, config):
    selected = [r for r in progress["history"] if r["selected_as_best"]]
    return {"status": "completed", "model": "aca_dense_highres_candidate", "paper_exact_reproduction": False,
            "output": str(output), "epochs": progress["epoch"], "optimizer_steps": progress["optimizer_steps"],
            "best_epoch": selected[-1]["epoch"] if selected else None,
            "best_validation": selected[-1]["validation"] if selected else None,
            "selection": "minimum_validation_CE_not_accuracy", "test_videos_opened": 0,
            "input_resolution_height_width": [192, 256], "fresh_ImageNet_initialization": True,
            "stop_reason": "early_stop" if progress["bad_epochs"] >= config.patience else "max_epochs"}


def checked_loader(loader, plan, split):
    for batch in loader:
        if not plan.get("engineering_only_synthetic_toy", False):
            x = batch["tensor"]
            require(tuple(x.shape[1:]) == (3, 50, 192, 256) and x.dtype == torch.float32
                    and torch.isfinite(x).all().item(), "Expected finite FP32 [B,3,50,192,256]")
            require(all(s == split for s in batch["split"]), "Cross-split batch")
        yield batch


def checkpoint_identity(plan):
    toy = bool(plan.get("engineering_only_synthetic_toy", False))
    real = bool(plan.get("engineering_only_real_probe", False))
    require(not (toy and real), "Ambiguous checkpoint engineering identity")
    return (True, "synthetic_toy") if toy else (True, "real_highres_engineering") if real else (False, "formal_aca_highres_candidate")


def check_adam(optimizer, steps):
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    require(steps >= 0 and (len(optimizer.state) == len(parameters) if steps else not optimizer.state),
            "Adam state coverage mismatch")
    for p, state in optimizer.state.items():
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"} and state["step"].item() == steps
                and state["exp_avg"].shape == state["exp_avg_sq"].shape == p.shape
                and torch.isfinite(state["exp_avg"]).all().item()
                and torch.isfinite(state["exp_avg_sq"]).all().item(), "Invalid Adam moments/step")


def validate_progress(progress, plan):
    cfg = CandidateConfig(**plan["config"])
    if not plan.get("engineering_only_synthetic_toy", False):
        require(plan["schema"] == SCHEMA and plan["input_shape_per_video"] == [3, 50, 192, 256],
                "Only the new high-resolution plan is accepted")
    epoch, cursor, steps = (progress[k] for k in ("epoch", "cursor", "optimizer_steps"))
    require(all(type(v) is int for v in (epoch, cursor, steps)) and 0 <= epoch <= cfg.epochs
            and 0 <= cursor <= plan["train_count"]
            and (cursor % cfg.effective_batch == 0 or cursor == plan["train_count"]), "Invalid boundary")
    require(len(progress["history"]) == epoch and progress["train_metrics"]["count"] == cursor
            and sum(map(sum, progress["train_metrics"]["confusion"])) == cursor
            and steps == epoch * math.ceil(plan["train_count"] / cfg.effective_batch)
                + math.ceil(cursor / cfg.effective_batch), "Invalid checkpoint counts")
    best, bad = None, 0
    for index, row in enumerate(progress["history"], 1):
        loss = row["validation"]["loss"]
        require(row["epoch"] == index and math.isfinite(loss) and loss >= 0
                and row["train"]["count"] == plan["train_count"]
                and row["validation"]["count"] == plan["validation_count"], "Invalid epoch history")
        improved = best is None or loss < best
        require(row["selected_as_best"] is improved, "Not strict min-CE/earliest tie selection")
        best = loss if improved else best
        bad = 0 if improved else bad + 1
        require(index == epoch or bad < cfg.patience, "History continued beyond early stop")
    require(progress["best_loss"] == best and progress["bad_epochs"] == bad
            and progress["finished"] is (epoch >= cfg.epochs or bad >= cfg.patience)
            and (not progress["finished"] or cursor == 0), "Best-loss/patience/finished mismatch")
