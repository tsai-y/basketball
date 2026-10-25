"""Explicitly gated ResNet50 baseline training. Default mode only prints a plan.

No test-set interface. ACA formal training is deliberately blocked pending its
architecture audit. Checkpoints are committed only at optimizer boundaries.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import tempfile
import time

import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset

from .baseline_preflight import freeze_batchnorm, training_mode
from .data import Basketball50Dataset, POLICY, VideoScope, file_sha256, require, ROOT
from .weights import initialized_baseline, pin, verify_payload, FOLDER

SCHEMA = "aca_baseline_resume_v1"


@dataclass(frozen=True)
class TrainConfig:
    variant: str = "baseline"
    seed: int = 20261025
    epochs: int = 30
    patience: int = 5
    effective_batch: int = 16
    micro_batch: int = 1
    workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.0
    clip_norm: float = 1.0
    checkpoint_every: int = 25

    def __post_init__(self):
        require(self.variant == "baseline", "Formal ACA training blocked by unresolved architecture audit")
        for name in ("seed", "epochs", "patience", "effective_batch", "micro_batch", "workers", "checkpoint_every"):
            require(type(getattr(self, name)) is int, f"{name} must be int")
        require(0 <= self.seed < 2**32 and 1 <= self.epochs <= 30, "Invalid seed/epochs")
        require(1 <= self.patience <= self.epochs, "Invalid early-stop patience")
        require(self.effective_batch == 16 and self.micro_batch in (1, 2, 4) and
                0 <= self.workers <= 8 and self.checkpoint_every > 0, "Invalid batching/workers/checkpoint cadence")
        require(self.lr == 1e-4 and self.weight_decay == 0 and self.clip_norm == 1,
                "Protocol changes require a new reviewed experiment config")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def source_hashes():
    from development.data_pipeline import pipeline
    paths = list(FOLDER.glob("*.py")) + [FOLDER / "imagenet_v1.json"]
    result = {p.name: file_sha256(p) for p in paths}
    for p in (Path(pipeline.__file__), Path(pipeline.gate1a.__file__)):
        result["repo/" + p.relative_to(ROOT).as_posix()] = file_sha256(p)
    return result


def source_inventory(scopes):
    """Check every allowed original descriptor, without reading video bytes."""
    from development.data_pipeline.pipeline import gate1a
    identities, total_bytes = [], 0
    for scope in scopes:
        for row in scope.rows:
            _, path = scope.authorize(row["video_id"])
            fd = gate1a.open_guarded(path)
            try:
                info = os.fstat(fd)
                require(info.st_size > 0, "Empty authorized source")
                identities.append([row["video_id"], info.st_dev, info.st_ino, info.st_size,
                                   info.st_mtime_ns, info.st_ctime_ns])
                total_bytes += info.st_size
            finally:
                os.close(fd)
    return {"count": len(identities), "identity_sha256": canonical_hash(identities),
            "total_bytes": total_bytes, "check": "guarded_open_fstat_no_video_bytes_no_full_decode"}


def runtime_identity():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "opencv": cv2.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "precision": "FP32_no_AMP_no_TF32", "deterministic_algorithms": True,
            "cpu_threads": 4, "opencv_threads_per_worker": 1}


def configure_runtime(seed, *, cuda=True):
    if cuda:
        require(torch.cuda.is_available(), "CUDA required; no silent CPU fallback")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8", "Unexpected CUBLAS configuration")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def build_plan(source_root, config):
    train = VideoScope(source_root=source_root, split="train")
    validation = VideoScope(source_root=source_root, split="validation")
    identity = pin()
    verify_payload(FOLDER / identity["local_relative_path"], identity)
    return {"schema": SCHEMA, "config": asdict(config),
            "source_root": str(Path(source_root).resolve()), "input_hashes": train.hashes,
            "train_count": len(train.rows), "validation_count": len(validation.rows),
            "original_source_inventory": source_inventory((train, validation)),
            "train_ordered_ids_sha256": canonical_hash([r["video_id"] for r in train.rows]),
            "validation_ordered_ids_sha256": canonical_hash([r["video_id"] for r in validation.rows]),
            "source_code_sha256": source_hashes(), "weight_identity": identity,
            "runtime": runtime_identity(), "policy": POLICY,
            "bn_policy": "all_affine_and_running_buffers_frozen",
            "selection": "strict_minimum_validation_cross_entropy; ties_keep_earlier_epoch",
            "early_stop": "patience_consecutive_epochs_without_strict_loss_improvement",
            "scheduler": "none", "augmentation": "none", "class_weights": "none",
            "project_choices_not_specified_by_paper": [f"patience={config.patience}", "frozen_BN", "clip_norm=1",
                                                        "fixed_source_group_split", "uniform_full_frame_sampling"],
            "test_videos_opened": 0, "formal_training_started": False,
            "optimizer_steps_per_full_epoch": math.ceil(len(train.rows) / config.effective_batch),
            "last_effective_batch_videos": len(train.rows) % config.effective_batch or config.effective_batch,
            "resume_policy": "optimizer_boundary; redo_only_work_after_last_committed_checkpoint"}


def epoch_order(count, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(count, generator=generator).tolist()


def worker_init(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def make_loader(dataset, indices, config, epoch):
    # Dedicated generator: iterator creation must not consume the model RNG.
    generator = torch.Generator().manual_seed(config.seed + epoch + 100000)
    return DataLoader(Subset(dataset, indices), batch_size=config.micro_batch,
                      shuffle=False, num_workers=config.workers, pin_memory=True,
                      drop_last=False, worker_init_fn=worker_init, generator=generator,
                      persistent_workers=False,
                      **({"prefetch_factor": 2, "multiprocessing_context": "spawn"} if config.workers else {}))


class Metrics:
    def __init__(self, state=None):
        self.state = state or {"count": 0, "loss_sum": 0., "confusion": [[0, 0], [0, 0]], "by_shot_type": {}}

    def add(self, logits, labels, loss_sum, original_classes):
        require(torch.isfinite(logits).all().item() and math.isfinite(float(loss_sum)), "Nonfinite metrics")
        predicted, actual = logits.detach().argmax(-1).cpu().tolist(), labels.cpu().tolist()
        for target, prediction, cls in zip(actual, predicted, original_classes):
            self.state["confusion"][target][prediction] += 1
            shot = self.state["by_shot_type"].setdefault(cls[:-1], [[0, 0], [0, 0]])
            shot[target][prediction] += 1
        self.state["count"] += len(actual)
        self.state["loss_sum"] += float(loss_sum)

    @staticmethod
    def from_confusion(matrix):
        count = sum(map(sum, matrix))
        recalls, f1 = [], []
        for cls in (0, 1):
            tp, fn, fp = matrix[cls][cls], matrix[cls][1-cls], matrix[1-cls][cls]
            recalls.append(tp / (tp + fn) if tp + fn else 0.)
            f1.append(2 * tp / (2 * tp + fn + fp) if 2 * tp + fn + fp else 0.)
        return {"count": count, "accuracy": (matrix[0][0] + matrix[1][1]) / count if count else 0.,
                "macro_f1": sum(f1) / 2, "miss_recall": recalls[0], "make_recall": recalls[1],
                "confusion_rows_true_columns_predicted": matrix}

    def result(self):
        require(self.state["count"] > 0, "Empty metric accumulator")
        return {**self.from_confusion(self.state["confusion"]),
                "loss": self.state["loss_sum"] / self.state["count"],
                "by_shot_type": {k: self.from_confusion(v) for k, v in self.state["by_shot_type"].items()}}


def capture_rng():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(),
            numpy_state[2], numpy_state[3], numpy_state[4]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_json(path, payload):
    fd, name = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_checkpoint(path, model, optimizer, progress, plan):
    payload = {"schema": SCHEMA, "engineering_only": False, "plan_sha256": canonical_hash(plan),
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "rng": capture_rng(), "progress": progress}
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
    require(path.is_file() and not path.is_symlink(), "Resume requires a regular checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("schema") == SCHEMA and payload.get("engineering_only") is False,
            "Engineering or incompatible checkpoints cannot resume formal training")
    require(payload["plan_sha256"] == canonical_hash(plan), "Resume protocol/code/data/runtime mismatch")
    progress = payload["progress"]
    config = TrainConfig(**plan["config"])
    require(0 <= progress["cursor"] <= plan["train_count"] and
            (progress["cursor"] % config.effective_batch == 0 or progress["cursor"] == plan["train_count"]),
            "Checkpoint is not at an optimizer boundary")
    require(progress["train_metrics"]["count"] == progress["cursor"], "Checkpoint metric/cursor mismatch")
    require(0 <= progress["epoch"] <= config.epochs and len(progress["history"]) == progress["epoch"] and
            progress["optimizer_steps"] == progress["epoch"] * math.ceil(plan["train_count"] / config.effective_batch)
            + math.ceil(progress["cursor"] / config.effective_batch), "Checkpoint epoch/step/history mismatch")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    return progress


def initial_progress():
    return {"epoch": 0, "cursor": 0, "optimizer_steps": 0, "best_loss": None,
            "bad_epochs": 0, "history": [], "train_metrics": Metrics().state, "finished": False}


def verify_preflight(path, plan):
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report.get("status") == "passed" and report.get("optimizer_steps") == 0 and
            report.get("formal_training_started") is False and
            report.get("plan_sha256") == canonical_hash(plan) and
            report.get("cuda_resume_bitwise_equal") is True,
            "Missing/stale/incompatible training-readiness preflight")


def validation_pass(model, loader, device, stop):
    model.eval()
    metrics = Metrics()
    with torch.no_grad():
        for batch in loader:
            x, y = batch["tensor"].to(device, non_blocking=True), batch["binary_label"].to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, reduction="sum")
            metrics.add(logits, y, loss.item(), batch["original_class"])
            if stop["requested"]:
                return None
    return metrics.result()


def train_run(source_root, config, preflight_report, *, resume=None):
    configure_runtime(config.seed)
    plan = build_plan(source_root, config)
    verify_preflight(preflight_report, plan)
    model, _ = initialized_baseline(config.seed)
    freeze_batchnorm(model)
    model = model.cuda()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config.lr, weight_decay=config.weight_decay)
    if resume:
        output = Path(resume).resolve()
        require(output.parent == (FOLDER / "artifacts").resolve() and output.name.startswith("baseline_train_"),
                "Resume must target this module's formal run directory")
        stored = json.loads((output / "config.json").read_text())
        require(canonical_hash(stored) == canonical_hash(plan), "Saved run config mismatch")
        progress = load_checkpoint(output / "last.pt", model, optimizer, plan)
    else:
        output = Path(tempfile.mkdtemp(prefix=f"baseline_train_seed{config.seed}_", dir=FOLDER / "artifacts"))
        atomic_json(output / "config.json", plan)
        shutil.copy2(preflight_report, output / "readiness.json")
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for name in source_hashes():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = ROOT / name.removeprefix("repo/") if name.startswith("repo/") else FOLDER / name
            shutil.copy2(source, target)
        progress = initial_progress()
        save_checkpoint(output / "last.pt", model, optimizer, progress, plan)
    print(f"RUN: {output}", flush=True)
    if progress["finished"]:
        return {"status": "already_finished", "output": str(output)}
    train_data = Basketball50Dataset(source_root=source_root, split="train")
    validation_data = Basketball50Dataset(source_root=source_root, split="validation")
    return fit(model, optimizer, train_data, validation_data, config, plan, output, progress, device="cuda")


def fit(model, optimizer, train_data, validation_data, config, plan, output, progress, *, device):
    """Shared production loop; CPU tiny-data tests exercise this exact loop too."""
    require(len(train_data) == plan["train_count"] and len(validation_data) == plan["validation_count"],
            "Dataset count mismatch")
    stop = {"requested": False}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def request_stop(signum, frame):
        stop["requested"] = True
        print("Stop requested: finish current effective batch and checkpoint; no partial update saved.", flush=True)

    for sig in previous:
        signal.signal(sig, request_stop)
    try:
        while progress["epoch"] < config.epochs and not progress["finished"]:
            epoch, cursor = progress["epoch"], progress["cursor"]
            order = epoch_order(len(train_data), config.seed, epoch)
            iterator = iter(make_loader(train_data, order[cursor:], config, epoch))
            metrics = Metrics(progress["train_metrics"])
            training_mode(model)
            while cursor < len(order):
                actual = min(config.effective_batch, len(order) - cursor)
                optimizer.zero_grad(set_to_none=True)
                consumed = 0
                started = time.perf_counter()
                while consumed < actual:
                    batch = next(iterator)
                    x = batch["tensor"].to(device, non_blocking=True)
                    y = batch["binary_label"].to(device)
                    expected_ids = [train_data.scope.rows[i]["video_id"]
                                    for i in order[cursor + consumed:cursor + consumed + len(y)]]
                    require(list(batch["video_id"]) == expected_ids and consumed + len(y) <= actual,
                            "Loader order or effective batch boundary mismatch")
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y, reduction="sum")
                    require(torch.isfinite(loss).item(), "Nonfinite training loss")
                    (loss / actual).backward()
                    metrics.add(logits, y, loss.item(), batch["original_class"])
                    consumed += len(y)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         config.clip_norm, error_if_nonfinite=True)
                optimizer.step()
                cursor += actual
                progress.update(cursor=cursor, optimizer_steps=progress["optimizer_steps"] + 1,
                                train_metrics=metrics.state)
                if progress["optimizer_steps"] % config.checkpoint_every == 0 or cursor == len(order) or stop["requested"]:
                    save_checkpoint(output / "last.pt", model, optimizer, progress, plan)
                    print(f"epoch={epoch+1} cursor={cursor}/{len(order)} steps={progress['optimizer_steps']} "
                          f"train_loss={metrics.result()['loss']:.6f} effective_batch_seconds={time.perf_counter()-started:.3f}", flush=True)
                if stop["requested"]:
                    return {"status": "stopped_at_optimizer_boundary", "output": str(output), **progress}
            del iterator
            validation = validation_pass(model, make_loader(validation_data, list(range(len(validation_data))), config, epoch),
                                         device, stop)
            if validation is None:
                save_checkpoint(output / "last.pt", model, optimizer, progress, plan)
                return {"status": "stopped_validation_will_repeat", "output": str(output)}
            require(validation["count"] == len(validation_data), "Incomplete validation")
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
            print(json.dumps(progress["history"][-1], ensure_ascii=False), flush=True)
        summary = {"status": "completed", "output": str(output), "epochs": progress["epoch"],
                   "optimizer_steps": progress["optimizer_steps"], "best_validation_loss": progress["best_loss"],
                   "selection": "best.pt_by_validation_loss", "test_videos_opened": 0,
                   "stop_reason": "early_stop" if progress["bad_epochs"] >= config.patience else "max_epochs"}
        atomic_json(output / "summary.json", summary)
        return summary
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=FOLDER / "baseline_config.json")
    parser.add_argument("--start-training", action="store_true", help="Explicit authorization gate; absent means plan only")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = TrainConfig(**json.loads(args.config.read_text()))
    if not args.start_training:
        require(not args.resume, "Resume requires --start-training")
        print(json.dumps(build_plan(args.source_root, config), ensure_ascii=False, indent=2))
        return
    require(args.preflight_report is not None, "Training requires a matching readiness report")
    print(json.dumps(train_run(args.source_root, config, args.preflight_report, resume=args.resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
