"""Training-readiness audit: no optimizer updates on real videos.

The tiny synthetic resume probe DOES update a toy classifier, never ResNet50.
The full dense ACA candidate receives one forward/backward only; no training.
"""
import argparse
import copy
from dataclasses import replace
import gc
import json
from pathlib import Path
import signal
import shutil
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from .baseline_preflight import (check_gradients, check_state, freeze_batchnorm,
                                 state_snapshot, training_mode)
from .data import (Basketball50Dataset, VideoScope, guarded_source_sha256,
                   require, select_smoke_ids)
from .model import VideoResNet50
from .train import (TrainConfig, Metrics, atomic_json, build_plan, canonical_hash,
                    configure_runtime, epoch_order, fit, initial_progress,
                    load_checkpoint, make_loader, validation_pass)
from .weights import FOLDER, initialized_baseline


class ToyDataset(Dataset):
    """Deterministic synthetic inputs, deliberately 19 samples (16+3)."""
    def __init__(self, count):
        self.count = count
        self.scope = SimpleNamespace(rows=[{"video_id": f"synthetic-{i}"} for i in range(count)])

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {"tensor": torch.tensor([index / 19, (index % 3) / 3, 1.]),
                "binary_label": index % 2, "video_id": f"synthetic-{index}",
                "original_class": f"2p{index % 2}"}


def toy_model(device):
    return nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Dropout(.3), nn.Linear(8, 2)).to(device)


def nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def synthetic_resume_probe(parent, device="cpu"):
    config = TrainConfig(epochs=2, patience=2, workers=0, checkpoint_every=1)
    plan = {"config": vars(config), "train_count": 19, "validation_count": 5,
            "engineering_only_synthetic_toy": True}
    train, validation = ToyDataset(19), ToyDataset(5)
    continuous, interrupted = parent / "continuous", parent / "interrupted"
    continuous.mkdir()
    interrupted.mkdir()
    configure_runtime(config.seed, cuda=device == "cuda")
    model = toy_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    fit(model, optimizer, train, validation, config, plan, continuous, initial_progress(), device=device)
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected = torch.load(continuous / "last.pt", map_location="cpu", weights_only=True)
    configure_runtime(config.seed, cuda=device == "cuda")
    model = toy_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    original_step = optimizer.step
    calls = 0

    def interrupted_step(*args, **kwargs):
        nonlocal calls
        result = original_step(*args, **kwargs)
        calls += 1
        if calls == 1:
            signal.raise_signal(signal.SIGINT)
        return result

    with patch.object(optimizer, "step", side_effect=interrupted_step):
        result = fit(model, optimizer, train, validation, config, plan, interrupted, initial_progress(), device=device)
    require(result["status"] == "stopped_at_optimizer_boundary" and result["cursor"] == 16,
            "Synthetic interruption did not save at expected cursor")
    model = toy_model(device)  # Deliberately consumes RNG; load must restore it.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    progress = load_checkpoint(interrupted / "last.pt", model, optimizer, plan)
    fit(model, optimizer, train, validation, config, plan, interrupted, progress, device=device)
    actual = torch.load(interrupted / "last.pt", map_location="cpu", weights_only=True)
    require(nested_equal(expected_model, model.state_dict()) and
            nested_equal(expected_optimizer, optimizer.state_dict()) and
            nested_equal(expected["rng"], actual["rng"]) and
            expected["progress"] == actual["progress"], "Resumed trajectory differs from uninterrupted run")
    require(actual["progress"]["optimizer_steps"] == 4 and
            all(row["train"]["count"] == 19 and row["validation"]["count"] == 5
                for row in actual["progress"]["history"]), "Last partial batch or validation omitted")
    return {"bitwise_equal": True, "device": device, "synthetic_optimizer_updates_executed": 8,
            "train_samples_per_epoch": 19, "last_effective_batch": 3,
            "dropout_active_during_training": True, "compared": ["model", "Adam", "RNG", "history", "cursor"]}


def aca_gpu_probe(tensor, label, initialization):
    """Full 50-frame dense candidate, pretrained backbone, zero updates."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(20261025)
    model = VideoResNet50(variant="aca", num_classes=2)
    state = torch.load(initialization["checkpoint"], map_location="cpu", weights_only=True)
    transfer = model.load_imagenet_state(state)
    del state
    freeze_batchnorm(model)
    before = state_snapshot(model)
    model = model.cuda()
    training_mode(model)
    torch.cuda.synchronize()
    started = time.perf_counter()
    logits = model(tensor.cuda())
    loss = nn.functional.cross_entropy(logits, label.cuda())
    require(torch.isfinite(loss).item(), "ACA candidate produced nonfinite loss")
    loss.backward()
    gradients = check_gradients(model)
    branches = {name: 0. for name in ("gate_value", "gate_control", "gate_output", "short_term", "kernel_generator", "tsci")}
    for name, p in model.named_parameters():
        if p.grad is not None:
            for branch in branches:
                if f".{branch}." in name:
                    branches[branch] += p.grad.abs().sum().item()
    require(all(value > 0 for value in branches.values()), "ACA branch received no gradient")
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    check_state(model, before)
    result = {"status": "passed", "variant": "aca_dense_candidate_not_author_verified",
              "parameters": sum(p.numel() for p in model.parameters()), "input_shape": list(tensor.shape),
              "logits": logits.detach().cpu().tolist(), "loss": loss.item(),
              "forward_backward_seconds": seconds,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "gradient_abs_sums": gradients, "branch_gradient_abs_sums": branches,
              "pretrained_transfer": transfer, "all_parameters_and_buffers_unchanged": True,
              "optimizer_steps": 0, "architecture_fidelity_gate_passed": False}
    del model, before, loss, logits
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(source_root, config, *, include_aca=True):
    configure_runtime(config.seed)
    plan = build_plan(source_root, config)
    free_bytes = shutil.disk_usage(FOLDER / "artifacts").free
    require(free_bytes >= 5 * 1024**3, "At least 5 GiB free space required for formal checkpoints")
    output = Path(tempfile.mkdtemp(prefix="training_readiness_", dir=FOLDER / "artifacts"))
    atomic_json(output / "plan.json", plan)
    print(f"OUTPUT: {output}", flush=True)
    resume = synthetic_resume_probe(output, "cuda")
    configure_runtime(config.seed)
    model, initialization = initialized_baseline(config.seed)
    freeze_batchnorm(model)
    before = state_snapshot(model)
    model = model.cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sources = {}
    all_ids = {}
    for split in ("train", "validation"):
        scope = VideoScope(source_root=source_root, split=split)
        all_ids[split] = select_smoke_ids(scope)
        sources[split] = {i: guarded_source_sha256(scope, i) for i in all_ids[split]}
    dataset = Basketball50Dataset(source_root=source_root, split="train", video_ids=all_ids["train"])
    observed_ids = []
    training_mode(model)
    model.zero_grad(set_to_none=True)
    metrics = Metrics()
    torch.cuda.synchronize()
    started = time.perf_counter()
    first_tensor = first_label = None
    for batch in make_loader(dataset, list(range(4)), config, 0):
        if first_tensor is None:
            first_tensor, first_label = batch["tensor"][:1].clone(), batch["binary_label"][:1].clone()
        x, y = batch["tensor"].cuda(non_blocking=True), batch["binary_label"].cuda()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y, reduction="sum")
        (loss / 4).backward()
        metrics.add(logits, y, loss.item(), batch["original_class"])
        observed_ids.extend(batch["video_id"])
    require(observed_ids == all_ids["train"], "Train loader order mismatch")
    gradients = check_gradients(model)
    check_state(model, before)
    torch.cuda.synchronize()
    baseline_seconds = time.perf_counter() - started
    baseline_memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    validation_data = Basketball50Dataset(source_root=source_root, split="validation", video_ids=all_ids["validation"])
    validation = validation_pass(model, make_loader(validation_data, list(range(4)), config, 0), "cuda", {"requested": False})
    require(validation["count"] == 4, "Validation loader failed")
    check_state(model, before)
    del model, before, logits, loss, x, y
    gc.collect()
    torch.cuda.empty_cache()
    aca = aca_gpu_probe(first_tensor, first_label, initialization) if include_aca else {"status": "not_run"}
    for split in sources:
        scope = VideoScope(source_root=source_root, split=split)
        require(all(guarded_source_sha256(scope, i) == digest for i, digest in sources[split].items()),
                "Source video changed during preflight")
    require(canonical_hash(build_plan(source_root, config)) == canonical_hash(plan), "Plan changed during preflight")
    report = {"status": "passed", "plan_sha256": canonical_hash(plan), "output": str(output),
              "optimizer_steps": 0, "formal_training_started": False,
              "cuda_resume_bitwise_equal": resume["bitwise_equal"], "synthetic_resume": resume,
              "baseline_real_train": {"count": 4, "gradient_abs_sums": gradients,
                   "all_parameters_and_buffers_unchanged": True, "metrics_not_generalization": metrics.result(),
                   "decode_loader_forward_backward_seconds": baseline_seconds, **baseline_memory},
              "baseline_validation": validation, "validation_is_four_video_engineering_only": True,
              "aca_gpu": aca, "video_ids": all_ids, "original_video_sha256": sources,
              "original_sources_unchanged": True, "test_videos_opened": 0,
              "disk_free_bytes_at_preflight": free_bytes,
              "full_train_validation_sources_guarded_stat_checked": plan["original_source_inventory"],
              "full_dataset_decoding_audit_completed": False,
              "formal_baseline_ready": True, "formal_aca_ready": False,
              "remaining_aca_blocker": "93.18M dense candidate differs from paper 26.8M; author settings unresolved"}
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=FOLDER / "baseline_config.json")
    parser.add_argument("--skip-aca", action="store_true")
    args = parser.parse_args()
    config = TrainConfig(**json.loads(args.config.read_text()))
    print(json.dumps(run(args.source_root, config, include_aca=not args.skip_aca), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
