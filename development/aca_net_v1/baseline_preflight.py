"""Train-only four-video ResNet50 engineering overfit and CUDA measurement.

This is a bounded engineering check. Its checkpoint must never initialize a
formal experiment. --dry-run has no optimizer, parameter updates, or file writes.
"""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import time

import cv2
import torch
import torchvision
from torch import nn
from torchvision.models import resnet50

from .data import (Basketball50Dataset, POLICY, VideoScope, file_sha256,
                   guarded_source_sha256, require, select_smoke_ids)
from .model import VideoResNet50
from .pipeline_smoke import write_json
from .weights import initialized_baseline

FOLDER = Path(__file__).parent


def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)


def training_mode(model):
    model.train()
    freeze_batchnorm(model)


def state_snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def frozen_snapshot(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in state_snapshot(model).items() if k not in names}


def check_state(model, snapshot):
    current = model.state_dict()
    require(all(torch.equal(current[k].detach().cpu(), v) for k, v in snapshot.items()),
            "Expected unchanged state was modified")


def accumulated_backward(model, x, y, micro_batch):
    """Summed CE divided by the actual number of videos, including last microbatch."""
    loss_sum = x.new_zeros(())
    for start in range(0, len(x), micro_batch):
        stop = min(start + micro_batch, len(x))
        logits = model(x[start:stop])
        loss = nn.functional.cross_entropy(logits, y[start:stop], reduction="sum")
        (loss / len(x)).backward()
        loss_sum += loss.detach()
    return loss_sum / len(x)


def check_gradients(model):
    sums = {"backbone": 0.0, "classifier": 0.0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            require(p.grad is None, "Frozen parameter unexpectedly has a gradient")
            continue
        require(p.grad is not None and torch.isfinite(p.grad).all().item(), f"Invalid gradient: {name}")
        key = "classifier" if name.startswith("fc.") else "backbone"
        sums[key] += p.grad.detach().abs().sum().item()
    require(all(v > 0 for v in sums.values()), "Backbone/head must both receive nonzero gradients")
    return sums


def evaluate(model, x, y, micro_batch):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(x[i:i + micro_batch]) for i in range(0, len(x), micro_batch)])
        loss = nn.functional.cross_entropy(logits, y).item()
    require(torch.isfinite(logits).all().item(), "Nonfinite eval logits")
    return {"loss": loss, "accuracy": (logits.argmax(-1) == y).float().mean().item(),
            "probability_make": logits.softmax(-1)[:, 1].cpu().tolist(),
            "logits": logits.detach().cpu()}


def run(source_root, *, dry_run, steps=40, micro_batch=1, seed=20261025):
    require(type(steps) is int and 1 <= steps <= 80, "Bounded probe requires 1..80 updates")
    require(micro_batch in (1, 2) and type(micro_batch) is int, "micro_batch must be 1 or 2")
    require(type(seed) is int and 0 <= seed < 2**32, "Invalid seed")
    require(torch.cuda.is_available(), "CUDA required; no CPU fallback for this probe")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8"), "Unexpected CUBLAS workspace")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, initialization = initialized_baseline(seed)
    scope = VideoScope(source_root=source_root, split="train")
    ids = select_smoke_ids(scope)
    dataset = Basketball50Dataset(source_root=source_root, split="train", video_ids=ids)
    started = time.perf_counter()
    sources_before = {i: guarded_source_sha256(scope, i) for i in ids}
    items = [dataset[i] for i in range(len(dataset))]
    decode_seconds = time.perf_counter() - started
    host_x = torch.stack([item["tensor"] for item in items])
    require(tuple(host_x.shape) == (4, 3, 50, 120, 160), "Expected four full-video inputs")
    labels = [item["binary_label"] for item in items]
    require(labels == [0, 1, 0, 1], "Expected fixed balanced engineering subset")
    for i in ids:
        require(guarded_source_sha256(scope, i) == sources_before[i], "Original source changed")
    model = model.cuda().eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    x = host_x.cuda()
    y = torch.tensor(labels, dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    transfer_seconds = time.perf_counter() - started

    # Match the reconstructed baseline against native frame-wise torchvision.
    reference = resnet50(weights=None)
    raw = torch.load(initialization["checkpoint"], map_location="cpu", weights_only=True)
    reference.load_state_dict(raw, strict=True)
    reference.fc = copy.deepcopy(model.fc).cpu()
    reference = reference.cuda().eval()
    with torch.no_grad():
        expected = reference(x[:1].permute(0, 2, 1, 3, 4).reshape(50, 3, 120, 160)).mean(0, keepdim=True)
        observed = model(x[:1])
    torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)
    parity_error = (observed - expected).abs().max().item()
    del reference, raw, expected, observed
    freeze_batchnorm(model)
    before = state_snapshot(model)
    frozen = frozen_snapshot(model)
    initial = evaluate(model, x, y, micro_batch)
    training_mode(model)
    model.zero_grad(set_to_none=True)
    accumulated_backward(model, x, y, micro_batch)
    gradients = check_gradients(model)
    check_state(model, before)
    model.zero_grad(set_to_none=True)
    setup = {
        "experiment": "resnet50_four_video_engineering_preflight_v1", "seed": seed,
        "initialization": initialization, "pretrained_weights_loaded": True,
        "model_variant": "baseline", "num_classes": 2,
        "model_config": asdict(model.config), "policy": POLICY,
        "input_hashes": scope.hashes, "train_video_ids": ids, "labels": labels,
        "source_sha256": sources_before,
        "tensor_sha256": {name: hashlib.sha256(item["tensor"].numpy().tobytes()).hexdigest()
                          for name, item in zip(ids, items)},
        "input_shape": list(x.shape), "decoded_once_then_cached_on_gpu": True,
        "decode_and_source_hash_seconds": decode_seconds, "host_to_device_seconds": transfer_seconds,
        "optimizer": "Adam", "lr": 1e-4, "weight_decay": 0.0, "grad_clip_norm": 1.0,
        "micro_batch_videos": micro_batch, "effective_batch_videos": len(x),
        "maximum_optimizer_updates": steps, "all_batchnorm_affine_and_buffers_frozen": True,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "python": platform.python_version(),
        "cudnn": torch.backends.cudnn.version(),
        "gpu_context_before_optimizer": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"],
            capture_output=True, text=True, check=True).stdout.strip(),
        "cuda": torch.version.cuda, "precision": "float32_no_amp_no_tf32",
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "validation_videos_opened": 0, "test_videos_opened": 0,
        "full_dataset_training_started": False,
        "source_code_sha256": {p.name: file_sha256(p) for p in FOLDER.glob("*.py")},
        "success_criterion": "final four-video eval accuracy=1 AND loss<0.05 AND loss<0.2*initial",
    }
    if dry_run:
        return {"status": "dry_run_passed", "optimizer_steps": 0, "files_written": False,
                "gradient_abs_sums": gradients, "native_parity_max_abs_error": parity_error,
                "all_parameters_and_buffers_unchanged": True, "config": setup}

    parent = FOLDER / "artifacts"
    output = Path(tempfile.mkdtemp(prefix=f"baseline_preflight_seed{seed}_", dir=parent))
    print(f"OUTPUT: {output}", flush=True)
    write_json(output / "config.json", {**setup, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    snapshot_dir = output / "source_snapshot"
    snapshot_dir.mkdir()
    for p in FOLDER.glob("*.py"):
        shutil.copy2(p, snapshot_dir / p.name)
    shutil.copy2(FOLDER / "imagenet_v1.json", snapshot_dir / "imagenet_v1.json")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    history = []
    for step in range(1, steps + 1):
        training_mode(model)
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = accumulated_backward(model, x, y, micro_batch)
        norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.,
                                       error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        row = {"step": step, "train_loss_before_update": loss.item(),
               "seconds": seconds, "unclipped_grad_norm": norm.item()}
        history.append(row)
        if step == 1 or step % 5 == 0 or step == steps:
            print(f'step {step:02d}: loss={row["train_loss_before_update"]:.6f}, seconds={seconds:.3f}', flush=True)
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    final = evaluate(model, x, y, micro_batch)
    check_state(model, frozen)
    changed = {key: not torch.equal(model.state_dict()[key].detach().cpu(), before[key])
               for key in ("conv1.weight", "layer4.2.conv3.weight", "fc.weight")}
    require(all(changed.values()), "Stem, last stage and task head must all update")
    for i in ids:
        require(guarded_source_sha256(scope, i) == sources_before[i], "Source changed after training")
    require(scope.hashes == VideoScope(source_root=source_root, split="train").hashes,
            "Manifest/split changed")
    checkpoint = output / "engineering_only.pt"
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_steps": steps, "config": setup,
                "engineering_only": True, "prohibited_as_formal_training_initialization": True}, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = VideoResNet50(num_classes=2).cuda().eval()
    restored.load_state_dict(loaded["model"], strict=True)
    restored_eval = evaluate(restored, x, y, micro_batch)
    torch.testing.assert_close(restored_eval["logits"], final["logits"], rtol=0, atol=0)
    durations = [r["seconds"] for r in history[2:]] or [r["seconds"] for r in history]
    median = statistics.median(durations)
    passed = final["accuracy"] == 1.0 and final["loss"] < .05 and final["loss"] < .2 * initial["loss"]
    summary = {
        "status": "passed" if passed else "completed_success_criterion_not_met",
        "output": str(output), "optimizer_steps": steps,
        "initial_subset": {k: v for k, v in initial.items() if k != "logits"},
        "final_subset": {k: v for k, v in final.items() if k != "logits"},
        "gradient_abs_sums_before_updates": gradients, "changed_parameters": changed,
        "all_frozen_bn_parameters_and_buffers_unchanged": True,
        "native_torchvision_parity_max_abs_error": parity_error,
        "checkpoint_roundtrip_logits_bitwise_equal": True,
        "checkpoint_sha256": file_sha256(checkpoint),
        "gpu_training_peak_allocated_bytes": allocated,
        "gpu_training_peak_reserved_bytes": reserved,
        "effective_step_seconds_median_excluding_first_two": median,
        "cached_training_videos_per_second": len(x) / median,
        "timing_scope": "GPU-resident four-video inputs; forward/backward/gradient clipping/Adam; excludes decode and checkpoint IO",
        "formal_epoch_time_estimate_available": False,
        "pretrained_weights_and_original_videos_unchanged": True,
        "validation_videos_opened": 0, "test_videos_opened": 0,
        "full_dataset_training_started": False,
        "checkpoint_may_initialize_formal_training": False,
        "pretrained_weights_loaded": True,
    }
    require(file_sha256(Path(initialization["checkpoint"])) == initialization["identity"]["sha256"],
            "Pretrained checkpoint changed")
    write_json(output / "history.json", history)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261025)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, dry_run=args.dry_run, steps=args.steps,
                         micro_batch=args.micro_batch, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
