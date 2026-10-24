"""Train-only two-video engineering check, not full-data training or model selection."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import mkdtemp
import time

import torch
from torch import nn

from development.data_pipeline.multiwindow_v1 import BasketballThreeWindowDataset
from development.data_pipeline.pipeline import load_gate_config
from development.partial_finetune_v1.initialization import ROOT, file_sha, load_initialized_model
from development.temporal_baseline_v1.train_temporal import CACHE, CACHE_RUN_ID, require


FOLDER = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Forward/parity/backward only; no updates or files")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.threads <= 0 or not 0 <= args.seed < 2**32:
        parser.error("Expected positive steps/threads and a seed in [0, 2**32)")
    return args


def frozen_snapshot(model):
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name not in names}


def check_frozen_state(model, snapshot):
    current = model.state_dict()
    for name, value in snapshot.items():
        require(torch.equal(current[name].detach().cpu(), value), f"Frozen state changed: {name}")


def check_gradients(model):
    totals = {"backbone": 0., "head": 0.}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(parameter.grad is not None, f"Missing trainable gradient: {name}")
            require(torch.isfinite(parameter.grad).all().item(), f"Non-finite gradient: {name}")
            totals[name.split(".")[0]] += float(parameter.grad.detach().abs().sum())
        else:
            require(parameter.grad is None, f"Frozen parameter has gradient: {name}")
    require(all(value > 0 for value in totals.values()), "Tail/head must both receive nonzero gradients")
    return totals


def reference_features(items):
    protocol = torch.load(CACHE / "run.pt", map_location="cpu", weights_only=True)
    require(hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == CACHE_RUN_ID,
            "Reference cache protocol changed")
    features = []
    for item in items:
        filename = hashlib.sha256(item["video_id"].encode()).hexdigest() + ".pt"
        record = torch.load(CACHE / "train" / filename, map_location="cpu", weights_only=True)
        require(record["split"] == "train" and record["video_id"] == item["video_id"],
                "Reference sample identity mismatch")
        require(record["run_id"] == CACHE_RUN_ID and record["binary_label"] == item["binary_label"],
                "Reference label/cache mismatch")
        require(record["windows"] == item["windows"], "Reference windows changed")
        x = record["features"]
        require(tuple(x.shape) == (3, 8, 1536) and x.dtype == torch.float32
                and torch.isfinite(x).all().item(), "Invalid reference feature tensor")
        require(hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() == record["features_sha256"],
                "Reference feature checksum mismatch")
        features.append(x)
    return torch.stack(features), protocol


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8"),
            "Expected a deterministic CUBLAS_WORKSPACE_CONFIG")
    # Fail before decoding/creating outputs, not silently fall back to CPU for real RGB work.
    require(torch.cuda.is_available(), "GPU unavailable; run the preflight where CUDA devices are exposed")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, initialization = load_initialized_model()
    config = load_gate_config()
    dataset = BasketballThreeWindowDataset(config, split="train", epoch=0)
    indices = [next(i for i, row in enumerate(dataset.base.rows)
                    if int(row["binary_label"]) == label) for label in (0, 1)]
    started = time.perf_counter()
    items = [dataset[i] for i in indices]
    decode_seconds = time.perf_counter() - started
    clips = torch.stack([item["tensor"] for item in items])
    labels = torch.tensor([item["binary_label"] for item in items], dtype=torch.float32, device="cuda")
    require(tuple(clips.shape) == (2, 3, 3, 64, 192, 256), "Unexpected real RGB input shape")
    require(torch.isfinite(clips).all().item() and clips.min() >= 0 and clips.max() <= 1,
            "RGB inputs must be finite and in [0,1]")
    cached, protocol = reference_features(items)
    require(protocol["gate_config_sha256"] == config.sha256, "Gate configuration changed")
    require(protocol["input_identities"] == config.raw["identities"], "Gate input identities changed")
    # Checkpoint must start with exactly the same head and frozen features as the baseline.
    model = model.cuda().eval()
    clips = clips.cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        features = model.forward_features(clips)
        expected = cached.cuda()
        torch.testing.assert_close(features, expected, rtol=1e-4, atol=1e-5)
        logits = model.head(features)
        cached_logits = model.head(expected)
        torch.testing.assert_close(logits, cached_logits, rtol=1e-4, atol=1e-5)
        initial_loss = float(nn.functional.binary_cross_entropy_with_logits(logits, labels))
        feature_max_error = float((features - expected).abs().max())
    del features, expected, cached_logits, logits
    frozen = frozen_snapshot(model)
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    groups = model.optimizer_groups()
    optimizer_ids = [id(p) for group in groups for p in group["params"]]
    require(len(optimizer_ids) == len(set(optimizer_ids)), "Duplicated optimizer parameters")
    require(set(optimizer_ids) == {id(p) for p in model.parameters() if p.requires_grad},
            "Optimizer groups do not exactly match trainable parameters")
    setup = {
        "experiment": "partial_mixnet_i3d_rgb_engineering_preflight_v1",
        "seed": args.seed, "initialization": initialization,
        "unfrozen_backbone": ["blocks.5 (excluding BatchNorm)", "conv_head"],
        "all_batchnorm_frozen": True, "input_shape": list(clips.shape),
        "window_policy": "three_window_coverage_v1", "window_weights": "equal",
        "train_video_ids": [item["video_id"] for item in items], "labels": labels.cpu().tolist(),
        "optimizer": "AdamW", "weight_decay": .0001, "max_grad_norm": 1.,
        "parameter_groups": [{"name": g["name"], "lr": g["lr"],
                              "parameters": sum(p.numel() for p in g["params"])} for g in groups],
        "device": torch.cuda.get_device_name(), "precision": "float32_no_amp_no_tf32",
        "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
        "deterministic_algorithms": True, "cpu_threads": args.threads,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "gate_config_sha256": config.sha256,
        "input_identities": config.raw["identities"],
        "quality_weighting_used": False, "balanced_sampling_used": False,
        "test_used": False, "validation_used": False, "full_dataset_training_started": False,
        "source_sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in sorted(FOLDER.glob("*.py"))},
    }
    if args.dry_run:
        model.train()
        loss = nn.functional.binary_cross_entropy_with_logits(model(clips), labels)
        require(torch.isfinite(loss).item(), "Non-finite dry-run loss")
        loss.backward()
        gradients = check_gradients(model)
        check_frozen_state(model, frozen)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                require(torch.equal(parameter.detach().cpu(), before[name]), "Dry-run weights changed")
        print(json.dumps({"status": "PASS", "dry_run": True, "optimizer_steps": 0,
                          "output_files_written": False, "feature_max_abs_error": feature_max_error,
                          "gradient_abs_sums": gradients, "config": setup}, indent=2), flush=True)
        return

    parent = FOLDER / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=f"preflight_seed{args.seed}_", dir=parent))
    with (output / "config.json").open("x", encoding="utf-8") as stream:
        json.dump(setup, stream, indent=2)
    print("Output:", output, flush=True)
    print("RGB/cached baseline parity: PASS; max feature error:", feature_max_error, flush=True)
    print("Initial eval loss:", initial_loss, flush=True)
    optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(clips)
        require(tuple(logits.shape) == (2,), "Unexpected training output")
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        require(torch.isfinite(loss).item(), "Non-finite training loss")
        loss.backward()
        gradients = check_gradients(model)
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.,
                                 error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        history.append((step, float(loss.detach()), elapsed))
        print(f"step {step:02d}: loss={float(loss.detach()):.6f}, seconds={elapsed:.3f}", flush=True)
    model.eval()
    with torch.no_grad():
        logits = model(clips)
        final_loss = float(nn.functional.binary_cross_entropy_with_logits(logits, labels))
        probabilities = logits.sigmoid().cpu().tolist()
    check_frozen_state(model, frozen)
    changed = {name: not torch.equal(parameter.detach().cpu(), before[name])
               for name, parameter in model.named_parameters() if parameter.requires_grad}
    require(any(value for name, value in changed.items() if name.startswith("backbone.blocks.5.")),
            "Last stage weights did not update")
    require(any(value for name, value in changed.items() if name.startswith("backbone.conv_head.")),
            "Backbone head weights did not update")
    require(any(value for name, value in changed.items() if name.startswith("head.")), "Temporal head did not update")
    require(final_loss < initial_loss, "Two-video eval loss did not decrease; do not start full training")
    summary = {"status": "PASS", "optimizer_steps": args.steps,
               "initial_two_train_video_eval_loss": initial_loss,
               "final_two_train_video_eval_loss": final_loss, "final_p_make": probabilities,
               "frozen_parameters_and_buffers_unchanged": True,
               "tail_and_head_updated": True, "feature_max_abs_error_before_updates": feature_max_error,
               "last_gradient_abs_sums_before_clipping": gradients,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
               "two_video_decode_seconds": decode_seconds,
               "median_two_video_step_seconds": sorted(r[2] for r in history)[len(history) // 2],
               "full_dataset_training_started": False,
               "validation_used": False, "test_used": False,
               "meaning": "Engineering learning check on two train videos only; not validation accuracy"}
    with (output / "history.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "train_loss", "seconds"])
        writer.writerows(history)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    # Engineering checkpoint only. Never use this two-video-fitted state to initialize a formal run.
    with (output / "engineering_only.pt").open("xb") as stream:
        torch.save({"model_state_dict": {n: v.detach().cpu() for n, v in model.state_dict().items()},
                    "config": setup, "summary": summary, "formal_initialization_allowed": False}, stream)
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()
