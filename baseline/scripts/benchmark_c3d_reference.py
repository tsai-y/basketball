#!/usr/bin/env python3
"""Short train-only C3D VRAM/throughput and DataLoader feasibility checks."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reference_c3d.aggregate import aggregate_fc6
from reference_c3d.data import C3DVideoDataset, ContinuousClipSampler
from reference_c3d.model import ReferenceC3D


def root_path(value):
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_dataset(benchmark, reference, limit):
    preprocessing = {"resize_height": reference["resize_height"],
                     "resize_width": reference["resize_width"],
                     "crop_size": reference["crop_size"],
                     "crop_mode": reference["eval_crop"],
                     "flip_probability": 0.0,
                     "pixel_preprocessing": reference["pixel_preprocessing"]}
    sampler = ContinuousClipSampler(reference["clip_length"], reference["clip_stride"],
                                    reference["tail_policy"])
    all_train = C3DVideoDataset(root_path(benchmark["manifest_path"]),
                                root_path(benchmark["dataset_path"]), "train",
                                sampler, preprocessing)
    # Keep the existing manifest order and TRAIN split. Never inspect test rows.
    ids = {row["video_id"] for row in all_train.rows[:limit]}
    return C3DVideoDataset(root_path(benchmark["manifest_path"]),
                           root_path(benchmark["dataset_path"]), "train", sampler,
                           preprocessing, ids)


def configure_mode(model, mode):
    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif mode == "frozen":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    elif mode == "partial":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for layer in (model.conv5a, model.conv5b, model.fc6):
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    else:
        raise ValueError(mode)


def one_step(model, head, optimizer, x, label, mode):
    optimizer.zero_grad(set_to_none=True)
    if mode == "frozen":
        with torch.no_grad():
            fc6 = model.extract_fc6(x)
    else:
        fc6 = model.extract_fc6(x)
    descriptor = aggregate_fc6(fc6)
    logits = head(descriptor.unsqueeze(0))
    loss = F.cross_entropy(logits, label)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Nonfinite loss")
    loss.backward()
    optimizer.step()
    return float(loss.item())


def bench_mode(config, reference, mode, output):
    sample = make_dataset(config, reference, 1)[0]
    results = []
    total_memory = torch.cuda.get_device_properties(0).total_memory
    for batch in config["clip_batches"]:
        seed_all(config["seed"])
        torch.cuda.empty_cache()
        model = ReferenceC3D(reference["num_classes"], reference["pool_ceil_mode"]).cuda()
        configure_mode(model, mode)
        model.train(mode != "frozen")
        head = nn.Linear(4096, 2).cuda()
        optimizer = torch.optim.AdamW((p for p in list(model.parameters()) + list(head.parameters())
                                       if p.requires_grad), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        x = sample["clips"][:batch].cuda()
        label = torch.tensor([sample["label"]], dtype=torch.long, device="cuda")
        if len(x) != batch:
            raise RuntimeError("Selected training video has too few clips")
        declared_trainable = sum(p.numel() for p in list(model.parameters()) + list(head.parameters())
                                 if p.requires_grad)
        record = {"mode": mode, "clip_batch_size": batch,
                  "input_shape": list(x.shape), "trainable_parameter_count": declared_trainable,
                  "warmup_iterations": config["warmup_iterations"],
                  "measured_iterations": config["measured_iterations"]}
        try:
            for _ in range(config["warmup_iterations"]):
                one_step(model, head, optimizer, x, label, mode)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            losses = []
            for _ in range(config["measured_iterations"]):
                torch.cuda.synchronize()
                start = time.perf_counter()
                losses.append(one_step(model, head, optimizer, x, label, mode))
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
            active = sum(p.numel() for p in list(model.parameters()) + list(head.parameters())
                         if p.grad is not None)
            seconds = sum(durations) / len(durations)
            record.update({"status": "PASS", "forward_pass": True, "backward_pass": True,
                           "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                           "average_iteration_seconds": seconds,
                           "clips_per_second": batch / seconds,
                           "active_gradient_parameter_count": active,
                           "loss_finite": all(np.isfinite(losses)),
                           "oom": False})
        except torch.cuda.OutOfMemoryError as error:
            record.update({"status": "OOM", "forward_pass": None, "backward_pass": None,
                           "oom": True, "error": str(error).split("\n")[0]})
        results.append(record)
        save_json(output / f"mode_{mode}.json", {"mode": mode, "results": results})
        print(json.dumps(record), flush=True)
        del x, label, optimizer, head, model
        torch.cuda.empty_cache()
        if record["status"] != "PASS":
            break
        if batch == 4 and record["peak_reserved_bytes"] > total_memory * 0.72:
            results.append({"mode": mode, "clip_batch_size": 8, "status": "SKIPPED",
                            "reason": "batch4_peak_reserved_above_72_percent_gpu_total"})
            save_json(output / f"mode_{mode}.json", {"mode": mode, "results": results})
            break


def identity_collate(batch):
    return batch[0]


def bench_loader(config, reference, output):
    dataset = make_dataset(config, reference, config["loader_video_count"])
    model = ReferenceC3D(reference["num_classes"], reference["pool_ceil_mode"]).cuda().eval()
    warmup = dataset[0]["clips"][:config["loader_clip_batch_size"]].cuda()
    with torch.no_grad():
        for _ in range(3):
            model.extract_fc6(warmup)
    torch.cuda.synchronize()
    del warmup
    records = []
    for workers in config["loader_workers"]:
        args = {"dataset": dataset, "batch_size": 1, "shuffle": False,
                "num_workers": workers, "collate_fn": identity_collate,
                "pin_memory": True, "persistent_workers": False}
        if workers:
            args["prefetch_factor"] = 2
        loader = DataLoader(**args)
        start = time.perf_counter()
        iterator = iter(loader)
        setup_seconds = time.perf_counter() - start
        waits, gpu_times, counts = [], [], []
        try:
            for _ in range(config["loader_video_count"]):
                start = time.perf_counter()
                sample = next(iterator)
                waits.append(time.perf_counter() - start)
                if sample["label"] not in (0, 1):
                    raise RuntimeError("Unexpected label")
                # Full video, chunked to bounded VRAM. Include host-to-device copy in GPU step.
                start = time.perf_counter()
                with torch.no_grad():
                    parts = []
                    for chunk in sample["clips"].split(config["loader_clip_batch_size"]):
                        parts.append(model.extract_fc6(chunk.cuda(non_blocking=True)))
                    aggregate_fc6(torch.cat(parts))
                torch.cuda.synchronize()
                gpu_times.append(time.perf_counter() - start)
                counts.append(len(sample["indices"]))
            record = {"status": "PASS", "workers": workers,
                      "video_count": len(waits), "clip_counts": counts,
                      "loader_setup_seconds": setup_seconds,
                      "average_data_wait_seconds": sum(waits) / len(waits),
                      "average_gpu_step_seconds": sum(gpu_times) / len(gpu_times),
                      "data_wait_fraction": sum(waits) / (sum(waits) + sum(gpu_times)),
                      "clip_batch_size_on_gpu": config["loader_clip_batch_size"],
                      "all_video_clips_processed": True,
                      "gpu_utilization_measured": False}
        except Exception as error:
            record = {"status": "FAIL", "workers": workers,
                      "video_count_completed": len(waits),
                      "error": str(error).split("\n")[0]}
        finally:
            del iterator, loader
        records.append(record)
        save_json(output / "dataloader_benchmark.json", {"results": records})
        print(json.dumps(record), flush=True)
    del model


def bench_video_memory(config, reference, output):
    sample = make_dataset(config, reference, 1)[0]
    clips = sample["clips"]
    label = torch.tensor([sample["label"]], dtype=torch.long, device="cuda")
    records = []
    for strategy in ("all_clips_one_batch_full_gradient",
                     "chunked_two_full_gradient",
                     "chunked_two_frozen_backbone"):
        seed_all(config["seed"])
        torch.cuda.empty_cache()
        model = ReferenceC3D(reference["num_classes"], reference["pool_ceil_mode"]).cuda()
        frozen = strategy == "chunked_two_frozen_backbone"
        configure_mode(model, "frozen" if frozen else "full")
        head = nn.Linear(4096, 2).cuda()
        optimizer = torch.optim.AdamW((p for p in list(model.parameters()) + list(head.parameters())
                                       if p.requires_grad), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        one_step(model, head, optimizer, clips[:1].cuda(), label,
                 "frozen" if frozen else "full")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        record = {"strategy": strategy, "clip_count": len(clips),
                  "clip_batch_size": len(clips) if strategy.startswith("all") else 2,
                  "retains_clip_graphs": not frozen}
        features = descriptor = loss = pieces = None
        try:
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            if strategy.startswith("all"):
                features = model.extract_fc6(clips.cuda())
            else:
                pieces = []
                for chunk in clips.split(2):
                    if frozen:
                        with torch.no_grad():
                            pieces.append(model.extract_fc6(chunk.cuda()))
                    else:
                        pieces.append(model.extract_fc6(chunk.cuda()))
                features = torch.cat(pieces)
            descriptor = aggregate_fc6(features)
            loss = F.cross_entropy(head(descriptor.unsqueeze(0)), label)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite video-level loss")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            record.update({"status": "PASS", "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                           "step_seconds": time.perf_counter() - start,
                           "gradient_reached_conv1a": model.conv1a.weight.grad is not None,
                           "gradient_reached_fc6": model.fc6.weight.grad is not None,
                           "descriptor_shape": list(descriptor.shape)})
        except torch.cuda.OutOfMemoryError as error:
            record.update({"status": "OOM", "error": str(error).split("\n")[0]})
        records.append(record)
        save_json(output / "video_memory_benchmark.json", {"results": records})
        print(json.dumps(record), flush=True)
        del optimizer, head, model, features, descriptor, loss, pieces
        torch.cuda.empty_cache()
    descriptor_storage = 10279 * 4096 * 4
    result = {"results": records,
              "frozen_float32_descriptor_storage_estimate_bytes_for_10279_videos": descriptor_storage,
              "estimate_excludes_ids_metadata_and_filesystem_overhead": True}
    save_json(output / "video_memory_benchmark.json", result)


def summarize(config, output):
    all_results = []
    modes = {}
    for mode in ("full", "frozen", "partial"):
        source = output / f"mode_{mode}.json"
        if source.exists():
            rows = json.loads(source.read_text())["results"]
            all_results.extend(rows)
            passing = [r for r in rows if r["status"] == "PASS"]
            modes[mode] = {"trainable_parameter_count": passing[0]["trainable_parameter_count"] if passing else None,
                           "active_gradient_parameter_count": passing[0]["active_gradient_parameter_count"] if passing else None,
                           "maximum_tested_safe_clip_batch": max((r["clip_batch_size"] for r in passing), default=None)}
    if not all_results:
        raise RuntimeError("No mode benchmark results")
    save_json(output / "vram_benchmark.json", {"results": all_results})
    save_json(output / "throughput_benchmark.json", {"results": [
        {k: r.get(k) for k in ("mode", "clip_batch_size", "status", "average_iteration_seconds",
                              "clips_per_second", "warmup_iterations", "measured_iterations")}
        for r in all_results]})
    save_json(output / "training_mode_comparison.json", {"modes": modes})
    print(json.dumps({"modes": modes}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c3d_reference_benchmark.yaml")
    parser.add_argument("--stage", choices=("mode", "loader", "video_memory", "summarize", "metadata"), required=True)
    parser.add_argument("--mode", choices=("full", "frozen", "partial"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    reference = yaml.safe_load(root_path(config["reference_config_path"]).read_text())
    output = root_path(config["output_dir"]).resolve()
    if output != (ROOT / "runs/full_video_c3d_reference_v1/feasibility").resolve():
        raise RuntimeError("Feasibility outputs must be separate from Experiment A and 3B")
    if not torch.cuda.is_available() and args.stage != "summarize":
        raise RuntimeError("CUDA required for GPU feasibility benchmark")
    seed_all(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    meta = {"config": config, "reference_config": reference,
            "manifest_sha256": hashlib.sha256(root_path(config["manifest_path"]).read_bytes()).hexdigest(),
            "split_summary_sha256": hashlib.sha256(root_path(config["split_summary_path"]).read_bytes()).hexdigest(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if args.stage != "summarize":
        save_json(output / "benchmark_config.json", meta)
    if args.stage == "mode":
        if args.mode is None:
            raise ValueError("--mode required")
        bench_mode(config, reference, args.mode, output)
    elif args.stage == "loader":
        bench_loader(config, reference, output)
    elif args.stage == "video_memory":
        bench_video_memory(config, reference, output)
    elif args.stage == "metadata":
        print(json.dumps({"status": "PASS", "gpu": meta["gpu"]}))
    else:
        summarize(config, output)


if __name__ == "__main__":
    main()
