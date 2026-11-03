"""Bounded four-arm engineering checks, never full-data training or test."""
import copy
from dataclasses import asdict
import fcntl
import gc
from pathlib import Path
import signal
import tempfile
import time
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from development.aca_net_v1.baseline_preflight import check_state, freeze_batchnorm
from development.aca_net_v1.data import Basketball50Dataset, VideoScope, guarded_source_sha256, require
from development.aca_net_v1.modules import ACAModule, LSTA, TSCI, ReconstructionConfig
from development.aca_net_v1.readiness import ToyDataset, nested_equal
from development.aca_net_v1.train import (
    Metrics, atomic_json, canonical_hash, configure_runtime, initial_progress, make_loader,
)
from . import engine
from .experiment import (
    ArmConfig, FOLDER, PREFLIGHT_SCHEMA, SEEDS, VARIANTS, arm_plan, assert_idle_gpu,
    campaign_plan, component_digests, initialized, load_imagenet,
)


class TinyVideoData(ToyDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        row["tensor"] = (torch.arange(144).float().reshape(3, 3, 4, 4)*.01 + index*.03).sin()
        return row


class TinyVideoModel(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.stem = nn.Conv3d(3, 4, 1)
        cfg = ReconstructionConfig(frames=3, reduction=2)
        constructors = {"baseline": nn.Identity, "lsta": lambda: LSTA(4, cfg),
                        "tsci": TSCI, "aca": lambda: ACAModule(4, cfg)}
        self.aca = constructors[variant]()
        self.dropout = nn.Dropout(.2)
        self.fc = nn.Linear(4, 2)
        freeze_batchnorm(self)

    def forward(self, x):
        return self.fc(self.dropout(self.aca(self.stem(x)).mean(dim=(2, 3, 4))))


def synthetic_resume_probe(parent, variant, *, seed=20261025, device="cpu"):
    """All variants use the common real fit loop; tail3 and active dropout."""
    cfg = ArmConfig(variant=variant, seed=seed, epochs=2, patience=2, workers=0, checkpoint_every=1)
    plan = arm_plan({"train_count": 19, "validation_count": 5, "toy": True}, cfg, synthetic=True)
    train, validation = TinyVideoData(19), TinyVideoData(5)
    a, b = parent / "continuous", parent / "resumed"
    a.mkdir(parents=True); b.mkdir(parents=True)

    def start():
        configure_runtime(seed, cuda=device == "cuda")
        model = TinyVideoModel(variant).to(device)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
        return model, optimizer

    model, optimizer = start()
    engine.fit(model, optimizer, train, validation, cfg, plan, a, initial_progress(), device=device)
    expected = torch.load(a / "last.pt", map_location="cpu", weights_only=True)
    model, optimizer = start()
    original_step = optimizer.step
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        result = original_step(*args, **kwargs)
        calls += 1
        if calls == 1:
            signal.raise_signal(signal.SIGINT)
        return result

    with patch.object(optimizer, "step", side_effect=interrupt):
        stopped = engine.fit(model, optimizer, train, validation, cfg, plan, b, initial_progress(), device=device)
    require(stopped["status"] == "stopped_at_optimizer_boundary" and stopped["cursor"] == 16,
            "Incorrect synthetic optimizer boundary")
    model, optimizer = start()
    progress = engine.load_checkpoint(b / "last.pt", model, optimizer, plan)
    engine.fit(model, optimizer, train, validation, cfg, plan, b, progress, device=device)
    actual = torch.load(b / "last.pt", map_location="cpu", weights_only=True)
    require(all(nested_equal(expected[k], actual[k]) for k in ("model", "optimizer", "rng", "progress")),
            "Resumed model/Adam/RNG/history differ")
    require(actual["progress"]["optimizer_steps"] == 4, "Synthetic tail update omitted")
    return {"bitwise_equal": True, "variant": variant, "device": device,
            "synthetic_updates_executed": 8, "tail_batch": 3, "dropout_enabled": True}


class CachedData(Dataset):
    """Only a bounded, already authorized readiness cache; never formal data."""
    def __init__(self, items, repeats=1):
        self.items = items * repeats

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return {k: item[k] for k in ("tensor", "binary_label", "video_id", "original_class")}


def frozen_state(model):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: t.detach().cpu().clone() for n, t in model.state_dict().items() if n not in trainable}


def gradients(model, variant):
    wanted = ["backbone", "head"]
    if variant in ("lsta", "aca"):
        wanted += ["gate_value", "gate_control", "gate_output", "short_term", "kernel_generator"]
    if variant in ("tsci", "aca"):
        wanted += ["tsci"]
    sums = dict.fromkeys(wanted, 0.)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            require(p.grad is None, "Frozen BN acquired gradient")
            continue
        require(p.grad is not None and torch.isfinite(p.grad).all().item(), f"Invalid gradient: {name}")
        key = "head" if name.startswith("fc.") else "backbone"
        if ".aca." in name:
            key = next((b for b in wanted if f".{b}." in name), "tsci")
        sums[key] += p.grad.abs().sum().item()
    require(all(value > 0 for value in sums.values()), f"A branch has no gradient: {sums}")
    return sums


def one_arm(output, campaign, prepared, state, cache, variant, seed):
    assert_idle_gpu()
    configure_runtime(seed)
    cfg = ArmConfig(variant=variant, seed=seed, workers=0)
    resume = synthetic_resume_probe(output / "toy", variant, seed=seed, device="cuda")
    gc.collect(); torch.cuda.empty_cache()
    configure_runtime(seed)
    model, transfer = initialized(variant, seed, state)
    before = component_digests(model)
    require(before == prepared["seed_initialization"][str(seed)][variant]["digests"],
            "GPU model initialization differs from frozen CPU preparation")
    frozen = frozen_state(model)
    model = model.cuda()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started = time.perf_counter()
    observed = {}
    data = CachedData(cache["train"], repeats=4)
    iterator = iter(DataLoader(data, batch_size=1, shuffle=False, num_workers=0))
    ids = [item["video_id"] for item in data.items]
    norm = engine.optimizer_update(model, optimizer, iterator, ids, 16, cfg, Metrics(), device="cuda",
                                   observe=lambda m: observed.update(gradients(m, variant)))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    check_state(model, frozen)
    after = component_digests(model)
    require(all(before[key] != after[key] for key in before), "A trainable component failed to update")
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        logits = [model(item["tensor"].unsqueeze(0).cuda()).detach().cpu() for item in cache["validation"]]
    require(all(tuple(x.shape) == (1, 2) and torch.isfinite(x).all().item() for x in logits),
            "Invalid engineering validation output")
    check_state(model, frozen)
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    checkpoint = output / "real_engineering_only.pt"
    torch.save({"schema": "aca_four_arm_real_engineering_v1", "engineering_only": True,
                "checkpoint_may_initialize_formal_training": False, "variant": variant, "seed": seed,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "optimizer_steps": 1}, checkpoint)
    del model, optimizer, data, iterator
    gc.collect(); torch.cuda.empty_cache()
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model, _ = initialized(variant, seed, state)
    model.load_state_dict(loaded["model"], strict=True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    optimizer.load_state_dict(loaded["optimizer"])
    require(nested_equal(optimizer.state_dict(), loaded["optimizer"]), "Adam reload mismatch")
    del optimizer, loaded
    model = model.cuda().eval()
    with torch.no_grad():
        restored = model(cache["validation"][0]["tensor"].unsqueeze(0).cuda()).cpu()
    torch.testing.assert_close(restored, logits[0], rtol=0, atol=0)
    del model, restored
    gc.collect(); torch.cuda.empty_cache()
    return {"status": "passed", "variant": variant, "seed": seed, "transfer": transfer,
            "initialization_matches_CPU_preparation": True, "initial_component_sha256": before,
            "post_engineering_component_sha256": after, "all_components_changed": True,
            "all_frozen_BN_state_unchanged": True, "branch_gradient_abs_sums": observed,
            "grad_norm_before_clip": norm.item(), "real_engineering_optimizer_updates": 1,
            "unique_train_videos": 4, "train_exposures": 16,
            "unique_validation_videos_for_finite_output_only": 4,
            "new_validation_accuracy_computed": False,
            "checkpoint_roundtrip_bitwise_equal": True, "Adam_reload_equal": True,
            "synthetic_resume_bitwise_equal": resume["bitwise_equal"], "synthetic_resume": resume,
            "peak_allocated_bytes_including_Adam": peak, "peak_reserved_bytes": reserved,
            "engineering_update_seconds": elapsed,
            "timing_scope": "one cached 16-exposure update incl. gradient auditing; not an epoch estimate"}


def run(source_root, preparation, *, seed=20261025):
    require(type(seed) is int and seed in SEEDS, "Unknown seed")
    require(torch.cuda.is_available(), "CUDA is required; no CPU fallback")
    assert_idle_gpu()
    campaign = campaign_plan(source_root)
    preparation = Path(preparation)
    import json
    prepared = json.loads((preparation / "report.json").read_text())
    stored = json.loads((preparation / "plan.json").read_text())
    require(prepared["status"] == "cpu_preparation_passed" and
            prepared["campaign_sha256"] == canonical_hash(campaign) == canonical_hash(stored),
            "Preparation is stale or incompatible")
    parent = FOLDER / "artifacts"; parent.mkdir(exist_ok=True)
    with (parent / "gpu_preflight.lock").open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure_runtime(seed)
        output = Path(tempfile.mkdtemp(prefix=f"gpu_readiness_seed{seed}_", dir=parent))
        print(f"OUTPUT: {output}", flush=True)
        atomic_json(output / "plan.json", campaign)
        progress = {"status": "running", "seed": seed, "active_variant": None, "arms": {},
                    "formal_training_started": False}
        atomic_json(output / "state.json", progress)
        try:
            scopes = {s: VideoScope(source_root=source_root, split=s) for s in ("train", "validation")}
            ids = campaign["selected_preflight_ids"]
            sources = {s: {i: guarded_source_sha256(scopes[s], i) for i in ids[s]} for s in scopes}
            cache = {}
            loader_cfg = ArmConfig(seed=seed)
            for split in scopes:
                dataset = Basketball50Dataset(source_root=source_root, split=split, video_ids=ids[split])
                cache[split] = []
                for batch in make_loader(dataset, list(range(4)), loader_cfg, 0):
                    cache[split].append({"tensor": batch["tensor"][0],
                        "binary_label": int(batch["binary_label"][0]),
                        "video_id": batch["video_id"][0], "original_class": batch["original_class"][0]})
                require([item["video_id"] for item in cache[split]] == ids[split],
                        "Production worker loader order differs from fixed metadata selection")
            input_sha = {s: {item["video_id"]: __import__('hashlib').sha256(
                memoryview(item["tensor"].numpy()).cast('B')).hexdigest() for item in cache[s]} for s in scopes}
            state, _ = load_imagenet()
            for variant in VARIANTS:
                progress["active_variant"] = variant
                atomic_json(output / "state.json", progress)
                folder = output / variant; folder.mkdir()
                print(f"GPU preflight variant={variant} seed={seed}", flush=True)
                row = one_arm(folder, campaign, prepared, state, cache, variant, seed)
                progress["arms"][variant] = row
                atomic_json(folder / "report.json", row)
                atomic_json(output / "state.json", progress)
                print(f"GPU {variant}: passed, peak={row['peak_allocated_bytes_including_Adam']/1024**3:.2f}GiB", flush=True)
            require(all(guarded_source_sha256(scopes[s], i) == h for s in sources for i, h in sources[s].items()),
                    "A selected original source changed")
            require(canonical_hash(campaign_plan(source_root)) == canonical_hash(campaign), "Campaign/code changed")
            report = {"schema": PREFLIGHT_SCHEMA, "status": "passed", "seed": seed,
                      "campaign_sha256": canonical_hash(campaign), "preparation": str(preparation),
                      "arms": progress["arms"], "engineering_only": True,
                      "checkpoint_may_initialize_formal_training": False, "formal_training_started": False,
                      "real_engineering_optimizer_updates": 4, "synthetic_optimizer_updates": 32,
                      "raw_video_decodes": {"train": 4, "validation": 4, "test": 0},
                      "production_data_loader_workers_verified": loader_cfg.workers,
                      "guarded_source_SHA_reads": 16, "input_tensor_sha256": input_sha,
                      "source_sha256": sources, "original_sources_and_protocol_unchanged": True,
                      "validation_metrics_not_computed": True, "other_seeds_require_own_GPU_readiness": True}
            atomic_json(output / "report.json", report)
            progress.update(status="completed", active_variant=None)
            atomic_json(output / "state.json", progress)
            return {"output": str(output), "status": "passed", "seed": seed, "arms": list(VARIANTS),
                    "formal_training_started": False}
        except BaseException as error:
            progress.update(status="failed", error=str(error))
            atomic_json(output / "state.json", progress)
            atomic_json(output / "failure.json", {"error": str(error), "automatic_retry": False,
                                                   "formal_training_started": False})
            raise
