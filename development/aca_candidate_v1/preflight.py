"""Bounded candidate GPU preflight, refused while baseline is unfinished.

Only --preflight invokes this. One engineering update on four real train clips;
its checkpoint is forbidden as a formal initialization. Test is never opened.
"""
import copy
import gc
from pathlib import Path
import signal
import tempfile
import time
from unittest.mock import patch

import torch
from torch import nn

from development.aca_net_v1.baseline_preflight import (accumulated_backward, check_gradients,
    check_state, frozen_snapshot, state_snapshot, training_mode)
from development.aca_net_v1.data import (Basketball50Dataset, VideoScope, guarded_source_sha256,
                                       require, select_smoke_ids)
from development.aca_net_v1.readiness import ToyDataset, nested_equal, toy_model
from development.aca_net_v1.train import (Metrics, atomic_json, canonical_hash, configure_runtime,
                                        initial_progress, make_loader)
from . import engine
from .experiment import (CandidateConfig, FOLDER, PREFLIGHT_SCHEMA, assert_launch_prerequisites,
                         build_plan, initialized_candidate, snapshot_sources)


def synthetic_resume_probe(parent, device="cpu", *, model_factory=toy_model, dataset_factory=ToyDataset):
    config = CandidateConfig(epochs=2, patience=2, workers=0, checkpoint_every=1)
    plan = {"config": vars(config), "train_count": 19, "validation_count": 5,
            "engineering_only_synthetic_toy": True}
    a, b = parent / "continuous", parent / "resumed"
    a.mkdir()
    b.mkdir()
    train, validation = dataset_factory(19), dataset_factory(5)
    configure_runtime(config.seed, cuda=device == "cuda")
    model = model_factory(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    engine.fit(model, optimizer, train, validation, config, plan, a, initial_progress(), device=device)
    expected_model, expected_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    expected = torch.load(a / "last.pt", map_location="cpu", weights_only=True)
    configure_runtime(config.seed, cuda=device == "cuda")
    model = model_factory(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    original_step = optimizer.step
    calls = 0

    def step(*args, **kwargs):
        nonlocal calls
        result = original_step(*args, **kwargs)
        calls += 1
        if calls == 1:
            signal.raise_signal(signal.SIGINT)
        return result

    with patch.object(optimizer, "step", side_effect=step):
        stopped = engine.fit(model, optimizer, train, validation, config, plan, b, initial_progress(), device=device)
    require(stopped["cursor"] == 16 and stopped["status"] == "stopped_at_optimizer_boundary", "Bad synthetic boundary")
    model = model_factory(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    progress = engine.load_checkpoint(b / "last.pt", model, optimizer, plan)
    engine.fit(model, optimizer, train, validation, config, plan, b, progress, device=device)
    actual = torch.load(b / "last.pt", map_location="cpu", weights_only=True)
    require(nested_equal(expected_model, model.state_dict()) and
            nested_equal(expected_optimizer, optimizer.state_dict()) and
            nested_equal(expected["rng"], actual["rng"]) and expected["progress"] == actual["progress"],
            "Candidate resume differs from uninterrupted synthetic trajectory")
    return {"bitwise_equal": True, "device": device, "synthetic_updates_executed": 8,
            "train_samples_per_epoch": 19, "last_effective_batch": 3, "dropout_enabled": True}


def run(source_root, config, *, accepted):
    assert_launch_prerequisites(accepted)
    require(torch.cuda.is_available(), "CUDA required for candidate preflight")
    configure_runtime(config.seed)
    plan = build_plan(source_root, config)
    parent = FOLDER / "artifacts"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="gpu_preflight_", dir=parent))
    print(f"OUTPUT: {output}", flush=True)
    atomic_json(output / "plan.json", plan)
    snapshot_sources(output, plan)
    resume = synthetic_resume_probe(output, "cuda")
    configure_runtime(config.seed)
    model, initialization = initialized_candidate(config.seed)
    before, frozen = state_snapshot(model), frozen_snapshot(model)
    model = model.cuda()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    scopes = {s: VideoScope(source_root=source_root, split=s) for s in ("train", "validation")}
    ids = {s: select_smoke_ids(scope) for s, scope in scopes.items()}
    sources = {s: {i: guarded_source_sha256(scopes[s], i) for i in ids[s]} for s in scopes}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    training_mode(model)
    optimizer.zero_grad(set_to_none=True)
    train_data = Basketball50Dataset(source_root=source_root, split="train", video_ids=ids["train"])
    seen = []
    for batch in make_loader(train_data, list(range(4)), config, 0):
        x, y = batch["tensor"].cuda(non_blocking=True), batch["binary_label"].cuda()
        loss = nn.functional.cross_entropy(model(x), y, reduction="sum")
        require(torch.isfinite(loss).item(), "Nonfinite engineering loss")
        (loss / 4).backward()
        seen.extend(batch["video_id"])
    require(seen == ids["train"], "Preflight loader IDs mismatch")
    gradients = check_gradients(model)
    branch_gradients = {}
    for branch in ("gate_value", "gate_control", "gate_output", "short_term", "kernel_generator", "tsci"):
        branch_gradients[branch] = sum(p.grad.abs().sum().item() for name, p in model.named_parameters()
                                     if f".{branch}." in name and p.grad is not None)
    require(all(v > 0 for v in branch_gradients.values()), "Missing candidate branch gradients")
    check_state(model, before)
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.clip_norm, error_if_nonfinite=True)
    optimizer.step()  # Exactly one ENGINEERING update, never a formal initialization.
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - started
    check_state(model, frozen)
    changed_keys = ("conv1.weight", "layer4.2.conv3.weight", "fc.weight",
                    "layer1.0.aca.lsta.gate_value.0.weight", "layer1.0.aca.tsci.channel.conv.weight")
    changed = {k: not torch.equal(model.state_dict()[k].detach().cpu(), before[k]) for k in changed_keys}
    require(all(changed.values()), "Backbone, head and ACA weights did not all update")
    del before
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    validation = Metrics()
    first_x = first_logits = None
    validation_data = Basketball50Dataset(source_root=source_root, split="validation", video_ids=ids["validation"])
    with torch.no_grad():
        for batch in make_loader(validation_data, list(range(4)), config, 0):
            x, y = batch["tensor"].cuda(non_blocking=True), batch["binary_label"].cuda()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, reduction="sum")
            validation.add(logits, y, loss.item(), batch["original_class"])
            if first_x is None:
                first_x, first_logits = x.detach().cpu(), logits.detach().cpu()
    peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    checkpoint = output / "engineering_only.pt"
    torch.save({"schema": "aca_dense_candidate_engineering_v1", "engineering_only": True,
                "prohibited_as_formal_training_initialization": True, "optimizer_steps": 1,
                "model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    del model, optimizer, x, y, logits, loss
    gc.collect()
    torch.cuda.empty_cache()
    restored, _ = initialized_candidate(config.seed)
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored.load_state_dict(value["model"], strict=True)
    restored = restored.cuda().eval()
    with torch.no_grad():
        actual = restored(first_x.cuda()).cpu()
    torch.testing.assert_close(actual, first_logits, rtol=0, atol=0)
    del restored, value, actual
    gc.collect()
    torch.cuda.empty_cache()
    for split in scopes:
        require(all(guarded_source_sha256(scopes[split], i) == digest for i, digest in sources[split].items()),
                "Original source changed during candidate preflight")
    require(canonical_hash(build_plan(source_root, config)) == canonical_hash(plan), "Plan changed during GPU preflight")
    report = {"schema": PREFLIGHT_SCHEMA, "status": "passed", "output": str(output),
              "plan_sha256": canonical_hash(plan), "engineering_only": True, "optimizer_steps": 1,
              "formal_training_started": False, "checkpoint_may_initialize_formal_training": False,
              "synthetic_cuda_resume_bitwise_equal": resume["bitwise_equal"], "synthetic_resume": resume,
              "checkpoint_roundtrip_logits_bitwise_equal": True, "initialization": initialization,
              "gradients": gradients, "branch_gradients": branch_gradients, "changed_parameters": changed,
              "all_frozen_BN_parameters_and_buffers_unchanged": True,
              "engineering_update_including_loader_startup_seconds": update_seconds,
              "peak_allocated_bytes_including_Adam": peak_allocated, "peak_reserved_bytes": peak_reserved,
              "measurement_scope": "one 4-video update and four-video validation; not steady-state epoch benchmark",
              "validation_engineering_subset_not_generalization": validation.result(),
              "decoded_train_videos": 4, "decoded_validation_videos": 4, "test_videos_opened": 0,
              "original_source_sha256": sources, "paper_exact_reproduction": False}
    atomic_json(output / "report.json", report)
    # Convenience mirror only; immutable per-run report remains in its own directory.
    atomic_json(parent / "latest_gpu_preflight.json", report)
    return report
