"""Budget extension, NOT a relaxed generic resume or a fresh training run.

Import the frozen dense training loop unchanged. Permit only epochs 5 -> 15,
preserving the parent model, Adam, RNG, epoch, history and early-stop counter.
Parent files stay read-only. Every extension output is in this module's artifacts.
"""
import argparse
import copy
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import shutil
import tempfile

import torch
from torch import nn

from development.aca_candidate_v1 import engine, experiment as exp
from development.aca_net_v1 import train as base
from development.aca_net_v1.baseline_preflight import check_gradients, check_state, state_snapshot, training_mode
from development.aca_net_v1.data import Basketball50Dataset, ROOT, VideoScope, file_sha256, require, select_smoke_ids
from development.aca_net_v1.readiness import nested_equal, ToyDataset, toy_model

FOLDER = Path(__file__).resolve().parent
PARENT = exp.FOLDER / "artifacts/aca_train_seed20261025_hmqhr2pg"
REPORT_SCHEMA = "aca_dense_extension15_preflight_v1"
PARENT_FILES = ("config.json", "summary.json", "history.json", "last.pt", "best.pt", "gpu_preflight.json")


def config():
    value = exp.config_from(FOLDER / "config.json")
    require(value.epochs == 15 and value.patience == 5, "Only total epochs=15 and patience=5 authorized")
    return value


def validate_extension(old, new):
    require(old["epochs"] == 5 and new["epochs"] == 15 and old["patience"] == new["patience"] == 5,
            "Only the completed five-epoch run may extend to total 15; patience stays 5")
    require({k: v for k, v in old.items() if k != "epochs"} ==
            {k: v for k, v in new.items() if k != "epochs"}, "Only the epoch budget may change")


def validate_parent_progress(progress, old, summary, history, train_count):
    require(summary.get("status") == "completed" and summary.get("stop_reason") == "max_epochs" and
            summary.get("model") == "aca_dense_candidate" and summary.get("test_videos_opened") == 0,
            "Need a successfully completed dense pilot, not an early-stop/failure/engineering run")
    require(progress["epoch"] == summary["epochs"] == old["epochs"] == len(history) == 5 and
            progress["cursor"] == 0 and progress["train_metrics"]["count"] == 0 and
            progress["finished"] is True and progress["history"] == history,
            "Parent is not the completed epoch-five boundary")
    require(progress["optimizer_steps"] == summary["optimizer_steps"] ==
            5 * math.ceil(train_count / old["effective_batch"]), "Parent optimizer step count mismatch")
    best_loss, bad, selected_epoch = None, 0, None
    for index, row in enumerate(history, 1):
        require(row["epoch"] == index and math.isfinite(row["validation"]["loss"]), "Invalid parent history")
        improved = best_loss is None or row["validation"]["loss"] < best_loss
        require(row["selected_as_best"] is improved, "Parent best selection mismatch")
        bad = 0 if improved else bad + 1
        if improved:
            best_loss, selected_epoch = row["validation"]["loss"], index
    require(progress["best_loss"] == best_loss and progress["bad_epochs"] == bad < old["patience"] and
            summary["best_epoch"] == selected_epoch and
            summary["best_validation"] == history[selected_epoch - 1]["validation"],
            "Parent early-stop/best state mismatch; do not reset its counter")


def reopen_progress(progress, old, new):
    """Only clear finished=max_epochs. Never reset epoch, bad_epochs, best or history."""
    validate_extension(old, new)
    require(progress["finished"] is True and progress["epoch"] == old["epochs"] and
            progress["cursor"] == 0 and progress["bad_epochs"] < old["patience"],
            "An early-stopped or partial run cannot be reopened by this budget extension")
    return lift_finished(progress)


def parent_identity():
    return {name: file_sha256(PARENT / name) for name in PARENT_FILES}


def build_plan(source_root):
    parent = json.loads((PARENT / "config.json").read_text())
    old = exp.CandidateConfig(**parent["config"])
    new = config()
    validate_extension(asdict(old), asdict(new))
    require(base.canonical_hash(exp.build_plan(source_root, old)) == base.canonical_hash(parent),
            "Parent source/config/data/runtime identity changed")
    exp.verify_gpu_report(PARENT / "gpu_preflight.json", parent)
    plan = copy.deepcopy(parent)
    plan["config"] = asdict(new)
    plan["initialization"] = "parent_last.pt_model_Adam_RNG_progress; not fresh ImageNet or best-weight restart"
    plan["extension"] = {"schema": "aca_dense_budget_extension_v1", "parent_run": str(PARENT),
        "parent_plan_sha256": base.canonical_hash(parent), "parent_file_sha256": parent_identity(),
        "allowed_config_changes": {"epochs": [5, 15]}, "start_epoch": 6, "max_total_epochs": 15,
        "post_hoc_budget_extension_after_inspecting_pilot_validation": True,
        "early_stop_counter_preserved": True, "parent_outputs_read_only": True}
    for path in sorted([*FOLDER.glob("*.py"), FOLDER / "config.json"]):
        plan["source_code_sha256"][path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return parent, plan


def check_adam(optimizer, steps):
    params = [p for group in optimizer.param_groups for p in group["params"]]
    require(len(optimizer.state) == len(params), "Missing parent Adam moments")
    for p in params:
        state = optimizer.state[p]
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"} and state["step"].item() == steps and
                state["exp_avg"].shape == state["exp_avg_sq"].shape == p.shape and
                torch.isfinite(state["exp_avg"]).all().item() and
                torch.isfinite(state["exp_avg_sq"]).all().item(), "Invalid parent Adam state")


def restore_parent(model, optimizer, parent, plan):
    summary = json.loads((PARENT / "summary.json").read_text())
    history = json.loads((PARENT / "history.json").read_text())
    progress = engine.load_checkpoint(PARENT / "last.pt", model, optimizer, parent)
    validate_parent_progress(progress, parent["config"], summary, history, parent["train_count"])
    check_adam(optimizer, progress["optimizer_steps"])
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite parent model")
    return reopen_progress(progress, parent["config"], plan["config"])


def new_model_optimizer(device):
    model, _ = exp.initialized_candidate(config().seed)
    model = model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config().lr, weight_decay=config().weight_decay)
    return model, optimizer


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def import_parent_checkpoints(output, model, optimizer, parent, plan):
    """Re-home inherited best and last under the new plan, never mutate originals."""
    best_progress = engine.load_checkpoint(PARENT / "best.pt", model, optimizer, parent)
    engine.save_checkpoint(output / "best.pt", model, optimizer, lift_finished(best_progress), plan)
    progress = restore_parent(model, optimizer, parent, plan)
    engine.save_checkpoint(output / "last.pt", model, optimizer, progress, plan)
    return progress


def synthetic_extension_probe(output, device="cpu"):
    """Completed-budget extension must exactly match an uninterrupted trajectory."""
    cfg = exp.CandidateConfig(epochs=15, patience=5, workers=0, checkpoint_every=1)
    pilot_cfg = exp.CandidateConfig(epochs=5, patience=5, workers=0, checkpoint_every=1)
    full_plan = {"config": asdict(cfg), "train_count": 19, "validation_count": 5,
                 "engineering_only_synthetic_toy": True}
    pilot_plan = {**full_plan, "config": asdict(pilot_cfg)}
    continuous, pilot, extended = (output / n for n in ("continuous", "pilot", "extended"))
    for folder in (continuous, pilot, extended):
        folder.mkdir()
    train, val = ToyDataset(19), ToyDataset(5)
    base.configure_runtime(cfg.seed, cuda=device == "cuda")
    model = toy_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    engine.fit(model, optimizer, train, val, cfg, full_plan, continuous, base.initial_progress(), device=device)
    expected = torch.load(continuous / "last.pt", map_location="cpu", weights_only=True)
    base.configure_runtime(cfg.seed, cuda=device == "cuda")
    model = toy_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    engine.fit(model, optimizer, train, val, pilot_cfg, pilot_plan, pilot, base.initial_progress(), device=device)
    model = toy_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    progress = engine.load_checkpoint(pilot / "last.pt", model, optimizer, pilot_plan)
    require(progress["finished"] and progress["epoch"] == 5, "Synthetic completed boundary missing")
    progress = reopen_progress(progress, asdict(pilot_cfg), asdict(cfg))
    engine.save_checkpoint(extended / "last.pt", model, optimizer, progress, full_plan)
    shutil.copy2(pilot / "best.pt", extended / "parent_best.pt")
    engine.fit(model, optimizer, train, val, cfg, full_plan, extended, progress, device=device)
    actual = torch.load(extended / "last.pt", map_location="cpu", weights_only=True)
    require(all(nested_equal(expected[k], actual[k]) for k in ("model", "optimizer", "rng", "progress")),
            "Completed-budget extension differs from uninterrupted trajectory")
    return {"bitwise_equal": True, "device": device, "compared": ["model", "Adam", "RNG", "history", "cursor"]}


def lift_finished(progress):
    value = copy.deepcopy(progress)
    value["finished"] = False
    return value


def preflight(source_root, accepted):
    exp.assert_launch_prerequisites(accepted)
    base.configure_runtime(config().seed)
    parent, plan = build_plan(source_root)
    (FOLDER / "artifacts").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="gpu_preflight_", dir=FOLDER / "artifacts"))
    base.atomic_json(output / "plan.json", plan)
    exp.snapshot_sources(output, plan)
    print(f"PREFLIGHT: {output}", flush=True)
    probe = synthetic_extension_probe(output, "cuda")
    model, optimizer = new_model_optimizer("cuda")
    progress = restore_parent(model, optimizer, parent, plan)
    saved_rng = base.capture_rng()
    engine.save_checkpoint(output / "imported_engineering_only.pt", model, optimizer, progress,
                           {**plan, "engineering_only_synthetic_toy": True})
    # Real checkpoint migration plus read-only gradient check; no real optimizer.step.
    before = state_snapshot(model)
    adam_before = cpu_tree(optimizer.state_dict())
    scope = VideoScope(source_root=source_root, split="train")
    ids = select_smoke_ids(scope)[:2]
    data = Basketball50Dataset(source_root=source_root, split="train", video_ids=ids)
    items = [data[index] for index in range(2)]
    training_mode(model)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        item = items[0]
        first_x = item["tensor"].unsqueeze(0).cuda()
        model.eval()
        expected_logits = model(first_x).cpu()
    training_mode(model)
    for item in items:
        x = item["tensor"].unsqueeze(0).cuda()
        y = torch.tensor([item["binary_label"]], device="cuda")
        (nn.functional.cross_entropy(model(x), y) / 2).backward()
    gradients = check_gradients(model)
    check_state(model, before)
    del x, y
    optimizer.zero_grad(set_to_none=True)
    # Reload the on-disk engineering migration through the unchanged strict loader.
    progress2 = engine.load_checkpoint(output / "imported_engineering_only.pt", model, optimizer,
                                     {**plan, "engineering_only_synthetic_toy": True})
    model.eval()
    with torch.no_grad():
        actual_logits = model(first_x).cpu()
    require(torch.equal(expected_logits, actual_logits) and progress == progress2 and
            nested_equal(saved_rng, base.capture_rng()), "Real migration changed logits/progress/RNG")
    check_state(model, before)
    require(nested_equal(adam_before, cpu_tree(optimizer.state_dict())), "Adam moments changed during migration")
    del before, adam_before
    check_adam(optimizer, 2250)
    require(parent_identity() == plan["extension"]["parent_file_sha256"], "Parent changed during preflight")
    report = {"schema": REPORT_SCHEMA, "status": "passed", "output": str(output),
              "plan_sha256": base.canonical_hash(plan), "synthetic_extension": probe,
              "real_checkpoint_logits_bitwise_equal": True, "parent_Adam_steps": 2250,
              "model_unchanged_during_backward": True, "parent_progress_RNG_preserved": True,
              "Adam_moments_bitwise_preserved": True,
              "gradients": gradients, "decoded_train_videos": 2, "real_optimizer_updates": 0,
              "test_videos_opened": 0, "engineering_only": True,
              "checkpoint_may_initialize_formal_training": False}
    base.atomic_json(output / "report.json", report)
    del model, optimizer, first_x
    gc.collect()
    torch.cuda.empty_cache()
    return report


def verify_report(path, plan):
    report = json.loads(Path(path).read_text())
    require(report.get("schema") == REPORT_SCHEMA and report.get("status") == "passed" and
            report.get("plan_sha256") == base.canonical_hash(plan) and
            report.get("real_checkpoint_logits_bitwise_equal") is True and
            report.get("parent_progress_RNG_preserved") is True and
            report.get("Adam_moments_bitwise_preserved") is True and
            report.get("synthetic_extension", {}).get("bitwise_equal") is True and
            report.get("engineering_only") is True and
            report.get("checkpoint_may_initialize_formal_training") is False and
            report.get("real_optimizer_updates") == 0 and report.get("test_videos_opened") == 0,
            "Missing/stale extension GPU preflight")


def run(source_root, report_path, accepted, *, resume=None):
    exp.assert_launch_prerequisites(accepted)
    base.configure_runtime(config().seed)
    parent, plan = build_plan(source_root)
    verify_report(report_path, plan)
    model, optimizer = new_model_optimizer("cuda")
    if resume:
        output = Path(resume).resolve()
        require(output.parent == (FOLDER / "artifacts").resolve() and output.name.startswith("aca_extend15_seed"),
                "Resume only this extension's formal run; never parent or engineering outputs")
        require(base.canonical_hash(json.loads((output / "config.json").read_text())) == base.canonical_hash(plan),
                "Extension resume plan mismatch")
        progress = engine.load_checkpoint(output / "last.pt", model, optimizer, plan)
    else:
        output = Path(tempfile.mkdtemp(prefix=f"aca_extend15_seed{config().seed}_", dir=FOLDER / "artifacts"))
        base.atomic_json(output / "config.json", plan)
        shutil.copy2(report_path, output / "gpu_preflight.json")
        exp.snapshot_sources(output, plan)
        # Carry the best-so-far model forward, even if all future epochs get worse.
        progress = import_parent_checkpoints(output, model, optimizer, parent, plan)
        base.atomic_json(output / "history.json", progress["history"])
        base.atomic_json(output / "continuation.json", {"parent_run": str(PARENT), "start_epoch": 6,
            "inherited_epochs": 5, "inherited_optimizer_steps": 2250, "total_epoch_cap": 15,
            "patience": 5, "inherited_bad_epochs": progress["bad_epochs"],
            "parent_file_sha256": plan["extension"]["parent_file_sha256"]})
    require(parent_identity() == plan["extension"]["parent_file_sha256"], "Parent changed during import")
    print(f"RUN: {output}", flush=True)
    print(f"CONTINUE: epoch={progress['epoch']+1} steps={progress['optimizer_steps']} bad_epochs={progress['bad_epochs']} "
          f"best_loss={progress['best_loss']} total_cap=15 patience=5", flush=True)
    train = Basketball50Dataset(source_root=source_root, split="train")
    val = Basketball50Dataset(source_root=source_root, split="validation")
    result = engine.fit(model, optimizer, train, val, config(), plan, output, progress, device="cuda")
    require(parent_identity() == plan["extension"]["parent_file_sha256"], "Parent changed during extension")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--accept-candidate-architecture", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--start-training", action="store_true")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    require(args.resume is None or args.start_training, "Resume requires --start-training")
    if args.preflight:
        value = preflight(args.source_root, args.accept_candidate_architecture)
    elif args.start_training:
        require(args.preflight_report is not None, "Extension preflight report required")
        value = run(args.source_root, args.preflight_report, args.accept_candidate_architecture, resume=args.resume)
    else:
        _, value = build_plan(args.source_root)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
