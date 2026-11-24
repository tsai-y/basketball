"""Isolated formal single-seed Swin3D-T comparison vs ACA temporal50 (seed20261024).

System comparison (architecture + Kinetics-400 pretraining source changed; NOT a
pure backbone ablation). Uses the audited shared numerical/checkpoint loop from
aca_binary_v1 and endpoint-stratified temporal50 train / fixed uniform50 val
from aca_temporal_v1. Spatial/color augmentation is disabled to match the
temporal50 primary control.

CLI commands:
- prepare: validates readiness evidence & writes formal_plan.json + review.json
  WITHOUT starting training.
- start / resume / stop: explicit lifecycle commands only; default never trains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision

from development.aca_binary_v1 import core, runtime
from development.aca_binary_v1.data import guard_fetch
from development.aca_eight_class_data_v1 import formal_train as base
from development.aca_reconstruction_v2 import formal_train as shell
from development.aca_spatial_pair_v1 import formal_train as gpu_policy_mod
from development.aca_temporal_v1 import sampling
from development.aca_temporal_v1.data import EpochStream, TemporalDataset
from development.adjudicated_input_v1 import adapter as labels

from .model import (
    HEAD_SEED,
    NORM_POLICY,
    SEED,
    VARIANT,
    WEIGHT_SHA,
    WEIGHTS,
    initialized,
    need,
    sha,
    training_mode,
    validate_architecture,
)

ROOT = Path(__file__).resolve().parents[2]
CURRENT = Path("/home/tsai/projects/basketball/outputs/01_current")
RUN = CURRENT / "swin3d_t_binary50_seed20261024_labelrev1_v1"
QA = CURRENT / "swin3d_t_binary50_formal_readiness_20261125_v1"
PRIOR_CPU = CURRENT / "swin3d_t_binary50_cpu_acceptance_20261124_v1"
PRIOR_PRETRAIN = CURRENT / "swin3d_t_binary50_pretrained_load_20261124_v1"
PRIOR_GPU = CURRENT / "swin3d_t_binary50_gpu_readiness_20261125_v1"
PARENT = CURRENT / "aca_binary_temporal50_seed20261024_labelrev1_v1"
PARENT_SHA = "ee331c184ca8b0cea54550098bd926929e9f1b74398528948c62410a70a2495a"
SCHEMA = "swin3d_t_binary50_formal_v1"

OPTIMIZER_SPEC = {
    "type": "Adam",
    "lr": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "foreach": False,
    "fused": False,
    "grad_clip_norm": 1.0,
    "grad_clip_norm_type": 2.0,
    "grad_clip_error_if_nonfinite": True,
    "lr_schedule": "constant_fixed_1e-4",
}


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes(parent: dict) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, expected in parent["code_sha256"].items():
        p = ROOT / name
        need(sha(p) == expected, f"Shared source changed: {name}")
        hashes[str(p)] = expected
    for folder in ("swin3d_t_binary50_v1", "swin3d_t_binary50_formal_v1"):
        for p in sorted((ROOT / "development" / folder).glob("*.py")):
            hashes[str(p)] = sha(p)
    for name, expected in labels.PINNED_CODE.items():
        p = labels.f.ROOT / name
        need(sha(p) == expected, f"Pinned data source changed: {name}")
        hashes[str(p)] = expected
    return dict(sorted(hashes.items()))


def draft_plan() -> dict:
    parent = read(PARENT / "plan.json")
    need(core.plan_digest(parent) == PARENT_SHA, "Primary control plan changed")
    reference = read(PARENT / "summary.json")
    history = read(PARENT / "history.json")
    need(
        reference["status"] == "completed"
        and reference["plan_sha256"] == PARENT_SHA
        and reference["test_run"] is False
        and len(history) == reference["completed_epochs"]
        and reference["selected_validation"] == min(history, key=lambda r: r["validation"]["ce"])["validation"],
        "Primary control result invalid",
    )
    rev = labels.LabelRevision()
    manifest = rev._manifest
    need(
        parent["reference"] == manifest["reference"]
        and parent["train_ids"] == manifest["train_ids"]
        and parent["validation_ids"] == manifest["validation_ids"],
        "Data protocol changed",
    )

    # Verify prior Stage B gate artifacts & locked weight SHA-256
    cpu_audit = read(PRIOR_CPU / "cpu_acceptance.json")
    pretrain_audit = read(PRIOR_PRETRAIN / "pretrained_load_acceptance.json")
    gpu_audit = read(PRIOR_GPU / "gpu_readiness.json")
    need(cpu_audit["status"] == "PASS_CPU_ONLY", "Prior CPU acceptance missing")
    need(
        pretrain_audit["status"] == "PASS_PRETRAINED_LOAD_CPU_ONLY"
        and pretrain_audit["checks"]["official_weight_acquisition_and_hash"]["full_sha256"] == WEIGHT_SHA,
        "Prior pretrained load acceptance invalid",
    )
    need(
        gpu_audit["status"] == "PASS_GPU_ENGINEERING_READINESS_ONLY"
        and gpu_audit["weight_provenance"]["full_sha256"] == WEIGHT_SHA
        and all(gpu_audit["resume_equivalence_post_selection"]["matches"].values()),
        "Prior GPU engineering readiness invalid",
    )
    need(WEIGHTS.is_file() and not WEIGHTS.is_symlink() and sha(WEIGHTS) == WEIGHT_SHA, "Locked weight changed")

    inherited_keys = (
        "seed",
        "max_epochs",
        "patience",
        "effective",
        "checkpoint_every",
        "device",
        "class_order",
        "train_ids",
        "validation_ids",
        "reference",
        "train_count",
        "validation_count",
        "updates_per_epoch",
        "tail_videos",
        "source_root",
        "data_version",
        "label_revision_sha256",
        "pixel_contract_parent_sha256",
        "label_patch_sha256",
        "task",
        "loss",
        "prediction",
        "selection",
        "temporal_sampling",
        "temporal_input_contract",
        "input_contract_sha256",
    )
    plan = {k: parent[k] for k in inherited_keys}
    need(plan["seed"] == SEED and plan["max_epochs"] == 30 and plan["patience"] == 5, "Unexpected training schedule")
    need(plan["effective"] == 16 and plan["tail_videos"] == 8 and plan["validation_count"] == 1547,
         "Unexpected batch/split counts")

    plan.update(
        schema=SCHEMA,
        variant=VARIANT,
        purpose="single_seed_system_comparison_Swin3D_T_dynamic_pos_Kinetics400_vs_ACA_temporal50",
        allowed_run_directory=str(RUN),
        model_display_name="含動態位置索引修正的 Swin3D-T",
        lsta_tsci_included=False,
        spatial_augmentation="disabled_matches_temporal50_control",
        normalization_policy=NORM_POLICY,
        precision="FP32_strict_no_AMP_no_TF32",
        optimizer=OPTIMIZER_SPEC,
        head={
            "type": "Linear",
            "in_features": 768,
            "out_features": 2,
            "initialization": "trunc_normal_std_0p02_zero_bias",
            "initialization_seed": HEAD_SEED,
            "cross_architecture_note": (
                "Same head_seed (20264024) does not imply identical initial weights across "
                "different feature dimensions (768 vs 2048/1536)."
            ),
        },
        initialization={
            "name": "torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1",
            "path": str(WEIGHTS),
            "sha256": WEIGHT_SHA,
            "load": "strict_400class_CPU_load_then_isolated_binary_head_swap",
            "head_seed": HEAD_SEED,
            "trained_weights_from_engineering_used": False,
        },
        comparison={
            "primary_control_run": str(PARENT),
            "primary_control_plan_sha256": PARENT_SHA,
            "primary_control_summary_sha256": sha(PARENT / "summary.json"),
            "comparison_type": "system_comparison_not_pure_backbone_ablation",
            "changed_factors": [
                "architecture_ResNet50_ACA_to_Swin3D_T_dynamic_pos",
                "pretraining_source_ImageNet1k_to_Kinetics400",
                "normalization_BN_to_LayerNorm_and_StochasticDepth",
            ],
            "unchanged_factors": [
                "train_seed_20261024",
                "50frame_120x160_RGB_single_ImageNet_norm",
                "train_endpoint_stratified50_no_spatial_aug",
                "val_fixed_uniform50_1547clips_zero_aug",
                "Adam_lr1e-4_wd0_clip1_FP32_micro1_effective16_tail8",
                "max30epochs_patience5_min_val_CE_earlier_tie",
            ],
            "historical_test_contact_disclosed": True,
        },
        prior_stage_evidence={
            "cpu_acceptance": {"path": str(PRIOR_CPU / "cpu_acceptance.json"), "sha256": sha(PRIOR_CPU / "cpu_acceptance.json")},
            "pretrained_load": {"path": str(PRIOR_PRETRAIN / "pretrained_load_acceptance.json"), "sha256": sha(PRIOR_PRETRAIN / "pretrained_load_acceptance.json")},
            "gpu_engineering_readiness": {
                "path": str(PRIOR_GPU / "gpu_readiness.json"),
                "sha256": sha(PRIOR_GPU / "gpu_readiness.json"),
                "cumulative_updates_across_retry_disclosed": 10,
                "max_peak_allocated_gib_across_phases": 3.273,
                "max_peak_reserved_gib_across_phases": 4.3613,
            },
        },
        runtime={**base.runtime_identity(), "torchvision": torchvision.__version__},
        source_inventory=base.source_inventory(plan["source_root"], plan["reference"]),
        code_sha256=source_hashes(parent),
        gpu_policy=gpu_policy_mod.make_gpu_policy(),
        test_enabled=False,
        automatic_restart=False,
        automatic_next_arm=False,
        checkpoint="single_atomic_last.pt_current_model_Adam_RNG_cudaRNG_history_best_model",
        notifications="no_hourly_reports",
    )
    need(plan["source_inventory"] == parent["source_inventory"], "Source inventory changed")
    return plan


def build_plan() -> dict:
    plan = draft_plan()
    for name in ("cpu_report.json", "gpu_report.json"):
        result = read(QA / name)
        need(
            result["status"] == "passed" and result["draft_plan_sha256"] == core.plan_digest(plan),
            f"Formal readiness {name} does not match current draft plan",
        )
    plan["formal_readiness"] = {name: sha(QA / name) for name in ("cpu_report.json", "gpu_report.json")}
    return plan


def verify_plan(plan: dict, digest: str) -> None:
    need(core.plan_digest(plan) == digest and plan == build_plan(), "Formal plan/code/data/runtime mismatch")


def install_runtime() -> None:
    # Process-local hooks only; never edit shared ACA source files.
    core.training_mode = training_mode
    runtime.training_mode = training_mode


def validate_optimizer_options(optimizer: torch.optim.Optimizer) -> None:
    need(isinstance(optimizer, torch.optim.Adam), f"Expected torch.optim.Adam, got {type(optimizer).__name__}")
    d = optimizer.defaults
    need(
        d["lr"] == 1e-4
        and tuple(d["betas"]) == (0.9, 0.999)
        and d["eps"] == 1e-8
        and d["weight_decay"] == 0.0
        and d["amsgrad"] is False
        and d["foreach"] is False
        and d["fused"] is False,
        f"Adam defaults diverge from contract: {d}",
    )


class TrainingLoop(runtime.TrainingLoop):
    def __init__(self, model, plan, fetch, *args, **kwargs):
        need(
            plan["schema"] == SCHEMA
            and plan["variant"] == VARIANT
            and plan["normalization_policy"] == NORM_POLICY
            and plan["spatial_augmentation"] == "disabled_matches_temporal50_control"
            and plan["temporal_sampling"] == sampling.POLICY
            and plan["seed"] == SEED
            and plan["optimizer"] == OPTIMIZER_SPEC,
            "Wrong formal Swin3D-T plan",
        )
        labels.require_revision_identity(plan)
        binary_fetch = guard_fetch(fetch, plan)

        def guarded(split, index):
            batch = binary_fetch(split, index)
            need(tuple(batch["tensor"].shape) == (1, 3, 50, 120, 160), "Wrong single clip shape")
            need("spatial_policy_sha256" not in batch, "Spatial augmentation must remain disabled")
            if split == "train":
                need(
                    batch.get("temporal_sampling_version") == sampling.VERSION
                    and all(r["metadata"]["epoch_index"] == self.state["epoch"] for r in batch["records"]),
                    "Stale or missing train temporal sampling",
                )
            else:
                need("temporal_sampling_version" not in batch, "Validation augmentation forbidden")
            return batch

        super().__init__(model, plan, guarded, *args, **kwargs)
        validate_architecture(model)
        validate_optimizer_options(self.optimizer)
        need(
            {id(p) for g in self.optimizer.param_groups for p in g["params"]}
            == {id(p) for p in model.parameters() if p.requires_grad},
            "Optimizer parameter coverage mismatch",
        )


def execute(path: Path, digest: str, *, resume: bool = False):
    install_runtime()
    shell.verify_plan = verify_plan
    shell.assert_idle_gpu = gpu_policy_mod.assert_idle_gpu
    shell.run_path = lambda variant: RUN if variant == VARIANT else None
    shell.require_predecessor = lambda plan: None  # Primary control verified inside build_plan
    shell.initialized = initialized
    shell.rt = SimpleNamespace(
        **{
            **vars(runtime),
            "VerifiedDataset": TemporalDataset,
            "EpochStream": EpochStream,
            "TrainingLoop": TrainingLoop,
        }
    )
    return shell.execute(path, digest, RUN, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    s = sub.add_parser("start")
    s.add_argument("--plan", type=Path, required=True)
    s.add_argument("--plan-sha256", required=True)
    r = sub.add_parser("resume")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--plan-sha256", required=True)
    sub.add_parser("stop")
    args = parser.parse_args()

    if args.command == "prepare":
        plan = build_plan()
        observed = gpu_policy_mod.assert_idle_gpu(plan["gpu_policy"])
        target_plan = QA / "formal_plan.json"
        target_review = QA / "review.json"
        need(not target_plan.exists(), f"Prepared plan already exists: {target_plan}")
        runtime.atomic_json(target_plan, plan)
        plan_sha = core.plan_digest(plan)
        review = {
            "status": "prepared_not_started",
            "schema": SCHEMA,
            "plan_sha256": plan_sha,
            "formal_plan_path": str(target_plan),
            "run_directory": str(RUN),
            "gpu_processes_observed": observed,
            "formal_training_started": False,
            "test_run": False,
            "historical_test_contact_disclosed": True,
        }
        runtime.atomic_json(target_review, review)
        print(json.dumps(review, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "start":
        execute(args.plan, args.plan_sha256, resume=False)
        return 0
    if args.command == "resume":
        execute(args.plan, args.plan_sha256, resume=True)
        return 0
    if args.command == "stop":
        need(RUN.exists() and (RUN / "plan.json").is_file(), "Run directory not active")
        identity = read(RUN / "run.json")
        need(read(RUN / "plan.json")["schema"] == SCHEMA, "Wrong run schema")
        runtime.atomic_json(
            RUN / "STOP.json",
            {"run_id": identity["run_id"], "requested_at_taiwan": runtime.now()},
        )
        print("Stop requested; train saves at next optimizer boundary.", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
