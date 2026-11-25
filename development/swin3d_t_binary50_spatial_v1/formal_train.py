"""Single-factor Swin3D-T seed20261024 comparison: add clip_flip_color_light_v2 spatial augmentation.

Only change vs swin3d_t_binary50_seed20261024_labelrev1_v1:
- Train clips pass through audited clip-consistent spatial augmentation
  (development.aca_spatial_v1.augment: flip p=0.5, brightness/contrast/saturation [0.95, 1.05]).
- Model init (locked Kinetics-400 + head_seed=20264024), temporal sampling (seed=20261024),
  validation (fixed uniform 50-frame, zero aug, 1,547 clips), optimizer (Adam lr=1e-4, wd=0),
  FP32, micro1/effective16/tail8, patience=5, lowest Val CE selection remain 100% identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torchvision

from development.aca_binary_v1 import core, runtime
from development.aca_binary_v1.data import guard_fetch
from development.aca_eight_class_data_v1 import formal_train as base
from development.aca_reconstruction_v2 import formal_train as shell
from development.aca_spatial_pair_v1 import formal_train as gpu_policy_mod
from development.aca_spatial_v1 import augment
from development.aca_spatial_v1.data import EpochStream as SpatialEpochStream
from development.aca_spatial_v1.data import SpatialDataset
from development.aca_temporal_v1 import sampling
from development.adjudicated_input_v1 import adapter as labels
from development.swin3d_t_binary50_formal_v1 import formal_train as parent_swin24
from development.swin3d_t_binary50_formal_v1.model import (
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
RUN = CURRENT / "swin3d_t_binary50_spatial_seed20261024_labelrev1_v1"
QA = CURRENT / "swin3d_t_binary50_spatial_seed20261024_readiness_20261125_v1"
PARENT_SWIN24 = CURRENT / "swin3d_t_binary50_seed20261024_labelrev1_v1"
PARENT_SWIN24_SHA = "6bd908ced780140ebf88fae132a566f338eb2b04e6ce76b5a9c44d5d6ff2787c"
SCHEMA = "swin3d_t_binary50_spatial_seed24_formal_v1"


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes(parent_plan: dict) -> dict[str, str]:
    hashes = dict(parent_plan["code_sha256"])
    for p_str, expected in hashes.items():
        need(sha(Path(p_str)) == expected, f"Parent source changed: {p_str}")
    for folder in ("aca_spatial_v1", "swin3d_t_binary50_spatial_v1"):
        for p in sorted((ROOT / "development" / folder).glob("*.py")):
            hashes[str(p)] = sha(p)
    return dict(sorted(hashes.items()))


def draft_plan() -> dict:
    parent = read(PARENT_SWIN24 / "plan.json")
    need(core.plan_digest(parent) == PARENT_SWIN24_SHA, "Parent Swin3D-T seed24 plan changed")
    summary = read(PARENT_SWIN24 / "summary.json")
    history = read(PARENT_SWIN24 / "history.json")
    need(
        summary["status"] == "completed"
        and summary["plan_sha256"] == PARENT_SWIN24_SHA
        and summary["test_run"] is False
        and len(history) == summary["completed_epochs"],
        "Parent Swin3D-T seed24 run invalid",
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
        "head",
        "initialization",
        "normalization_policy",
        "precision",
        "optimizer",
    )
    plan = {k: parent[k] for k in inherited_keys}
    plan.update(
        schema=SCHEMA,
        variant=VARIANT,
        purpose="single_factor_spatial_augmentation_ablation_on_Swin3D_T_seed20261024",
        allowed_run_directory=str(RUN),
        model_display_name="含動態位置索引修正的 Swin3D-T（＋空間增強）",
        lsta_tsci_included=False,
        spatial_augmentation=augment.POLICY,
        spatial_policy_sha256=augment.POLICY_SHA,
        ablation={
            "parent_swin3d_t_run": str(PARENT_SWIN24),
            "parent_swin3d_t_plan_sha256": PARENT_SWIN24_SHA,
            "parent_swin3d_t_summary_sha256": sha(PARENT_SWIN24 / "summary.json"),
            "only_change": "add_train_only_clip_flip_color_light_v2_spatial_augmentation",
            "historical_test_contact_disclosed": True,
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
            f"Readiness {name} does not match current draft plan",
        )
    plan["formal_readiness"] = {name: sha(QA / name) for name in ("cpu_report.json", "gpu_report.json")}
    return plan


def verify_plan(plan: dict, digest: str) -> None:
    need(core.plan_digest(plan) == digest and plan == build_plan(), "Plan/code/data/runtime mismatch")


def install_runtime() -> None:
    core.training_mode = training_mode
    runtime.training_mode = training_mode


class SpatialTrainingLoop(runtime.TrainingLoop):
    def __init__(self, model, plan, fetch, *args, **kwargs):
        need(
            plan["schema"] == SCHEMA
            and plan["variant"] == VARIANT
            and plan["normalization_policy"] == NORM_POLICY
            and plan["spatial_augmentation"] == augment.POLICY
            and plan["spatial_policy_sha256"] == augment.POLICY_SHA
            and plan["temporal_sampling"] == sampling.POLICY
            and plan["seed"] == SEED
            and plan["optimizer"] == parent_swin24.OPTIMIZER_SPEC,
            "Wrong formal Swin3D-T spatial plan",
        )
        labels.require_revision_identity(plan)
        binary_fetch = guard_fetch(fetch, plan)

        def guarded(split, index):
            batch = binary_fetch(split, index)
            need(tuple(batch["tensor"].shape) == (1, 3, 50, 120, 160), "Wrong single clip shape")
            if split == "train":
                need(
                    batch.get("temporal_sampling_version") == sampling.VERSION
                    and batch.get("spatial_policy_sha256") == augment.POLICY_SHA
                    and all(r["metadata"]["epoch_index"] == self.state["epoch"] for r in batch["records"]),
                    "Stale or missing train temporal/spatial augmentation",
                )
            else:
                need(
                    "temporal_sampling_version" not in batch and "spatial_policy_sha256" not in batch,
                    "Validation augmentation forbidden",
                )
            return batch

        super().__init__(model, plan, guarded, *args, **kwargs)
        validate_architecture(model)
        parent_swin24.validate_optimizer_options(self.optimizer)
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
    shell.require_predecessor = lambda plan: None
    shell.initialized = initialized
    shell.rt = SimpleNamespace(
        **{
            **vars(runtime),
            "VerifiedDataset": SpatialDataset,
            "EpochStream": SpatialEpochStream,
            "TrainingLoop": SpatialTrainingLoop,
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
