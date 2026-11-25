"""Paired seed20261025 Swin3D-T + clip_flip_color_light_v2 formal run.

Only change vs swin3d_t_binary50_seed20261025_labelrev1_v1:
- Train clips pass through audited clip-consistent spatial augmentation
  (clip_flip_color_light_v2, seed=20261025 via SpatialDataset25 / SpatialCollator25 / EpochStream25).
- Model init (locked Kinetics-400 + head_seed=20264025), temporal sampling (seed=20261025),
  validation (fixed uniform 50-frame, zero aug, 1,547 clips), Adam(lr=1e-4, wd=0),
  FP32, micro1/effective16/tail8, max30/patience5, lowest Val CE remain 100% identical.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision

from development.aca_binary_v1 import core, runtime
from development.aca_binary_v1.data import guard_fetch
from development.aca_eight_class_data_v1 import formal_train as base
from development.aca_reconstruction_v2 import formal_train as shell
from development.aca_spatial_pair_v1 import formal_train as gpu_policy_mod
from development.aca_spatial_pair_v1.data import EpochStream25, SpatialDataset25
from development.aca_spatial_v1 import augment
from development.aca_temporal_v1 import sampling
from development.adjudicated_input_v1 import adapter as labels
from development.swin3d_t_binary50_formal_v1 import formal_train as s24_formal
from development.swin3d_t_binary50_pair_v1.model import (
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
RUN = CURRENT / "swin3d_t_binary50_spatial_seed20261025_labelrev1_v1"
QA = CURRENT / "swin3d_t_binary50_spatial_seed20261025_readiness_20261125_v1"
PARENT_SWIN25 = CURRENT / "swin3d_t_binary50_seed20261025_labelrev1_v1"
PARENT_SWIN25_SHA = "7737737be27b526a8e474fbc16d87bae6f2e187b4936ae8a80d7a32ac9840f54"
SEED24_SPATIAL_SWIN = CURRENT / "swin3d_t_binary50_spatial_seed20261024_labelrev1_v1"
SEED24_SPATIAL_SWIN_SHA = "2318e391357792bd4ba69d9a614e64c227b2b2e7a9b46d5491451468c5a53490"
SCHEMA = "swin3d_t_binary50_spatial_seed25_formal_v1"
_ORIGINAL_CONFIGURE = shell.configure


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configure() -> None:
    _ORIGINAL_CONFIGURE()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def source_hashes(parent_plan: dict) -> dict[str, str]:
    hashes = dict(parent_plan["code_sha256"])
    for p_str, expected in hashes.items():
        need(sha(Path(p_str)) == expected, f"Parent source changed: {p_str}")
    for folder in ("aca_spatial_v1", "aca_spatial_pair_v1", "swin3d_t_binary50_spatial_pair_v1"):
        for p in sorted((ROOT / "development" / folder).glob("*.py")):
            hashes[str(p)] = sha(p)
    return dict(sorted(hashes.items()))


def _verify_completed(directory: Path, expected_plan_sha: str) -> dict:
    plan = read(directory / "plan.json")
    need(core.plan_digest(plan) == expected_plan_sha, f"Plan mismatch: {directory}")
    s = read(directory / "summary.json")
    h = read(directory / "history.json")
    need(
        s["status"] == "completed"
        and s["plan_sha256"] == expected_plan_sha
        and s["test_run"] is False
        and len(h) == s["completed_epochs"],
        f"Run invalid: {directory}",
    )
    return {
        "directory": str(directory),
        "plan_sha256": expected_plan_sha,
        "summary_sha256": sha(directory / "summary.json"),
        "best_epoch": s["best_epoch"],
        "selected_ce": s["selected_validation"]["ce"],
        "selected_accuracy": s["selected_validation"]["accuracy"],
    }


def draft_plan() -> dict:
    parent = read(PARENT_SWIN25 / "plan.json")
    parent25_ref = _verify_completed(PARENT_SWIN25, PARENT_SWIN25_SHA)
    spatial24_ref = _verify_completed(SEED24_SPATIAL_SWIN, SEED24_SPATIAL_SWIN_SHA)
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
    need(plan["seed"] == SEED and plan["head"]["initialization_seed"] == HEAD_SEED, "Wrong seed25 setup")
    plan.update(
        schema=SCHEMA,
        variant=VARIANT,
        purpose="paired_second_seed_confirmation_Swin3D_T_spatial_augmentation_seed20261025",
        allowed_run_directory=str(RUN),
        model_display_name="含動態位置索引修正的 Swin3D-T（＋空間增強, seed25）",
        lsta_tsci_included=False,
        spatial_augmentation=augment.POLICY,
        spatial_policy_sha256=augment.POLICY_SHA,
        ablation={
            "parent_swin3d_t_seed25": parent25_ref,
            "seed24_spatial_swin3d_t_reference": spatial24_ref,
            "only_change_vs_parent_seed25": "add_train_only_clip_flip_color_light_v2_seed20261025",
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


class SpatialTrainingLoop25(runtime.TrainingLoop):
    def __init__(self, model, plan, fetch, *args, **kwargs):
        need(
            plan["schema"] == SCHEMA
            and plan["variant"] == VARIANT
            and plan["normalization_policy"] == NORM_POLICY
            and plan["spatial_augmentation"] == augment.POLICY
            and plan["spatial_policy_sha256"] == augment.POLICY_SHA
            and plan["temporal_sampling"] == sampling.POLICY
            and plan["seed"] == SEED
            and plan["optimizer"] == s24_formal.OPTIMIZER_SPEC,
            "Wrong formal Swin3D-T spatial seed25 plan",
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
                    and batch.get("spatial_seed") == SEED
                    and all(r["metadata"]["epoch_index"] == self.state["epoch"] for r in batch["records"])
                    and all(r["metadata"]["temporal_seed"] == SEED for r in batch["records"]),
                    "Stale or wrong-seed train temporal/spatial augmentation",
                )
            else:
                need(
                    "temporal_sampling_version" not in batch
                    and "spatial_policy_sha256" not in batch
                    and "spatial_seed" not in batch,
                    "Validation augmentation forbidden",
                )
            return batch

        super().__init__(model, plan, guarded, *args, **kwargs)
        validate_architecture(model)
        s24_formal.validate_optimizer_options(self.optimizer)
        need(
            {id(p) for g in self.optimizer.param_groups for p in g["params"]}
            == {id(p) for p in model.parameters() if p.requires_grad},
            "Optimizer parameter coverage mismatch",
        )


def execute(path: Path, digest: str, *, resume: bool = False):
    install_runtime()
    shell.configure = configure
    shell.verify_plan = verify_plan
    shell.assert_idle_gpu = gpu_policy_mod.assert_idle_gpu
    shell.run_path = lambda variant: RUN if variant == VARIANT else None
    shell.require_predecessor = lambda plan: None
    shell.initialized = initialized
    shell.rt = SimpleNamespace(
        **{
            **vars(runtime),
            "VerifiedDataset": SpatialDataset25,
            "EpochStream": EpochStream25,
            "TrainingLoop": SpatialTrainingLoop25,
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
