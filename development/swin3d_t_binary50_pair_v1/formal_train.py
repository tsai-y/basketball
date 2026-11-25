"""Paired second-seed (seed=20261025) formal Swin3D-T run vs ACA temporal50 seed25."""

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
from development.aca_temporal_pair_v1.data import EpochStream as EpochStream25
from development.aca_temporal_pair_v1.data import TemporalDataset as TemporalDataset25
from development.aca_temporal_v1 import sampling
from development.adjudicated_input_v1 import adapter as labels
from development.swin3d_t_binary50_formal_v1 import formal_train as s24_formal

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
RUN = CURRENT / "swin3d_t_binary50_seed20261025_labelrev1_v1"
QA = CURRENT / "swin3d_t_binary50_pair_seed20261025_readiness_20261125_v1"
CONTROL_ACA25 = CURRENT / "aca_binary_temporal50_seed20261025_labelrev1_v1"
CONTROL_ACA25_SHA = "7648128de39039f9aac16ea316a3eca8c9e7d2412de37adc70995037122857a4"
SEED24_SWIN = CURRENT / "swin3d_t_binary50_seed20261024_labelrev1_v1"
SEED24_SWIN_SHA = "6bd908ced780140ebf88fae132a566f338eb2b04e6ce76b5a9c44d5d6ff2787c"
SCHEMA = "swin3d_t_binary50_pair_seed25_formal_v1"
_ORIGINAL_CONFIGURE = shell.configure


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configure() -> None:
    _ORIGINAL_CONFIGURE()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def source_hashes(control: dict) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, expected in control["code_sha256"].items():
        p = ROOT / name
        need(sha(p) == expected, f"Shared source changed: {name}")
        hashes[str(p)] = expected
    for folder in ("swin3d_t_binary50_v1", "swin3d_t_binary50_formal_v1", "swin3d_t_binary50_pair_v1"):
        for p in sorted((ROOT / "development" / folder).glob("*.py")):
            hashes[str(p)] = sha(p)
    for name, expected in labels.PINNED_CODE.items():
        p = labels.f.ROOT / name
        need(sha(p) == expected, f"Pinned data source changed: {name}")
        hashes[str(p)] = expected
    return dict(sorted(hashes.items()))


def _verify_completed_run(directory: Path, expected_plan_sha: str) -> dict:
    plan = read(directory / "plan.json")
    need(core.plan_digest(plan) == expected_plan_sha, f"Plan SHA mismatch: {directory}")
    summary = read(directory / "summary.json")
    history = read(directory / "history.json")
    need(
        summary["status"] == "completed"
        and summary["plan_sha256"] == expected_plan_sha
        and summary["test_run"] is False
        and len(history) == summary["completed_epochs"]
        and summary["selected_validation"] == min(history, key=lambda r: r["validation"]["ce"])["validation"],
        f"Completed run invalid: {directory}",
    )
    return {
        "directory": str(directory),
        "plan_sha256": expected_plan_sha,
        "summary_sha256": sha(directory / "summary.json"),
        "history_sha256": sha(directory / "history.json"),
        "best_epoch": summary["best_epoch"],
        "selected_ce": summary["selected_validation"]["ce"],
        "selected_accuracy": summary["selected_validation"]["accuracy"],
    }


def draft_plan() -> dict:
    control = read(CONTROL_ACA25 / "plan.json")
    aca25_ref = _verify_completed_run(CONTROL_ACA25, CONTROL_ACA25_SHA)
    swin24_ref = _verify_completed_run(SEED24_SWIN, SEED24_SWIN_SHA)

    rev = labels.LabelRevision()
    manifest = rev._manifest
    need(
        control["reference"] == manifest["reference"]
        and control["train_ids"] == manifest["train_ids"]
        and control["validation_ids"] == manifest["validation_ids"],
        "Data protocol changed",
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
    plan = {k: control[k] for k in inherited_keys}
    need(plan["seed"] == SEED and plan["max_epochs"] == 30 and plan["patience"] == 5, "Unexpected schedule")
    need(plan["effective"] == 16 and plan["tail_videos"] == 8 and plan["validation_count"] == 1547,
         "Unexpected batch/split counts")

    plan.update(
        schema=SCHEMA,
        variant=VARIANT,
        purpose="paired_second_seed_confirmation_Swin3D_T_dynamic_pos_Kinetics400_vs_ACA_temporal50",
        allowed_run_directory=str(RUN),
        model_display_name="含動態位置索引修正的 Swin3D-T",
        lsta_tsci_included=False,
        spatial_augmentation="disabled_matches_temporal50_control",
        normalization_policy=NORM_POLICY,
        precision="FP32_strict_no_AMP_no_TF32",
        optimizer=s24_formal.OPTIMIZER_SPEC,
        head={
            "type": "Linear",
            "in_features": 768,
            "out_features": 2,
            "initialization": "trunc_normal_std_0p02_zero_bias",
            "initialization_seed": HEAD_SEED,
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
            "primary_control_seed25": aca25_ref,
            "seed24_swin3d_t_reference": swin24_ref,
            "comparison_type": "paired_second_seed_system_comparison",
            "historical_test_contact_disclosed": True,
        },
        runtime={**base.runtime_identity(), "torchvision": torchvision.__version__},
        source_inventory=base.source_inventory(plan["source_root"], plan["reference"]),
        code_sha256=source_hashes(control),
        gpu_policy=gpu_policy_mod.make_gpu_policy(),
        test_enabled=False,
        automatic_restart=False,
        automatic_next_arm=False,
        checkpoint="single_atomic_last.pt_current_model_Adam_RNG_cudaRNG_history_best_model",
        notifications="no_hourly_reports",
    )
    need(plan["source_inventory"] == control["source_inventory"], "Source inventory changed")
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


class TrainingLoop25(runtime.TrainingLoop):
    def __init__(self, model, plan, fetch, *args, **kwargs):
        need(
            plan["schema"] == SCHEMA
            and plan["variant"] == VARIANT
            and plan["normalization_policy"] == NORM_POLICY
            and plan["spatial_augmentation"] == "disabled_matches_temporal50_control"
            and plan["temporal_sampling"] == sampling.POLICY
            and plan["seed"] == SEED
            and plan["optimizer"] == s24_formal.OPTIMIZER_SPEC,
            "Wrong formal Swin3D-T seed25 plan",
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
                    and all(r["metadata"]["epoch_index"] == self.state["epoch"] for r in batch["records"])
                    and all(r["metadata"]["temporal_seed"] == SEED for r in batch["records"]),
                    "Stale or wrong-seed train temporal sampling",
                )
            else:
                need("temporal_sampling_version" not in batch, "Validation augmentation forbidden")
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
            "VerifiedDataset": TemporalDataset25,
            "EpochStream": EpochStream25,
            "TrainingLoop": TrainingLoop25,
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
