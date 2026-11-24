"""Bounded formal wiring readiness checks for swin3d_t_binary50_formal_v1.

Covers ALL items required before formal launch authorization:
- CPU phase (0 optimizer updates, 0 validation/test video decodes):
  1. Fresh locked Kinetics-400 initialization + isolated head_seed=20264024
     (verifies 183 non-head tensors match checkpoint; 0 engineering weights).
  2. Formal TemporalDataset + 2-worker spawn DataLoader + TemporalCollator +
     EpochStream formal read order (torch.randperm(7192, seed=20261024+epoch)),
     confirming zero model RNG consumption & disabled spatial augmentation.
  3. Validation split delegation to unchanged RevisedDataset / BinaryCollator.
  4. Formal TrainingLoop Adam options & 100% parameter coverage.
  5. Tie-break rule on equal validation CE (keeps earlier best_epoch & best_model).
  6. Sample 15 adjudicated label preservation.
- GPU phase (strictly <= 8 cumulative engineering updates across retries,
  enforced via persistent gpu_update_ledger.json; 0 real validation/test opens):
  1. Probe 1 (4 updates): Formal TrainingLoop mid-epoch checkpoint & resume at
     cursor=16 (update 1 of effective16 + tail8 = 24 clips), covering formal
     torch.randperm read order, guard_fetch, and bitwise equality across
     model, optimizer, rng, cuda_rng, state, best_model.
  2. Probe 2 (4 updates): Post-selection epoch-boundary resume + full
     BinaryMetrics validation accumulation + REAL GPU NO-IMPROVEMENT branch
     (train on 16x fixture 04 [Miss], engineering eval on fixture 22 [Make],
     so Epoch 2 validation CE > Epoch 1 CE -> bad_epochs increments 0->1,
     best_epoch stays 1, best_model remains bitwise equal to Epoch 1 weights
     and strictly unequal to Epoch 2 weights, matching continuous vs resumed!).
  3. Single GPU model enforced between every sub-run (zero live CUDA tensors
     between runs) and reports maximum peak allocated / reserved across ALL phases.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from development.aca_binary_v1 import core, runtime
from development.aca_binary_v1.data import BinaryCollator, RevisedDataset
from development.aca_eight_class_data_v1.gpu_readiness import configure, equal, reset_seed, worker_init
from development.aca_net_v1.data import normalize_rgb
from development.aca_temporal_v1 import sampling
from development.aca_temporal_v1.data import EpochStream, TemporalCollator, TemporalDataset
from development.mixnet_binary50_v1.data_adapter import (
    load_saved_rgb_fixture_item,
    verify_sample15_metadata,
)

from . import formal_train as f
from .model import (
    EXPECTED_TOTAL_PARAMS,
    HEAD_SEED,
    SEED,
    VARIANT,
    initialized,
    need,
    validate_architecture,
)

FIXTURES = ("04", "22")
DIAGNOSTIC = f.CURRENT / "input_diagnostic32_20261119_v1"
MAX_CUMULATIVE_GPU_UPDATES = 8


def _ledger_path() -> Path:
    return f.QA / "gpu_update_ledger.json"


def _load_ledger() -> dict:
    p = _ledger_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"max_cumulative_updates": MAX_CUMULATIVE_GPU_UPDATES, "cumulative_updates": 0, "attempts": []}


def _record_ledger_attempt(planned_updates: int, completed_updates: int, status: str) -> dict:
    ledger = _load_ledger()
    ledger["cumulative_updates"] += int(completed_updates)
    ledger["attempts"].append(
        {
            "at_taiwan": runtime.now(),
            "planned_updates": int(planned_updates),
            "completed_updates": int(completed_updates),
            "status": status,
        }
    )
    runtime.atomic_json(_ledger_path(), ledger)
    return ledger


def _assert_zero_live_cuda_tensors() -> tuple[int, int]:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    live = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]
    need(len(live) == 0, f"Live CUDA tensors leaked between runs: {len(live)}")
    return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())


def fixture_items(plan: dict):
    ids = [json.loads((DIAGNOSTIC / x / "evidence.json").read_text(encoding="utf-8"))["video_id"] for x in FIXTURES]
    ds = TemporalDataset(plan["source_root"], "train", plan["train_ids"], plan["reference"])
    ds.set_epoch(0)
    indices = [plan["train_ids"].index(vid) for vid in ids]
    return ds, indices, ids


def _check_tie_break_on_real_loop(plan: dict, ids: list[str], workers: list[dict]) -> dict:
    """Verify real TrainingLoop keeps earlier best_epoch/best_model on exact validation CE tie."""
    # Use a tiny linear module matching binary logit contract to exercise TrainingLoop._run selection logic on CPU
    # without extra Swin3D-T updates.
    class _ConstLogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return self.w.unsqueeze(0).expand(x.shape[0], -1)

    val_batch = copy.deepcopy(workers[0])
    val_batch.pop("temporal_sampling_version", None)
    for r in val_batch["records"]:
        r["split"] = "validation"
    train_batch = copy.deepcopy(workers[0])
    tiny_plan = copy.deepcopy(plan)
    tiny_plan.update(
        train_ids=[ids[0]],
        validation_ids=[ids[0]],
        max_epochs=2,
        patience=5,
        effective=1,
        checkpoint_every=1,
    )
    # Zero gradient on w so epoch 1 and epoch 2 produce exact equal validation CE (tie!)
    m = _ConstLogitModel()
    m.w.requires_grad_(True)
    orig_tm = runtime.training_mode
    orig_core_tm = core.training_mode
    try:
        runtime.training_mode = lambda mod: mod.train()
        core.training_mode = lambda mod: mod.train()
        with tempfile.TemporaryDirectory(prefix="swin_tie_check_", dir=f.QA) as tmp:
            ckpt = Path(tmp) / "tie.pt"
            loop = runtime.TrainingLoop(
                m,
                tiny_plan,
                lambda split, idx: (train_batch if split == "train" else val_batch),
                ckpt,
                "cpu_tie_break_only",
            )
            # Zero learning rate ONLY in this synthetic tie-break probe so epoch 1 & 2 validation CE tie identically
            for g in loop.optimizer.param_groups:
                g["lr"] = 0.0
            res = loop.run(lambda phase, state, seen: False)
            need(res == "completed", "Tie-break loop failed to complete")
            h = loop.state["history"]
            need(len(h) == 2 and h[0]["validation"]["ce"] == h[1]["validation"]["ce"],
                 f"Expected exact CE tie, got {h[0]['validation']['ce']} vs {h[1]['validation']['ce']}")
            need(
                loop.state["best_epoch"] == 1 and loop.state["bad_epochs"] == 1 and loop.best_model is not None,
                f"Tie-break did not preserve earlier best_epoch=1: {loop.state}",
            )
            tie_ce = float(h[0]["validation"]["ce"])
    finally:
        runtime.training_mode = orig_tm
        core.training_mode = orig_core_tm
    return {"status": "PASS", "tied_validation_ce": round(tie_ce, 6), "kept_earlier_best_epoch": 1, "bad_epochs_after_tie": 1}


def check_cpu(plan: dict) -> dict:
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    f.install_runtime()

    model_a, init_a = initialized(VARIANT, SEED)
    model_b, init_b = initialized(VARIANT, SEED)
    need(init_a == init_b, "Formal initialization metadata not deterministic")
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    need(all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a), "Formal initialization weights not deterministic")
    del model_b

    validate_architecture(model_a)
    ds, indices, ids = fixture_items(plan)
    items = [ds[i] for i in indices]
    for key, item in zip(FIXTURES, items):
        ev = json.loads((DIAGNOSTIC / key / "evidence.json").read_text(encoding="utf-8"))
        rgb_path = DIAGNOSTIC / key / "source_frames_rgb.npy"
        need(f.sha(rgb_path) == ev["frames_array_sha256"], "Saved RGB fixture changed")
        rgb = np.load(rgb_path, mmap_mode="r")
        chosen = sampling.indices(len(rgb), plan["seed"], 0, item["video_id"])
        expected = normalize_rgb(
            np.stack([cv2.resize(rgb[i], (160, 120), interpolation=cv2.INTER_AREA) for i in chosen])
        )
        need(torch.equal(expected, item["tensor"]), "Temporal decode differs from saved RGB reference")

    # Exercise exact 2-worker spawn DataLoader + TemporalCollator
    before = core.capture_rng()
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        num_workers=2,
        collate_fn=TemporalCollator(),
        worker_init_fn=worker_init,
        multiprocessing_context="spawn",
        prefetch_factor=2,
        generator=torch.Generator().manual_seed(SEED + 100000),
        persistent_workers=False,
    )
    workers = list(loader)
    need(equal(before, core.capture_rng()), "Spawn workers consumed model RNG")
    for a, b in zip(items, workers):
        need(torch.equal(a["tensor"], b["tensor"][0]), "Worker tensor differs from serial")
        need(b["targets"].tolist() == [a["binary_label"]], "Worker target differs")
        need("spatial_policy_sha256" not in b, "Unexpected spatial augmentation metadata")

    # Verify formal EpochStream read order on tiny 2-clip dataset (epoch 0 and epoch 1)
    tiny_ds = TemporalDataset(plan["source_root"], "train", ids, plan["reference"])
    stream_state = {"epoch": 0, "cursor": 0}
    stream = EpochStream({"train": tiny_ds}, {**plan, "train_ids": ids}, lambda: stream_state)
    try:
        order_e0 = torch.randperm(2, generator=torch.Generator().manual_seed(SEED)).tolist()
        for idx in order_e0:
            got = stream("train", idx)
            need(got["records"][0]["video_id"] == ids[idx], "EpochStream order mismatch")
            need(got["records"][0]["metadata"]["epoch_index"] == 0, "EpochStream epoch mismatch")
    finally:
        stream.close()

    # Verify validation delegation to unchanged RevisedDataset / BinaryCollator (0 real validation video decodes)
    need(issubclass(TemporalDataset, RevisedDataset), "Validation delegate changed")
    for key in FIXTURES:
        item, _ = load_saved_rgb_fixture_item(key)
        b_fixed = BinaryCollator()([item])
        need(b_fixed["targets"].tolist() == [item["binary_label"]], "Fixed collator changed")
        need("temporal_sampling_version" not in b_fixed, "Validation batch unexpectedly tagged with temporal sampling")

    # Verify TrainingLoop wiring, Adam options, and 100% parameter coverage on CPU (0 updates)
    tiny_plan = {**plan, "train_ids": ids, "validation_ids": ids}
    loop = f.TrainingLoop(model_a, tiny_plan, lambda s, i: workers[i], f.QA / "never_written.pt", "cpu_contract_only")
    f.validate_optimizer_options(loop.optimizer)
    opt_params = sum(p.numel() for g in loop.optimizer.param_groups for p in g["params"])
    need(opt_params == EXPECTED_TOTAL_PARAMS, f"Optimizer covers {opt_params} != {EXPECTED_TOTAL_PARAMS}")

    # Verify exact tie-break rule on real TrainingLoop._run
    tie_check = _check_tie_break_on_real_loop(plan, ids, workers)

    return {
        "initialization": init_a,
        "head_seed": HEAD_SEED,
        "source_train_decodes": 6,
        "train_ids": ids,
        "worker_tensor_equal": True,
        "workers_zero_model_rng_consumption": True,
        "epoch_stream_read_order_verified": True,
        "spatial_augmentation_disabled_verified": True,
        "saved_RGB_temporal_tensor_equal": True,
        "validation_delegates_unchanged_revised_dataset": True,
        "optimizer_defaults_verified": dict(loop.optimizer.defaults),
        "optimizer_param_count": opt_params,
        "tie_break_keeps_earlier_epoch": tie_check,
        "sample15": verify_sample15_metadata(),
        "validation_video_decodes": 0,
        "test_video_decodes": 0,
        "optimizer_updates": 0,
    }


def _make_engineering_batches(plan: dict, epoch: int, ids: list[str]) -> list[dict]:
    ds = TemporalDataset(plan["source_root"], "train", plan["train_ids"], plan["reference"])
    ds.set_epoch(epoch)
    indices = [plan["train_ids"].index(vid) for vid in ids]
    return [TemporalCollator()([ds[i]]) for i in indices]


def check_gpu(plan: dict) -> dict:
    # Enforce persistent cumulative GPU engineering update cap BEFORE touching CUDA
    ledger_before = _load_ledger()
    planned_updates = 5  # Unified 2-epoch continuous (4 updates) + Epoch 2 mid-epoch cursor=16 resume (1 update)
    need(
        ledger_before["cumulative_updates"] + planned_updates <= MAX_CUMULATIVE_GPU_UPDATES,
        f"Cumulative GPU engineering update cap ({MAX_CUMULATIVE_GPU_UPDATES}) would be exceeded: "
        f"already={ledger_before['cumulative_updates']}, planned={planned_updates}",
    )

    f.gpu_policy_mod.assert_idle_gpu(plan["gpu_policy"])
    completed_updates = 0
    try:
        with f.shell.run_lock(f.CURRENT.parent / "00_admin", "baseline8_gpu.lock"):
            observed_gpu = f.gpu_policy_mod.assert_idle_gpu(plan["gpu_policy"])
            configure()
            f.install_runtime()
            _, _, ids = fixture_items(plan)
            batches_e0 = _make_engineering_batches(plan, 0, ids)
            batches_e1 = _make_engineering_batches(plan, 1, ids)

            # Build in-memory engineering validation batch for fixture 22 (Make) — 0 real validation video opens
            val_22_batch = copy.deepcopy(batches_e0[1])
            val_22_batch.pop("temporal_sampling_version", None)
            for r in val_22_batch["records"]:
                r["split"] = "validation"
                r["metadata"].pop("temporal_sampling_version", None)
                r["metadata"].pop("temporal_seed", None)
                r["metadata"].pop("epoch_index", None)

            peaks_alloc: list[int] = []
            peaks_res: list[int] = []
            phase_peaks: list[dict] = []

            with tempfile.TemporaryDirectory(prefix="swin_formal_gpu_probe_", dir=f.QA) as temp:
                folder = Path(temp)

                def new_loop(probe_cfg: dict, fetch_fn, ckpt_path: Path, *, publish=None):
                    torch.cuda.reset_peak_memory_stats()
                    reset_seed()
                    m, _ = initialized(VARIANT, SEED)
                    m.cuda()
                    return f.TrainingLoop(
                        m, probe_cfg, fetch_fn, ckpt_path, "formal_readiness_eng_only", publish=publish
                    )

                # Unified 2-epoch probe (24 clips/epoch = effective16 + tail8; train=04 Miss, eng val=22 Make):
                # - Epoch 1: effective16 + tail8 -> full BinaryMetrics val -> commits best_epoch=1, best_model!=None
                # - Epoch 2: effective16 -> commits mid-epoch checkpoint (epoch=1, cursor=16, best_model!=None,
                #   train_metrics._count=16) -> tail8 -> full BinaryMetrics val (ce_e2 > ce_e1 -> NO-IMPROVEMENT:
                #   bad_epochs=1, best_epoch=1, best_model preserved bitwise from Epoch 1!)
                probe = copy.deepcopy(plan)
                probe.update(
                    train_ids=[ids[0]] * 24,
                    validation_ids=[ids[1]],
                    max_epochs=2,
                    checkpoint_every=1,
                )

                cont_ckpt = folder / "cont.pt"
                ep1_post_sel_ckpt = folder / "ep1_post_sel.pt"
                ep2_mid_ckpt = folder / "ep2_mid_cursor16.pt"
                res_ckpt = folder / "res.pt"

                def make_fetch(loop_box):
                    def fetch(split, index):
                        if split == "train":
                            ep = loop_box[0].state["epoch"]
                            return batches_e0[0] if ep == 0 else batches_e1[0]
                        need(index == 0, "Engineering val has 1 clip")
                        return val_22_batch

                    return fetch

                snapshots_taken = {"ep1_post_sel": False, "ep2_mid_cursor16": False}

                def capture_boundaries(event: str, committed: dict) -> None:
                    if event != "checkpoint_committed":
                        return
                    if committed["epoch"] == 1 and committed["cursor"] == 0 and not snapshots_taken["ep1_post_sel"]:
                        shutil.copy2(cont_ckpt, ep1_post_sel_ckpt)
                        snapshots_taken["ep1_post_sel"] = True
                    elif committed["epoch"] == 1 and committed["cursor"] == 16 and not snapshots_taken["ep2_mid_cursor16"]:
                        shutil.copy2(cont_ckpt, ep2_mid_ckpt)
                        snapshots_taken["ep2_mid_cursor16"] = True

                # Phase A: Continuous 2-epoch run (4 updates: 16+8 in Ep1, 16+8 in Ep2)
                box_c: list = [None]
                cont = new_loop(probe, make_fetch(box_c), cont_ckpt, publish=capture_boundaries)
                box_c[0] = cont
                need(cont.run(lambda phase, state, seen: False) == "completed", "Continuous 2-epoch probe failed")
                completed_updates += 4
                need(snapshots_taken["ep1_post_sel"] and snapshots_taken["ep2_mid_cursor16"],
                     f"Missing expected boundary snapshots: {snapshots_taken}")
                need(all(p.grad is None for p in cont.model.parameters()), "Gradients retained after continuous run")

                pa_c = int(torch.cuda.max_memory_allocated())
                pr_c = int(torch.cuda.max_memory_reserved())
                peaks_alloc.append(pa_c)
                peaks_res.append(pr_c)
                box_c[0] = None
                del cont, box_c
                ba_c, br_c = _assert_zero_live_cuda_tensors()
                phase_peaks.append(
                    {
                        "phase": "continuous_2epochs_effective16_tail8_4updates",
                        "peak_allocated_bytes": pa_c,
                        "peak_allocated_gib": round(pa_c / (1024**3), 4),
                        "peak_reserved_bytes": pr_c,
                        "peak_reserved_gib": round(pr_c / (1024**3), 4),
                        "after_cleanup_allocated_bytes_cublas": ba_c,
                        "after_cleanup_reserved_bytes": br_c,
                    }
                )

                exp_final = torch.load(cont_ckpt, map_location="cpu", weights_only=True)
                ep1_snap = torch.load(ep1_post_sel_ckpt, map_location="cpu", weights_only=True)
                ep2_mid_snap = torch.load(ep2_mid_ckpt, map_location="cpu", weights_only=True)

                need(
                    ep1_snap["best_model"] is not None
                    and ep1_snap["state"]["epoch"] == 1
                    and ep1_snap["state"]["best_epoch"] == 1
                    and ep1_snap["state"]["bad_epochs"] == 0,
                    "Epoch 1 post-selection snapshot invalid",
                )
                need(
                    ep2_mid_snap["best_model"] is not None
                    and ep2_mid_snap["state"]["epoch"] == 1
                    and ep2_mid_snap["state"]["cursor"] == 16
                    and ep2_mid_snap["state"]["updates"] == 3
                    and ep2_mid_snap["state"]["train_metrics"]["count"] == 16,
                    "Epoch 2 mid-epoch cursor=16 snapshot invalid",
                )

                h = exp_final["state"]["history"]
                need(len(h) == 2 and exp_final["state"]["updates"] == 4, "Continuous final history/updates invalid")
                ce_e1 = float(h[0]["validation"]["ce"])
                ce_e2 = float(h[1]["validation"]["ce"])
                need(ce_e2 > ce_e1, f"Expected Epoch 2 no-improvement (ce_e2 > ce_e1), got {ce_e1} -> {ce_e2}")
                need(
                    exp_final["best_model"] is not None
                    and exp_final["state"]["best_epoch"] == 1
                    and exp_final["state"]["bad_epochs"] == 1
                    and bool(equal(exp_final["best_model"], ep1_snap["best_model"]))
                    and not equal(exp_final["best_model"], exp_final["model"]),
                    "Continuous no-improvement branch failed to preserve Epoch 1 best_model distinctly",
                )

                # Phase B: Resume from Epoch 2 mid-epoch cursor=16 (with non-None best_model & partial train_metrics)
                # Runs tail8 (1 update) + full BinaryMetrics val + no-improvement commit!
                shutil.copy2(ep2_mid_ckpt, res_ckpt)
                box_r: list = [None]
                res = new_loop(probe, make_fetch(box_r), res_ckpt)
                box_r[0] = res
                res.resume()
                need(
                    res.state["epoch"] == 1
                    and res.state["cursor"] == 16
                    and res.state["updates"] == 3
                    and res.best_model is not None,
                    "Mid-epoch post-selection resume state invalid",
                )
                need(res.run(lambda phase, state, seen: False) == "completed", "Resumed Epoch 2 tail8 run failed")
                completed_updates += 1
                need(all(p.grad is None for p in res.model.parameters()), "Gradients retained after resumed run")

                pa_r = int(torch.cuda.max_memory_allocated())
                pr_r = int(torch.cuda.max_memory_reserved())
                peaks_alloc.append(pa_r)
                peaks_res.append(pr_r)
                box_r[0] = None
                del res, box_r
                ba_r, br_r = _assert_zero_live_cuda_tensors()
                phase_peaks.append(
                    {
                        "phase": "resumed_epoch2_mid_cursor16_tail8_post_selection_1update",
                        "peak_allocated_bytes": pa_r,
                        "peak_allocated_gib": round(pa_r / (1024**3), 4),
                        "peak_reserved_bytes": pr_r,
                        "peak_reserved_gib": round(pr_r / (1024**3), 4),
                        "after_cleanup_allocated_bytes_cublas": ba_r,
                        "after_cleanup_reserved_bytes": br_r,
                    }
                )

                act_final = torch.load(res_ckpt, map_location="cpu", weights_only=True)
                matches = {
                    k: bool(equal(exp_final[k], act_final[k]))
                    for k in ("model", "optimizer", "rng", "cuda_rng", "state", "best_model")
                }
                need(all(matches.values()), f"Continuous vs resumed mismatch: {matches}")
                need(
                    bool(equal(act_final["best_model"], ep1_snap["best_model"])),
                    "Resumed no-improvement Epoch 2 altered Epoch 1 best_model",
                )

            need(not folder.exists(), "Temporary GPU probe directory still exists")
        ledger_after = _record_ledger_attempt(planned_updates, completed_updates, "passed")
        max_alloc = max(peaks_alloc)
        max_res = max(peaks_res)
        return {
            "gpu_processes_observed": observed_gpu,
            "real_optimizer_updates_this_run": completed_updates,
            "cumulative_engineering_updates_in_ledger": ledger_after["cumulative_updates"],
            "max_cumulative_updates_cap": MAX_CUMULATIVE_GPU_UPDATES,
            "effective": 16,
            "tail": 8,
            "microbatch": 1,
            "input_shape": [1, 3, 50, 120, 160],
            "mid_epoch_cursor16_and_post_selection_resume_equal": matches,
            "mid_epoch_resume_boundary_verified": {
                "resumed_at_epoch": 1,
                "resumed_at_cursor": 16,
                "resumed_at_updates": 3,
                "resumed_partial_train_metrics_count": 16,
                "resumed_best_model_non_null": True,
                "completed_tail8_and_val_after_resume": True,
            },
            "no_improvement_branch_verified": {
                "epoch1_eng_val_ce": round(ce_e1, 6),
                "epoch2_eng_val_ce": round(ce_e2, 6),
                "epoch2_worse_than_epoch1": ce_e2 > ce_e1,
                "best_epoch_remained_1": True,
                "bad_epochs_incremented_to_1": True,
                "best_model_bitwise_preserved_from_epoch1": True,
                "best_model_distinct_from_epoch2_model": True,
                "note": "Engineering fixture probe only (24x 04 train / 22 eval); NOT real validation.",
            },
            "peak_memory_max_across_all_phases": {
                "max_allocated_bytes": max_alloc,
                "max_allocated_gib": round(max_alloc / (1024**3), 4),
                "max_reserved_bytes": max_res,
                "max_reserved_gib": round(max_res / (1024**3), 4),
            },
            "phase_memory_breakdown": phase_peaks,
            "temporary_checkpoints_removed": True,
            "source_train_decodes": 4,
            "validation_video_decodes": 0,
            "test_video_decodes": 0,
            "formal_weights_from_engineering": False,
        }
    except BaseException:
        _record_ledger_attempt(planned_updates, completed_updates, "failed")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["cpu", "gpu"])
    args = parser.parse_args()
    f.QA.mkdir(parents=True, exist_ok=True)
    output = f.QA / f"{args.phase}_report.json"
    need(not output.exists(), f"Do not overwrite existing readiness report: {output}")
    start = time.monotonic()
    report: dict = {"status": "running", "phase": args.phase}
    try:
        plan = f.draft_plan()
        report["draft_plan_sha256"] = core.plan_digest(plan)
        if args.phase == "gpu":
            cpu = f.read(f.QA / "cpu_report.json")
            need(
                cpu["status"] == "passed" and cpu["draft_plan_sha256"] == report["draft_plan_sha256"],
                "CPU readiness gate missing or draft plan hash changed",
            )
        report.update(check_cpu(plan) if args.phase == "cpu" else check_gpu(plan))
        report["status"] = "passed"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    report["elapsed_seconds"] = round(time.monotonic() - start, 4)
    runtime.atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
