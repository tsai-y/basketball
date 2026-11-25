"""Bounded CPU/GPU readiness for Swin3D-T + clip_flip_color_light_v2 (seed=20261025)."""

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

import cv2
import torch
from torch.utils.data import DataLoader, Subset

from development.aca_binary_v1 import core, runtime
from development.aca_binary_v1.data import BinaryCollator, RevisedDataset
from development.aca_eight_class_data_v1.gpu_readiness import equal, worker_init
from development.aca_spatial_pair_v1.data import SpatialCollator25, SpatialDataset25
from development.aca_spatial_v1 import augment
from development.mixnet_binary50_v1.data_adapter import (
    load_saved_rgb_fixture_item,
    verify_sample15_metadata,
)
from development.swin3d_t_binary50_formal_v1.model import EXPECTED_TOTAL_PARAMS
from development.swin3d_t_binary50_pair_v1.model import (
    HEAD_SEED,
    SEED,
    VARIANT,
    initialized,
    need,
    validate_architecture,
)

from . import formal_train as f

FIXTURES = ("04", "22")
DIAGNOSTIC = f.CURRENT / "input_diagnostic32_20261119_v1"
MAX_CUMULATIVE_GPU_UPDATES = 8


def _ledger_path() -> Path:
    return f.QA / "gpu_update_ledger.json"


def _load_ledger() -> dict:
    p = _ledger_path()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"max_cumulative_updates": MAX_CUMULATIVE_GPU_UPDATES, "cumulative_updates": 0, "events": []}


def _increment_ledger_atomic(delta: int, note: str) -> dict:
    ledger = _load_ledger()
    ledger["cumulative_updates"] += int(delta)
    ledger["events"].append({"at_taiwan": runtime.now(), "delta": int(delta), "note": note})
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
    ds = SpatialDataset25(plan["source_root"], "train", plan["train_ids"], plan["reference"])
    ds.set_epoch(0)
    indices = [plan["train_ids"].index(vid) for vid in ids]
    return ds, indices, ids


def check_cpu(plan: dict) -> dict:
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    f.install_runtime()

    model, init = initialized(VARIANT, SEED)
    validate_architecture(model)
    ds, indices, ids = fixture_items(plan)
    items = [ds[i] for i in indices]

    # Verify seed25 spatial parameters differ from seed24
    need(augment.parameters(SEED, 0, ids[0]) != augment.parameters(20261024, 0, ids[0]),
         "Seed25 spatial params identical to seed24")

    before = core.capture_rng()
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        num_workers=2,
        collate_fn=SpatialCollator25(),
        worker_init_fn=worker_init,
        multiprocessing_context="spawn",
        prefetch_factor=2,
        generator=torch.Generator().manual_seed(SEED + 100000),
        persistent_workers=False,
    )
    workers = list(loader)
    need(equal(before, core.capture_rng()), "Workers consumed model RNG")
    for a, b in zip(items, workers):
        need(torch.equal(a["tensor"], b["tensor"][0]), "Worker spatial25 tensor differs from serial")
        need(b.get("spatial_policy_sha256") == augment.POLICY_SHA and b.get("spatial_seed") == SEED,
             "Missing seed25 spatial metadata")

    need(issubclass(SpatialDataset25, RevisedDataset), "Validation delegate changed")
    for key in FIXTURES:
        item, _ = load_saved_rgb_fixture_item(key)
        b_val = BinaryCollator()([item])
        need("spatial_policy_sha256" not in b_val and "temporal_sampling_version" not in b_val,
             "Validation batch contaminated")

    tiny_plan = {**plan, "train_ids": ids, "validation_ids": ids}
    loop = f.SpatialTrainingLoop25(model, tiny_plan, lambda s, i: workers[i], f.QA / "never.pt", "cpu_only")
    opt_params = sum(p.numel() for g in loop.optimizer.param_groups for p in g["params"])
    need(opt_params == EXPECTED_TOTAL_PARAMS, "Optimizer param count mismatch")

    return {
        "initialization": init,
        "head_seed": HEAD_SEED,
        "spatial_policy_sha256": augment.POLICY_SHA,
        "spatial_seed_distinct_from_seed24": True,
        "worker_tensor_equal": True,
        "workers_zero_model_rng_consumption": True,
        "optimizer_param_count": opt_params,
        "sample15": verify_sample15_metadata(),
        "validation_video_decodes": 0,
        "test_video_decodes": 0,
        "swin3d_t_optimizer_updates": 0,
    }


def _make_batches(plan: dict, epoch: int, ids: list[str]) -> list[dict]:
    ds = SpatialDataset25(plan["source_root"], "train", plan["train_ids"], plan["reference"])
    ds.set_epoch(epoch)
    indices = [plan["train_ids"].index(vid) for vid in ids]
    return [SpatialCollator25()([ds[i]]) for i in indices]


def check_gpu(plan: dict) -> dict:
    ledger = _load_ledger()
    planned = 5
    need(
        ledger["cumulative_updates"] + planned <= MAX_CUMULATIVE_GPU_UPDATES,
        f"Cumulative cap ({MAX_CUMULATIVE_GPU_UPDATES}) exceeded",
    )
    f.gpu_policy_mod.assert_idle_gpu(plan["gpu_policy"])
    with f.shell.run_lock(f.CURRENT.parent / "00_admin", "baseline8_gpu.lock"):
        observed_gpu = f.gpu_policy_mod.assert_idle_gpu(plan["gpu_policy"])
        f.configure()
        f.install_runtime()
        _, _, ids = fixture_items(plan)
        batches_e0 = _make_batches(plan, 0, ids)
        batches_e1 = _make_batches(plan, 1, ids)

        val_22 = copy.deepcopy(batches_e0[1])
        for k in ("temporal_sampling_version", "spatial_policy_sha256", "spatial_seed"):
            val_22.pop(k, None)
        for r in val_22["records"]:
            r["split"] = "validation"
            for k in ("temporal_sampling_version", "temporal_seed", "epoch_index", "spatial_augmentation"):
                r["metadata"].pop(k, None)

        peaks_alloc: list[int] = []
        peaks_res: list[int] = []
        with tempfile.TemporaryDirectory(prefix="swin_sp25_gpu_probe_", dir=f.QA) as temp:
            folder = Path(temp)
            probe = copy.deepcopy(plan)
            probe.update(train_ids=[ids[0]] * 24, validation_ids=[ids[1]], max_epochs=2, checkpoint_every=1)
            cont_ckpt = folder / "cont.pt"
            ep1_ckpt = folder / "ep1.pt"
            ep2_mid_ckpt = folder / "ep2_mid.pt"
            res_ckpt = folder / "res.pt"

            def make_fetch(box):
                def fetch(split, index):
                    if split == "train":
                        return batches_e0[0] if box[0].state["epoch"] == 0 else batches_e1[0]
                    return val_22

                return fetch

            snaps = {"ep1": False, "ep2_mid": False}
            prev_u_c = [0]

            def pub_c(event: str, committed: dict) -> None:
                if event != "checkpoint_committed":
                    return
                du = committed["updates"] - prev_u_c[0]
                if du > 0:
                    _increment_ledger_atomic(du, f"cont_ep{committed['epoch']}_u{committed['updates']}")
                    prev_u_c[0] = committed["updates"]
                if committed["epoch"] == 1 and committed["cursor"] == 0 and not snaps["ep1"]:
                    shutil.copy2(cont_ckpt, ep1_ckpt)
                    snaps["ep1"] = True
                elif committed["epoch"] == 1 and committed["cursor"] == 16 and not snaps["ep2_mid"]:
                    shutil.copy2(cont_ckpt, ep2_mid_ckpt)
                    snaps["ep2_mid"] = True

            def new_loop(fetch_fn, ckpt: Path, pub=None):
                torch.cuda.reset_peak_memory_stats()
                f.configure()
                m, _ = initialized(VARIANT, SEED)
                m.cuda()
                return f.SpatialTrainingLoop25(m, probe, fetch_fn, ckpt, "sp25_readiness_eng", publish=pub)

            box_c: list = [None]
            cont = new_loop(make_fetch(box_c), cont_ckpt, pub_c)
            box_c[0] = cont
            need(cont.run(lambda p, s, seen: False) == "completed", "Spatial25 cont probe failed")
            peaks_alloc.append(int(torch.cuda.max_memory_allocated()))
            peaks_res.append(int(torch.cuda.max_memory_reserved()))
            box_c[0] = None
            del cont, box_c
            _assert_zero_live_cuda_tensors()

            exp = torch.load(cont_ckpt, map_location="cpu", weights_only=True)
            ep1 = torch.load(ep1_ckpt, map_location="cpu", weights_only=True)
            ce1, ce2 = exp["state"]["history"][0]["validation"]["ce"], exp["state"]["history"][1]["validation"]["ce"]
            need(
                ce2 > ce1
                and exp["state"]["best_epoch"] == 1
                and exp["state"]["bad_epochs"] == 1
                and equal(exp["best_model"], ep1["best_model"])
                and not equal(exp["best_model"], exp["model"]),
                "Spatial25 no-improvement branch failed",
            )

            shutil.copy2(ep2_mid_ckpt, res_ckpt)
            prev_u_r = [3]

            def pub_r(event: str, committed: dict) -> None:
                if event == "checkpoint_committed":
                    du = committed["updates"] - prev_u_r[0]
                    if du > 0:
                        _increment_ledger_atomic(du, f"res_ep{committed['epoch']}_u{committed['updates']}")
                        prev_u_r[0] = committed["updates"]

            box_r: list = [None]
            res = new_loop(make_fetch(box_r), res_ckpt, pub_r)
            box_r[0] = res
            res.resume()
            need(res.run(lambda p, s, seen: False) == "completed", "Spatial25 resumed probe failed")
            peaks_alloc.append(int(torch.cuda.max_memory_allocated()))
            peaks_res.append(int(torch.cuda.max_memory_reserved()))
            box_r[0] = None
            del res, box_r
            _assert_zero_live_cuda_tensors()

            act = torch.load(res_ckpt, map_location="cpu", weights_only=True)
            matches = {k: bool(equal(exp[k], act[k])) for k in ("model", "optimizer", "rng", "cuda_rng", "state", "best_model")}
            need(all(matches.values()), f"Spatial25 resume mismatch: {matches}")

        final_ledger = _load_ledger()
        return {
            "gpu_processes_observed": observed_gpu,
            "cumulative_updates_atomic_ledger": final_ledger["cumulative_updates"],
            "max_cumulative_updates_cap": MAX_CUMULATIVE_GPU_UPDATES,
            "mid_epoch_and_post_selection_no_improvement_matches": matches,
            "no_improvement_ce_e1_to_e2": [round(float(ce1), 6), round(float(ce2), 6)],
            "max_allocated_gib": round(max(peaks_alloc) / (1024**3), 4),
            "max_reserved_gib": round(max(peaks_res) / (1024**3), 4),
            "validation_video_decodes": 0,
            "test_video_decodes": 0,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["cpu", "gpu"])
    args = parser.parse_args()
    f.QA.mkdir(parents=True, exist_ok=True)
    output = f.QA / f"{args.phase}_report.json"
    need(not output.exists(), f"Report exists: {output}")
    start = time.monotonic()
    report: dict = {"status": "running", "phase": args.phase}
    try:
        plan = f.draft_plan()
        report["draft_plan_sha256"] = core.plan_digest(plan)
        if args.phase == "gpu":
            cpu = f.read(f.QA / "cpu_report.json")
            need(cpu["status"] == "passed" and cpu["draft_plan_sha256"] == report["draft_plan_sha256"], "CPU gate mismatch")
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
