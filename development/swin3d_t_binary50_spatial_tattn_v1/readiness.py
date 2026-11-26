"""Bounded CPU/GPU readiness for Swin3D-T + Gated Temporal Attention Pooling (seed=20261024)."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from development.aca_binary_v1 import core, runtime
from development.aca_binary_v1.data import BinaryCollator, RevisedDataset
from development.aca_eight_class_data_v1.gpu_readiness import configure, equal, reset_seed, worker_init
from development.aca_spatial_v1 import augment
from development.aca_spatial_v1.data import SpatialCollator, SpatialDataset
from development.mixnet_binary50_v1.data_adapter import (
    load_saved_rgb_fixture_item,
    verify_sample15_metadata,
)
from development.swin3d_t_binary50_formal_v1.model import (
    VARIANT as BASE_VARIANT,
    initialized as base_initialized,
)

from . import formal_train as f
from .model import (
    ATTN_PARAM_KEYS,
    ATTN_SEED,
    EXPECTED_TOTAL_PARAMS_TATTN,
    HEAD_SEED,
    SEED,
    VARIANT,
    GatedTemporalAttentionPool,
    initialized,
    need,
    validate_architecture,
)

FIXTURES = ("04", "22")
DIAGNOSTIC = f.CURRENT / "input_diagnostic32_20261119_v1"
MAX_CUMULATIVE_GPU_UPDATES = 8
PROXIMITY_RTOL = 1e-5
PROXIMITY_ATOL = 1e-6


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
    ds = SpatialDataset(plan["source_root"], "train", plan["train_ids"], plan["reference"])
    ds.set_epoch(0)
    indices = [plan["train_ids"].index(vid) for vid in ids]
    return ds, indices, ids


def check_cpu(plan: dict) -> dict:
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    f.install_runtime()

    # 1. Verify RNG isolation during GatedTemporalAttentionPool construction + init
    rng_before_attn = core.capture_rng()
    _ = GatedTemporalAttentionPool(768, 128, attn_seed=ATTN_SEED)
    need(equal(rng_before_attn, core.capture_rng()), "GatedTemporalAttentionPool construction mutated global RNG")

    # 2. Build both baseline and new model fresh from locked K400 weights + head_seed=20264024
    base_model, _ = base_initialized(BASE_VARIANT, SEED)
    model, init = initialized(VARIANT, SEED)
    validate_architecture(model)

    # Bitwise equality of all 185 base tensors by key name
    base_sd = base_model.state_dict()
    new_sd = model.state_dict()
    need(len(base_sd) == 185 and len(new_sd) == 190, "Unexpected state_dict sizes")
    need(set(new_sd) - set(base_sd) == set(ATTN_PARAM_KEYS), "Unexpected extra state_dict keys")
    for k, v in base_sd.items():
        need(torch.equal(v, new_sd[k]), f"Base tensor mismatch on key {k}")

    # 3. Data & worker parity + validation zero-aug check (no real validation video decodes)
    ds, indices, ids = fixture_items(plan)
    items = [ds[i] for i in indices]
    before_loader = core.capture_rng()
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        num_workers=2,
        collate_fn=SpatialCollator(),
        worker_init_fn=worker_init,
        multiprocessing_context="spawn",
        prefetch_factor=2,
        generator=torch.Generator().manual_seed(SEED + 100000),
        persistent_workers=False,
    )
    workers = list(loader)
    need(equal(before_loader, core.capture_rng()), "Workers consumed model RNG")
    for a, b in zip(items, workers):
        need(torch.equal(a["tensor"], b["tensor"][0]), "Worker spatial tensor differs from serial")
        need(b.get("spatial_policy_sha256") == augment.POLICY_SHA, "Missing spatial policy SHA")

    need(issubclass(SpatialDataset, RevisedDataset), "Validation delegate changed")
    for key in FIXTURES:
        item, _ = load_saved_rgb_fixture_item(key)
        b_val = BinaryCollator()([item])
        need(
            "spatial_policy_sha256" not in b_val and "temporal_sampling_version" not in b_val,
            "Validation batch contaminated",
        )

    # 4. Initial uniform alpha + numerical proximity vs baseline Swin3D-T (eval mode)
    base_model.eval()
    model.eval()
    x0 = workers[0]["tensor"]
    with torch.no_grad():
        base_logits = base_model(x0)
        rng_pre_fwd = core.capture_rng()
        new_logits = model(x0)
        rng_post_fwd = core.capture_rng()
        diag_logits, alpha0 = model.forward_with_attention(x0)
        rng_post_diag = core.capture_rng()

    need(torch.equal(new_logits, diag_logits), "forward_with_attention altered logits vs forward")
    need(equal(rng_pre_fwd, rng_post_fwd) and equal(rng_post_fwd, rng_post_diag), "Eval forward/diag altered RNG")
    need(tuple(alpha0.shape) == (1, 25), f"Unexpected alpha shape: {tuple(alpha0.shape)}")
    alpha_max_diff_from_uniform = float((alpha0 - (1.0 / 25.0)).abs().max().item())
    alpha_sum_err = float(abs(float(alpha0.sum().item()) - 1.0))
    need(alpha_max_diff_from_uniform <= 1e-7 and alpha_sum_err <= 1e-6, "Initial alpha not uniform 1/25")

    logits_max_abs_diff = float((new_logits - base_logits).abs().max().item())
    need(
        torch.allclose(new_logits, base_logits, rtol=PROXIMITY_RTOL, atol=PROXIMITY_ATOL),
        f"Initial logits diverged beyond rtol={PROXIMITY_RTOL}, atol={PROXIMITY_ATOL}: max_abs_diff={logits_max_abs_diff}",
    )

    # 5. Initial pooled feature & input-feature gradient parity vs mean pooling + 2-step synthetic grad flow
    syn_gen = torch.Generator(device="cpu").manual_seed(12345)
    h_syn_a = torch.randn(2, 25, 768, generator=syn_gen, dtype=torch.float32, requires_grad=True)
    h_syn_b = h_syn_a.detach().clone().requires_grad_(True)
    y_syn = torch.tensor([0, 1], dtype=torch.long)

    pool_probe = GatedTemporalAttentionPool(768, 128, attn_seed=ATTN_SEED)
    head_probe = copy.deepcopy(model.head)

    z_attn, alpha_syn = pool_probe(h_syn_a)
    z_mean = h_syn_b.mean(dim=1)
    pooled_max_abs_diff = float((z_attn.detach() - z_mean.detach()).abs().max().item())
    need(
        torch.allclose(z_attn.detach(), z_mean.detach(), rtol=PROXIMITY_RTOL, atol=PROXIMITY_ATOL),
        f"Initial pooled features diverged: max_abs_diff={pooled_max_abs_diff}",
    )

    loss_attn_1 = F.cross_entropy(head_probe(z_attn), y_syn)
    loss_mean_1 = F.cross_entropy(head_probe(z_mean), y_syn)
    loss_attn_1.backward()
    loss_mean_1.backward()

    input_grad_max_abs_diff = float((h_syn_a.grad - h_syn_b.grad).abs().max().item())
    need(
        torch.allclose(h_syn_a.grad, h_syn_b.grad, rtol=PROXIMITY_RTOL, atol=PROXIMITY_ATOL),
        f"Initial input-feature gradient diverged: max_abs_diff={input_grad_max_abs_diff}",
    )

    # Step 1 backward expectations: V/U grads == 0, w grad finite & norm > 0
    step1_norms = {k: float(p.grad.norm().item()) for k, p in pool_probe.named_parameters()}
    need(
        all(step1_norms[k] == 0.0 for k in ("V.weight", "V.bias", "U.weight", "U.bias"))
        and torch.isfinite(pool_probe.w.weight.grad).all().item()
        and step1_norms["w.weight"] > 0.0,
        f"Unexpected step-1 synthetic grad norms: {step1_norms}",
    )

    # Run 2 synthetic Adam updates on standalone pool_probe + head_probe (0 Swin3D-T backbone updates!)
    syn_opt = torch.optim.Adam(
        list(pool_probe.parameters()) + list(head_probe.parameters()),
        lr=1e-4,
        foreach=False,
        fused=False,
    )
    syn_opt.step()  # synthetic update #1 (moves w away from 0)
    syn_opt.zero_grad(set_to_none=True)

    z_attn_2, _ = pool_probe(h_syn_a.detach())
    F.cross_entropy(head_probe(z_attn_2), y_syn).backward()
    step2_norms = {k: float(p.grad.norm().item()) for k, p in pool_probe.named_parameters()}
    need(
        all(torch.isfinite(p.grad).all().item() and step2_norms[k] > 0.0 for k, p in pool_probe.named_parameters()),
        f"Expected finite non-zero step-2 grad norms for all V/U/w: {step2_norms}",
    )
    syn_opt.step()  # synthetic update #2 (confirms finite updated parameters)
    for k, p in pool_probe.named_parameters():
        need(torch.isfinite(p).all().item(), f"Non-finite param after synthetic update #2: {k}")

    # 6. Verify loop wiring & exact optimizer parameter coverage (all 28,049,000 params once)
    tiny_plan = {**plan, "train_ids": ids, "validation_ids": ids}
    loop = f.TAttnSpatialTrainingLoop(model, tiny_plan, lambda s, i: workers[i], f.QA / "never.pt", "cpu_only")
    opt_params = sum(p.numel() for g in loop.optimizer.param_groups for p in g["params"])
    need(opt_params == EXPECTED_TOTAL_PARAMS_TATTN, "Optimizer param count mismatch")

    return {
        "initialization": init,
        "head_seed": HEAD_SEED,
        "attn_seed": ATTN_SEED,
        "base_185_tensors_bitwise_equal": True,
        "tattn_rng_isolated": True,
        "initial_alpha_max_abs_diff_from_uniform": alpha_max_diff_from_uniform,
        "initial_alpha_sum_err": alpha_sum_err,
        "initial_logits_max_abs_diff_vs_baseline": logits_max_abs_diff,
        "initial_pooled_max_abs_diff_vs_mean": pooled_max_abs_diff,
        "initial_input_grad_max_abs_diff_vs_mean": input_grad_max_abs_diff,
        "proximity_tolerances": {"rtol": PROXIMITY_RTOL, "atol": PROXIMITY_ATOL},
        "diagnostic_interface_non_interfering": True,
        "synthetic_2step_grad_norms": {"step1": step1_norms, "step2": step2_norms},
        "cpu_synthetic_pool_optimizer_updates": 2,
        "swin3d_t_full_model_cpu_optimizer_updates": 0,
        "spatial_policy_sha256": augment.POLICY_SHA,
        "worker_tensor_equal": True,
        "workers_zero_model_rng_consumption": True,
        "validation_zero_aug_fixture_check_passed": True,
        "optimizer_param_count": opt_params,
        "sample15": verify_sample15_metadata(),
        "validation_video_decodes": 0,
        "test_video_decodes": 0,
    }


def _make_batches(plan: dict, epoch: int, ids: list[str]) -> list[dict]:
    ds = SpatialDataset(plan["source_root"], "train", plan["train_ids"], plan["reference"])
    ds.set_epoch(epoch)
    indices = [plan["train_ids"].index(vid) for vid in ids]
    return [SpatialCollator()([ds[i]]) for i in indices]


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
        configure()
        f.install_runtime()
        _, _, ids = fixture_items(plan)
        batches_e0 = _make_batches(plan, 0, ids)
        batches_e1 = _make_batches(plan, 1, ids)

        val_22 = copy.deepcopy(batches_e0[1])
        val_22.pop("temporal_sampling_version", None)
        val_22.pop("spatial_policy_sha256", None)
        for r in val_22["records"]:
            r["split"] = "validation"
            for k in ("temporal_sampling_version", "temporal_seed", "epoch_index", "spatial_augmentation"):
                r["metadata"].pop(k, None)

        # Capture initial temporal_attn tensors to confirm they actually update on GPU!
        init_m, _ = initialized(VARIANT, SEED)
        init_tattn = {k: init_m.state_dict()[k].clone() for k in ATTN_PARAM_KEYS}
        tattn_param_indices = [
            idx for idx, (name, _) in enumerate(init_m.named_parameters()) if name in ATTN_PARAM_KEYS
        ]
        need(len(tattn_param_indices) == 5, f"Expected 5 temporal_attn param indices, got {tattn_param_indices}")
        del init_m
        gc.collect()

        peaks_alloc: list[int] = []
        peaks_res: list[int] = []
        with tempfile.TemporaryDirectory(prefix="swin_sp24_tattn_gpu_probe_", dir=f.QA) as temp:
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
                reset_seed()
                m, _ = initialized(VARIANT, SEED)
                m.cuda()
                return f.TAttnSpatialTrainingLoop(m, probe, fetch_fn, ckpt, "sp24_tattn_readiness_eng", publish=pub)

            box_c: list = [None]
            cont = new_loop(make_fetch(box_c), cont_ckpt, pub_c)
            box_c[0] = cont
            need(cont.run(lambda p, s, seen: False) == "completed", "TAttn cont probe failed")
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
                "TAttn no-improvement branch failed",
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
            need(res.run(lambda p, s, seen: False) == "completed", "TAttn resumed probe failed")
            peaks_alloc.append(int(torch.cuda.max_memory_allocated()))
            peaks_res.append(int(torch.cuda.max_memory_reserved()))
            box_r[0] = None
            del res, box_r
            _assert_zero_live_cuda_tensors()

            act = torch.load(res_ckpt, map_location="cpu", weights_only=True)
            matches = {k: bool(equal(exp[k], act[k])) for k in ("model", "optimizer", "rng", "cuda_rng", "state", "best_model")}
            need(all(matches.values()), f"TAttn resume mismatch: {matches}")

            # Explicit Codex checks on the 5 new temporal_attn parameters & their Adam states
            tattn_updated = {k: not torch.equal(exp["model"][k], init_tattn[k]) for k in ATTN_PARAM_KEYS}
            need(all(tattn_updated.values()), f"Expected all 5 temporal_attn tensors to update after 4 steps: {tattn_updated}")
            tattn_model_eq = all(torch.equal(exp["model"][k], act["model"][k]) for k in ATTN_PARAM_KEYS)
            tattn_best_eq = all(torch.equal(exp["best_model"][k], act["best_model"][k]) for k in ATTN_PARAM_KEYS)
            tattn_adam_eq = all(
                equal(exp["optimizer"]["state"][idx], act["optimizer"]["state"][idx])
                and int(exp["optimizer"]["state"][idx]["step"].item()) == 4
                for idx in tattn_param_indices
            )
            need(tattn_model_eq and tattn_best_eq and tattn_adam_eq, "New temporal_attn resume/Adam state mismatch")

        final_ledger = _load_ledger()
        return {
            "gpu_processes_observed": observed_gpu,
            "cumulative_updates_atomic_ledger": final_ledger["cumulative_updates"],
            "max_cumulative_updates_cap": MAX_CUMULATIVE_GPU_UPDATES,
            "mid_epoch_and_post_selection_no_improvement_matches": matches,
            "new_tattn_checks": {
                "param_keys": list(ATTN_PARAM_KEYS),
                "param_indices": tattn_param_indices,
                "all_5_updated_from_init_after_4_updates": tattn_updated,
                "model_5_tensors_bitwise_equal_on_resume": tattn_model_eq,
                "best_model_5_tensors_bitwise_equal_on_resume": tattn_best_eq,
                "adam_state_5_tensors_bitwise_equal_and_step4": tattn_adam_eq,
            },
            "no_improvement_ce_e1_to_e2": [round(float(ce1), 6), round(float(ce2), 6)],
            "max_allocated_gib": round(max(peaks_alloc) / (1024**3), 4),
            "max_reserved_gib": round(max(peaks_res) / (1024**3), 4),
            "validation_video_decodes": 0,
            "test_video_decodes": 0,
            "gpu_probe_note": "Simulated validation branch uses train fixture; zero-aug validation path verified in CPU gate",
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
