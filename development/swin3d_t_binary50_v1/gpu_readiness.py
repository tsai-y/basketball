"""Bounded GPU engineering readiness acceptance for Swin3D-T (50-frame, 120x160).

Strictly engineering readiness ONLY — NOT formal training:
- Enforces existing idle-GPU scheduling policy BEFORE touching CUDA; never kills
  or interrupts other GPU processes.
- Uses ONLY the already-downloaded official Kinetics-400 weight file after
  verifying its locked full 64-char SHA-256 (zero network downloads).
- Fixed [1, 3, 50, 120, 160], FP32 only (no AMP/TF32/resolution/batch fallback).
- Reuses ONLY saved RGB fixtures 04 (Miss=0) and 22 (Make=1); zero video opens,
  zero real validation or test split access.
- Exercises microbatch=1 with full accum=16 AND tail accum=8 averaging across
  total 6 engineering optimizer updates (<= 8 cap):
  - Continuous run: 2 simulated epochs x (16-accum + 8-tail) = 4 updates.
  - Resumed run (from epoch-1 checkpoint with non-None best_model):
    epoch 2 (16-accum + 8-tail) = 2 updates.
  - Strictly deletes continuous GPU model & optimizer + empty_cache() BEFORE
    constructing the resumed model so only ONE model resides on GPU during VRAM
    measurement.
- Restores checkpoint CPU/CUDA RNG strictly AFTER constructing and loading the
  resumed model and optimizer, proving bitwise resume equivalence across model,
  Adam optimizer states, CPU/CUDA RNG, progress, and non-None best_model with
  StochasticDepth active.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torchvision.models.video import swin_transformer as tv_swin

from development.aca_binary_v1 import core as binary_core
from development.aca_eight_class_data_v1.gpu_readiness import equal
from development.aca_spatial_pair_v1 import formal_train as gpu_policy_mod
from development.mixnet_binary50_v1.data_adapter import (
    ALLOWED_PIXEL_FIXTURE_IDS,
    collate_saved_binary_fixtures,
)
from .model import (
    BINARY_CLASSES,
    CONTRACT_FRAMES,
    CONTRACT_HEIGHT,
    CONTRACT_WIDTH,
    DEFAULT_LOCAL_WEIGHT_PATH,
    MODEL_VARIANT_NAME,
    OFFICIAL_KINETICS400_URL,
    build_swin3d_t_binary50_dynamic_pos,
)

LOCKED_WEIGHT_SHA256 = "7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130"
DEFAULT_OUT_DIR = Path(
    "/home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_gpu_readiness_20261125_v1"
)
ENGINEERING_TRAIN_SEED = 20261125
ENGINEERING_HEAD_SEED = 20261124  # Engineering probe only; NOT locked as formal setting.
EFFECTIVE_ACCUM = 16
TAIL_ACCUM = 8
CLIPS_PER_SIM_EPOCH = EFFECTIVE_ACCUM + TAIL_ACCUM  # 24 clips -> 2 updates/epoch
MAX_TOTAL_ENGINEERING_UPDATES = 8
ENGINEERING_LR = 1e-4
ENGINEERING_GRAD_CLIP_NORM = 1.0


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_source_hashes() -> dict[str, str]:
    pkg_dir = Path(__file__).resolve().parent
    return {
        p.name: _sha256_file(p)
        for p in sorted(pkg_dir.iterdir())
        if p.is_file() and p.suffix in (".py", ".md")
    }


def _configure_deterministic_fp32() -> None:
    _need(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
        "CUBLAS_WORKSPACE_CONFIG must be set to :4096:8 or :16:8 before CUDA ops",
    )
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _capture_full_rng() -> dict[str, Any]:
    return {
        "cpu_rng": binary_core.capture_rng(),
        "cuda_rng": torch.cuda.get_rng_state().cpu().clone(),
    }


def _restore_full_rng(bundle: dict[str, Any]) -> None:
    binary_core.restore_rng(bundle["cpu_rng"])
    torch.cuda.set_rng_state(bundle["cuda_rng"].cpu())


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _cpu_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    raw = optimizer.state_dict()
    cpu_state: dict[int, dict[str, Any]] = {}
    for idx, entry in raw["state"].items():
        cpu_state[idx] = {
            k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
            for k, v in entry.items()
        }
    return {"state": cpu_state, "param_groups": copy.deepcopy(raw["param_groups"])}


def _assert_all_params_finite(model: torch.nn.Module) -> None:
    for name, p in model.named_parameters():
        _need(bool(torch.isfinite(p).all().item()), f"Non-finite parameter after update: {name}")


def _simulated_eval_on_fixtures(
    model: torch.nn.Module,
    clips_cpu: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    """Zero-augmentation simulated eval on 04/22 ONLY (NOT real validation)."""
    was_training = model.training
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers_before = {n: b.detach().clone() for n, b in model.named_buffers()}
    model.eval()
    loss_sum = 0.0
    correct = 0
    logits_list: list[list[float]] = []
    with torch.no_grad():
        for x_cpu, y_cpu in clips_cpu:
            x = x_cpu.cuda(non_blocking=False)
            y = y_cpu.cuda(non_blocking=False)
            logits = model(x)
            _need(
                logits.dtype == torch.float32
                and tuple(logits.shape) == (1, 2)
                and bool(torch.isfinite(logits).all().item()),
                "Invalid or non-finite eval logits",
            )
            ce_sum = binary_core.summed_ce(logits, y)
            loss_sum += float(ce_sum.item())
            pred = int(torch.argmax(logits, dim=-1).item())
            correct += int(pred == int(y.item()))
            logits_list.append([round(float(v), 6) for v in logits[0].detach().cpu().tolist()])

    # Confirm zero parameter and buffer mutation during eval
    for name, p in model.named_parameters():
        _need(torch.equal(p, params_before[name]), f"Eval mutated parameter: {name}")
    for name, b in model.named_buffers():
        _need(torch.equal(b, buffers_before[name]), f"Eval mutated buffer: {name}")
    model.train(was_training)

    mean_ce = loss_sum / float(len(clips_cpu))
    return {
        "simulated_selection_check_NOT_real_validation": True,
        "fixture_count": len(clips_cpu),
        "mean_ce": mean_ce,
        "correct_count": correct,
        "logits": logits_list,
        "params_unchanged_during_eval": True,
        "buffers_unchanged_during_eval": True,
        "zero_augmentation": True,
    }


def _run_simulated_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clips_cpu: list[tuple[torch.Tensor, torch.Tensor]],
    state: dict[str, Any],
    best_model_box: list[dict[str, torch.Tensor] | None],
    *,
    epoch_index: int,
    update_counter: list[int],
) -> dict[str, Any]:
    """Run one 24-clip simulated epoch: one accum=16 step + one tail accum=8 step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    windows = [
        ("effective16", 0, EFFECTIVE_ACCUM),
        ("tail8", EFFECTIVE_ACCUM, CLIPS_PER_SIM_EPOCH),
    ]
    step_records: list[dict[str, Any]] = []

    for label, start_idx, end_idx in windows:
        window_size = end_idx - start_idx
        _need(window_size in (EFFECTIVE_ACCUM, TAIL_ACCUM), f"Unexpected window size {window_size}")
        _need(
            update_counter[0] + 1 <= MAX_TOTAL_ENGINEERING_UPDATES,
            f"Exceeded max engineering optimizer updates cap ({MAX_TOTAL_ENGINEERING_UPDATES})",
        )
        raw_ce_sum = 0.0
        scaled_loss_sum = 0.0
        for clip_idx in range(start_idx, end_idx):
            x_cpu, y_cpu = clips_cpu[(clip_idx + epoch_index) % len(clips_cpu)]
            x = x_cpu.cuda(non_blocking=False)
            y = y_cpu.cuda(non_blocking=False)
            logits = model(x)
            _need(
                logits.dtype == torch.float32
                and tuple(logits.shape) == (1, 2)
                and bool(torch.isfinite(logits).all().item()),
                f"Non-finite train logits at epoch {epoch_index} clip {clip_idx}",
            )
            ce_sum = binary_core.summed_ce(logits, y)
            raw_ce_sum += float(ce_sum.item())
            scaled = ce_sum / float(window_size)
            _need(bool(torch.isfinite(scaled).all().item()), "Non-finite scaled loss")
            scaled_loss_sum += float(scaled.item())
            scaled.backward()

        # Verify finite gradients across all parameters before clipping & stepping
        for name, p in model.named_parameters():
            _need(p.grad is not None and bool(torch.isfinite(p.grad).all().item()), f"Non-finite grad: {name}")

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), ENGINEERING_GRAD_CLIP_NORM).item())
        _need(np.isfinite(grad_norm), f"Non-finite grad_norm: {grad_norm}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _assert_all_params_finite(model)

        # Verify Adam optimizer states exist and are finite after step
        adam_state_count = 0
        for p in model.parameters():
            st = optimizer.state.get(p)
            _need(st is not None and "exp_avg" in st and "exp_avg_sq" in st, "Missing Adam moment state")
            _need(
                bool(torch.isfinite(st["exp_avg"]).all().item())
                and bool(torch.isfinite(st["exp_avg_sq"]).all().item()),
                "Non-finite Adam state",
            )
            adam_state_count += 1

        update_counter[0] += 1
        state["updates"] += 1
        mean_ce_direct = raw_ce_sum / float(window_size)
        _need(
            abs(mean_ce_direct - scaled_loss_sum) < 1e-5,
            f"Averaging mismatch for {label}: direct={mean_ce_direct} scaled_sum={scaled_loss_sum}",
        )
        step_records.append(
            {
                "window_label": label,
                "microbatch": 1,
                "accum_divisor": window_size,
                "mean_ce": round(mean_ce_direct, 6),
                "scaled_loss_sum": round(scaled_loss_sum, 6),
                "pre_clip_grad_norm": round(grad_norm, 6),
                "adam_state_tensors_finite": adam_state_count,
                "allocated_mib_after_step": round(torch.cuda.memory_allocated() / (1024**2), 2),
                "reserved_mib_after_step": round(torch.cuda.memory_reserved() / (1024**2), 2),
                "peak_allocated_mib_so_far": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
                "peak_reserved_mib_so_far": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
            }
        )

    sim_eval = _simulated_eval_on_fixtures(model, clips_cpu)
    state["epoch"] = epoch_index + 1
    if state["best_sim_ce"] is None or sim_eval["mean_ce"] < state["best_sim_ce"]:
        state["best_sim_ce"] = sim_eval["mean_ce"]
        state["best_epoch"] = state["epoch"]
        state["bad_epochs"] = 0
        best_model_box[0] = _cpu_state_dict(model)
    else:
        state["bad_epochs"] += 1
    state["sim_eval_history"].append({"epoch": state["epoch"], **sim_eval})
    return {"epoch": state["epoch"], "steps": step_records, "simulated_eval": sim_eval}


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    best_model: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    payload = {
        "model": _cpu_state_dict(model),
        "optimizer": _cpu_optimizer_state(optimizer),
        "rng": _capture_full_rng(),
        "state": copy.deepcopy(state),
        "best_model": copy.deepcopy(best_model),
    }
    binary_core.atomic_save(payload, path)
    return payload


def _build_fresh_gpu_model_and_opt(
    raw_sd: dict[str, torch.Tensor],
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = build_swin3d_t_binary50_dynamic_pos(
        strict_contract_shape=True,
        pretrained_400_state_dict=raw_sd,
        head_seed=ENGINEERING_HEAD_SEED,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=ENGINEERING_LR)
    return model, optimizer


def run(out_dir: Path, weight_path: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1. GPU scheduling guard FIRST (before any CUDA initialization)
    policy = gpu_policy_mod.make_gpu_policy()
    observed_gpu_processes = gpu_policy_mod.assert_idle_gpu(policy)

    # 2. Verify locked weight SHA-256 and block any network download
    _need(weight_path.is_file() and not weight_path.is_symlink(), f"Weight file missing: {weight_path}")
    actual_weight_sha = _sha256_file(weight_path)
    _need(
        actual_weight_sha == LOCKED_WEIGHT_SHA256,
        f"Weight SHA-256 mismatch: got {actual_weight_sha}, expected {LOCKED_WEIGHT_SHA256}",
    )
    tv_swin_path = Path(tv_swin.__file__).resolve()
    tv_swin_sha_before = _sha256_file(tv_swin_path)
    pkg_hashes = _package_source_hashes()

    # 3. Load ONLY saved RGB fixtures 04 and 22 (zero video opens, zero real val/test)
    with patch.object(cv2, "VideoCapture", side_effect=AssertionError("Video decode forbidden")), patch.object(
        tv_swin.Swin3D_T_Weights.KINETICS400_V1,
        "get_state_dict",
        side_effect=AssertionError("Weight download forbidden in GPU readiness"),
    ):
        batch, provenances = collate_saved_binary_fixtures(ALLOWED_PIXEL_FIXTURE_IDS)
        _need(all(p["video_open_count"] == 0 and p["video_decode_count"] == 0 for p in provenances), "Video opened")
        clips_cpu = [
            (batch["tensor"][i : i + 1].contiguous(), batch["targets"][i : i + 1].contiguous())
            for i in range(len(ALLOWED_PIXEL_FIXTURE_IDS))
        ]
        for x_c, y_c in clips_cpu:
            _need(
                tuple(x_c.shape) == (1, 3, CONTRACT_FRAMES, CONTRACT_HEIGHT, CONTRACT_WIDTH)
                and x_c.dtype == torch.float32
                and y_c.dtype == torch.long,
                "Fixture tensor shape/dtype invalid",
            )

        _configure_deterministic_fp32()
        _need(torch.cuda.is_available(), "CUDA device required for GPU readiness")
        device_name = torch.cuda.get_device_name(0)
        raw_sd = torch.load(weight_path, map_location="cpu", weights_only=True)

        update_counter = [0]
        start = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="swin3d_gpu_readiness_", dir=out_dir) as tmpdir:
            tmp = Path(tmpdir)
            ckpt_e1_path = tmp / "ckpt_epoch1.pt"
            ckpt_cont_final_path = tmp / "ckpt_cont_final.pt"
            ckpt_res_final_path = tmp / "ckpt_res_final.pt"

            # --- Phase A: Continuous 2-epoch run (single GPU model; 4 updates) ---
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            _seed_all(ENGINEERING_TRAIN_SEED)
            cont_model, cont_opt = _build_fresh_gpu_model_and_opt(raw_sd)
            cont_state: dict[str, Any] = {
                "epoch": 0,
                "updates": 0,
                "best_epoch": None,
                "best_sim_ce": None,
                "bad_epochs": 0,
                "sim_eval_history": [],
            }
            cont_best: list[dict[str, torch.Tensor] | None] = [None]

            # Re-seed right before Epoch 1 training loop for explicit reproducibility
            _seed_all(ENGINEERING_TRAIN_SEED)
            ep1_info = _run_simulated_epoch(
                cont_model, cont_opt, clips_cpu, cont_state, cont_best, epoch_index=0, update_counter=update_counter
            )
            _need(cont_best[0] is not None and cont_state["best_epoch"] == 1, "Epoch 1 did not commit best_model")
            ckpt_e1 = _save_checkpoint(ckpt_e1_path, cont_model, cont_opt, cont_state, cont_best[0])

            ep2_cont_info = _run_simulated_epoch(
                cont_model, cont_opt, clips_cpu, cont_state, cont_best, epoch_index=1, update_counter=update_counter
            )
            cont_final = _save_checkpoint(ckpt_cont_final_path, cont_model, cont_opt, cont_state, cont_best[0])
            cont_peak_alloc = int(torch.cuda.max_memory_allocated())
            cont_peak_reserved = int(torch.cuda.max_memory_reserved())

            # Explicitly release continuous model & optimizer BEFORE resumed run (single-model VRAM hygiene)
            del cont_model, cont_opt, cont_best
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            live_cuda_tensors = [
                obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda
            ]
            _need(
                len(live_cuda_tensors) == 0,
                f"Live CUDA tensors leaked after deleting continuous model: {len(live_cuda_tensors)}",
            )
            between_runs_allocated_bytes = int(torch.cuda.memory_allocated())
            between_runs_reserved_bytes = int(torch.cuda.memory_reserved())

            # --- Phase B: Resumed run from Epoch 1 checkpoint (single GPU model; 2 updates) ---
            torch.cuda.reset_peak_memory_stats()
            loaded_ckpt = torch.load(ckpt_e1_path, map_location="cpu", weights_only=False)
            _need(loaded_ckpt["best_model"] is not None, "Saved Epoch 1 checkpoint missing non-None best_model")

            # Construct & load fresh model and optimizer FIRST, then restore RNG AFTERWARDS (Codex rule)
            res_model, res_opt = _build_fresh_gpu_model_and_opt(raw_sd)
            res_model.load_state_dict(loaded_ckpt["model"], strict=True)
            res_opt.load_state_dict(loaded_ckpt["optimizer"])
            res_state = copy.deepcopy(loaded_ckpt["state"])
            res_best: list[dict[str, torch.Tensor] | None] = [copy.deepcopy(loaded_ckpt["best_model"])]
            _restore_full_rng(loaded_ckpt["rng"])

            ep2_res_info = _run_simulated_epoch(
                res_model, res_opt, clips_cpu, res_state, res_best, epoch_index=1, update_counter=update_counter
            )
            res_final = _save_checkpoint(ckpt_res_final_path, res_model, res_opt, res_state, res_best[0])
            res_peak_alloc = int(torch.cuda.max_memory_allocated())
            res_peak_reserved = int(torch.cuda.max_memory_reserved())

            # Confirm StochasticDepth actually uses CUDA RNG during GPU train mode (sanity check: RNG advanced)
            cuda_rng_changed_across_ep2 = not torch.equal(
                loaded_ckpt["rng"]["cuda_rng"], res_final["rng"]["cuda_rng"]
            )
            _need(cuda_rng_changed_across_ep2, "Expected CUDA RNG to advance during Epoch 2 training")

            del res_model, res_opt, res_best
            gc.collect()
            torch.cuda.empty_cache()

        elapsed_s = time.monotonic() - start
        _need(not tmp.exists(), "Temporary GPU readiness directory was not cleaned up")

    # 4. Strict bitwise comparison across all resume-critical state trees
    resume_matches = {
        "model": bool(equal(cont_final["model"], res_final["model"])),
        "optimizer": bool(equal(cont_final["optimizer"], res_final["optimizer"])),
        "cpu_rng": bool(equal(cont_final["rng"]["cpu_rng"], res_final["rng"]["cpu_rng"])),
        "cuda_rng": bool(torch.equal(cont_final["rng"]["cuda_rng"], res_final["rng"]["cuda_rng"])),
        "progress_and_selection_state": bool(equal(cont_final["state"], res_final["state"])),
        "best_model_non_null": cont_final["best_model"] is not None and res_final["best_model"] is not None,
        "best_model_bitwise_equal": bool(equal(cont_final["best_model"], res_final["best_model"])),
    }
    _need(all(resume_matches.values()), f"Resume equivalence mismatch: {resume_matches}")

    tv_swin_sha_after = _sha256_file(tv_swin_path)
    _need(tv_swin_sha_before == tv_swin_sha_after, "Installed torchvision file changed!")

    report = {
        "schema": "swin3d_t_binary50_gpu_readiness_v1",
        "status": "PASS_GPU_ENGINEERING_READINESS_ONLY",
        "model_variant": MODEL_VARIANT_NAME,
        "display_name": "含動態位置索引修正的 Swin3D-T",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(elapsed_s, 4),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device_name": device_name,
            "precision": "FP32_strict_no_AMP_no_TF32",
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torchvision_swin_file": str(tv_swin_path),
            "torchvision_swin_sha256": tv_swin_sha_after,
            "torchvision_unmodified": True,
        },
        "gpu_policy": {
            "schema": policy["schema"],
            "observed_processes": observed_gpu_processes,
        },
        "package_code_sha256": pkg_hashes,
        "weight_provenance": {
            "official_url": OFFICIAL_KINETICS400_URL,
            "local_path": str(weight_path),
            "full_sha256": actual_weight_sha,
            "matches_locked_sha256": True,
            "downloaded_in_this_run": False,
        },
        "scope": {
            "engineering_readiness_only_NOT_formal_training": True,
            "input_shape": [1, 3, CONTRACT_FRAMES, CONTRACT_HEIGHT, CONTRACT_WIDTH],
            "microbatch": 1,
            "effective_accum": EFFECTIVE_ACCUM,
            "tail_accum": TAIL_ACCUM,
            "total_engineering_optimizer_updates": update_counter[0],
            "max_total_engineering_updates_cap": MAX_TOTAL_ENGINEERING_UPDATES,
            "fixture_ids": list(ALLOWED_PIXEL_FIXTURE_IDS),
            "binary_classes": list(BINARY_CLASSES),
            "video_opens": 0,
            "real_validation_access": 0,
            "real_test_access": 0,
            "historical_test_contact_disclosed": True,
            "engineering_head_seed_NOT_formal_setting": ENGINEERING_HEAD_SEED,
            "engineering_train_seed_NOT_formal_setting": ENGINEERING_TRAIN_SEED,
            "engineering_lr_NOT_formal_setting": ENGINEERING_LR,
            "two_sample_ce_is_wiring_only_NOT_accuracy_claim": True,
        },
        "memory_single_model_on_gpu": {
            "single_model_enforced_zero_live_cuda_tensors_between_runs": True,
            "between_runs_allocated_bytes_cublas_workspace": between_runs_allocated_bytes,
            "between_runs_reserved_bytes": between_runs_reserved_bytes,
            "continuous_run_peak_allocated_bytes": cont_peak_alloc,
            "continuous_run_peak_allocated_gib": round(cont_peak_alloc / (1024**3), 4),
            "continuous_run_peak_reserved_bytes": cont_peak_reserved,
            "continuous_run_peak_reserved_gib": round(cont_peak_reserved / (1024**3), 4),
            "resumed_run_peak_allocated_bytes": res_peak_alloc,
            "resumed_run_peak_allocated_gib": round(res_peak_alloc / (1024**3), 4),
            "resumed_run_peak_reserved_bytes": res_peak_reserved,
            "resumed_run_peak_reserved_gib": round(res_peak_reserved / (1024**3), 4),
        },
        "runs": {
            "continuous_epoch1": ep1_info,
            "continuous_epoch2": ep2_cont_info,
            "resumed_epoch2": ep2_res_info,
        },
        "resume_equivalence_post_selection": {
            "status": "PASS",
            "checkpoint_best_model_non_null_before_resume": ckpt_e1["best_model"] is not None,
            "rng_restored_strictly_after_model_and_optimizer_load": True,
            "cuda_rng_advanced_during_epoch2": cuda_rng_changed_across_ep2,
            "matches": resume_matches,
        },
    }
    out_json = out_dir / "gpu_readiness.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded GPU engineering readiness for swin3d_t_binary50_v1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--weight-path", type=Path, default=DEFAULT_LOCAL_WEIGHT_PATH)
    args = parser.parse_args()
    report = run(args.out_dir, args.weight_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "total_engineering_updates": report["scope"]["total_engineering_optimizer_updates"],
                "peak_allocated_gib": report["memory_single_model_on_gpu"]["continuous_run_peak_allocated_gib"],
                "peak_reserved_gib": report["memory_single_model_on_gpu"]["continuous_run_peak_reserved_gib"],
                "resume_matches": report["resume_equivalence_post_selection"]["matches"],
                "output_json": str(args.out_dir / "gpu_readiness.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
