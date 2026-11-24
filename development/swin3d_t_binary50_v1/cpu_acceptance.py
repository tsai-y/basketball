"""Bounded CPU-only acceptance for Swin3D-T (50-frame, 120x160, dynamic pos index).

Strict guardrails enforced:
- weights=None only; ZERO weight downloads and ZERO network access.
- CPU only (CUDA_VISIBLE_DEVICES=""); ZERO GPU execution.
- ZERO training/optimizer updates; ZERO validation/test split access.
- ZERO video opens/decodes (cv2.VideoCapture patched to raise AssertionError);
  uses ONLY saved RGB fixtures 04 (Miss) and 22 (Make).
- ZERO modification to installed torchvision or protected baseline files.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torchvision.models.video import swin_transformer as tv_swin

from development.mixnet_binary50_v1.data_adapter import (
    ALLOWED_PIXEL_FIXTURE_IDS,
    binary_cross_entropy_from_logits,
    collate_saved_binary_fixtures,
    verify_sample15_metadata,
)
from .model import (
    BINARY_CLASSES,
    CONTRACT_FRAMES,
    CONTRACT_HEIGHT,
    CONTRACT_MEAN,
    CONTRACT_STD,
    CONTRACT_WIDTH,
    MODEL_VARIANT_NAME,
    SWIN3D_T_NUM_BLOCKS,
    SWIN3D_T_WINDOW_SIZE,
    DynamicPosShiftedWindowAttention3d,
    build_swin3d_t_400_dynamic_pos,
    build_swin3d_t_binary50_dynamic_pos,
    independent_relative_position_index,
    replace_head_for_binary,
    select_dynamic_relative_position_index,
)

DEFAULT_OUT_DIR = Path(
    "/home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_cpu_acceptance_20261124_v1"
)
SEED = 20261124
MAX_RUNTIME_SECONDS = 120.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_geometry_and_windows() -> dict[str, Any]:
    """Check 1: Verify (8,4,5) exactness & every runtime window in 50x120x160 and 8x64x64."""
    ref_400 = tv_swin.swin3d_t(weights=None)
    base_index = ref_400.features[6][0].attn.relative_position_index.clone()
    base_window = list(SWIN3D_T_WINDOW_SIZE)

    # Explicit (8, 4, 5) audit matching Codex's independent calculation
    indep_845 = independent_relative_position_index(base_window, (8, 4, 5))
    dyn_845 = select_dynamic_relative_position_index(base_index, base_window, (8, 4, 5))
    naive_845 = base_index[:160, :160]
    _need(tuple(indep_845.shape) == (160, 160), "Unexpected (8,4,5) index shape")
    _need(torch.equal(dyn_845, indep_845), "Dynamic (8,4,5) index does not match independent formula")
    naive_mismatches = int((naive_845 != indep_845).sum().item())
    _need(naive_mismatches == 24980, f"Expected 24980 naive mismatches on (8,4,5), got {naive_mismatches}")

    # Capture every runtime window actually encountered on [1, 3, 50, 120, 160] eval forward
    torch.manual_seed(SEED)
    model_contract = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=True).eval()
    x_full = torch.randn(1, 3, CONTRACT_FRAMES, CONTRACT_HEIGHT, CONTRACT_WIDTH)
    t0 = time.monotonic()
    with torch.no_grad():
        logits_full = model_contract(x_full)
    full_eval_s = time.monotonic() - t0
    _need(tuple(logits_full.shape) == (1, 2) and bool(torch.isfinite(logits_full).all()), "Full eval invalid")
    full_geoms = model_contract.collect_last_runtime_geometries()
    _need(len(full_geoms) == SWIN3D_T_NUM_BLOCKS, "Missing geometry records on full forward")

    # Verify Stage 3 errata detail explicitly: block 0 shift=(0,0,0), block 1 shift=(4,0,0)
    stage3_geoms = [g for g in full_geoms if g["module"].startswith("features.6.")]
    _need(len(stage3_geoms) == 2, "Expected 2 blocks in Stage 3")
    _need(
        stage3_geoms[0]["geometry"]["actual_window_size"] == [8, 4, 5]
        and stage3_geoms[0]["geometry"]["actual_shift_size"] == [0, 0, 0],
        f"Unexpected Stage 3 block 0 geometry: {stage3_geoms[0]}",
    )
    _need(
        stage3_geoms[1]["geometry"]["actual_window_size"] == [8, 4, 5]
        and stage3_geoms[1]["geometry"]["actual_shift_size"] == [4, 0, 0],
        f"Unexpected Stage 3 block 1 geometry: {stage3_geoms[1]}",
    )

    # Capture every runtime window actually encountered on [2, 3, 8, 64, 64] small train backward
    torch.manual_seed(SEED)
    model_small = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=False).train()
    x_small = torch.randn(2, 3, 8, 64, 64)
    y_small = torch.tensor([0, 1], dtype=torch.long)
    logits_small = model_small(x_small)
    loss_small = binary_cross_entropy_from_logits(logits_small, y_small)
    loss_small.backward()
    small_geoms = model_small.collect_last_runtime_geometries()
    _need(len(small_geoms) == SWIN3D_T_NUM_BLOCKS, "Missing geometry records on small backward")

    encountered_windows = sorted(
        {
            tuple(rec["geometry"]["actual_window_size"])
            for rec in (full_geoms + small_geoms)
        }
    )
    window_verifications: list[dict[str, Any]] = []
    for win in encountered_windows:
        indep = independent_relative_position_index(base_window, win)
        dyn = select_dynamic_relative_position_index(base_index, base_window, win)
        vol = win[0] * win[1] * win[2]
        naive = base_index[:vol, :vol]
        dyn_exact = bool(torch.equal(dyn, indep))
        _need(dyn_exact, f"Dynamic index mismatch on encountered window {win}")
        window_verifications.append(
            {
                "window_dhw": list(win),
                "total_pairs": int(vol * vol),
                "dynamic_exact_match_vs_independent": dyn_exact,
                "upstream_naive_slice_mismatches": int((naive != indep).sum().item()),
            }
        )

    return {
        "status": "PASS",
        "stage3_845_total_pairs": 25600,
        "stage3_845_dynamic_exact_match": True,
        "stage3_845_upstream_naive_mismatches": naive_mismatches,
        "full_50x120x160_eval_seconds": round(full_eval_s, 4),
        "full_50x120x160_stage3_blocks": stage3_geoms,
        "encountered_windows_verified": window_verifications,
        "full_50x120x160_block_geometries": full_geoms,
        "small_8x64x64_block_geometries": small_geoms,
    }


def _check_full_window_equivalence() -> dict[str, Any]:
    """Check 2: Under full (8,7,7) window across all 12 blocks, match upstream output & grads."""
    torch.manual_seed(SEED)
    upstream = tv_swin.swin3d_t(weights=None)
    replace_head_for_binary(upstream, num_classes=2)

    dynamic = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=False)
    dynamic.load_state_dict(upstream.state_dict(), strict=True)

    # [1, 3, 16, 224, 224] -> Stage 0..3 D=8, Stage 3 H=7, W=7 -> exact (8,7,7) in all 12 blocks
    torch.manual_seed(SEED + 1)
    x = torch.randn(1, 3, 16, 224, 224)
    y = torch.tensor([1], dtype=torch.long)

    # 2a. Eval forward exactness
    upstream.eval()
    dynamic.eval()
    t0 = time.monotonic()
    with torch.no_grad():
        out_up_eval = upstream(x)
        out_dyn_eval = dynamic(x)
    eval_s = time.monotonic() - t0
    eval_max_diff = float((out_up_eval - out_dyn_eval).abs().max().item())
    eval_exact = bool(torch.equal(out_up_eval, out_dyn_eval))
    _need(eval_exact, f"Full-window eval output differs: max_abs_diff={eval_max_diff}")

    geoms = dynamic.collect_last_runtime_geometries()
    _need(len(geoms) == SWIN3D_T_NUM_BLOCKS, "Missing geometry records on full-window check")
    all_full_877 = all(
        g["geometry"]["actual_window_size"] == [8, 7, 7] and not g["geometry"]["window_clipped"]
        for g in geoms
    )
    _need(all_full_877, f"Expected all 12 blocks to run unclipped (8,7,7), got {geoms}")

    # 2b. Train-mode forward + backward exactness (fixed RNG so StochasticDepth masks match)
    upstream.train()
    dynamic.train()
    upstream.zero_grad(set_to_none=True)
    dynamic.zero_grad(set_to_none=True)

    t1 = time.monotonic()
    torch.manual_seed(SEED + 2)
    out_up_train = upstream(x)
    loss_up = binary_cross_entropy_from_logits(out_up_train, y)
    loss_up.backward()

    torch.manual_seed(SEED + 2)
    out_dyn_train = dynamic(x)
    loss_dyn = binary_cross_entropy_from_logits(out_dyn_train, y)
    loss_dyn.backward()
    train_bwd_s = time.monotonic() - t1

    train_out_max_diff = float((out_up_train - out_dyn_train).abs().max().item())
    train_loss_diff = float((loss_up - loss_dyn).abs().item())
    _need(torch.equal(out_up_train, out_dyn_train), f"Train logits differ: {train_out_max_diff}")
    _need(torch.equal(loss_up, loss_dyn), f"Train loss differs: {train_loss_diff}")

    max_grad_diff = 0.0
    grad_checked = 0
    all_grads_bitwise_equal = True
    up_params = dict(upstream.named_parameters())
    for name, p_dyn in dynamic.named_parameters():
        p_up = up_params[name]
        _need(p_dyn.grad is not None and p_up.grad is not None, f"Missing grad for {name}")
        _need(bool(torch.isfinite(p_dyn.grad).all()), f"Non-finite grad for {name}")
        diff = float((p_dyn.grad - p_up.grad).abs().max().item())
        max_grad_diff = max(max_grad_diff, diff)
        if not torch.equal(p_dyn.grad, p_up.grad):
            all_grads_bitwise_equal = False
        grad_checked += 1

    _need(all_grads_bitwise_equal, f"Train gradients differ under (8,7,7): max_grad_diff={max_grad_diff}")

    return {
        "status": "PASS",
        "probe_input_shape": [1, 3, 16, 224, 224],
        "all_12_blocks_unclipped_877": all_full_877,
        "eval_bitwise_equal": eval_exact,
        "eval_max_abs_diff": eval_max_diff,
        "train_logits_bitwise_equal": True,
        "train_loss_bitwise_equal": True,
        "train_loss_abs_diff": train_loss_diff,
        "train_grad_tensors_checked": grad_checked,
        "train_all_grads_bitwise_equal": all_grads_bitwise_equal,
        "train_max_grad_abs_diff": max_grad_diff,
        "eval_seconds": round(eval_s, 4),
        "train_backward_seconds": round(train_bwd_s, 4),
    }


def _check_state_dict_compatibility() -> dict[str, Any]:
    """Check 3: state_dict keys/shapes/dtypes match upstream 400-class except binary head."""
    upstream_400 = tv_swin.swin3d_t(weights=None)
    dynamic_bin = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=True)

    up_sd = upstream_400.state_dict()
    dyn_sd = dynamic_bin.state_dict()

    up_keys = list(up_sd.keys())
    dyn_keys = list(dyn_sd.keys())
    _need(up_keys == dyn_keys, "state_dict key names or ordering differ from upstream swin3d_t")

    shape_diffs: dict[str, dict[str, Any]] = {}
    dtype_diffs: dict[str, dict[str, str]] = {}
    for k in up_keys:
        if tuple(up_sd[k].shape) != tuple(dyn_sd[k].shape):
            shape_diffs[k] = {
                "upstream_400": list(up_sd[k].shape),
                "dynamic_binary": list(dyn_sd[k].shape),
            }
        if up_sd[k].dtype != dyn_sd[k].dtype:
            dtype_diffs[k] = {
                "upstream_400": str(up_sd[k].dtype),
                "dynamic_binary": str(dyn_sd[k].dtype),
            }

    expected_head_diffs = {
        "head.weight": {"upstream_400": [400, 768], "dynamic_binary": [2, 768]},
        "head.bias": {"upstream_400": [400], "dynamic_binary": [2]},
    }
    _need(shape_diffs == expected_head_diffs, f"Unexpected shape diffs: {shape_diffs}")
    _need(not dtype_diffs, f"Unexpected dtype diffs: {dtype_diffs}")

    # Verify LayerNorm, BatchNorm, StochasticDepth, bias table, and index buffer counts/shapes
    layernorm_count = sum(1 for m in dynamic_bin.modules() if isinstance(m, torch.nn.LayerNorm))
    batchnorm_count = sum(
        1 for m in dynamic_bin.modules() if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
    )
    dyn_attn_count = sum(
        1 for m in dynamic_bin.modules() if isinstance(m, DynamicPosShiftedWindowAttention3d)
    )
    _need(layernorm_count == 29 and batchnorm_count == 0, "Unexpected norm layer counts")
    _need(dyn_attn_count == SWIN3D_T_NUM_BLOCKS, "Unexpected dynamic attention block count")

    # Dry-run the future pretrained loading protocol: load 400-class strict=True, then swap head
    reloaded = build_swin3d_t_binary50_dynamic_pos(
        strict_contract_shape=True,
        pretrained_400_state_dict=up_sd,
    )
    for k in up_keys:
        if k not in ("head.weight", "head.bias"):
            _need(
                torch.equal(reloaded.state_dict()[k], up_sd[k]),
                f"Dry-run 400-class strict load altered tensor {k}",
            )

    # Also verify 400-class dynamic builder loads upstream 400 state_dict strict=True verbatim
    dyn_400 = build_swin3d_t_400_dynamic_pos(strict_contract_shape=False)
    dyn_400.load_state_dict(up_sd, strict=True)

    return {
        "status": "PASS",
        "total_state_dict_entries": len(dyn_keys),
        "exact_key_list_match_vs_upstream": True,
        "dtype_diffs": dtype_diffs,
        "shape_diffs": shape_diffs,
        "total_params_binary": sum(p.numel() for p in dynamic_bin.parameters()),
        "layernorm_count": layernorm_count,
        "batchnorm_count": batchnorm_count,
        "patched_dynamic_attn_blocks": dyn_attn_count,
        "dry_run_400class_strict_load_then_binary_head_swap": "PASS",
    }


def _check_determinism_and_train_repro() -> dict[str, Any]:
    """Check 4: Same-seed eval determinism & fixed-RNG train backward reproducibility."""
    x_full = torch.randn(1, 3, CONTRACT_FRAMES, CONTRACT_HEIGHT, CONTRACT_WIDTH, generator=torch.Generator().manual_seed(11))

    torch.manual_seed(SEED)
    m1 = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=True).eval()
    with torch.no_grad():
        y1 = m1(x_full)

    torch.manual_seed(SEED)
    m2 = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=True).eval()
    with torch.no_grad():
        y2 = m2(x_full)

    _need(torch.equal(y1, y2), "Same-seed eval forward is not bitwise deterministic")

    # Fixed-RNG train backward reproducibility (including StochasticDepth)
    x_small = torch.randn(2, 3, 8, 64, 64, generator=torch.Generator().manual_seed(12))
    targets = torch.tensor([0, 1], dtype=torch.long)

    def run_train_bwd(init_seed: int, forward_seed: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        torch.manual_seed(init_seed)
        model = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=False).train()
        torch.manual_seed(forward_seed)
        logits = model(x_small)
        loss = binary_cross_entropy_from_logits(logits, targets)
        loss.backward()
        grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        return logits.detach().clone(), loss.detach().clone(), grads

    t0 = time.monotonic()
    l1, loss1, g1 = run_train_bwd(SEED, SEED + 100)
    l2, loss2, g2 = run_train_bwd(SEED, SEED + 100)
    l3, loss3, _ = run_train_bwd(SEED, SEED + 101)
    repro_s = time.monotonic() - t0

    _need(torch.equal(l1, l2) and torch.equal(loss1, loss2), "Fixed-RNG train forward/loss not reproducible")
    _need(bool(torch.isfinite(loss1).all()), "Train loss non-finite")
    for name in g1:
        _need(bool(torch.isfinite(g1[name]).all()), f"Non-finite grad in {name}")
        _need(torch.equal(g1[name], g2[name]), f"Fixed-RNG grad mismatch in {name}")

    # Confirm StochasticDepth is genuinely active in train mode (different forward RNG -> different logits)
    stochastic_depth_active = not torch.equal(l1, l3)
    _need(stochastic_depth_active, "Expected StochasticDepth to vary train output across RNG seeds")

    return {
        "status": "PASS",
        "same_seed_eval_50x120x160_bitwise_equal": True,
        "fixed_rng_train_8x64x64_logits_bitwise_equal": True,
        "fixed_rng_train_8x64x64_loss_bitwise_equal": True,
        "fixed_rng_train_8x64x64_all_grads_bitwise_equal": True,
        "train_grad_tensors_verified": len(g1),
        "train_loss_value": round(float(loss1.item()), 6),
        "stochastic_depth_active_in_train_mode": stochastic_depth_active,
        "different_rng_logits_max_abs_diff": round(float((l1 - l3).abs().max().item()), 6),
        "optimizer_updates": 0,
        "elapsed_seconds": round(repro_s, 4),
    }


def _check_fixtures_and_no_double_norm() -> dict[str, Any]:
    """Check 5: Fixed 04/22 saved RGB fixtures, no double normalization, sample 15 metadata."""
    with patch.object(cv2, "VideoCapture", side_effect=AssertionError("Video decode forbidden")):
        sample15 = verify_sample15_metadata()
        batch, provenances = collate_saved_binary_fixtures(ALLOWED_PIXEL_FIXTURE_IDS)

    inputs = batch["tensor"]
    targets = batch["targets"]
    _need(tuple(inputs.shape) == (2, 3, CONTRACT_FRAMES, CONTRACT_HEIGHT, CONTRACT_WIDTH), f"Bad batch shape: {inputs.shape}")
    _need(inputs.dtype == torch.float32, f"Bad batch dtype: {inputs.dtype}")
    _need(targets.dtype == torch.long and targets.tolist() == [0, 1], f"Bad binary targets: {targets}")
    _need(all(p["video_open_count"] == 0 and p["video_decode_count"] == 0 for p in provenances), "Video opened!")

    # Independent single-normalization check against official Swin3D_T_Weights mean/std
    mean_t = torch.tensor(CONTRACT_MEAN, dtype=torch.float32).view(1, 3, 1, 1, 1)
    std_t = torch.tensor(CONTRACT_STD, dtype=torch.float32).view(1, 3, 1, 1, 1)
    unnorm = inputs * std_t + mean_t
    unnorm_min = float(unnorm.min().item())
    unnorm_max = float(unnorm.max().item())
    _need(-1e-5 <= unnorm_min and unnorm_max <= 1.0 + 1e-5, f"Unnormalized range out of [0,1]: [{unnorm_min}, {unnorm_max}]")

    # Verify against raw saved uint8 RGB fixture 04 directly (single normalization vs double normalization)
    rgb04 = np.load(provenances[0]["rgb_npy_path"], mmap_mode="r")
    idx04 = batch["records"][0]["metadata"]["selected_indices"]
    resized04 = np.stack(
        [cv2.resize(np.asarray(rgb04[i]), (CONTRACT_WIDTH, CONTRACT_HEIGHT), interpolation=cv2.INTER_AREA) for i in idx04]
    )
    single_norm_04 = (
        torch.from_numpy(resized04).permute(3, 0, 1, 2).float().div(255.0)
        - torch.tensor(CONTRACT_MEAN).view(3, 1, 1, 1)
    ) / torch.tensor(CONTRACT_STD).view(3, 1, 1, 1)
    double_norm_04 = (single_norm_04 - torch.tensor(CONTRACT_MEAN).view(3, 1, 1, 1)) / torch.tensor(CONTRACT_STD).view(3, 1, 1, 1)
    single_err = float((inputs[0] - single_norm_04).abs().max().item())
    double_err = float((inputs[0] - double_norm_04).abs().max().item())
    _need(single_err < 1e-6, f"Single normalization mismatch: {single_err}")
    _need(double_err > 1.0, f"Unexpectedly small double-norm gap: {double_err}")

    # Deterministic CPU eval forward on real [2, 3, 50, 120, 160] fixture batch
    # Also hook patch_embed.proj to verify the model performs zero internal re-normalization
    t0 = time.monotonic()
    torch.manual_seed(SEED)
    model_a = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=True).eval()
    seen_first_conv_inputs: list[torch.Tensor] = []

    def _capture_first_conv(_module, hook_inputs):
        seen_first_conv_inputs.append(hook_inputs[0].detach().clone())

    hook_handle = model_a.patch_embed.proj.register_forward_pre_hook(_capture_first_conv)
    try:
        with torch.no_grad():
            logits_a = model_a(inputs)
            ce_a = binary_cross_entropy_from_logits(logits_a, targets)
    finally:
        hook_handle.remove()

    _need(len(seen_first_conv_inputs) == 1, "First conv hook did not fire once")
    _need(torch.equal(seen_first_conv_inputs[0], inputs), "Model altered input tensor before patch_embed.proj!")

    torch.manual_seed(SEED)
    model_b = build_swin3d_t_binary50_dynamic_pos(strict_contract_shape=True).eval()
    with torch.no_grad():
        logits_b = model_b(inputs)
        ce_b = binary_cross_entropy_from_logits(logits_b, targets)
    fixture_eval_s = time.monotonic() - t0

    _need(tuple(logits_a.shape) == (2, 2) and bool(torch.isfinite(logits_a).all()), "Invalid fixture logits")
    _need(bool(torch.isfinite(ce_a).all()), "Non-finite fixture CE")
    _need(torch.equal(logits_a, logits_b) and torch.equal(ce_a, ce_b), "Fixture eval not deterministic")

    return {
        "status": "PASS",
        "fixture_ids": list(ALLOWED_PIXEL_FIXTURE_IDS),
        "binary_classes": list(BINARY_CLASSES),
        "batch_shape": list(inputs.shape),
        "targets": targets.tolist(),
        "video_open_count": 0,
        "video_decode_count": 0,
        "unnormalized_pixel_range": [round(unnorm_min, 6), round(unnorm_max, 6)],
        "single_norm_max_abs_diff_vs_official_swin_mean_std": single_err,
        "double_norm_max_abs_diff_counterfactual": round(double_err, 6),
        "first_conv_input_bitwise_equal_to_batch": True,
        "no_double_normalization_verified": True,
        "fixture_logits": [[round(float(v), 6) for v in row] for row in logits_a.tolist()],
        "fixture_binary_ce": round(float(ce_a.item()), 6),
        "fixture_same_seed_bitwise_equal": True,
        "fixture_eval_two_runs_seconds": round(fixture_eval_s, 4),
        "sample15_metadata": sample15,
        "fixture_provenances": provenances,
    }


def _check_weight_cache_unchanged() -> dict[str, Any]:
    """Confirm no swin3d_t / 7615ae03 weight file exists in monitored roots."""
    roots = [
        Path("/home/tsai/.cache"),
        Path("/home/tsai/projects/basketball"),
        Path("/home/tsai/.codex/worktrees"),
    ]
    found: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for pat in ("*swin3d*pth*", "*7615ae03*"):
            for match in root.rglob(pat):
                found.append(str(match))
    _need(not found, f"Unexpected swin3d weight file found: {found}")
    return {
        "status": "PASS",
        "local_weight_downloaded": False,
        "matching_files_found": found,
    }


def run_acceptance(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    _need(not torch.cuda.is_available(), "GPU must be hidden/disabled for CPU acceptance")

    tv_swin_path = Path(tv_swin.__file__).resolve()
    tv_swin_sha_before = _sha256_file(tv_swin_path)

    # Hard-block torchvision download helper during the entire CPU acceptance run
    start = time.monotonic()
    with patch.object(
        tv_swin.Swin3D_T_Weights.KINETICS400_V1,
        "get_state_dict",
        side_effect=AssertionError("Pretrained weight download is forbidden in CPU acceptance"),
    ):
        checks = {
            "geometry_and_dynamic_window_index": _check_geometry_and_windows(),
            "full_window_877_upstream_equivalence": _check_full_window_equivalence(),
            "state_dict_compatibility": _check_state_dict_compatibility(),
            "determinism_and_train_reproducibility": _check_determinism_and_train_repro(),
            "fixed_fixtures_04_22_and_no_double_norm": _check_fixtures_and_no_double_norm(),
            "no_weight_download_verified": _check_weight_cache_unchanged(),
        }
    total_s = time.monotonic() - start
    _need(total_s <= MAX_RUNTIME_SECONDS, f"CPU acceptance exceeded time cap: {total_s:.2f}s > {MAX_RUNTIME_SECONDS}s")

    tv_swin_sha_after = _sha256_file(tv_swin_path)
    _need(tv_swin_sha_before == tv_swin_sha_after, "Installed torchvision file changed!")

    report = {
        "schema": "swin3d_t_binary50_cpu_acceptance_v1",
        "status": "PASS_CPU_ONLY",
        "model_variant": MODEL_VARIANT_NAME,
        "display_name": "含動態位置索引修正的 Swin3D-T",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(total_s, 4),
        "time_cap_seconds": MAX_RUNTIME_SECONDS,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available_in_process": torch.cuda.is_available(),
            "deterministic_algorithms": True,
            "torch_num_threads": torch.get_num_threads(),
            "torchvision_swin_file": str(tv_swin_path),
            "torchvision_swin_sha256": tv_swin_sha_after,
            "torchvision_unmodified": True,
        },
        "scope": {
            "weights": None,
            "weights_downloaded": False,
            "gpu_executed": False,
            "optimizer_updates": 0,
            "video_opens": 0,
            "validation_access": 0,
            "test_access": 0,
            "historical_test_contact_disclosed": True,
        },
        "contract": {
            "input_shape": ["B", 3, CONTRACT_FRAMES, CONTRACT_HEIGHT, CONTRACT_WIDTH],
            "normalization_mean": list(CONTRACT_MEAN),
            "normalization_std": list(CONTRACT_STD),
            "binary_classes": list(BINARY_CLASSES),
            "head": "Linear(768, 2)",
            "dynamic_window_index_policy": (
                "Select actual clipped-window (t,h,w) token coordinates from original "
                "[392, 392] relative_position_index buffer; keep original bias table, "
                "buffer shape, parameter names, LayerNorm, and StochasticDepth untouched."
            ),
            "errata_recorded": [
                "Stage 3 spatial clipping triggers when ceil(H/32) < 7 or ceil(W/32) < 7 (i.e. H <= 192 or W <= 192), not for every size below 224.",
                "In Stage 3 for 50x120x160, block 0 (features.6.0) has shift_size=(0,0,0) and block 1 (features.6.1) has shift_size=(4,0,0).",
            ],
        },
        "checks": checks,
    }
    (out_dir / "cpu_acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded CPU acceptance for swin3d_t_binary50_v1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    report = run_acceptance(args.out_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "model_variant": report["model_variant"],
                "elapsed_seconds": report["elapsed_seconds"],
                "checks": {k: v["status"] for k, v in report["checks"].items()},
                "output_json": str(args.out_dir / "cpu_acceptance.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
