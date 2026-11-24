"""Bounded CPU-only acceptance runner for development.mixnet_binary50_v1.

Strictly read-only with respect to dataset/videos/checkpoints:
- No GPU execution
- No training or optimizer updates
- No validation or test dataset sweeps
- No video opening or decoding (reuses saved RGB fixtures 04 and 22 only)
- No pretrained weight loading or network downloads (pretrained=False only)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import timm
import torch
from torch.nn import functional as F

from development.mixnet_binary50_v1.data_adapter import (
    binary_cross_entropy_from_logits,
    collate_saved_binary_fixtures,
    verify_sample15_metadata,
)
from development.mixnet_binary50_v1.model import (
    CLASS_ORDER,
    EXPECTED_INPUT_SHAPE,
    MODEL_NAME,
    UPSTREAM_NORMALIZATION,
    build_mixnet_binary50,
    compare_bn3d_snapshots,
    parameter_summary,
    snapshot_bn3d_buffers,
    summarize_bn3d,
    trace_stage_shapes,
)

TZ_TAIPEI = timezone(timedelta(hours=8))
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
MAIN_REPO_ROOT = Path("/home/tsai/projects/basketball")
DEFAULT_SEED = 20261123


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(repo: Path) -> dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        text=True,
    ).strip()
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return {"path": str(repo), "branch": branch, "head": head}


def _inspect_pretrained_candidates_metadata_only() -> dict[str, Any]:
    """Record local pretrained candidate paths/hashes WITHOUT loading or downloading."""
    m2d = timm.create_model(MODEL_NAME, pretrained=False)
    default_cfg = dict(m2d.default_cfg)
    hf_symlink = Path(
        "/home/tsai/.cache/huggingface/hub/models--timm--mixnet_s.ft_in1k/"
        "snapshots/94149f6f14da590e98eeaa4dd9867b97a23bd3f4/model.safetensors"
    )
    hf_entry: dict[str, Any] = {"path": str(hf_symlink), "exists": hf_symlink.exists(), "loaded": False}
    if hf_symlink.exists():
        resolved = hf_symlink.resolve()
        hf_entry["resolved_path"] = str(resolved)
        hf_entry["size_bytes"] = resolved.stat().st_size
        hf_entry["sha256"] = _sha256_file(resolved)

    hist_frozen = Path(
        "/home/tsai/projects/basketball/outputs/04_history/earlier_research/"
        "mixnet_i3d/frozen_extractor_v1.pt"
    )
    hist_entry: dict[str, Any] = {
        "path": str(hist_frozen),
        "exists": hist_frozen.exists(),
        "loaded": False,
        "status": "historical_three_window_frozen_extractor_not_used",
    }
    if hist_frozen.exists():
        hist_entry["size_bytes"] = hist_frozen.stat().st_size
        hist_entry["sha256"] = _sha256_file(hist_frozen)

    return {
        "weights_loaded_in_this_run": False,
        "network_download_performed": False,
        "timm_model_name": MODEL_NAME,
        "timm_default_cfg": {
            "hf_hub_id": default_cfg.get("hf_hub_id"),
            "url": default_cfg.get("url"),
            "mean": default_cfg.get("mean"),
            "std": default_cfg.get("std"),
            "input_size": default_cfg.get("input_size"),
            "interpolation": default_cfg.get("interpolation"),
            "crop_pct": default_cfg.get("crop_pct"),
            "license": default_cfg.get("license"),
        },
        "local_hf_cache_candidate": hf_entry,
        "historical_frozen_extractor_artifact": hist_entry,
        "formal_pretrained_approval_status": (
            "UNVERIFIED_PENDING_CODEX_REVIEW: MixNet-S ImageNet-1k weights "
            "(timm/mixnet_s.ft_in1k, inflated to 3D via inflation.py) are not "
            "equivalent to torchvision ResNet50_Weights.IMAGENET1K_V1; formal "
            "provenance, normalization, and comparability require Codex/user sign-off."
        ),
    }


def _run_check(name: str, fn) -> dict[str, Any]:
    t0 = time.perf_counter()
    entry: dict[str, Any] = {"name": name, "status": "NOT_RUN"}
    try:
        details = fn()
        entry["status"] = "PASS"
        entry["details"] = details
    except Exception as exc:
        entry["status"] = "FAIL"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    entry["elapsed_seconds"] = round(time.perf_counter() - t0, 4)
    return entry


def run_acceptance(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    total_t0 = time.perf_counter()
    started_at = datetime.now(TZ_TAIPEI).isoformat()
    command = " ".join([sys.executable, "-B", *sys.argv])

    # Shared deterministic eval model built with pretrained=False.
    model = build_mixnet_binary50(pretrained=False, seed=seed, strict_clip_shape=True, eval_mode=True)
    synth_gen = torch.Generator().manual_seed(seed + 1)
    synth_50 = torch.randn((1, *EXPECTED_INPUT_SHAPE), generator=synth_gen, dtype=torch.float32)

    checks: dict[str, dict[str, Any]] = {
        key: {"name": key, "status": "NOT_RUN"}
        for key in (
            "single_clip_no_double_normalization",
            "rgb_temporal_axes_and_stage_shapes",
            "full_50frame_cpu_forward",
            "small_synthetic_backward_finite_gradients",
            "binary_target_and_raw_logit_ce_contract",
            "sample15_adjudicated_metadata_not_overwritten",
            "same_seed_reproducibility",
            "eval_forward_bn3d_buffers_unchanged",
        )
    }

    # Preload fixed saved-RGB fixtures 04 (Miss) and 22 (Make) once (0 video opens).
    fixture_batch, fixture_provs = collate_saved_binary_fixtures(("04", "22"))

    # 1. Single clip, no double normalization
    def _check_no_double_norm() -> dict[str, Any]:
        assert not hasattr(model, "rgb_mean") and not hasattr(model, "rgb_std")
        rejected_cases = {}
        cases = {
            "pretrained_true": lambda: build_mixnet_binary50(pretrained=True),
            "six_dim_three_window": lambda: model(torch.zeros(1, 3, 3, 50, 120, 160)),
            "swapped_T_and_C": lambda: model(torch.zeros(1, 50, 3, 120, 160)),
            "wrong_T_64": lambda: model(torch.zeros(1, 3, 64, 120, 160)),
            "wrong_HW_240x320": lambda: model(torch.zeros(1, 3, 50, 240, 320)),
            "raw_uint8_input": lambda: model(torch.zeros((1, 3, 50, 120, 160), dtype=torch.uint8)),
        }
        for label, fn in cases.items():
            raised = None
            try:
                fn()
            except ValueError as exc:
                raised = str(exc)
            assert raised is not None, f"Case {label} did not raise ValueError"
            rejected_cases[label] = raised

        with torch.inference_mode():
            _, trace_synth = trace_stage_shapes(model, synth_50)
            _, trace_fix04 = trace_stage_shapes(model, fixture_batch["tensor"][0:1])
        assert trace_synth["backbone_input_same_object"] is True
        assert trace_synth["backbone_input_bitwise_equal"] is True
        assert trace_synth["backbone_input_max_abs_diff"] == 0.0
        assert trace_fix04["backbone_input_same_object"] is True
        assert trace_fix04["backbone_input_bitwise_equal"] is True
        assert trace_fix04["backbone_input_max_abs_diff"] == 0.0
        return {
            "has_internal_rgb_mean_or_std_buffers": False,
            "upstream_normalization_contract": UPSTREAM_NORMALIZATION,
            "rejected_invalid_inputs": rejected_cases,
            "synthetic_backbone_hook_bitwise_equal": True,
            "synthetic_backbone_hook_max_abs_diff": 0.0,
            "fixture04_backbone_hook_bitwise_equal": True,
            "fixture04_backbone_hook_max_abs_diff": 0.0,
        }

    checks["single_clip_no_double_normalization"] = _run_check(
        "single_clip_no_double_normalization", _check_no_double_norm
    )

    # 2. RGB / T axes & stage shapes + 3. Full [1,3,50,120,160] CPU forward
    full_forward_cache: dict[str, Any] = {}

    def _check_stage_shapes() -> dict[str, Any]:
        with torch.inference_mode():
            logits, trace = trace_stage_shapes(model, synth_50)
        expected_t = {
            "input": 50,
            "stem_bn1": 25,
            "stage_0": 25,
            "stage_1": 25,
            "stage_2": 25,
            "stage_3": 13,
            "stage_4": 13,
            "stage_5": 7,
            "head_bn2": 7,
        }
        assert trace["temporal_progression"] == expected_t
        assert trace["stage_shapes"]["head_bn2"] == [1, 1536, 7, 4, 5]
        assert trace["stage_shapes"]["global_avg_pool_thw"] == [1, 1536]
        assert trace["stage_shapes"]["logits"] == [1, 2]
        assert model.backbone.num_features == 1536
        full_forward_cache["synthetic_logits"] = logits
        full_forward_cache["synthetic_trace"] = trace
        return {
            "backbone_num_features_measured": model.backbone.num_features,
            "stage_shapes": trace["stage_shapes"],
            "temporal_progression": trace["temporal_progression"],
        }

    checks["rgb_temporal_axes_and_stage_shapes"] = _run_check(
        "rgb_temporal_axes_and_stage_shapes", _check_stage_shapes
    )

    def _check_full_50frame_forward() -> dict[str, Any]:
        with torch.inference_mode():
            synth_logits = model(synth_50)
            fix04_logits = model(fixture_batch["tensor"][0:1])
            fix22_logits = model(fixture_batch["tensor"][1:2])

        for name, out in (
            ("synthetic", synth_logits),
            ("fixture_04", fix04_logits),
            ("fixture_22", fix22_logits),
        ):
            assert tuple(out.shape) == (1, 2), f"{name} shape={tuple(out.shape)}"
            assert out.dtype == torch.float32
            assert torch.isfinite(out).all().item(), f"{name} non-finite"

        return {
            "input_shape": [1, 3, 50, 120, 160],
            "output_shape": [1, 2],
            "device": "cpu",
            "dtype": "torch.float32",
            "synthetic_logits": [round(v, 6) for v in synth_logits[0].tolist()],
            "synthetic_all_finite": True,
            "saved_rgb_fixtures_tested": [
                {
                    "fixture_id": fixture_provs[0]["fixture_id"],
                    "video_id": fixture_provs[0]["video_id"],
                    "binary_label": fixture_provs[0]["binary_label"],
                    "tensor_sha256": fixture_provs[0]["tensor_sha256"],
                    "logits_shape": list(fix04_logits.shape),
                    "logits": [round(v, 6) for v in fix04_logits[0].tolist()],
                    "all_finite": True,
                    "video_open_count": 0,
                },
                {
                    "fixture_id": fixture_provs[1]["fixture_id"],
                    "video_id": fixture_provs[1]["video_id"],
                    "binary_label": fixture_provs[1]["binary_label"],
                    "tensor_sha256": fixture_provs[1]["tensor_sha256"],
                    "logits_shape": list(fix22_logits.shape),
                    "logits": [round(v, 6) for v in fix22_logits[0].tolist()],
                    "all_finite": True,
                    "video_open_count": 0,
                },
            ],
            "accuracy_reported": False,
            "note": "Randomly initialized pretrained=False engineering wiring check only; no accuracy claim.",
        }

    checks["full_50frame_cpu_forward"] = _run_check(
        "full_50frame_cpu_forward", _check_full_50frame_forward
    )

    # 4. Small synthetic backward finite gradients (separate from full 50-frame forward)
    def _check_small_backward() -> dict[str, Any]:
        small_shape = (2, 3, 8, 64, 64)
        small_model = build_mixnet_binary50(
            pretrained=False,
            seed=seed,
            strict_clip_shape=False,
            eval_mode=True,
        )
        g = torch.Generator().manual_seed(seed + 2)
        small_x = torch.randn(small_shape, generator=g, dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)

        before_bn = snapshot_bn3d_buffers(small_model)
        small_model.zero_grad(set_to_none=True)
        logits = small_model(small_x)
        loss = binary_cross_entropy_from_logits(logits, targets)
        loss.backward()
        after_bn = snapshot_bn3d_buffers(small_model)

        assert tuple(logits.shape) == (2, 2)
        assert torch.isfinite(logits).all().item()
        assert torch.isfinite(loss).item()

        grad_count = 0
        max_grad_abs = 0.0
        missing = []
        non_finite = []
        for pname, param in small_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                missing.append(pname)
            else:
                grad_count += 1
                if not torch.isfinite(param.grad).all().item():
                    non_finite.append(pname)
                else:
                    max_grad_abs = max(max_grad_abs, float(param.grad.abs().max().item()))

        assert not missing, f"Missing gradients: {missing[:5]}"
        assert not non_finite, f"Non-finite gradients: {non_finite[:5]}"
        bn_cmp = compare_bn3d_snapshots(before_bn, after_bn)
        assert bn_cmp["all_buffers_bitwise_unchanged"]

        return {
            "synthetic_input_shape": list(small_shape),
            "targets": targets.tolist(),
            "target_dtype": str(targets.dtype),
            "model_mode_during_backward": "eval",
            "logits_shape": list(logits.shape),
            "loss_ce": round(float(loss.item()), 6),
            "loss_finite": True,
            "trainable_param_tensors_with_finite_grad": grad_count,
            "classifier_weight_grad_norm": round(
                float(small_model.classifier.weight.grad.norm().item()), 6
            ),
            "conv_stem_weight_grad_norm": round(
                float(small_model.backbone.conv_stem.weight.grad.norm().item()), 6
            ),
            "max_abs_grad_across_all_params": round(max_grad_abs, 6),
            "eval_bn3d_buffers_unchanged": bn_cmp["all_buffers_bitwise_unchanged"],
            "optimizer_updates": 0,
            "note": "Small synthetic backward only; not a substitute for full 50-frame forward.",
        }

    checks["small_synthetic_backward_finite_gradients"] = _run_check(
        "small_synthetic_backward_finite_gradients", _check_small_backward
    )

    # 5. Binary target (long, Miss=0/Make=1) & raw-logit CE contract
    def _check_target_ce_contract() -> dict[str, Any]:
        logits = torch.tensor([[1.25, -0.35], [-0.75, 0.85]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        ce = binary_cross_entropy_from_logits(logits, targets)
        manual = -F.log_softmax(logits, dim=1)[torch.arange(2), targets].mean()
        diff_manual = float((ce - manual).abs().item())
        assert diff_manual < 1e-6

        ce_after_softmax = F.cross_entropy(F.softmax(logits, dim=1), targets)
        ce_after_sigmoid = F.cross_entropy(torch.sigmoid(logits), targets)
        assert abs(float(ce.item()) - float(ce_after_softmax.item())) > 1e-2
        assert abs(float(ce.item()) - float(ce_after_sigmoid.item())) > 1e-2

        rejected = {}
        for name, fn in {
            "float_targets": lambda: binary_cross_entropy_from_logits(logits, targets.float()),
            "out_of_range_target": lambda: binary_cross_entropy_from_logits(
                logits, torch.tensor([0, 2], dtype=torch.long)
            ),
            "single_logit_column": lambda: binary_cross_entropy_from_logits(
                torch.randn(2, 1), targets
            ),
        }.items():
            err = None
            try:
                fn()
            except ValueError as exc:
                err = str(exc)
            assert err is not None, f"{name} did not raise ValueError"
            rejected[name] = err

        assert fixture_batch["task"] == "basketball51_direct_binary"
        assert fixture_batch["targets"].dtype == torch.long
        assert fixture_batch["targets"].tolist() == [0, 1]
        assert fixture_batch["binary_targets"].tolist() == [0, 1]
        assert fixture_batch["eight_class_targets"].tolist() == [0, 5]
        return {
            "class_order": list(CLASS_ORDER),
            "raw_logit_ce": round(float(ce.item()), 6),
            "manual_log_softmax_nll_abs_diff": diff_manual,
            "softmax_then_ce_value": round(float(ce_after_softmax.item()), 6),
            "sigmoid_then_ce_value": round(float(ce_after_sigmoid.item()), 6),
            "rejected_invalid_contracts": rejected,
            "collated_fixtures_04_22": {
                "task": fixture_batch["task"],
                "tensor_shape": list(fixture_batch["tensor"].shape),
                "targets": fixture_batch["targets"].tolist(),
                "targets_dtype": str(fixture_batch["targets"].dtype),
                "eight_class_targets": fixture_batch["eight_class_targets"].tolist(),
                "data_version": fixture_batch["data_version"],
                "label_revision_sha256": fixture_batch["label_revision_sha256"],
                "fixture_provenances": fixture_provs,
            },
        }

    checks["binary_target_and_raw_logit_ce_contract"] = _run_check(
        "binary_target_and_raw_logit_ce_contract", _check_target_ce_contract
    )

    # 6. Sample 15 metadata check (effective 2p1 not overwritten by 3p1 filename)
    checks["sample15_adjudicated_metadata_not_overwritten"] = _run_check(
        "sample15_adjudicated_metadata_not_overwritten", verify_sample15_metadata
    )

    # 7. Same-seed reproducibility
    def _check_reproducibility() -> dict[str, Any]:
        m_a = build_mixnet_binary50(pretrained=False, seed=seed, eval_mode=True)
        m_b = build_mixnet_binary50(pretrained=False, seed=seed, eval_mode=True)
        sd_a, sd_b = m_a.state_dict(), m_b.state_dict()
        assert list(sd_a.keys()) == list(sd_b.keys())
        for key in sd_a:
            assert torch.equal(sd_a[key], sd_b[key]), f"state_dict differs at {key}"
        with torch.inference_mode():
            out_a = m_a(synth_50)
            out_b = m_b(synth_50)
        max_diff = float((out_a - out_b).abs().max().item())
        assert torch.equal(out_a, out_b) and max_diff == 0.0
        return {
            "seed": seed,
            "pretrained": False,
            "state_dict_tensor_count": len(sd_a),
            "all_state_dict_tensors_bitwise_equal": True,
            "input_shape": [1, 3, 50, 120, 160],
            "outputs_bitwise_equal": True,
            "max_abs_diff": max_diff,
            "logits": [round(v, 6) for v in out_a[0].tolist()],
        }

    checks["same_seed_reproducibility"] = _run_check(
        "same_seed_reproducibility", _check_reproducibility
    )

    # 8. Eval forward does not modify BN3d buffers
    def _check_eval_bn_invariance() -> dict[str, Any]:
        bn_before = snapshot_bn3d_buffers(model)
        with torch.inference_mode():
            logits = model(synth_50)
        bn_after = snapshot_bn3d_buffers(model)
        cmp_res = compare_bn3d_snapshots(bn_before, bn_after)
        assert cmp_res["all_buffers_bitwise_unchanged"]
        assert torch.isfinite(logits).all().item()
        return {
            "input_shape": [1, 3, 50, 120, 160],
            "model_training_mode": bool(model.training),
            **cmp_res,
        }

    checks["eval_forward_bn3d_buffers_unchanged"] = _run_check(
        "eval_forward_bn3d_buffers_unchanged", _check_eval_bn_invariance
    )

    pkg_dir = Path(__file__).resolve().parent
    new_source_hashes = {
        str(p): _sha256_file(p)
        for p in sorted(pkg_dir.glob("*.py")) + sorted(pkg_dir.glob("*.md"))
    }

    all_passed = all(c["status"] == "PASS" for c in checks.values())
    total_elapsed = round(time.perf_counter() - total_t0, 4)

    return {
        "schema": "mixnet_binary50_cpu_acceptance_v1",
        "started_at": started_at,
        "completed_at": datetime.now(TZ_TAIPEI).isoformat(),
        "command": command,
        "overall_status": "PASS_CPU_ONLY" if all_passed else "FAIL",
        "total_elapsed_seconds": total_elapsed,
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "timm": timm.__version__,
            "cv2": cv2.__version__,
            "numpy": np.__version__,
            "worktree": _git_info(WORKTREE_ROOT),
            "main_repo": _git_info(MAIN_REPO_ROOT),
        },
        "model_contract": {
            "model_class": "development.mixnet_binary50_v1.model.MixNetBinary50Model",
            "builder": "development.mixnet_binary50_v1.model.build_mixnet_binary50",
            "backbone_class": "development.mixnet_i3d.backbone.MixNetI3DBackbone",
            "backbone_timm_name": MODEL_NAME,
            "backbone_num_features": model.backbone.num_features,
            "input_shape_single_clip": [1, *EXPECTED_INPUT_SHAPE],
            "input_dtype": "torch.float32",
            "upstream_normalization": UPSTREAM_NORMALIZATION,
            "pooling": "features_3d.mean(dim=(2, 3, 4))",
            "head": f"nn.Linear({model.backbone.num_features}, 2)",
            "class_order": list(CLASS_ORDER),
            "lsta_tsci_added": False,
            "cpu_init_seed": seed,
            "pretrained": False,
            "parameter_summary": parameter_summary(model),
            "bn3d_summary": summarize_bn3d(model),
        },
        "pretrained_provenance_audit": _inspect_pretrained_candidates_metadata_only(),
        "resource_access_log": {
            "new_video_opens": 0,
            "new_video_decodes": 0,
            "saved_rgb_fixtures_used": ["04", "22"],
            "sample15_metadata_only": True,
            "validation_inference_calls": 0,
            "test_access_calls": 0,
            "gpu_executions": 0,
            "optimizer_updates": 0,
            "weights_loaded": "none (pretrained=False random init only)",
            "network_access": False,
        },
        "new_package_source_sha256": new_source_hashes,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded CPU-only acceptance check for MixNetBinary50Model."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Destination path for cpu_acceptance.json (will not overwrite unless --overwrite).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic CPU init seed for pretrained=False acceptance.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output JSON file.",
    )
    args = parser.parse_args(argv)

    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without --overwrite: {args.output_json}"
        )

    report = run_acceptance(seed=args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "total_elapsed_seconds": report["total_elapsed_seconds"],
                "output_json": str(args.output_json),
                "checks": {k: v["status"] for k, v in report["checks"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["overall_status"] == "PASS_CPU_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
