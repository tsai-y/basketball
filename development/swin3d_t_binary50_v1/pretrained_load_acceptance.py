"""Single official Swin3D-T Kinetics-400 weight download & CPU load acceptance.

Approved scope:
1. Single download of official Swin3D_T_Weights.KINETICS400_V1 weight file
   (only when --allow-single-download is passed and file does not yet exist).
2. Verify SHA256 starts with '7615ae03' and record full 64-char SHA256 & source URL.
3. Verify 400-class strict=True load into dynamic-pos Swin3D-T (and full (8,7,7)
   window bitwise output equivalence vs upstream torchvision swin3d_t).
4. Replace head with Linear(768, 2) using an explicit, independent head_seed
   (without mutating global RNG) and verify ALL 183 non-head weights/buffers
   remain bitwise equal to the downloaded checkpoint.
5. Run deterministic CPU forward acceptance on fixed saved RGB fixtures 04 (Miss)
   and 22 (Make) with zero video opens and verified single normalization.
6. Record package source file SHA256 hashes and torchvision SHA256.

Strictly forbidden in this stage:
- GPU execution (CUDA_VISIBLE_DEVICES="").
- Training or optimizer updates.
- Validation or test split access.
- Modifying installed torchvision or protected baseline files.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
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
    DEFAULT_LOCAL_WEIGHT_PATH,
    MODEL_VARIANT_NAME,
    OFFICIAL_HASH_PREFIX,
    OFFICIAL_KINETICS400_URL,
    build_swin3d_t_400_dynamic_pos,
    build_swin3d_t_binary50_dynamic_pos,
    replace_head_for_binary,
)

DEFAULT_OUT_DIR = Path(
    "/home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_pretrained_load_20261124_v1"
)
HEAD_SEED = 20261124
ALT_HEAD_SEED = 20261125
DOWNLOAD_TIMEOUT_SECONDS = 90.0


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
    hashes: dict[str, str] = {}
    for p in sorted(pkg_dir.iterdir()):
        if p.is_file() and p.suffix in (".py", ".md"):
            hashes[p.name] = _sha256_file(p)
    return hashes


def _acquire_official_weight(weight_path: Path, *, allow_single_download: bool) -> dict[str, Any]:
    """Download official weight file at most once, verify prefix & record full SHA256."""
    _need(
        tv_swin.Swin3D_T_Weights.KINETICS400_V1.url == OFFICIAL_KINETICS400_URL,
        f"Official torchvision URL drifted: {tv_swin.Swin3D_T_Weights.KINETICS400_V1.url}",
    )
    downloaded_now = False
    http_status: int | None = None
    download_seconds = 0.0

    if not weight_path.exists():
        _need(
            allow_single_download,
            f"Local weight file missing at {weight_path}; pass --allow-single-download to fetch once.",
        )
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = weight_path.with_suffix(weight_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()
        t0 = time.monotonic()
        req = urllib.request.Request(OFFICIAL_KINETICS400_URL, headers={"User-Agent": "torchvision-swin3d-verify/1.0"})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
            http_status = int(getattr(resp, "status", 200))
            _need(http_status == 200, f"Unexpected HTTP status: {http_status}")
            with tmp_path.open("wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
        tmp_path.replace(weight_path)
        download_seconds = time.monotonic() - t0
        downloaded_now = True

    _need(weight_path.is_file() and not weight_path.is_symlink(), f"Expected regular non-symlink file: {weight_path}")
    size_bytes = int(weight_path.stat().st_size)
    full_sha256 = _sha256_file(weight_path)
    prefix_match = full_sha256.startswith(OFFICIAL_HASH_PREFIX)
    _need(
        prefix_match,
        f"Downloaded weight SHA256 prefix mismatch: got {full_sha256[:16]}, expected prefix {OFFICIAL_HASH_PREFIX}",
    )

    return {
        "status": "PASS",
        "weights_enum": "torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1",
        "official_url": OFFICIAL_KINETICS400_URL,
        "local_path": str(weight_path),
        "regular_non_symlink_file": True,
        "downloaded_in_this_run": downloaded_now,
        "http_status": http_status,
        "download_seconds": round(download_seconds, 4),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024.0 * 1024.0), 4),
        "expected_hash_prefix": OFFICIAL_HASH_PREFIX,
        "hash_prefix_match": prefix_match,
        "full_sha256": full_sha256,
    }


def _verify_400class_strict_load_and_equivalence(raw_sd: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Verify 400-class strict=True load and full-window (8,7,7) equivalence with upstream."""
    _need(isinstance(raw_sd, dict) and len(raw_sd) > 0, "Checkpoint did not load as non-empty state_dict")
    all_finite_in_ckpt = all(bool(torch.isfinite(v).all()) for v in raw_sd.values())
    _need(all_finite_in_ckpt, "Downloaded checkpoint contains non-finite values")

    model_400 = build_swin3d_t_400_dynamic_pos(strict_contract_shape=False)
    pre_load_rel_pos_buffers = {
        name: buf.clone()
        for name, buf in model_400.named_buffers()
        if name.endswith("relative_position_index")
    }
    incompatible = model_400.load_state_dict(raw_sd, strict=True)
    _need(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        f"400-class strict load reported keys: {incompatible}",
    )

    # Confirm all 12 relative_position_index buffers in checkpoint match generated buffers verbatim
    rel_pos_buffers_identical = all(
        torch.equal(model_400.state_dict()[name], orig_buf)
        for name, orig_buf in pre_load_rel_pos_buffers.items()
    )
    _need(rel_pos_buffers_identical, "Checkpoint relative_position_index buffers differ from model buffers")

    # Full-window (8,7,7) eval equivalence vs upstream swin3d_t loaded with same checkpoint
    upstream_400 = tv_swin.swin3d_t(weights=None)
    upstream_400.load_state_dict(raw_sd, strict=True)
    model_400.eval()
    upstream_400.eval()

    x_full_win = torch.randn(1, 3, 16, 224, 224, generator=torch.Generator().manual_seed(404))
    t0 = time.monotonic()
    with torch.no_grad():
        logits_dyn = model_400(x_full_win)
        logits_up = upstream_400(x_full_win)
    equiv_s = time.monotonic() - t0

    _need(tuple(logits_dyn.shape) == (1, 400) and bool(torch.isfinite(logits_dyn).all()), "Invalid 400-class logits")
    geoms = model_400.collect_last_runtime_geometries()
    all_877 = all(
        g["geometry"]["actual_window_size"] == [8, 7, 7] and not g["geometry"]["window_clipped"]
        for g in geoms
    )
    _need(all_877, "Expected all 12 blocks unclipped (8,7,7) for 16x224x224")
    max_diff = float((logits_dyn - logits_up).abs().max().item())
    bitwise_eq = bool(torch.equal(logits_dyn, logits_up))
    _need(bitwise_eq, f"400-class full-window output differs from upstream: max_abs_diff={max_diff}")

    return {
        "status": "PASS",
        "checkpoint_tensor_count": len(raw_sd),
        "all_checkpoint_tensors_finite": all_finite_in_ckpt,
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "relative_position_index_buffers_count": len(pre_load_rel_pos_buffers),
        "relative_position_index_buffers_bitwise_equal_to_ckpt": rel_pos_buffers_identical,
        "full_window_16x224x224_all_12_blocks_877": all_877,
        "full_window_400class_logits_bitwise_equal_vs_upstream": bitwise_eq,
        "full_window_400class_max_abs_diff": max_diff,
        "equiv_eval_seconds": round(equiv_s, 4),
    }


def _verify_binary_head_replacement(raw_sd: dict[str, torch.Tensor]) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Verify explicit head_seed isolation and bitwise preservation of all 183 non-head tensors."""
    # Check global RNG non-mutation directly on replace_head_for_binary
    m_probe = build_swin3d_t_400_dynamic_pos(strict_contract_shape=True)
    m_probe.load_state_dict(raw_sd, strict=True)
    torch.manual_seed(987654)
    global_rng_before = torch.get_rng_state().clone()
    replace_head_for_binary(m_probe, num_classes=2, head_seed=HEAD_SEED)
    global_rng_unchanged = bool(torch.equal(global_rng_before, torch.get_rng_state()))
    _need(global_rng_unchanged, "replace_head_for_binary mutated global PyTorch RNG state")

    # Independent second build with same head_seed & third build with different head_seed
    m_same = build_swin3d_t_binary50_dynamic_pos(
        strict_contract_shape=True,
        pretrained_400_state_dict=raw_sd,
        head_seed=HEAD_SEED,
    )
    m_diff = build_swin3d_t_binary50_dynamic_pos(
        strict_contract_shape=True,
        pretrained_400_state_dict=raw_sd,
        head_seed=ALT_HEAD_SEED,
    )

    same_head_weight = bool(torch.equal(m_probe.head.weight, m_same.head.weight))
    same_head_bias = bool(torch.equal(m_probe.head.bias, m_same.head.bias))
    diff_head_weight = not torch.equal(m_probe.head.weight, m_diff.head.weight)
    bias_all_zero = bool(torch.equal(m_probe.head.bias, torch.zeros_like(m_probe.head.bias)))
    _need(same_head_weight and same_head_bias, "Same head_seed did not reproduce identical binary head")
    _need(diff_head_weight, "Different head_seed did not change binary head weights")
    _need(bias_all_zero, "Expected zero-initialized binary head bias")

    # Verify ALL non-head weights & buffers remain bitwise equal to downloaded checkpoint
    non_head_keys = [k for k in raw_sd.keys() if k not in ("head.weight", "head.bias")]
    probe_sd = m_probe.state_dict()
    mismatched_non_head = [k for k in non_head_keys if not torch.equal(probe_sd[k], raw_sd[k])]
    _need(not mismatched_non_head, f"Non-head tensors altered after head swap: {mismatched_non_head}")

    head_weight_sha = hashlib.sha256(m_probe.head.weight.detach().numpy().tobytes()).hexdigest()
    return m_probe, {
        "status": "PASS",
        "head_seed": HEAD_SEED,
        "alt_head_seed_checked": ALT_HEAD_SEED,
        "global_rng_unchanged_by_head_init": global_rng_unchanged,
        "same_head_seed_bitwise_equal": same_head_weight and same_head_bias,
        "different_head_seed_changes_weight": diff_head_weight,
        "head_weight_shape": list(m_probe.head.weight.shape),
        "head_bias_shape": list(m_probe.head.bias.shape),
        "head_bias_all_zero": bias_all_zero,
        "head_weight_sha256": head_weight_sha,
        "non_head_tensors_checked": len(non_head_keys),
        "non_head_tensors_all_bitwise_equal_to_ckpt": True,
        "total_params_binary": sum(p.numel() for p in m_probe.parameters()),
    }


def _verify_fixtures_pretrained_cpu_forward(
    model_a: torch.nn.Module,
    raw_sd: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Run deterministic CPU eval forward on fixed 04/22 saved RGB fixtures with pretrained weights."""
    with patch.object(cv2, "VideoCapture", side_effect=AssertionError("Video decode forbidden")):
        sample15 = verify_sample15_metadata()
        batch, provenances = collate_saved_binary_fixtures(ALLOWED_PIXEL_FIXTURE_IDS)

    inputs = batch["tensor"]
    targets = batch["targets"]
    _need(tuple(inputs.shape) == (2, 3, CONTRACT_FRAMES, CONTRACT_HEIGHT, CONTRACT_WIDTH), f"Bad shape: {inputs.shape}")
    _need(targets.tolist() == [0, 1], f"Bad binary targets: {targets}")
    _need(all(p["video_open_count"] == 0 and p["video_decode_count"] == 0 for p in provenances), "Video opened!")

    # Verify patch_embed.proj receives batch tensor verbatim (no internal re-normalization)
    model_a.eval()
    seen_first_conv: list[torch.Tensor] = []

    def _capture(_mod, hook_inputs):
        seen_first_conv.append(hook_inputs[0].detach().clone())

    handle = model_a.patch_embed.proj.register_forward_pre_hook(_capture)
    t0 = time.monotonic()
    try:
        with torch.no_grad():
            logits_a = model_a(inputs)
            ce_a = binary_cross_entropy_from_logits(logits_a, targets)
    finally:
        handle.remove()

    # Independent second pretrained model instance with identical head_seed
    model_b = build_swin3d_t_binary50_dynamic_pos(
        strict_contract_shape=True,
        pretrained_400_state_dict=raw_sd,
        head_seed=HEAD_SEED,
    ).eval()
    with torch.no_grad():
        logits_b = model_b(inputs)
        ce_b = binary_cross_entropy_from_logits(logits_b, targets)
    eval_s = time.monotonic() - t0

    _need(len(seen_first_conv) == 1 and torch.equal(seen_first_conv[0], inputs), "Input altered before first conv!")
    _need(tuple(logits_a.shape) == (2, 2) and bool(torch.isfinite(logits_a).all()), "Non-finite fixture logits")
    _need(bool(torch.isfinite(ce_a).all()), "Non-finite fixture CE")
    _need(torch.equal(logits_a, logits_b) and torch.equal(ce_a, ce_b), "Pretrained fixture eval not deterministic")

    geoms = model_a.collect_last_runtime_geometries()
    stage3_geoms = [g for g in geoms if g["module"].startswith("features.6.")]
    _need(
        stage3_geoms[0]["geometry"]["actual_window_size"] == [8, 4, 5]
        and stage3_geoms[0]["geometry"]["actual_shift_size"] == [0, 0, 0]
        and stage3_geoms[1]["geometry"]["actual_window_size"] == [8, 4, 5]
        and stage3_geoms[1]["geometry"]["actual_shift_size"] == [4, 0, 0],
        f"Unexpected Stage 3 geometry on fixtures: {stage3_geoms}",
    )

    probs = torch.softmax(logits_a, dim=-1)
    return {
        "status": "PASS",
        "fixture_ids": list(ALLOWED_PIXEL_FIXTURE_IDS),
        "binary_classes": list(BINARY_CLASSES),
        "batch_shape": list(inputs.shape),
        "targets": targets.tolist(),
        "video_open_count": 0,
        "video_decode_count": 0,
        "first_conv_input_bitwise_equal_to_batch": True,
        "no_double_normalization_verified": True,
        "normalization_mean": list(CONTRACT_MEAN),
        "normalization_std": list(CONTRACT_STD),
        "stage3_runtime_geometries": stage3_geoms,
        "fixture_logits": [[round(float(v), 6) for v in row] for row in logits_a.tolist()],
        "fixture_probs": [[round(float(v), 6) for v in row] for row in probs.tolist()],
        "fixture_binary_ce": round(float(ce_a.item()), 6),
        "two_independent_loads_bitwise_equal": True,
        "eval_two_runs_seconds": round(eval_s, 4),
        "sample15_metadata": sample15,
        "fixture_provenances": provenances,
    }


def run(out_dir: Path, weight_path: Path, *, allow_single_download: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    _need(not torch.cuda.is_available(), "GPU must be hidden/disabled for CPU pretrained load acceptance")

    tv_swin_path = Path(tv_swin.__file__).resolve()
    tv_swin_sha_before = _sha256_file(tv_swin_path)
    pkg_hashes = _package_source_hashes()

    start = time.monotonic()
    weight_info = _acquire_official_weight(weight_path, allow_single_download=allow_single_download)
    print(
        f"[weight] path={weight_info['local_path']} bytes={weight_info['size_bytes']} "
        f"sha256={weight_info['full_sha256']} downloaded={weight_info['downloaded_in_this_run']}",
        flush=True,
    )

    # Hard-block any further torchvision automatic download attempts during verification
    with patch.object(
        tv_swin.Swin3D_T_Weights.KINETICS400_V1,
        "get_state_dict",
        side_effect=AssertionError("Unexpected torchvision download attempt after local acquisition"),
    ):
        raw_sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        check_400 = _verify_400class_strict_load_and_equivalence(raw_sd)
        print(
            f"[400class] strict_load=PASS tensors={check_400['checkpoint_tensor_count']} "
            f"full_877_bitwise_equal={check_400['full_window_400class_logits_bitwise_equal_vs_upstream']}",
            flush=True,
        )
        model_bin, check_head = _verify_binary_head_replacement(raw_sd)
        print(
            f"[head_swap] head_seed={check_head['head_seed']} "
            f"non_head_exact={check_head['non_head_tensors_checked']}/{check_head['non_head_tensors_checked']} "
            f"global_rng_unchanged={check_head['global_rng_unchanged_by_head_init']}",
            flush=True,
        )
        check_fixtures = _verify_fixtures_pretrained_cpu_forward(model_bin, raw_sd)
        print(
            f"[fixtures] 04/22 bitwise_equal={check_fixtures['two_independent_loads_bitwise_equal']} "
            f"binary_ce={check_fixtures['fixture_binary_ce']} video_opens=0",
            flush=True,
        )

    total_s = time.monotonic() - start
    tv_swin_sha_after = _sha256_file(tv_swin_path)
    _need(tv_swin_sha_before == tv_swin_sha_after, "Installed torchvision file changed!")

    report = {
        "schema": "swin3d_t_binary50_pretrained_load_acceptance_v1",
        "status": "PASS_PRETRAINED_LOAD_CPU_ONLY",
        "model_variant": MODEL_VARIANT_NAME,
        "display_name": "含動態位置索引修正的 Swin3D-T",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(total_s, 4),
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
        "package_code_sha256": pkg_hashes,
        "scope": {
            "single_official_weight_download_only": True,
            "gpu_executed": False,
            "optimizer_updates": 0,
            "video_opens": 0,
            "validation_access": 0,
            "test_access": 0,
            "historical_test_contact_disclosed": True,
        },
        "checks": {
            "official_weight_acquisition_and_hash": weight_info,
            "400class_strict_load_and_upstream_equivalence": check_400,
            "binary_head_replacement_and_backbone_preservation": check_head,
            "fixed_fixtures_04_22_pretrained_cpu_forward": check_fixtures,
        },
    }
    out_json = out_dir / "pretrained_load_acceptance.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Swin3D-T official weight download & CPU load acceptance")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--weight-path", type=Path, default=DEFAULT_LOCAL_WEIGHT_PATH)
    parser.add_argument(
        "--allow-single-download",
        action="store_true",
        help="Allow downloading official Kinetics-400 weight once if missing locally.",
    )
    args = parser.parse_args()
    report = run(args.out_dir, args.weight_path, allow_single_download=args.allow_single_download)
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "full_sha256": report["checks"]["official_weight_acquisition_and_hash"]["full_sha256"],
                "checks": {k: v["status"] for k, v in report["checks"].items()},
                "output_json": str(args.out_dir / "pretrained_load_acceptance.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
