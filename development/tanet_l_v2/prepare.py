"""CPU-only preparation/check entry. Deliberately has NO training/test mode."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import tempfile
import time

import cv2
import torch
import torchvision

from .config import FOLDER, InputConfig, load_config, need, sha256, verify_upstream
from .data import VideoScope, decode
from .model import TANetBinary
from .upstream.temporal_module import TAM


def source_hashes():
    files = [p for p in FOLDER.rglob("*") if p.is_file() and
             "artifacts" not in p.parts and "__pycache__" not in p.parts]
    result = {str(p.relative_to(FOLDER)): sha256(p) for p in sorted(files)}
    root = FOLDER.parents[1]
    for relative in ("scripts/audit_i3d_temporal_feasibility.py", "development/data_pipeline/pipeline.py"):
        result["project/" + relative] = sha256(root / relative)
    return result


def build_plan():
    need(not torch.cuda.is_initialized(), "CPU preparation must not inherit an initialized CUDA context")
    cfg, upstream = load_config(), verify_upstream()
    train, validation = VideoScope("train"), VideoScope("validation")
    need(len(train.rows) == 7193 and len(validation.rows) == 1547, "Fixed split counts changed")
    need(not ({r["source_group"] for r in train.rows} &
              {r["source_group"] for r in validation.rows}), "Train/validation source groups overlap")
    with torch.device("meta"):
        shape_model = TANetBinary(frames=cfg["frames"], dropout=cfg["dropout"]).eval()
        shape = list(shape_model(torch.empty(1, 3, cfg["frames"], cfg["height"], cfg["width"])).shape)
        parameters = sum(p.numel() for p in shape_model.parameters())
    need(shape == [1, 2], "Full-resolution shape check failed")
    tam_count = sum(isinstance(m, TAM) for m in shape_model.modules())
    need(tam_count == 16, "Wrong author TAM inventory")
    return {"schema": cfg["schema"], "status": "preparation_only_formal_training_not_authorized",
            "config": cfg, "upstream": upstream, "source_code_sha256": source_hashes(),
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torchvision": str(torchvision.__version__), "opencv": cv2.__version__,
                        "compiled_cuda": torch.version.cuda,
                        "mmcv_installed": importlib.util.find_spec("mmcv") is not None,
                        "mmengine_installed": importlib.util.find_spec("mmengine") is not None,
                        "mmaction_installed": importlib.util.find_spec("mmaction") is not None,
                        "cuda_context_initialized": False, "package_installations": 0},
            "data_identity": train.identity, "train_count": len(train.rows),
            "validation_count": len(validation.rows), "train_validation_source_groups_disjoint": True,
            "parameters": parameters, "meta_input_shape": [1, 3, 50, 192, 256],
            "meta_output_shape": shape, "tam_count": tam_count,
            "normalization": "external_RGB_float32_ImageNet_V1_once",
            "input_policy": "full_video_uniform50_no_temporal_selector_no_augmentation",
            "short_input_policy": "reject_no_repetition_no_black_padding",
            "loss_interface": "two_raw_logits_for_cross_entropy_Miss0_Make1",
            "validation_metrics_required_later": ["accuracy", "macro_f1", "CE"],
            "formal_recipe_status": "pending_temporal_selection_review_GPU_readiness_and_human_start",
            "not_reproduced": ["author_complete_dataset_training_recipe", "ACA_Net", "paper_results"],
            "future_50_frame_weight_policy": "ImageNet_backbone_only_fresh_TAM_no_8_or_16_frame_video_checkpoint",
            "optimizer_steps": 0, "test_videos_opened": 0, "formal_training_started": False}


def select_smoke_ids(scope):
    # Deterministic metadata-only engineering samples, not performance selection.
    ids, groups = [], set()
    for original_class in ("2p0", "2p1"):
        row = next(r for r in scope.rows if r["original_class"] == original_class
                   and r["source_group"] not in groups)
        ids.append(row["video_id"])
        groups.add(row["source_group"])
    return ids


def prepare(*, smoke_videos=False):
    started = time.monotonic()
    torch.set_num_threads(2)
    plan = build_plan()
    cfg = plan["config"]
    torch.manual_seed(cfg["seed"])
    model = TANetBinary(frames=cfg["frames"], dropout=cfg["dropout"],
                        checkpoint=cfg["imagenet_checkpoint"],
                        checkpoint_sha256=cfg["imagenet_sha256"],
                        checkpoint_size=cfg["imagenet_size_bytes"]).cpu().eval()
    need(sum(isinstance(m, TAM) for m in model.modules()) == 16, "Wrong TAM inventory")
    need(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite model parameters")
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 50, 32, 32))
    need(tuple(logits.shape) == (1, 2) and torch.isfinite(logits).all().item(), "CPU forward failed")
    checks = []
    if smoke_videos:
        scope = VideoScope("train")
        for video_id in select_smoke_ids(scope):
            sample = decode(scope, video_id, InputConfig())
            with torch.inference_mode():
                result = model(sample["tensor"].unsqueeze(0))
            need(tuple(result.shape) == (1, 2) and torch.isfinite(result).all().item(),
                 "Native-resolution train smoke forward failed")
            checks.append({k: v for k, v in sample.items() if k != "tensor"} |
                          {"input_shape": list(sample["tensor"].shape), "output_shape": list(result.shape),
                           "finite_output": True, "accuracy_computed": False})
            del sample, result
    need(not torch.cuda.is_initialized(), "Preparation unexpectedly initialized CUDA")
    need(source_hashes() == plan["source_code_sha256"], "Source changed during preparation")
    report = {"status": "cpu_preparation_passed_NOT_formal_training_ready", "plan": plan,
              "initialization": model.initialization, "synthetic_cpu_forward_passed": True,
              "train_video_smoke_checks": checks, "train_videos_opened": len(checks),
              "validation_videos_opened": 0, "test_videos_opened": 0,
              "optimizer_steps": 0, "cuda_context_initialized": False,
              "formal_training_started": False, "trained_model_saved": False,
              "elapsed_seconds": time.monotonic() - started,
              "remaining_gates": ["temporal_selection_policy_review", "formal_recipe_confirmation",
                                  "separately_authorized_GPU_readiness", "explicit_formal_training_start"]}
    parent = FOLDER / "artifacts"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="cpu_prepare_", dir=parent))
    report["output"] = str(output)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Verified local ImageNet + CPU-only checks")
    parser.add_argument("--smoke-videos", action="store_true", help="Explicitly decode exactly two train videos")
    args = parser.parse_args(argv)
    need(not args.smoke_videos or args.prepare, "--smoke-videos requires --prepare")
    result = prepare(smoke_videos=args.smoke_videos) if args.prepare else build_plan()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
