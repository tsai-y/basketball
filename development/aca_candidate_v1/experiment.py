"""Candidate identity, CPU preparation and strict launch prerequisites.

Imports the existing reconstruction read-only. Never modify its modules or
borrow its trained baseline/engineering checkpoint for initialization.
"""
import copy
import csv
from dataclasses import asdict, dataclass
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import torch
import torchvision

from development.aca_net_v1.baseline_preflight import freeze_batchnorm
from development.aca_net_v1.data import ROOT, POLICY, VideoScope, file_sha256, require
from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.modules import ReconstructionConfig
from development.aca_net_v1 import train as baseline
from development.aca_net_v1 import weights

FOLDER = Path(__file__).resolve().parent
SCHEMA = "aca_dense_candidate_resume_v1"
PREFLIGHT_SCHEMA = "aca_dense_candidate_gpu_preflight_v1"
PARAMETERS = 93163870
REFERENCE_RUN = weights.FOLDER / "artifacts/baseline_train_seed20261025_yqk5pn63"


@dataclass(frozen=True)
class CandidateConfig:
    variant: str = "aca_dense_candidate"
    seed: int = 20261025
    epochs: int = 30
    patience: int = 5
    effective_batch: int = 16
    micro_batch: int = 1
    workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.0
    clip_norm: float = 1.0
    checkpoint_every: int = 25

    def __post_init__(self):
        require(self.variant == "aca_dense_candidate", "Only the explicitly labelled dense candidate is accepted")
        # Reuse numeric protocol validation, NOT the baseline architecture gate.
        baseline.TrainConfig(**{**asdict(self), "variant": "baseline"})
        require(self.micro_batch == 1, "Candidate micro-batch is pinned to 1 until separately GPU-validated")


def config_from(path):
    return CandidateConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def environment():
    # NVML CLI metadata only: do not initialize a Python CUDA context during prepare.
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True)
    rows = list(csv.reader(io.StringIO(result.stdout)))
    require(len(rows) == 1 and len(rows[0]) == 4, "This experiment requires one identified GPU")
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "opencv": cv2.__version__, "numpy": np.__version__,
            "cuda_build": torch.version.cuda, "gpu": [v.strip() for v in rows[0]],
            "precision": "FP32_no_AMP_no_TF32", "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8", "cpu_threads": 4}


def source_hashes():
    paths = {p.relative_to(ROOT).as_posix(): p for p in FOLDER.glob("*.py")}
    paths[(FOLDER / "config.json").relative_to(ROOT).as_posix()] = FOLDER / "config.json"
    for name in baseline.source_hashes():
        p = ROOT / name.removeprefix("repo/") if name.startswith("repo/") else weights.FOLDER / name
        paths[p.relative_to(ROOT).as_posix()] = p
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def build_plan(source_root, config):
    train = VideoScope(source_root=source_root, split="train")
    validation = VideoScope(source_root=source_root, split="validation")
    identity = weights.pin()
    weights.verify_payload(weights.FOLDER / identity["local_relative_path"], identity)
    require(REFERENCE_RUN.is_dir(), "Missing fixed baseline reference run")
    return {"schema": SCHEMA, "config": asdict(config), "architecture": asdict(ReconstructionConfig()),
            "model_class": "VideoResNet50(variant=aca,num_classes=2)", "expected_parameters": PARAMETERS,
            "source_root": str(Path(source_root).resolve()), "input_hashes": train.hashes,
            "train_count": len(train.rows), "validation_count": len(validation.rows),
            "train_ordered_ids_sha256": baseline.canonical_hash([r["video_id"] for r in train.rows]),
            "validation_ordered_ids_sha256": baseline.canonical_hash([r["video_id"] for r in validation.rows]),
            "original_source_inventory": baseline.source_inventory((train, validation)),
            "policy": POLICY, "weight_identity": identity, "runtime": environment(),
            "source_code_sha256": source_hashes(), "baseline_reference_run": str(REFERENCE_RUN),
            "baseline_reference_config_sha256": file_sha256(REFERENCE_RUN / "config.json"),
            "initialization": "pinned_ImageNet_V1_backbone_plus_fresh_ACA; same_fresh_head_as_baseline_seed",
            "bn_policy": "all_affine_and_running_buffers_frozen_including_new_TSCI_BN",
            "optimizer": "Adam_all_non_BN_parameters_lr1e-4_wd0", "scheduler": "none",
            "selection": "strict_minimum_validation_CE_ties_keep_earlier_epoch",
            "test_policy": "no_test_interface_no_test_video_access",
            "paper_exact_reproduction": False, "paper_reported_parameters_approx": 26800000,
            "unresolved": ["dense_gating_parameter_mismatch", "unverified_hyperparameters_r16_K3_beta025",
                           "figure_equation_sigmoid_and_rescale_ambiguity", "author_stride_BN_pooling_preprocessing_unknown"],
            "optimizer_steps_per_full_epoch": 450, "last_effective_batch_videos": 9,
            "resume_policy": "optimizer_boundary_with_model_Adam_RNG_partial_epoch_metrics",
            "formal_training_started": False}


def initialized_candidate(seed):
    reference, initialization = weights.initialized_baseline(seed)
    head = copy.deepcopy(reference.fc.state_dict())
    del reference
    torch.manual_seed(seed)
    model = VideoResNet50(variant="aca", num_classes=2, config=ReconstructionConfig())
    state = torch.load(initialization["checkpoint"], map_location="cpu", weights_only=True)
    transfer = model.load_imagenet_state(state)
    require(len(transfer["legacy_bn_counters_initialized_to_zero"]) == 53 and
            transfer["backbone_tensors_loaded"] == 265, "Unexpected ImageNet transfer coverage")
    model.fc.load_state_dict(head, strict=True)
    freeze_batchnorm(model)
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS, "Candidate architecture changed")
    return model, {**initialization, "transfer": transfer, "fresh_head_matches_baseline_same_seed": True,
                   "trained_baseline_checkpoint_loaded": False, "engineering_checkpoint_loaded": False}


def snapshot_sources(output, plan):
    for name, expected in plan["source_code_sha256"].items():
        source = ROOT / name
        require(file_sha256(source) == expected, "Source changed before snapshot")
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def prepare(source_root, config):
    require(not torch.cuda.is_initialized(), "CPU preparation must start without a CUDA context")
    torch.set_num_threads(2)
    plan = build_plan(source_root, config)
    model, initialization = initialized_candidate(config.seed)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite initialized candidate")
    reference, _ = weights.initialized_baseline(config.seed)
    require(all(torch.equal(v, reference.fc.state_dict()[k]) for k, v in model.fc.state_dict().items()),
            "Fresh classification head does not match paired baseline initialization")
    del reference
    del model
    with torch.device("meta"):
        shape_model = VideoResNet50(variant="aca", num_classes=2)
        logits = shape_model(torch.empty(1, 3, 50, 120, 160))
    require(tuple(logits.shape) == (1, 2), "Full candidate meta shape failed")
    require(not torch.cuda.is_initialized(), "Preparation unexpectedly initialized CUDA")
    parent = FOLDER / "artifacts"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="cpu_preparation_", dir=parent))
    baseline.atomic_json(output / "plan.json", plan)
    snapshot_sources(output, plan)
    report = {"status": "cpu_preparation_passed_gpu_preflight_pending", "output": str(output),
              "plan_sha256": baseline.canonical_hash(plan), "parameters": PARAMETERS,
              "trainable_parameters": trainable, "initialization": initialization,
              "all_initialized_parameters_finite": True, "fresh_head_pairing_verified": True,
              "meta_output_shape": list(logits.shape), "cuda_context_initialized": False,
              "video_decodes": 0, "test_videos_opened": 0, "optimizer_steps": 0,
              "formal_training_started": False, "paper_exact_reproduction": False,
              "baseline_completion_required_before_gpu_preflight": True,
              "may_start_training_without_gpu_preflight": False}
    baseline.atomic_json(output / "report.json", report)
    return report


def assert_launch_prerequisites(accepted):
    require(accepted, "Pass --accept-candidate-architecture; this is not verified author ACA-Net")
    summary_path = REFERENCE_RUN / "summary.json"
    require(summary_path.is_file(), "Baseline is not complete; leave its GPU run untouched")
    summary = json.loads(summary_path.read_text())
    require(summary.get("status") == "completed", "Baseline did not finish successfully")
    # Refuse any non-desktop CUDA process, not just the known baseline PID.
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    desktop = {"gnome-remote-desktop-daemon", "Discord", "nautilus", "gnome-text-editor", "ptyxis",
               "gnome-shell", "Xwayland", "chrome", "codex", "Codex"}
    busy = []
    for row in result.stdout.splitlines():
        pid = int(row.strip())
        if pid == os.getpid():
            continue
        try:
            name = (Path("/proc") / str(pid) / "exe").resolve(strict=True).name
        except FileNotFoundError:
            continue  # Process exited between NVML query and /proc inspection.
        except PermissionError:
            name = "unknown_permission_denied"
        if name not in desktop:
            busy.append({"pid": pid, "name": name})
    require(not busy, f"Other GPU compute processes are active: {busy}; none were stopped")
    require(shutil.disk_usage(FOLDER).free >= 8 * 1024**3,
            "At least 8 GiB free space required for candidate checkpoints")


def verify_gpu_report(path, plan):
    report = json.loads(Path(path).read_text())
    require(report.get("schema") == PREFLIGHT_SCHEMA and report.get("status") == "passed" and
            report.get("plan_sha256") == baseline.canonical_hash(plan) and
            report.get("engineering_only") is True and report.get("formal_training_started") is False and
            report.get("checkpoint_may_initialize_formal_training") is False and
            report.get("checkpoint_roundtrip_logits_bitwise_equal") is True and
            report.get("synthetic_cuda_resume_bitwise_equal") is True,
            "Missing/stale/incompatible candidate GPU preflight; CPU/meta or baseline report is insufficient")
