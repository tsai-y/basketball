"""Paired initialization and fixed experiment identity for all four arms."""
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
import platform
import cv2
import numpy as np
import torchvision
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
from torch import nn

from development.aca_net_v1 import train as base, weights
from development.aca_net_v1.baseline_preflight import freeze_batchnorm
from development.aca_net_v1.data import POLICY, ROOT, VideoScope, file_sha256, require, select_smoke_ids
from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.modules import ACAModule, LSTA, TSCI, ReconstructionConfig

FOLDER = Path(__file__).resolve().parent
SCHEMA = "aca_four_arm_resume_v1"
PREFLIGHT_SCHEMA = "aca_four_arm_gpu_readiness_v1"
VARIANTS = ("baseline", "lsta", "tsci", "aca")
SEEDS = (20261024, 20261025, 20261026)
PARAMETERS = dict(baseline=23512130, lsta=93159070, tsci=23516930, aca=93163870)


@dataclass(frozen=True)
class ArmConfig:
    variant: str = "baseline"
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
        require(self.variant in VARIANTS, "Only baseline/lsta/tsci/aca are in this controlled protocol")
        require(type(self.seed) is int and self.seed in SEEDS, "Only the three prospective seeds are allowed")
        base.TrainConfig(**{**asdict(self), "variant": "baseline"})
        require(self.micro_batch == 1, "Physical batch is pinned to one")


def component_seed(seed, block, component):
    text = f"aca_four_arm_init_v1:{seed}:{block}:{component}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def load_imagenet():
    identity = weights.pin()
    path = weights.FOLDER / identity["local_relative_path"]
    weights.verify_payload(path, identity)
    return torch.load(path, map_location="cpu", weights_only=True), identity


def initialized(variant, seed, state):
    """The same backbone/head and per-block shared-component RNG in every arm.

    A common baseline is constructed first. Added modules cannot consume RNG
    before the task head. Each LSTA/TSCI has its own named, stable RNG stream.
    The caller must load and verify ImageNet with load_imagenet() in real runs.
    """
    require(variant in VARIANTS and type(seed) is int and seed in SEEDS, "Unknown arm/seed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = VideoResNet50(variant="baseline", num_classes=2)
        transfer = model.load_imagenet_state(state)
        cfg = ReconstructionConfig()
        for stage in ("layer1", "layer2", "layer3", "layer4"):
            for index, block in enumerate(getattr(model, stage)):
                label = f"{stage}.{index}"
                channels = block.conv1.out_channels
                lsta = tsci = None
                if variant in ("lsta", "aca"):
                    torch.manual_seed(component_seed(seed, label, "lsta"))
                    lsta = LSTA(channels, cfg)
                if variant in ("tsci", "aca"):
                    torch.manual_seed(component_seed(seed, label, "tsci"))
                    tsci = TSCI()
                if variant == "aca":
                    # Use the original class/key layout; its temporary children
                    # are immediately replaced by the independently paired ones.
                    module = ACAModule(channels, cfg)
                    module.lsta, module.tsci = lsta, tsci
                else:
                    module = lsta if lsta is not None else tsci
                if module is not None:
                    block.aca = module
        model.variant = variant
    freeze_batchnorm(model)
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS[variant], "Architecture count changed")
    return model, transfer


def component_digests(model):
    hashes = {key: hashlib.sha256() for key in ("backbone", "head", "lsta", "tsci")}
    counts = {key: 0 for key in hashes}
    for name, tensor in model.state_dict().items():
        if name.startswith("fc."):
            component, canonical = "head", name
        elif ".aca." not in name:
            component, canonical = "backbone", name
        else:
            prefix, suffix = name.split(".aca.", 1)
            if model.variant == "aca":
                component, suffix = suffix.split(".", 1)
            else:
                component = model.variant
            canonical = prefix + "." + suffix
        hashes[component].update(canonical.encode())
        value = tensor.detach().cpu().contiguous()
        hashes[component].update(str((str(value.dtype), tuple(value.shape))).encode())
        hashes[component].update(memoryview(value.numpy()).cast("B"))
        counts[component] += 1
    return {key: hashes[key].hexdigest() for key in hashes if counts[key]}


def source_hashes():
    from development.data_pipeline import pipeline
    paths = list(FOLDER.glob("*.py")) + [FOLDER / "protocol.json"]
    paths += list(weights.FOLDER.glob("*.py")) + [weights.PIN_FILE]
    paths += [Path(pipeline.__file__), Path(pipeline.gate1a.__file__)]
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(set(paths))}


def environment():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                             "--format=csv,noheader"], check=True, capture_output=True, text=True)
    rows = list(csv.reader(io.StringIO(result.stdout)))
    require(len(rows) == 1 and len(rows[0]) == 4, "Exactly one identified GPU is required")
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "opencv": cv2.__version__, "numpy": np.__version__,
            "cudnn": torch.backends.cudnn.version(), "cuda_build": torch.version.cuda,
            "GPU": [x.strip() for x in rows[0]], "precision": "FP32_no_AMP_no_TF32",
            "cublas_workspace": ":4096:8", "deterministic_algorithms": True}


def assert_idle_gpu():
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True)
    desktop = {"Discord", "gnome-shell", "Xwayland", "chrome", "codex", "Codex",
               "gnome-remote-desktop-daemon", "nautilus", "gnome-text-editor", "ptyxis"}
    busy = []
    for text in query.stdout.splitlines():
        pid = int(text.strip())
        if pid == os.getpid():
            continue
        try:
            name = (Path('/proc') / str(pid) / 'exe').resolve(strict=True).name
        except FileNotFoundError:
            continue
        except PermissionError:
            name = "unknown_permission_denied"
        if name not in desktop:
            busy.append((pid, name))
    require(not busy, f"Other GPU compute processes active: {busy}; none were stopped")
    require(shutil.disk_usage(FOLDER).free >= 8 * 1024**3, "Need at least 8 GiB free for engineering checkpoints")


def campaign_plan(source_root):
    train = VideoScope(source_root=source_root, split="train")
    validation = VideoScope(source_root=source_root, split="validation")
    identity = weights.pin()
    weights.verify_payload(weights.FOLDER / identity["local_relative_path"], identity)
    return {"schema": "aca_four_arm_campaign_plan_v1", "protocol": json.loads((FOLDER / "protocol.json").read_text()),
            "source_root": str(Path(source_root).resolve()), "input_hashes": train.hashes,
            "train_count": len(train.rows), "validation_count": len(validation.rows),
            "train_ordered_ids_sha256": base.canonical_hash([r["video_id"] for r in train.rows]),
            "validation_ordered_ids_sha256": base.canonical_hash([r["video_id"] for r in validation.rows]),
            "source_inventory": base.source_inventory((train, validation)),
            "selected_preflight_ids": {"train": select_smoke_ids(train), "validation": select_smoke_ids(validation)},
            "source_code_sha256": source_hashes(), "runtime": environment(), "weight_identity": identity,
            "input_policy": POLICY, "arms": [asdict(ArmConfig(variant=v, seed=s)) for s in SEEDS for v in VARIANTS],
            "parameters_by_variant": PARAMETERS, "formal_training_started": False,
            "optimizer_steps_per_epoch": 450, "tail_batch_videos": 9,
            "checkpoint_kind_isolation": "formal, synthetic engineering and real engineering are distinct",
            "historical_runs_reused": False, "test_interface": False, "paper_exact_reproduction": False}


def arm_plan(campaign, config, *, synthetic=False, train_count=None, validation_count=None):
    return {"schema": SCHEMA, "campaign_sha256": base.canonical_hash(campaign), "config": asdict(config),
            "train_count": campaign["train_count"] if train_count is None else train_count,
            "validation_count": campaign["validation_count"] if validation_count is None else validation_count,
            "engineering_only_synthetic_toy": synthetic}


def verify_preflight(path, campaign, config):
    value = json.loads(Path(path).read_text())
    require(value.get("schema") == PREFLIGHT_SCHEMA and value.get("status") == "passed" and
            value.get("campaign_sha256") == base.canonical_hash(campaign) and
            value.get("seed") == config.seed and value.get("formal_training_started") is False and
            value.get("engineering_only") is True and
            value.get("checkpoint_may_initialize_formal_training") is False,
            "Missing/stale/cross-seed GPU readiness")
    arm = value.get("arms", {}).get(config.variant, {})
    require(arm.get("status") == "passed" and arm.get("checkpoint_roundtrip_bitwise_equal") is True and
            arm.get("synthetic_resume_bitwise_equal") is True, "Arm-specific GPU readiness did not pass")


def prepare(source_root):
    require(not torch.cuda.is_initialized(), "Preparation must start without CUDA context")
    torch.set_num_threads(4)
    plan = campaign_plan(source_root)
    state, _ = load_imagenet()
    seeds = {}
    for seed in SEEDS:
        rows = {}
        for variant in VARIANTS:
            model, transfer = initialized(variant, seed, state)
            require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite initialized parameter")
            rows[variant] = {"digests": component_digests(model), "transfer": transfer,
                             "total_parameters": sum(p.numel() for p in model.parameters()),
                             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            del model
        require(len({r["digests"]["backbone"] for r in rows.values()}) == 1 and
                len({r["digests"]["head"] for r in rows.values()}) == 1, "Backbone/head pairing failed")
        require(rows["lsta"]["digests"]["lsta"] == rows["aca"]["digests"]["lsta"], "LSTA pairing failed")
        require(rows["tsci"]["digests"]["tsci"] == rows["aca"]["digests"]["tsci"], "TSCI pairing failed")
        seeds[str(seed)] = rows
        print(f"CPU seed={seed}: all four arms and shared modules paired", flush=True)
    require(not torch.cuda.is_initialized(), "Preparation unexpectedly initialized CUDA")
    parent = FOLDER / "artifacts"; parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="cpu_prepare_", dir=parent))
    base.atomic_json(output / "plan.json", plan)
    report = {"status": "cpu_preparation_passed", "campaign_sha256": base.canonical_hash(plan),
              "seed_initialization": seeds, "paired_backbone_head_and_overlapping_modules": True,
              "CUDA_context_initialized": False, "video_decodes": 0, "optimizer_updates": 0,
              "formal_training_started": False, "output": str(output)}
    base.atomic_json(output / "report.json", report)
    for name, expected in plan["source_code_sha256"].items():
        require(file_sha256(ROOT / name) == expected, "Source changed before preparation snapshot")
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, target)
    print(f"OUTPUT: {output}", flush=True)
    return report
