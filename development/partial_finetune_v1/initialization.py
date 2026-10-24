"""Strict offline initialization from the existing extractor and fixed-seed head."""

import hashlib
from pathlib import Path

import timm
import torch

from development.mixnet_i3d.backbone import MixNetI3DBackbone
from development.partial_finetune_v1.model import PartialFineTuneModel
from development.temporal_baseline_v1.model import WindowTemporalTransformer
from development.temporal_baseline_v1.train_temporal import CACHE_RUN_ID, require


ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR = ROOT / "development/mixnet_i3d/artifacts/frozen_extractor_v1.pt"
HEAD = ROOT / "development/temporal_baseline_v1/artifacts/temporal_seed20261024_y7wv4h0p/best.pt"
EXTRACTOR_SHA = "d8a8162d92ebb5270770d745c7a424126329f3a69a43edeeba0d3a99d72416d4"
HEAD_SHA = "50146a9ab7967573c78d9fa6bf379d7bde16143c7ae0573365a35967cf5a029c"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_initialized_model():
    require(file_sha(EXTRACTOR) == EXTRACTOR_SHA, "Extractor checkpoint identity mismatch")
    require(file_sha(HEAD) == HEAD_SHA, "Temporal head checkpoint identity mismatch")
    frozen = torch.load(EXTRACTOR, map_location="cpu", weights_only=True)
    checkpoint = torch.load(HEAD, map_location="cpu", weights_only=True)
    require(frozen["schema_version"] == "frozen_extractor_v1", "Unexpected extractor schema")
    require(frozen["timm_version"] == timm.__version__, "timm version changed")
    require(checkpoint["config"]["cache_run_id"] == CACHE_RUN_ID, "Head/cache identity mismatch")
    require(checkpoint["config"]["seed"] == 20261024, "Expected the predeclared fixed seed")
    require(checkpoint["config"]["experiment"] == "frozen_mixnet_i3d_window_transformer_v1",
            "Expected ordinary-sampling head, not balanced head")
    for name, digest in frozen["source_sha256"].items():
        require(file_sha(ROOT / "development/mixnet_i3d" / name) == digest,
                f"Extractor source changed: {name}")
    for name, digest in checkpoint["config"]["source_sha256"].items():
        require(file_sha(ROOT / name) == digest, f"Head training source changed: {name}")
    old = timm.create_model(frozen["model_name"], pretrained=False)
    backbone = MixNetI3DBackbone(old)
    require(set(frozen["state_dict"]) == {"rgb_mean", "rgb_std"} | {
        "backbone." + key for key in backbone.state_dict()}, "Unexpected extractor state keys")
    backbone.load_state_dict({key[len("backbone."):]: value
                              for key, value in frozen["state_dict"].items()
                              if key.startswith("backbone.")}, strict=True)
    head = WindowTemporalTransformer()
    head.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = PartialFineTuneModel(backbone, head, mean=frozen["mean"], std=frozen["std"])
    for name in ("rgb_mean", "rgb_std"):
        torch.testing.assert_close(getattr(model, name), frozen["state_dict"][name], rtol=0, atol=0)
    return model, {
        "extractor_path": str(EXTRACTOR), "extractor_sha256": EXTRACTOR_SHA,
        "head_path": str(HEAD), "head_sha256": HEAD_SHA,
        "head_seed": 20261024, "head_best_epoch": checkpoint["epoch"],
        "head_selection": "fixed seed 20261024, its minimum-validation-loss checkpoint",
        "cache_run_id": CACHE_RUN_ID, "pretrained_downloads": False,
        "mean": list(frozen["mean"]), "std": list(frozen["std"]),
    }
