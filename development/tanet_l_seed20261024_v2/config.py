"""Pinned preparation configuration, not a formal-training prescription."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
COMMIT = "5fcb46fe671ead2f7958d8ffc3321a0e274d6491"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class InputConfig:
    frames: int = 50
    height: int = 192
    width: int = 256

    def __post_init__(self):
        for name in ("frames", "height", "width"):
            need(type(getattr(self, name)) is int and getattr(self, name) >= 2,
                 f"{name} must be an integer >= 2")


def load_config():
    cfg = json.loads((FOLDER / "config.json").read_text())
    need(cfg["schema"] == "tanet_binary_preparation_v1", "Unknown preparation schema")
    need((cfg["frames"], cfg["height"], cfg["width"]) == (50, 192, 256), "Input contract changed")
    need(cfg["classes"] == ["Miss", "Make"] and cfg["sampling"] == "full_video_uniform_endpoint_floor",
         "Task/sampling contract changed")
    need(type(cfg["seed"]) is int and 0 <= cfg["seed"] < 2**32, "Invalid seed")
    need(cfg["dropout"] == 0.5, "Dropout contract changed")
    need(cfg["rgb_mean"] == [0.485, 0.456, 0.406] and cfg["rgb_std"] == [0.229, 0.224, 0.225],
         "Normalization contract changed")
    need(cfg["imagenet_sha256"] == "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
         and cfg["imagenet_size_bytes"] == 102530333, "ImageNet identity changed")
    need(cfg["formal_training_authorized"] is False and cfg["test_video_access"] is False,
         "Preparation cannot grant training/test authority")
    return cfg


def verify_upstream():
    root = FOLDER / "upstream"
    pin = json.loads((root / "PROVENANCE.json").read_text())
    need(pin["commit"] == COMMIT and pin["license"] == "Apache-2.0", "Upstream identity changed")
    for item in pin["files"].values():
        path = root / item["local_file"]
        need(path.is_file() and not path.is_symlink(), "Missing/symlinked upstream file")
        need(path.stat().st_size == item["size_bytes"] and sha256(path) == item["sha256"],
             "Upstream file changed; do not silently patch author code")
    return pin
