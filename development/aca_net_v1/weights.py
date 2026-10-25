"""Pinned local ResNet50 V1 checkpoint installation and initialization.

No implicit downloads. --install-from only copies a verified official payload
to this module's ignored weights folder; an existing valid file is reused.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch

from .data import file_sha256, require, MEAN, STD
from .model import VideoResNet50

FOLDER = Path(__file__).parent
PIN_FILE = FOLDER / "imagenet_v1.json"


def pin():
    value = json.loads(PIN_FILE.read_text(encoding="utf-8"))
    require(value["sha256"].startswith("0676ba61") and value["filename"] == "resnet50-0676ba61.pth",
            "Initialization pin mismatch")
    require(tuple(value["rgb_mean"]) == MEAN and tuple(value["rgb_std"]) == STD,
            "Data normalization differs from pinned weights")
    require(value["local_relative_path"] == "artifacts/weights/resnet50-0676ba61.pth",
            "Unexpected checkpoint destination")
    return value


def verify_payload(path, identity):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Expected a regular checkpoint file")
    require(path.stat().st_size == identity["size_bytes"], "Checkpoint byte count mismatch")
    require(file_sha256(path) == identity["sha256"], "Checkpoint SHA256 mismatch")


def install_from(source):
    identity = pin()
    verify_payload(source, identity)
    target = FOLDER / identity["local_relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        verify_payload(target, identity)
    else:
        with Path(source).open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        verify_payload(target, identity)
    return {"status": "verified_local_checkpoint", "path": str(target),
            "sha256": identity["sha256"], "size_bytes": identity["size_bytes"]}


def initialized_baseline(seed):
    identity = pin()
    path = FOLDER / identity["local_relative_path"]
    verify_payload(path, identity)
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(len(state) == identity["source_tensor_count"], "Unexpected source tensor inventory")
    torch.manual_seed(seed)
    model = VideoResNet50(variant="baseline", num_classes=2)
    transfer = model.load_imagenet_state(state)
    require(len(transfer["legacy_bn_counters_initialized_to_zero"]) == identity["missing_bn_counter_count"],
            "Unexpected legacy BatchNorm inventory")
    return model, {"checkpoint": str(path), "identity": identity,
                   "pin_file_sha256": file_sha256(PIN_FILE), "transfer": transfer}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-from", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install_from(args.install_from), indent=2))


if __name__ == "__main__":
    main()
