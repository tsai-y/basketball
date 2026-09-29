#!/usr/bin/env python3
"""Compare official C3D Caffe pool3d shape formula with PyTorch ceil pooling."""

import argparse
from math import ceil
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference_c3d.model import ReferenceC3D

OFFICIAL_SOURCE = "https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/src/caffe/layers/pool3d_layer.cpp"
OFFICIAL_NETWORK = "https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/examples/c3d_feature_extraction/prototxt/c3d_sport1m_feature_extractor_video.prototxt"


def official_caffe_pool_shape(shape, kernel, stride):
    # pool3d_layer.cpp Reshape, lines 52-57: ceil((size - kernel)/stride)+1;
    # official prototxt uses no pool padding.
    return tuple(ceil((size - k) / s) + 1 for size, k, s in zip(shape, kernel, stride))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c3d_reference_benchmark.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output_dir"]
    if output.resolve() != (ROOT / "runs/full_video_c3d_reference_v1/feasibility").resolve():
        raise RuntimeError("Shape audit output must be the independent feasibility directory")
    official = (16, 112, 112)
    torch_shape = official
    record = []
    for i in range(1, 6):
        kernel = (1, 2, 2) if i == 1 else (2, 2, 2)
        official = official_caffe_pool_shape(official, kernel, kernel)
        # Tiny tensor only: verifies PyTorch's actual ceil-mode shape, independent of full network.
        with torch.no_grad():
            tensor = torch.zeros((1, 1, *torch_shape))
            torch_shape = tuple(torch.nn.functional.max_pool3d(
                tensor, kernel_size=kernel, stride=kernel, ceil_mode=True).shape[2:])
        record.append({"pool": i, "official_formula_t_h_w": official,
                       "pytorch_actual_t_h_w": torch_shape, "match": official == torch_shape})
    model = ReferenceC3D(num_classes=2, pool_ceil_mode=True)
    flatten = 512 * official[0] * official[1] * official[2]
    result = {"status": "PASS" if all(x["match"] for x in record)
              and flatten == model.fc6.in_features else "FAIL",
              "official_pool_source": OFFICIAL_SOURCE,
              "official_network_definition": OFFICIAL_NETWORK,
              "source_formula": "ceil((input + 2*pad - kernel)/stride) + 1; pool pad=0",
              "pools": record, "pool5_c_h_w": [512, *official],
              "flatten_size": flatten, "fc6_in_features": model.fc6.in_features,
              "fc6_out_features": model.fc6.out_features}
    output.mkdir(parents=True, exist_ok=True)
    (output / "shape_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("Caffe/PyTorch shape mismatch")


if __name__ == "__main__":
    main()
