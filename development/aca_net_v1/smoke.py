"""Synthetic CPU/module probes and meta-device full-shape/parameter audit.

No videos, training optimizer, model downloads, checkpoint or report writes.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import torch

from .model import VideoResNet50
from .modules import ACAModule, ReconstructionConfig


def run():
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    config = ReconstructionConfig()
    rows = []
    # Real 50x120x160 shape without allocating full model activations/weights.
    for variant in ("baseline", "lsta", "tsci", "aca"):
        with torch.device("meta"):
            model = VideoResNet50(variant=variant, config=config)
            output = model(torch.empty(1, 3, 50, 120, 160))
        count = sum(p.numel() for p in model.parameters())
        rows.append({"variant": variant, "parameters_8_classes": count,
                     "output_shape": list(output.shape), "execution": "meta_shape_only",
                     "difference_from_paper_26_8M": count - 26_800_000})

    # Actual numerical gradients through all three reconstructed ACA branches.
    small = replace(config, frames=4, reduction=2)
    module = ACAModule(4, small).train()
    x = torch.randn(2, 4, 4, 5, 7, requires_grad=True)
    out = module(x)
    out.square().mean().backward()
    gradients = [p.grad for p in module.parameters() if p.requires_grad]
    if not all(g is not None and torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Invalid ACA module gradient")

    # Numerical full ResNet50 baseline forward/backward at a bounded small shape.
    baseline = VideoResNet50(config=replace(config, frames=2), num_classes=2).eval()
    logits = baseline(torch.randn(1, 3, 2, 32, 40))
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in baseline.parameters()):
        raise RuntimeError("Invalid ResNet50 baseline gradient")
    root = Path(__file__).parent
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("modules.py", "model.py", "smoke.py")}
    return {"status": "engineering_probes_passed", "seed": 20261025,
            "config": asdict(config), "parameter_audit": rows,
            "aca_numeric_shape": list(out.shape), "aca_gradients_finite": True,
            "baseline_logits_shape": list(logits.shape), "baseline_backward_finite": True,
            "full_aca_numeric_forward_tested": False,
            "paper_parameter_count_reproduced": False,
            "source_sha256": hashes, "videos_opened": 0,
            "optimizer_steps": 0, "pretrained_weights_loaded": False,
            "files_written": False, "formal_training_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
