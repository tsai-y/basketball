"""Exact parameter decomposition and shape audit; no videos, downloads or GPU."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from .model import VideoResNet50
from .modules import ReconstructionConfig


def category(name):
    if ".lsta.gate_" in name or ".aca.gate_" in name:
        return "lsta_gated_3d_convolutions"
    if ".lsta.short_term." in name or ".aca.short_term." in name:
        return "lsta_short_term"
    if ".lsta.kernel_generator." in name or ".aca.kernel_generator." in name:
        return "lsta_dynamic_kernel_generator"
    if ".aca.tsci." in name or ".aca.channel." in name or ".aca.height." in name or ".aca.width." in name:
        return "tsci"
    return "classifier" if name.startswith("fc.") else "resnet50_backbone"


def parameter_breakdown(model):
    groups, stages = defaultdict(int), defaultdict(int)
    for name, p in model.named_parameters():
        key = category(name)
        groups[key] += p.numel()
        if key == "lsta_gated_3d_convolutions":
            stages[name.split(".")[0]] += p.numel()
    total = sum(p.numel() for p in model.parameters())
    assert sum(groups.values()) == total
    return {"total": total, "categories": dict(groups), "gated_conv_by_stage": dict(stages)}


def run():
    cfg = ReconstructionConfig()
    counts = {}
    for variant in ("baseline", "lsta", "tsci", "aca"):
        with torch.device("meta"):
            model = VideoResNet50(variant=variant, num_classes=8, config=cfg)
        counts[variant] = parameter_breakdown(model)
    shapes, aca_shapes = {}, {}
    with torch.device("meta"):
        model = VideoResNet50(variant="aca", num_classes=2, config=cfg)
        for name in ("conv1", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            getattr(model, name).register_forward_hook(
                lambda m, inputs, output, key=name: shapes.__setitem__(key, list(output.shape)))
        for name, module in model.named_modules():
            if name.endswith(".aca"):
                module.register_forward_hook(
                    lambda m, inputs, output, key=name: aca_shapes.__setitem__(key, list(output.shape)))
        logits = model(torch.empty(1, 3, 50, 120, 160))
        binary_count = sum(p.numel() for p in model.parameters())
        alternative = VideoResNet50(variant="aca", num_classes=8,
                                   config=replace(cfg, gate_groups="depthwise"))
        depthwise = parameter_breakdown(alternative)
    channels, blocks = (64, 128, 256, 512), (3, 4, 6, 3)
    gate_formula = sum(n * 55 * c * c for n, c in zip(blocks, channels))
    assert gate_formula == counts["aca"]["categories"]["lsta_gated_3d_convolutions"]
    return {
        "status": "parameter_and_shape_audit_complete", "config": asdict(cfg),
        "variants_8_classes": counts, "aca_2_class_parameters": binary_count,
        "shape_trace_N_T_folded_for_2d_stem": shapes, "aca_module_output_shapes": aca_shapes,
        "binary_logits_shape": list(logits.shape),
        "gated_conv_formula": "sum(blocks_in_stage * (1+27+27) * C**2)",
        "gated_conv_formula_total": gate_formula,
        "paper_reported_approximate_total": 26_800_000,
        "dense_difference_from_paper": counts["aca"]["total"] - 26_800_000,
        "depthwise_hypothesis_only": depthwise,
        "depthwise_is_verified_author_architecture": False,
        "full_aca_numeric_inference": False, "videos_opened": 0, "optimizer_steps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing targets rejected")
    args = parser.parse_args()
    result = run()
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
