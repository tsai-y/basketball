"""Parameter/shape and synthetic operator audit; no videos, weights or CUDA.

The proposal mirror below is for auditing the user's pasted implementation,
not an author implementation or a new trainable experiment configuration.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.modules import (
    ACAModule, ReconstructionConfig, TSCI, dynamic_temporal_conv,
)
from development.aca_dw3d_v1.model import DW3DVideoResNet50


TAM_SHA256 = "b437fe46eb69198c824e1f2ecba286e7a8bf0bc5fc46e47462d68b94b7c7fb24"
CHANNELS = (64, 128, 256, 512)
BLOCKS = (3, 4, 6, 3)


class ProposalLSTA(nn.Module):
    """Audit mirror of TAM_to_LSTA defaults supplied by the user.

    Preserves dense 1x1 gate, depthwise 3x3 gates, r4, beta2, short k3/k1,
    bias-free temporal layers, max(C//r,16), and channel-wise long BN.
    """
    def __init__(self, channels, num_frames=50, r=4, beta=2, K=3,
                 depthwise_gate=True, short_k2=1, long_bn=True):
        super().__init__()
        self.num_frames, self.K = num_frames, K
        groups = channels if depthwise_gate else 1

        def gate(kernel, group):
            return nn.Sequential(
                nn.Conv3d(channels, channels, kernel, padding=kernel // 2,
                          groups=group, bias=False),
                nn.InstanceNorm3d(channels), nn.ReLU(inplace=True),
            )

        self.gate_branch_3x3 = nn.Sequential(*gate(3, groups), nn.Sigmoid())
        self.gate_branch_1x1 = gate(1, 1)
        self.gate_proj = gate(3, groups)
        mid_c = max(channels // r, 16)
        self.short_branch = nn.Sequential(
            nn.Conv1d(channels, mid_c, 3, padding=1, bias=False),
            nn.BatchNorm1d(mid_c), nn.ReLU(inplace=True),
            nn.Conv1d(mid_c, channels, short_k2, padding=short_k2 // 2,
                      bias=False), nn.Sigmoid(),
        )
        hidden_t = int(beta * num_frames)
        self.long_fc1 = nn.Linear(num_frames, hidden_t, bias=False)
        self.long_bn = nn.BatchNorm1d(channels) if long_bn else nn.Identity()
        self.long_relu = nn.ReLU(inplace=True)
        self.long_fc2 = nn.Linear(hidden_t, K, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n, c, t, h, w = x.shape
        g = x + self.gate_proj(self.gate_branch_3x3(x) * self.gate_branch_1x1(x))
        p = g.mean(dim=(-2, -1))
        z = g * self.short_branch(p)[..., None, None]
        theta = self.softmax(self.long_fc2(self.long_relu(self.long_bn(self.long_fc1(p)))))
        return grouped_temporal_conv(z, theta)


def grouped_temporal_conv(x, kernels):
    n, c, t, h, w = x.shape
    k = kernels.shape[-1]
    out = F.conv2d(x.reshape(1, n * c, t, h * w),
                   kernels.reshape(n * c, 1, k, 1),
                   padding=(k // 2, 0), groups=n * c)
    return out.reshape(n, c, t, h, w)


def build(variant, classes):
    if variant == "dw3d_existing":
        return DW3DVideoResNet50(num_classes=classes)
    if variant == "all_gates_depthwise_diagnostic":
        return VideoResNet50(variant="aca", num_classes=classes,
                             config=replace(ReconstructionConfig(), gate_groups="depthwise"))
    if variant == "proposal_default_audit_only":
        model = VideoResNet50(variant="baseline", num_classes=classes)
        for name in ("layer1", "layer2", "layer3", "layer4"):
            for block in getattr(model, name):
                module = ACAModule(block.conv1.out_channels, ReconstructionConfig())
                module.lsta = ProposalLSTA(block.conv1.out_channels)
                block.aca = module
        return model
    return VideoResNet50(variant=variant, num_classes=classes)


def breakdown(model):
    groups = dict(backbone=0, classifier=0, gating=0, short_term=0,
                  dynamic_kernel=0, tsci=0)
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            key = "classifier"
        elif ".aca." not in name:
            key = "backbone"
        elif "gate_" in name:
            key = "gating"
        elif "short_" in name:
            key = "short_term"
        elif "kernel_generator" in name or ".long_" in name:
            key = "dynamic_kernel"
        else:
            key = "tsci"
        groups[key] += param.numel()
    assert sum(groups.values()) == sum(p.numel() for p in model.parameters())
    return groups


def parameter_audit():
    variants = ("baseline", "lsta", "tsci", "aca", "dw3d_existing",
                "all_gates_depthwise_diagnostic", "proposal_default_audit_only")
    rows = []
    for variant in variants:
        for classes in (2, 8):
            with torch.device("meta"):
                model = build(variant, classes)
                shape = list(model(torch.empty(1, 3, 50, 120, 160)).shape)
            parts = breakdown(model)
            rows.append(dict(variant=variant, classes=classes,
                             total=sum(parts.values()), components=parts,
                             meta_logits_shape=shape,
                             difference_vs_paper_26_8M=sum(parts.values()) - 26_800_000))
    c2 = sum(n * c * c for n, c in zip(BLOCKS, CHANNELS))
    c1 = sum(n * c for n, c in zip(BLOCKS, CHANNELS))
    return {"rows": rows, "gate_formulas": {
        "sum_blocks_C_squared": c2, "sum_blocks_C": c1,
        "dense_55_C_squared": 55 * c2,
        "mixed_C_squared_plus_54_C": c2 + 54 * c1,
        "all_depthwise_55_C": 55 * c1,
    }, "class_head_8_minus_2": 6 * 2049,
        "paper_count_is_approximate_not_an_identifying_constraint": True}


def proposal_signature(path):
    """Parse definitions only: never execute attached source text."""
    raw = path.read_bytes()
    text = raw.decode()
    start = text.index("class TAM_to_LSTA(")
    code = text[start:]
    if "\n在完成" in code:
        code = code.split("\n在完成", 1)[0]
    tree = ast.parse(code)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    names = [n.arg for n in init.args.args][-len(init.args.defaults):]
    defaults = dict(zip(names, map(ast.literal_eval, init.args.defaults)))
    expected = dict(r=4, beta=2, K=3, depthwise_gate=True, short_k2=1, long_bn=True)
    if defaults != expected:
        raise ValueError("Attachment defaults changed; audit mirror must be reviewed")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "class": cls.name,
            "defaults": defaults, "attachment_executed": False,
            "mirror_scope": "manually reviewed architecture; defaults verified with AST"}


def synthetic_operator_audit(tam_path):
    if hashlib.sha256(tam_path.read_bytes()).hexdigest() != TAM_SHA256:
        raise ValueError("TAM does not match the pinned author source")
    spec = importlib.util.spec_from_file_location("pinned_tam_for_architecture_audit", tam_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(1937)
        x = torch.randn(2, 4, 5, 3, 2, dtype=torch.float64)
        kernels = torch.randn(2, 4, 3, dtype=torch.float64).softmax(-1)
        a = grouped_temporal_conv(x, kernels)
        b = dynamic_temporal_conv(x, kernels)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        with contextlib.redirect_stdout(io.StringIO()):
            tam = module.TAM(4, 5).double().train()
        proposal = ProposalLSTA(4, num_frames=5).double().train()
        proposal.long_fc1.weight.copy_(tam.G[0].weight)
        proposal.long_fc2.weight.copy_(tam.G[3].weight)
        p = torch.randn(2, 4, 5, dtype=torch.float64)
        official = tam.G(p.reshape(-1, 5)).reshape(2, 4, 3)
        other = proposal.softmax(proposal.long_fc2(
            proposal.long_relu(proposal.long_bn(proposal.long_fc1(p)))))
        gap = float((official - other).abs().max())
        assert gap > 1e-6
    return {
        "grouped_conv2d_vs_existing_einsum_max_abs": float((a - b).abs().max()),
        "synthetic_shape": [2, 4, 5, 3, 2],
        "identical_fc_weights_but_different_BN_kernel_max_abs": gap,
        "official_G_BN_num_features": tam.G[1].num_features,
        "proposal_G_BN_num_features": proposal.long_bn.num_features,
        "official_BN_reduces_over": "N*C for each hidden temporal feature",
        "proposal_BN_reduces_over": "N*hidden_T for each C",
        "BN_outputs_are_not_equivalent": True,
        "gate_sigmoid_after_ReLU_range": [0.5, 1.0],
        "gate_upper_bound_is_limit": True,
        "real_model_forward": False,
        "synthetic_toy_forward_only": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--tam", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "schema": "aca_architecture_audit_v1", "status": "completed_with_unresolved_author_settings",
        "exact_author_reproduction_confirmed": False,
        "proposal": proposal_signature(args.proposal),
        "parameters": parameter_audit(),
        "operators": synthetic_operator_audit(args.tam),
        "scope": {"device": "meta_and_cpu", "pretrained_weights_loaded": 0,
                  "train_videos_opened": 0, "validation_videos_opened": 0,
                  "test_videos_opened": 0, "optimizer_updates": 0,
                  "new_classification_metrics": False,
                  "hyperparameter_search": False},
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "parameter_rows": len(result["parameters"]["rows"]),
                      "scope": result["scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
