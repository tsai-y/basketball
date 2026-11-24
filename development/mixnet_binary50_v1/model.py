"""Single-clip 50-frame direct binary wrapper around MixNetI3DBackbone.

Contract:
- Input: already ImageNet-normalized FP32 tensor [B, 3, 50, 120, 160] (C, T, H, W).
- No internal /255 scaling, no internal mean/std normalization, no three-window
  expansion, no resizing/cropping, no LSTA/TSCI insertion.
- Reuses development.mixnet_i3d.backbone.MixNetI3DBackbone without modifying it.
- Global average pooling explicitly over (T, H, W) = dim=(2, 3, 4).
- Linear(backbone.num_features, 2) outputs raw logits for CrossEntropyLoss with
  torch.long targets: Miss=0, Make=1.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import timm
import torch
from torch import nn

from development.mixnet_i3d.backbone import MixNetI3DBackbone

MODEL_NAME = "mixnet_s.ft_in1k"
CLASS_ORDER = ("Miss", "Make")
EXPECTED_CHANNELS = 3
EXPECTED_FRAMES = 50
EXPECTED_HEIGHT = 120
EXPECTED_WIDTH = 160
EXPECTED_INPUT_SHAPE = (
    EXPECTED_CHANNELS,
    EXPECTED_FRAMES,
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
)
UPSTREAM_NORMALIZATION = {
    "scale": 255.0,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "applied_upstream_once": True,
    "applied_in_model": False,
}


class MixNetBinary50Model(nn.Module):
    """Single-clip 5D MixNet-I3D binary classifier outputting 2 raw logits.

    Unlike development.mixnet_i3d.video_classifier.ThreeWindowMakeMissModel:
    - Accepts 5D [B, 3, 50, 120, 160] (never 6D three-window clips).
    - Expects input already ImageNet-normalized once upstream (no mean/std buffers).
    - Outputs [B, 2] raw logits (Miss=0, Make=1) for unweighted CrossEntropyLoss.
    - Reads backbone.num_features dynamically (never hardcodes ACA's 2048).
    """

    def __init__(
        self,
        backbone: MixNetI3DBackbone,
        *,
        strict_clip_shape: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, MixNetI3DBackbone):
            raise TypeError(
                f"Expected MixNetI3DBackbone, got {type(backbone).__name__}"
            )
        num_features = getattr(backbone, "num_features", None)
        if not isinstance(num_features, int) or num_features <= 0:
            raise ValueError(f"Invalid backbone.num_features: {num_features!r}")

        self.backbone = backbone
        self.num_features = num_features
        self.num_classes = 2
        self.class_order = CLASS_ORDER
        self.strict_clip_shape = bool(strict_clip_shape)
        self.classifier = nn.Linear(self.num_features, self.num_classes)

    def _validate_input(self, clip: torch.Tensor) -> None:
        if not isinstance(clip, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(clip).__name__}")
        if clip.ndim != 5:
            raise ValueError(
                "Expected 5D single-clip tensor [B, 3, T, H, W]; "
                f"got ndim={clip.ndim} shape={tuple(clip.shape)}"
            )
        if clip.dtype != torch.float32:
            raise ValueError(
                f"Expected already-normalized float32 input, got dtype={clip.dtype}"
            )
        batch, channels, frames, height, width = clip.shape
        if batch < 1:
            raise ValueError(f"Expected batch size >= 1, got {batch}")
        if channels != EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected RGB channels=3 at dim=1, got shape={tuple(clip.shape)}"
            )
        if self.strict_clip_shape and (
            channels,
            frames,
            height,
            width,
        ) != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"Expected exact single-clip shape [B, 3, 50, 120, 160], "
                f"got {tuple(clip.shape)}"
            )
        if frames < 1 or height < 1 or width < 1:
            raise ValueError(f"Invalid spatial-temporal dims: {tuple(clip.shape)}")
        if not torch.isfinite(clip).all().item():
            raise ValueError("Input tensor contains non-finite values")

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        self._validate_input(clip)
        features_3d = self.backbone(clip)
        if features_3d.ndim != 5 or features_3d.shape[1] != self.num_features:
            raise RuntimeError(
                f"Unexpected backbone output shape: {tuple(features_3d.shape)}"
            )
        pooled = features_3d.mean(dim=(2, 3, 4))
        logits = self.classifier(pooled)
        if logits.ndim != 2 or logits.shape != (clip.shape[0], self.num_classes):
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        return logits


def build_mixnet_binary50(
    *,
    pretrained: bool = False,
    seed: int | None = None,
    strict_clip_shape: bool = True,
    eval_mode: bool = True,
) -> MixNetBinary50Model:
    """Build CPU-verifiable MixNetBinary50Model reusing MixNetI3DBackbone.

    Formal pretrained loading is intentionally blocked in this CPU-acceptance
    version until provenance, hash, and normalization comparability are approved.
    """
    if pretrained:
        raise ValueError(
            "pretrained=True is disabled in mixnet_binary50_v1 CPU acceptance; "
            "no weight download or unapproved pretrained loading is permitted."
        )
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError(f"seed must be an int or None, got {seed!r}")

    def _construct() -> MixNetBinary50Model:
        model_2d = timm.create_model(MODEL_NAME, pretrained=False)
        backbone = MixNetI3DBackbone(model_2d)
        model = MixNetBinary50Model(
            backbone,
            strict_clip_shape=strict_clip_shape,
        )
        if eval_mode:
            model.eval()
        else:
            model.train()
        return model

    if seed is None:
        return _construct()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _construct()


def trace_stage_shapes(
    model: MixNetBinary50Model,
    clip: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run one forward pass with read-only hooks to verify passthrough & shapes."""
    captured: dict[str, Any] = {}
    stage_shapes: OrderedDict[str, list[int]] = OrderedDict()
    stage_shapes["input"] = list(clip.shape)
    handles = []

    def _backbone_pre_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        received = inputs[0]
        captured["backbone_input_same_object"] = received is clip
        captured["backbone_input_bitwise_equal"] = bool(torch.equal(received, clip))
        captured["backbone_input_max_abs_diff"] = float(
            (received - clip).abs().max().item()
        )
        captured["backbone_input_shape"] = list(received.shape)
        captured["backbone_input_dtype"] = str(received.dtype)

    def _record_output(name: str):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            stage_shapes[name] = list(output.shape)
        return _hook

    def _classifier_pre_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        stage_shapes["global_avg_pool_thw"] = list(inputs[0].shape)

    handles.append(model.backbone.register_forward_pre_hook(_backbone_pre_hook))
    handles.append(model.backbone.bn1.register_forward_hook(_record_output("stem_bn1")))
    for idx, stage in enumerate(model.backbone.blocks):
        handles.append(stage.register_forward_hook(_record_output(f"stage_{idx}")))
    handles.append(model.backbone.bn2.register_forward_hook(_record_output("head_bn2")))
    handles.append(model.classifier.register_forward_pre_hook(_classifier_pre_hook))
    handles.append(model.classifier.register_forward_hook(_record_output("logits")))

    try:
        logits = model(clip)
    finally:
        for handle in handles:
            handle.remove()

    captured["stage_shapes"] = dict(stage_shapes)
    captured["temporal_progression"] = {
        name: shape[2]
        for name, shape in stage_shapes.items()
        if len(shape) == 5
    }
    return logits, captured


def summarize_bn3d(model: nn.Module) -> dict[str, Any]:
    """Summarize all BatchNorm3d layers without mutating module state."""
    bn_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm3d)
    ]
    return {
        "count": len(bn_layers),
        "training_modes": sorted({bool(m.training) for _, m in bn_layers}),
        "affine_values": sorted({bool(m.affine) for _, m in bn_layers}),
        "track_running_stats_values": sorted(
            {bool(m.track_running_stats) for _, m in bn_layers}
        ),
        "momentum_values": sorted({m.momentum for _, m in bn_layers}),
        "eps_values": sorted({m.eps for _, m in bn_layers}),
        "layer_names": [name for name, _ in bn_layers],
        "formal_train_bn_policy_approved": False,
        "engineering_note": (
            "All 49 BatchNorm3d layers inherit timm mixnet_s.ft_in1k defaults "
            "(momentum=0.1, eps=1e-5, affine=True, track_running_stats=True). "
            "CPU eval verification does not approve a formal training BN policy."
        ),
    }


def snapshot_bn3d_buffers(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Clone running_mean, running_var, num_batches_tracked for all BatchNorm3d."""
    snap: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm3d):
            entry: dict[str, torch.Tensor] = {}
            for buf_name in ("running_mean", "running_var", "num_batches_tracked"):
                buf = getattr(module, buf_name, None)
                if buf is not None:
                    entry[buf_name] = buf.detach().clone()
            snap[name] = entry
    return snap


def compare_bn3d_snapshots(
    before: dict[str, dict[str, torch.Tensor]],
    after: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Compare two BN3d buffer snapshots for bitwise identity."""
    if set(before.keys()) != set(after.keys()):
        raise ValueError("BN3d snapshot layer sets do not match")
    changed: list[str] = []
    max_float_diff = 0.0
    for layer_name in sorted(before.keys()):
        if set(before[layer_name].keys()) != set(after[layer_name].keys()):
            changed.append(layer_name)
            continue
        for buf_name, b_tensor in before[layer_name].items():
            a_tensor = after[layer_name][buf_name]
            if not torch.equal(b_tensor, a_tensor):
                changed.append(f"{layer_name}.{buf_name}")
                if b_tensor.is_floating_point():
                    diff = float((a_tensor - b_tensor).abs().max().item())
                    max_float_diff = max(max_float_diff, diff)
    return {
        "bn3d_layer_count": len(before),
        "all_buffers_bitwise_unchanged": len(changed) == 0,
        "changed_buffers": changed,
        "max_float_abs_diff": max_float_diff,
    }


def parameter_summary(model: MixNetBinary50Model) -> dict[str, Any]:
    """Count total and trainable parameters explicitly with mode provenance."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_total = sum(p.numel() for p in model.backbone.parameters())
    backbone_trainable = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )
    head_total = sum(p.numel() for p in model.classifier.parameters())
    head_trainable = sum(
        p.numel() for p in model.classifier.parameters() if p.requires_grad
    )
    return {
        "model_training_mode": bool(model.training),
        "total_params": total,
        "trainable_params": trainable,
        "backbone_total_params": backbone_total,
        "backbone_trainable_params": backbone_trainable,
        "classifier_total_params": head_total,
        "classifier_trainable_params": head_trainable,
        "classifier_weight_shape": list(model.classifier.weight.shape),
        "classifier_bias_shape": list(model.classifier.bias.shape),
        "measurement_note": (
            "requires_grad=True on all parameters during CPU acceptance "
            "(backbone unfrozen, no optimizer updates executed)."
        ),
    }
