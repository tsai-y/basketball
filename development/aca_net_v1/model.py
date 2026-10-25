"""Frame-wise ResNet50 with optional ACA modules after conv1/bn1/ReLU.

No downloads occur in this module. ImageNet initialization must be an explicitly
supplied local state dict; checkpoints from synthetic probes are not pretrained.
"""
import torch
from torch import nn
from torchvision.models import resnet50

from .modules import ACAModule, LSTA, TSCI, ReconstructionConfig


class VideoBottleneck(nn.Module):
    def __init__(self, block, module):
        super().__init__()
        for name in ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3",
                     "relu", "downsample"):
            setattr(self, name, getattr(block, name))
        self.stride = block.stride
        self.aca = module

    @staticmethod
    def framewise(module, x):
        n, c, t, h, w = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
        out = module(flat)
        return out.reshape(n, t, *out.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x):
        identity = x
        x = self.relu(self.framewise(self.bn1, self.framewise(self.conv1, x)))
        x = self.aca(x)
        x = self.relu(self.framewise(self.bn2, self.framewise(self.conv2, x)))
        x = self.framewise(self.bn3, self.framewise(self.conv3, x))
        if self.downsample is not None:
            identity = self.framewise(self.downsample, identity)
        return self.relu(x + identity)


class VideoResNet50(nn.Module):
    def __init__(self, *, variant="baseline", num_classes=8, config=None):
        super().__init__()
        if variant not in {"baseline", "lsta", "tsci", "aca"}:
            raise ValueError("Unknown variant")
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.config = config or ReconstructionConfig()
        self.variant = variant
        backbone = resnet50(weights=None)
        self.conv1, self.bn1, self.relu = backbone.conv1, backbone.bn1, backbone.relu
        self.maxpool = backbone.maxpool
        for stage_name in ("layer1", "layer2", "layer3", "layer4"):
            converted = []
            for block in getattr(backbone, stage_name):
                channels = block.conv1.out_channels
                if variant == "aca":
                    module = ACAModule(channels, self.config)
                elif variant == "lsta":
                    module = LSTA(channels, self.config)
                elif variant == "tsci":
                    module = TSCI()
                else:
                    module = nn.Identity()
                converted.append(VideoBottleneck(block, module))
            setattr(self, stage_name, nn.Sequential(*converted))
        self.fc = nn.Linear(2048, num_classes)

    def load_imagenet_state(self, state):
        """Require all ResNet50 weights/statistics; allow legacy absent BN counters.

        Exact key/shape checks precede any mutation. Never accept a partial
        backbone, unexpected module parameters, or an unreported random fallback.
        Missing num_batches_tracked buffers are explicitly zeroed and reported.
        """
        target = self.state_dict()
        keys = {k for k in target if ".aca." not in k and not k.startswith("fc.")}
        expected = keys | {"fc.weight", "fc.bias"}
        counters = {k for k in keys if k.endswith(".num_batches_tracked")}
        missing = expected - set(state)
        if not missing <= counters or set(state) - expected:
            raise ValueError("Expected a complete torchvision ResNet50 state dict")
        if tuple(state["fc.weight"].shape) != (1000, 2048) or tuple(state["fc.bias"].shape) != (1000,):
            raise ValueError("Expected the ImageNet 1000-class head")
        copied = keys & set(state)
        for key in copied:
            if state[key].shape != target[key].shape:
                raise ValueError(f"Shape mismatch: {key}")
            if state[key].dtype != target[key].dtype:
                raise ValueError(f"Dtype mismatch: {key}")
            if state[key].is_floating_point() and not torch.isfinite(state[key]).all():
                raise ValueError(f"Nonfinite source: {key}")
        # Keep new task/module tensors explicitly. A partial load with no state
        # metadata can cause BatchNorm to supply missing num_batches_tracked
        # buffers implicitly, making missing-key audits misleading.
        merged = dict(target)
        merged.update({k: state[k] for k in copied})
        merged.update({k: torch.zeros_like(target[k]) for k in missing})
        self.load_state_dict(merged, strict=True)
        return {"backbone_tensors_loaded": len(copied),
                "legacy_bn_counters_initialized_to_zero": sorted(missing),
                "task_head_reinitialized": True,
                "aca_parameters_reinitialized": self.variant != "baseline"}

    def forward_features(self, x):
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[2] != self.config.frames:
            raise ValueError("Expected [N,3,configured_T,H,W]")
        fw = VideoBottleneck.framewise
        x = fw(self.maxpool, self.relu(fw(self.bn1, fw(self.conv1, x))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(self, name)(x)
        return x

    def forward(self, x):
        return self.fc(self.forward_features(x).mean(dim=(2, 3, 4)))
