"""ACA-DW3D: only the two gating 3x3x3 convolutions become depthwise.

The original dense implementation remains untouched. Its backbone, pointwise
gate, short branch, dynamic kernels and TSCI keep their implementation/state keys.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from development.aca_net_v1.data import require
from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.modules import ReconstructionConfig


@dataclass(frozen=True)
class DW3DConfig(ReconstructionConfig):
    gate_groups: str = "depthwise_3x3"

    def __post_init__(self):
        require(self.gate_groups == "depthwise_3x3", "DW3D config cannot silently select dense or all-depthwise")
        # Validate the unchanged scalar settings through the original contract.
        ReconstructionConfig(**{**asdict(self), "gate_groups": "dense"})


def convert_lsta_gating(lsta):
    """Replace only Conv3D objects; preserve each original IN/ReLU block."""
    require(not getattr(lsta, "_dw3d_converted", False), "Gating already converted")
    value = lsta.gate_value[0]
    require(value.kernel_size == (1, 1, 1) and value.groups == 1, "Value branch must remain dense 1x1x1")
    for branch in (lsta.gate_control, lsta.gate_output):
        old = branch[0]
        require(old.kernel_size == (3, 3, 3) and old.groups == 1 and old.in_channels == old.out_channels,
                "Expected the original C-to-C dense 3x3x3 branch")
        branch[0] = nn.Conv3d(old.in_channels, old.out_channels, old.kernel_size,
                             stride=old.stride, padding=old.padding, dilation=old.dilation,
                             groups=old.in_channels, bias=old.bias is not None,
                             padding_mode=old.padding_mode, device=old.weight.device, dtype=old.weight.dtype)
    lsta._dw3d_converted = True


class DW3DVideoResNet50(VideoResNet50):
    def __init__(self, *, variant="aca", num_classes=2, config=None):
        require(variant == "aca", "DW3D factory only constructs the full ACA candidate")
        config = config or DW3DConfig()
        require(isinstance(config, DW3DConfig), "An explicit DW3D architecture config is required")
        # The original LSTA treats the new label as groups=1 at construction;
        # immediately replace just its two spatial-temporal kernels below.
        super().__init__(variant="aca", num_classes=num_classes, config=config)
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in stage:
                convert_lsta_gating(block.aca.lsta)
        self.variant = "aca_dw3d_candidate"


def gating_audit(model):
    rows = []
    for name, module in model.named_modules():
        if not name.endswith(".aca.lsta"):
            continue
        value, control, output = module.gate_value[0], module.gate_control[0], module.gate_output[0]
        channels = value.in_channels
        require(value.groups == 1 and value.kernel_size == (1, 1, 1) and
                control.groups == output.groups == channels and
                control.kernel_size == output.kernel_size == (3, 3, 3), "Actual gate groups differ from declared DW3D")
        rows.append({"module": name, "channels": channels, "value_groups": value.groups,
                     "control_groups": control.groups, "output_groups": output.groups,
                     "value_weight_shape": list(value.weight.shape),
                     "control_weight_shape": list(control.weight.shape),
                     "output_weight_shape": list(output.weight.shape)})
    require(len(rows) == 16, "Expected 16 DW3D ACA modules")
    return {"gate_mode": "dense_1x1x1_plus_two_depthwise_3x3x3", "ACA_modules": 16,
            "dense_1x1x1_convolutions": 16, "depthwise_3x3x3_convolutions": 32, "layers": rows}
