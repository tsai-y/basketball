"""C3D topology from Tran et al., ICCV 2015, Fig. 3 and official Caffe prototxt.

Basketball-51 adaptations: two-output fc8, PyTorch ceil pooling convention,
and fc6 descriptor extraction after ReLU but before dropout. See
docs/c3d_reference_implementation.md. No pretrained weights are loaded.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceC3D(nn.Module):
    def __init__(self, num_classes=2, pool_ceil_mode=True):
        super().__init__()
        specs = (("conv1a", 3, 64), ("conv2a", 64, 128),
                 ("conv3a", 128, 256), ("conv3b", 256, 256),
                 ("conv4a", 256, 512), ("conv4b", 512, 512),
                 ("conv5a", 512, 512), ("conv5b", 512, 512))
        self.conv_names = tuple(name for name, _, _ in specs)
        for name, channels_in, channels_out in specs:
            setattr(self, name, nn.Conv3d(channels_in, channels_out, 3, stride=1, padding=1, bias=True))
        pool_specs = ((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))
        for i, kernel in enumerate(pool_specs, 1):
            setattr(self, f"pool{i}", nn.MaxPool3d(kernel, stride=kernel, ceil_mode=pool_ceil_mode))
        # For 3x16x112x112 with ceil pooling: pool5 = 512x1x4x4.
        self.fc6 = nn.Linear(512 * 1 * 4 * 4, 4096, bias=True)
        self.drop6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 4096, bias=True)
        self.drop7 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(4096, num_classes, bias=True)
        self._initialize_official_fillers()

    def _initialize_official_fillers(self):
        for name in self.conv_names:
            layer = getattr(self, name)
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.constant_(layer.bias, 0.0 if name == "conv1a" else 1.0)
        for layer in (self.fc6, self.fc7):
            nn.init.normal_(layer.weight, mean=0.0, std=0.005)
            nn.init.constant_(layer.bias, 1.0)
        nn.init.normal_(self.fc8.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc8.bias, 0.0)

    def feature_maps(self, x, shapes=None):
        for group in (("conv1a", "pool1"), ("conv2a", "pool2"),
                      ("conv3a", "conv3b", "pool3"),
                      ("conv4a", "conv4b", "pool4"),
                      ("conv5a", "conv5b", "pool5")):
            for name in group:
                layer = getattr(self, name)
                x = layer(x) if name.startswith("pool") else F.relu(layer(x))
                if shapes is not None:
                    shapes[name] = list(x.shape)
        if tuple(x.shape[1:]) != (512, 1, 4, 4):
            raise ValueError(f"Unexpected pool5 shape: {tuple(x.shape)}")
        return x

    def extract_fc6(self, x, shapes=None):
        """PROJECT_ADAPTATION: post-fc6 ReLU, before drop6; [batch,4096]."""
        x = self.feature_maps(x, shapes).flatten(1)
        x = F.relu(self.fc6(x))
        if shapes is not None:
            shapes["fc6"] = list(x.shape)
        return x

    def forward(self, x):
        fc6 = self.extract_fc6(x)
        fc7 = F.relu(self.fc7(self.drop6(fc6)))
        return self.fc8(self.drop7(fc7))
