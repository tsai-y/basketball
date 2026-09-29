"""Modified C3D-style network inspired by Tran et al., ICCV 2015.

Reference: https://openaccess.thecvf.com/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html
This uses narrower channels and a small global-pooling binary head; it is not exact C3D.
"""

import torch.nn as nn


class C3DStyle(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        layers = []
        widths = ((3, 32), (32, 64), (64, 128), (128, 128),
                  (128, 256), (256, 256), (256, 256), (256, 256))
        pool_after = {0: (1, 2, 2), 1: (2, 2, 2), 3: (2, 2, 2),
                      5: (2, 2, 2), 7: (2, 2, 2)}
        for index, (input_channels, output_channels) in enumerate(widths):
            layers.extend([nn.Conv3d(input_channels, output_channels, 3, padding=1),
                           nn.BatchNorm3d(output_channels), nn.ReLU(inplace=True)])
            if index in pool_after:
                layers.append(nn.MaxPool3d(pool_after[index], stride=pool_after[index]))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, video):
        return self.head(self.pool(self.features(video)))
