"""PatchConv encoder: CNN backbone for single raster encoding.

Baseline encoder that processes a single raster image through a
ResNet18 backbone, producing a spatial feature map and a globally
pooled hidden vector.

Use case: static rasters (DEM, land cover, flood zones) or
single-frame snapshots of dynamic data.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from georsct.encoders.spatial_base import SpatialEncoder, SpatialEncoderOutput


class PatchConvEncoder(SpatialEncoder):
    """CNN encoder for single raster images.

    Architecture:
        input (batch, C, H, W)
        -> channel adapter (C -> 3 if needed, for pretrained backbone)
        -> ResNet18 (truncated after layer4)
        -> feature_map (batch, 512, h/32, w/32)
        -> global average pool
        -> linear projection -> hidden (batch, hidden_dim)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        pretrained: bool = False,
    ):
        super().__init__(in_channels, hidden_dim)

        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.channel_adapter = nn.Identity()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self._backbone_dim = 512

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(self._backbone_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> SpatialEncoderOutput:
        x = self.channel_adapter(x)
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        hidden = self.project(pooled)
        return SpatialEncoderOutput(hidden=hidden, feature_map=feat)
