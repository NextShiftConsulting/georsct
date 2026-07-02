"""SiameseDiff encoder: change detection from image pairs.

Processes two raster frames through a shared-weight CNN backbone,
computes explicit feature-space differences, and produces a hidden
vector encoding the CHANGE between frames.

Use case: before/after satellite pairs (Sentinel-1 SAR flood detection),
temporal raster differencing where the change IS the signal.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from georsct.encoders.spatial_base import SpatialEncoder, SpatialEncoderOutput


class SiameseDiffEncoder(SpatialEncoder):
    """Siamese encoder with explicit frame differencing.

    Architecture:
        input (batch, 2, C, H, W) -- pair of rasters
        -> split into frame_a, frame_b
        -> shared CNN backbone -> feat_a, feat_b
        -> diff = feat_b - feat_a (explicit change signal)
        -> concat [feat_b, diff]
        -> MLP -> hidden (batch, hidden_dim)
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

        fusion_dim = self._backbone_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dt_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> SpatialEncoderOutput:
        frame_a = x[:, 0]
        frame_b = x[:, 1]

        frame_a = self.channel_adapter(frame_a)
        frame_b = self.channel_adapter(frame_b)

        feat_a = self.features(frame_a)
        feat_b = self.features(frame_b)

        diff_feat = feat_b - feat_a

        pooled_b = self.pool(feat_b).flatten(1)
        pooled_diff = self.pool(diff_feat).flatten(1)

        fused = torch.cat([pooled_b, pooled_diff], dim=-1)
        hidden = self.fusion(fused)

        if timestamps is not None:
            dt = (timestamps[:, 1] - timestamps[:, 0]).unsqueeze(-1)
            dt_scale = self.dt_embed(dt)
            hidden = hidden * (1.0 + dt_scale)

        return SpatialEncoderOutput(hidden=hidden, feature_map=diff_feat)
