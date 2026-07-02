"""RasterSeq encoder: temporal raster sequence encoding.

Processes a stack of T raster frames through factorized space-time
attention, producing a hidden vector that captures both spatial
structure and temporal dynamics.

Use case: MRMS hourly precipitation sequences, HRRR forecast grids,
any temporal stack of spatial rasters with irregular frame spacing.
"""

import torch
import torch.nn as nn

from georsct.encoders.spatial_base import SpatialEncoder, SpatialEncoderOutput


class ContinuousTimePosEncoding(nn.Module):
    """Continuous-time positional encoding for irregular timestamps."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.freq = nn.Parameter(torch.randn(embed_dim // 2) * 0.01)
        self.phase = nn.Parameter(torch.zeros(embed_dim // 2))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        t = timestamps.unsqueeze(-1)
        angles = t * self.freq + self.phase
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class PatchEmbedding(nn.Module):
    """Convert a raster frame into a sequence of patch tokens."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class RasterSeqEncoder(SpatialEncoder):
    """Sequence encoder for multi-frame raster stacks.

    Architecture:
        input (batch, T, C, H, W)
        -> per-frame patch embedding -> (batch, T, n_patches, embed_dim)
        -> add continuous-time positional encoding (per frame)
        -> spatial self-attention (within each frame)
        -> temporal cross-attention (across frames)
        -> pool over patches and time -> hidden (batch, hidden_dim)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        embed_dim: int = 128,
        patch_size: int = 8,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__(in_channels, hidden_dim)
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.time_pos = ContinuousTimePosEncoding(embed_dim)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.spatial_attn = nn.TransformerEncoder(spatial_layer, num_layers=n_layers)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(temporal_layer, num_layers=n_layers)

        self.project = nn.Linear(embed_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> SpatialEncoderOutput:
        batch, T, C, H, W = x.shape

        x_flat = x.reshape(batch * T, C, H, W)
        patches = self.patch_embed(x_flat)
        n_patches = patches.shape[1]

        patches = self.spatial_attn(patches)

        patches = patches.reshape(batch, T, n_patches, self.embed_dim)

        if timestamps is None:
            timestamps = torch.arange(T, device=x.device, dtype=x.dtype)
            timestamps = timestamps.unsqueeze(0).expand(batch, -1)

        time_emb = self.time_pos(timestamps)
        patches = patches + time_emb.unsqueeze(2)

        patches = patches.permute(0, 2, 1, 3).reshape(batch * n_patches, T, self.embed_dim)
        patches = self.temporal_attn(patches)

        pooled_time = patches.mean(dim=1)
        pooled_time = pooled_time.reshape(batch, n_patches, self.embed_dim)
        pooled = pooled_time.mean(dim=1)

        hidden = self.project(pooled)

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        last_frame = patches.reshape(batch, n_patches, T, self.embed_dim)[:, :, -1, :]
        feat_map = last_frame.transpose(1, 2).reshape(batch, self.embed_dim, h_patches, w_patches)

        return SpatialEncoderOutput(hidden=hidden, feature_map=feat_map)
