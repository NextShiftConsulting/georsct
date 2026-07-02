"""Abstract base for spatial raster encoders."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class SpatialEncoderOutput:
    """Standard output from any spatial encoder.

    Attributes:
        hidden: Compressed spatial representation (batch, hidden_dim).
        feature_map: Spatial feature map (batch, hidden_dim, h, w).
            Retains spatial structure for downstream analysis.
    """
    hidden: torch.Tensor
    feature_map: torch.Tensor


class SpatialEncoder(nn.Module, ABC):
    """Abstract base class for spatial raster encoders.

    All encoders share the same interface:
        - Input: raster tensor + optional timestamps
        - Output: SpatialEncoderOutput with hidden state and feature map

    Subclasses handle different input formats:
        - Single raster: (batch, C, H, W)
        - Image pair: (batch, 2, C, H, W)
        - Raster sequence: (batch, T, C, H, W)
    """

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> SpatialEncoderOutput:
        """Encode spatial raster data.

        Args:
            x: Raster input. Shape depends on encoder type.
            timestamps: Optional timestamps for temporal encoding.

        Returns:
            SpatialEncoderOutput with hidden state and feature map.
        """
