"""Adapter: spatial encoder hidden state -> rotor-compatible embedding.

Boundary contract between spatial encoders and yrsn rotor.
Emits a fixed-dimension numpy array that yrsn's rotor consumes.
"""

import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray

from georsct.encoders.spatial_base import SpatialEncoder, SpatialEncoderOutput


class SpatialToEmbedding(nn.Module):
    """Projects spatial encoder output to a fixed-dimension embedding.

    The embedding dimension must match what the downstream rotor expects.
    Default is 64 (P19: hardware compression -- 64-dim must always work).

    Architecture: hidden_state -> LayerNorm -> Linear -> tanh -> Linear -> embedding
    """

    def __init__(self, hidden_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, encoder_output: SpatialEncoderOutput) -> torch.Tensor:
        h = self.norm(encoder_output.hidden)
        return self.project(h)

    def to_numpy(self, encoder_output: SpatialEncoderOutput) -> NDArray:
        """Produce a numpy embedding for yrsn rotor consumption.

        Args:
            encoder_output: Output from any SpatialEncoder.

        Returns:
            (batch, embedding_dim) numpy array, float64.
        """
        with torch.no_grad():
            embedding = self.forward(encoder_output)
        return embedding.cpu().numpy().astype(np.float64)


class SpatialPipeline:
    """Convenience wrapper: encoder + adapter in one call.

    Usage:
        pipeline = SpatialPipeline(encoder, adapter)
        embedding = pipeline.encode(raster)
    """

    def __init__(self, encoder: SpatialEncoder, adapter: SpatialToEmbedding):
        self.encoder = encoder
        self.adapter = adapter

    def encode(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> NDArray:
        """Encode spatial raster data to rotor-compatible embedding.

        Args:
            x: Input raster (shape depends on encoder type).
            timestamps: Optional timestamps for temporal encoders.

        Returns:
            (batch, embedding_dim) numpy array ready for yrsn rotor.
        """
        self.encoder.eval()
        self.adapter.eval()
        with torch.no_grad():
            encoder_output = self.encoder(x, timestamps)
            return self.adapter.to_numpy(encoder_output)
