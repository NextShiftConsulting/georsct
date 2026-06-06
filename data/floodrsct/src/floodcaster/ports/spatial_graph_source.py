"""Port: spatial graph / weights matrix construction.

Contract for building spatial weights from geometries or adjacency lists.
Adapter impls: libpysal from GeoDataFrame, PostGIS adjacency query, S3 edge list.
Pattern: Queen/KNN weights (AGDS Ch.8).
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class SpatialGraphSource(ABC):
    """Abstract port for spatial weights construction."""

    @abstractmethod
    def build_weights(
        self,
        source_id: str,
        method: str = "queen",
        k: Optional[int] = None,
    ) -> "libpysal.weights.W":
        """Build spatial weights matrix.

        Args:
            source_id: Identifier for the boundary dataset.
            method: 'queen', 'rook', or 'knn'.
            k: Number of neighbors (required for knn).
        """

    @abstractmethod
    def get_sparse_connectivity(
        self,
        source_id: str,
    ) -> "scipy.sparse.csr_matrix":
        """Return sparse connectivity matrix for clustering.

        Pattern: w.sparse (AGDS Ch.8).
        """
