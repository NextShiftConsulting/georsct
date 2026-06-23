"""Adapter: TJEPA embedding source implementing the EmbeddingSource port.

Wraps a trained TabularJEPA model to provide embeddings via the
standard port interface (georsct.ports.embedding_source).
"""

from __future__ import annotations

import numpy as np

from georsct.encoders.tjepa import TabularJEPA
from georsct.ports.embedding_source import EmbeddingSource


class TJEPAEmbeddingSource(EmbeddingSource):
    """EmbeddingSource backed by a trained TabularJEPA encoder.

    Args:
        model: A fitted TabularJEPA instance.
        features: (n_units, n_features) feature matrix, aligned with unit_ids.
        unit_ids: Ordered list of unit identifiers (e.g., ZCTA IDs).
    """

    def __init__(
        self,
        model: TabularJEPA,
        features: np.ndarray,
        unit_ids: list[str],
    ):
        self._model = model
        self._unit_ids = list(unit_ids)
        self._id_to_idx = {uid: i for i, uid in enumerate(self._unit_ids)}

        # Pre-compute embeddings (no masking for production use)
        self._embeddings = model.encode(features)

    def get_embedding(self, unit_id: str) -> np.ndarray:
        idx = self._id_to_idx.get(unit_id)
        if idx is None:
            raise KeyError(f"Unknown unit_id: {unit_id}")
        return self._embeddings[idx]

    def get_embeddings_batch(self, unit_ids: list[str]) -> np.ndarray:
        indices = []
        for uid in unit_ids:
            idx = self._id_to_idx.get(uid)
            if idx is None:
                raise KeyError(f"Unknown unit_id: {uid}")
            indices.append(idx)
        return self._embeddings[indices]

    @property
    def embed_dim(self) -> int:
        return self._embeddings.shape[1]

    @property
    def n_units(self) -> int:
        return len(self._unit_ids)
