"""Port: embedding vector access for similarity/novelty scoring."""

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingSource(ABC):

    @abstractmethod
    def get_embedding(self, unit_id: str) -> np.ndarray: ...

    @abstractmethod
    def get_embeddings_batch(self, unit_ids: list[str]) -> np.ndarray:
        """Return (n_units, embedding_dim) array."""
