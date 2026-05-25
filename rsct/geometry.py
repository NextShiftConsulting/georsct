"""263-dim geometry feature extraction from embedding pairs."""

import numpy as np
from typing import Optional


class GeometryExtractor:
    """Extract geometric features from (query, passage) embedding pairs.

    Features (263-dim):
    - u, v: projected embeddings (64 + 64 = 128)
    - |u-v|: absolute difference (64)
    - u*v: elementwise product (64)
    - scalars: cosine, norms, angles (7)
    """

    def __init__(self, projection_dim: int = 64):
        self.projection_dim = projection_dim
        self._projection_matrix: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray) -> "GeometryExtractor":
        """Fit projection matrix via PCA."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.projection_dim)
        pca.fit(embeddings)
        self._projection_matrix = pca.components_.T
        return self

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding to lower dimension."""
        if self._projection_matrix is None:
            raise ValueError("Must call fit() first or load projection matrix")
        return embedding @ self._projection_matrix

    def extract(
        self,
        query_emb: np.ndarray,
        passage_emb: np.ndarray,
        projected: bool = False
    ) -> np.ndarray:
        """Extract 263-dim geometry features.

        Args:
            query_emb: Query embedding (D,) or projected (64,)
            passage_emb: Passage embedding (D,) or projected (64,)
            projected: If True, skip projection step

        Returns:
            features: (263,) geometry feature vector
        """
        if not projected:
            u = self.project(query_emb)
            v = self.project(passage_emb)
        else:
            u = query_emb
            v = passage_emb

        # Pairwise features
        diff = u - v
        prod = u * v

        # Scalar features
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        diff_norm = np.linalg.norm(diff)

        cosine = np.dot(u, v) / (u_norm * v_norm + 1e-8)

        # Projection scalars
        u_proj_on_diff = np.dot(u, diff) / (diff_norm + 1e-8)
        v_proj_on_diff = np.dot(v, diff) / (diff_norm + 1e-8)
        directional_asymmetry = u_proj_on_diff - v_proj_on_diff

        scalars = np.array([
            cosine,
            u_norm,
            v_norm,
            diff_norm,
            u_proj_on_diff,
            v_proj_on_diff,
            directional_asymmetry
        ])

        # Concatenate: [u, v, |u-v|, u*v, scalars]
        features = np.concatenate([u, v, np.abs(diff), prod, scalars])

        return features

    def extract_batch(
        self,
        query_embs: np.ndarray,
        passage_embs: np.ndarray,
        projected: bool = False
    ) -> np.ndarray:
        """Extract features for batch of pairs.

        Args:
            query_embs: (N, D) query embeddings
            passage_embs: (N, D) passage embeddings
            projected: If True, inputs are already projected

        Returns:
            features: (N, 263) geometry features
        """
        n_samples = len(query_embs)
        features = np.zeros((n_samples, 263))

        for i in range(n_samples):
            features[i] = self.extract(
                query_embs[i],
                passage_embs[i],
                projected=projected
            )

        return features

    def save(self, path: str) -> None:
        """Save projection matrix."""
        np.save(path, self._projection_matrix)

    def load(self, path: str) -> "GeometryExtractor":
        """Load projection matrix."""
        self._projection_matrix = np.load(path)
        return self
