"""Compatibility assessment via spatial lag and clustering.

Pure functions -- no I/O, no S3, no SQL.
Source patterns: AGDS Ch.8 (spatially-constrained clustering),
AGDS Ch.6 (spatial lag computation).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompatibilityResult:
    """Spatial compatibility assessment for a variable set."""

    cluster_labels: np.ndarray
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    n_clusters: int


def compute_spatial_lag(
    values: np.ndarray,
    w: "libpysal.weights.W",
) -> np.ndarray:
    """Spatially lagged values using weights matrix.

    Pattern: weights.spatial_lag.lag_spatial (AGDS Ch.6).
    """
    from pysal.lib.weights.spatial_lag import lag_spatial

    return lag_spatial(w, values)


def cluster_spatially_constrained(
    X: np.ndarray,
    w: "libpysal.weights.W",
    n_clusters: int = 5,
    scale: bool = True,
) -> CompatibilityResult:
    """Spatially-constrained agglomerative clustering.

    Uses Queen/KNN sparse connectivity matrix to enforce spatial contiguity.
    Pattern: AgglomerativeClustering(connectivity=w.sparse) (AGDS Ch.8).
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import robust_scale

    X_scaled = robust_scale(X) if scale else X

    model = AgglomerativeClustering(
        linkage="ward",
        connectivity=w.sparse,
        n_clusters=n_clusters,
    )
    labels = model.fit_predict(X_scaled)

    return CompatibilityResult(
        cluster_labels=labels,
        silhouette=float(silhouette_score(X_scaled, labels)),
        calinski_harabasz=float(calinski_harabasz_score(X_scaled, labels)),
        davies_bouldin=float(davies_bouldin_score(X_scaled, labels)),
        n_clusters=n_clusters,
    )
