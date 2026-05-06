"""
Shared representation loader for Series 019 experiments.

Convention:
  *_transform.npz  — Store transform params (scaler + PCA components).
                     Applied at runtime to raw features.
                     Required keys: scaler_mean, scaler_scale, components,
                                    component_mean, feature_schema
  *_latents.npz    — Pre-computed embedding arrays (e.g. GNN).
                     Ready to use directly.
                     Required keys: Z, ids
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _apply_transform(art_path: Path, df: pd.DataFrame) -> np.ndarray:
    """Load a _transform.npz artifact and apply to raw features from df."""
    art = np.load(art_path, allow_pickle=True)

    # Resolve feature columns from stored schema
    stored_schema = [str(s) for s in art["feature_schema"]]
    n_features = len(stored_schema)

    X = np.zeros((len(df), n_features), dtype=np.float64)
    missing = []
    for i, col in enumerate(stored_schema):
        if col in df.columns:
            X[:, i] = df[col].values.astype(np.float64)
        else:
            missing.append(col)

    if missing:
        log.warning(
            "%s: %d/%d schema columns missing (zero-filled): %s",
            art_path.name, len(missing), n_features, missing[:5],
        )

    # NaN imputation (median)
    for j in range(n_features):
        nans = np.isnan(X[:, j])
        if nans.any():
            X[nans, j] = np.nanmedian(X[:, j])

    # Apply stored transform: scale then project
    scaler_mean = art["scaler_mean"]
    scaler_scale = art["scaler_scale"]

    # Key names: support both old ('pca_components','pca_mean') and new ('components','component_mean')
    if "components" in art:
        components = art["components"]
        component_mean = art["component_mean"]
    elif "pca_components" in art:
        components = art["pca_components"]
        component_mean = art["pca_mean"]
    else:
        raise KeyError(f"{art_path.name}: missing 'components' or 'pca_components' key")

    X_scaled = (X - scaler_mean) / scaler_scale
    Z = (X_scaled - component_mean) @ components.T

    log.info(
        "%s: applied transform (%d features -> %dd)",
        art_path.name, n_features, Z.shape[1],
    )
    return Z


def _load_latents(art_path: Path) -> np.ndarray:
    """Load a _latents.npz artifact (pre-computed embeddings)."""
    art = np.load(art_path, allow_pickle=True)

    # Support both standard key 'Z' and legacy key 'latents'
    if "Z" in art:
        Z = art["Z"]
    elif "latents" in art:
        Z = art["latents"]
    else:
        raise KeyError(f"{art_path.name}: missing 'Z' or 'latents' key")

    log.info("%s: loaded latents (shape %s)", art_path.name, Z.shape)
    return Z


# Canonical artifact filenames per embedding family
REPR_ARTIFACTS = {
    "pca_v1": "pca32_v1_transform.npz",
    "spatial_lag_v1": "spatial_lag_v1_transform.npz",
    "gnn_v2": "gnn_v2_latents.npz",
}

# Legacy filenames (fallback if new names not yet deployed)
REPR_ARTIFACTS_LEGACY = {
    "pca_v1": "pca32_v1.npz",
    "spatial_lag_v1": "spatial_lag_v1.npz",
    "gnn_v2": "zcta_latents_v1.npz",
}


def load_representations(
    df: pd.DataFrame,
    acs_cols: List[str],
    repr_dir: Optional[Path],
    embeddings_requested: List[str],
) -> Dict[str, np.ndarray]:
    """Load all requested representations.

    Args:
        df: DataFrame with raw features.
        acs_cols: ACS feature column names (for PCA fallback).
        repr_dir: Path to representation artifacts directory.
        embeddings_requested: List of embedding keys to load.

    Returns:
        Dict mapping embedding name -> (n_samples, dim) array.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    embeddings: Dict[str, np.ndarray] = {}

    for emb_name in embeddings_requested:
        loaded = False

        if repr_dir:
            # Try canonical name first, then legacy
            for artifact_map in (REPR_ARTIFACTS, REPR_ARTIFACTS_LEGACY):
                filename = artifact_map.get(emb_name)
                if filename and (repr_dir / filename).exists():
                    path = repr_dir / filename
                    if "_transform" in filename or filename in (
                        "pca32_v1.npz", "spatial_lag_v1.npz"
                    ):
                        embeddings[emb_name] = _apply_transform(path, df)
                    else:
                        embeddings[emb_name] = _load_latents(path)
                    loaded = True
                    break

        if not loaded:
            # Fallback: fit locally
            if emb_name == "pca_v1":
                X = df[acs_cols].values.astype(np.float64)
                X = _impute_nan(X)
                scaler = StandardScaler().fit(X)
                pca = PCA(n_components=32, random_state=42).fit(scaler.transform(X))
                embeddings[emb_name] = pca.transform(scaler.transform(X))
                log.info("pca_v1: fitted locally (no artifact)")
            elif emb_name == "spatial_lag_v1":
                # Use all numeric features including lag/enrich
                all_cols = acs_cols + sorted(
                    c for c in df.columns
                    if (c.startswith("lag_") or c.startswith("enrich_"))
                    and pd.api.types.is_numeric_dtype(df[c])
                )
                X = df[all_cols].values.astype(np.float64)
                X = _impute_nan(X)
                scaler = StandardScaler().fit(X)
                pca = PCA(n_components=32, random_state=42).fit(scaler.transform(X))
                embeddings[emb_name] = pca.transform(scaler.transform(X))
                log.info("spatial_lag_v1: fitted locally from %d features", len(all_cols))
            elif emb_name == "gnn_v2":
                log.warning("gnn_v2: no artifact found, using pca_v1 as fallback")
                if "pca_v1" in embeddings:
                    embeddings[emb_name] = embeddings["pca_v1"].copy()
                else:
                    raise FileNotFoundError(f"No artifact for {emb_name} and no fallback")
            else:
                raise FileNotFoundError(f"Unknown embedding: {emb_name}")

    return embeddings


def _impute_nan(X: np.ndarray) -> np.ndarray:
    """Median-impute NaN values column-wise."""
    for j in range(X.shape[1]):
        nans = np.isnan(X[:, j])
        if nans.any():
            X[nans, j] = np.nanmedian(X[:, j])
    return X
