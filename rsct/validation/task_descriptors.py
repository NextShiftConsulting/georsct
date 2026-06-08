"""Task descriptor utilities for deployment-aligned spatial validation.

This module treats each held-out row as a validation task. A task is described by
ordinary covariates plus difficulty descriptors such as distance from the nearest
training location in the corresponding fold.

Expected minimum inputs
-----------------------
features table:
    zcta, event_id, centroid_lat, centroid_lon, <descriptor columns...>

folds table:
    zcta, event_id, fold_id

The functions are intentionally pandas-first so they can be dropped into an
existing GeoRSCT/FloodRSCT experiment pipeline without requiring geopandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TaskDescriptorConfig:
    """Column configuration for row-level validation task descriptors."""

    id_cols: tuple[str, ...] = ("zcta", "event_id")
    fold_col: str = "fold_id"
    lat_col: str = "centroid_lat"
    lon_col: str = "centroid_lon"
    n_bins: int = 5
    distance_col: str = "nearest_train_km"


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str = "dataframe") -> None:
    """Raise a helpful error if required columns are missing."""

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Return dense pairwise haversine distances in kilometers.

    Parameters
    ----------
    lat1, lon1:
        Arrays for validation points.
    lat2, lon2:
        Arrays for training points.
    chunk_size:
        Maximum number of validation points processed per chunk. This keeps peak
        memory lower for large validation folds.
    """

    radius_km = 6371.0088
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    if len(lat1) == 0 or len(lat2) == 0:
        return np.empty((len(lat1), len(lat2)), dtype=float)

    lat2_rad = np.radians(lat2)[None, :]
    lon2_rad = np.radians(lon2)[None, :]

    chunks: list[np.ndarray] = []
    for start in range(0, len(lat1), chunk_size):
        end = min(start + chunk_size, len(lat1))
        lat1_rad = np.radians(lat1[start:end])[:, None]
        lon1_rad = np.radians(lon1[start:end])[:, None]

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        )
        chunks.append(2.0 * radius_km * np.arcsin(np.sqrt(a)))

    return np.vstack(chunks)


def add_nearest_training_distance(
    df: pd.DataFrame,
    cfg: TaskDescriptorConfig = TaskDescriptorConfig(),
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add nearest-training distance for each validation row.

    For every fold, validation rows are rows in that fold and training rows are
    rows outside that fold. If group_cols are supplied, distances are computed
    separately inside each group. This is useful when a file contains multiple
    scenarios or targets that should not borrow training rows from each other.
    """

    require_columns(df, [cfg.fold_col, cfg.lat_col, cfg.lon_col], "task dataframe")
    out = df.copy()
    out[cfg.distance_col] = np.nan

    if group_cols:
        require_columns(out, group_cols, "task dataframe")
        group_iter = out.groupby(list(group_cols), dropna=False).groups.items()
    else:
        group_iter = [("__all__", out.index)]

    for _, group_index in group_iter:
        group = out.loc[group_index]
        for fold_id, val_idx in group.groupby(cfg.fold_col, dropna=False).groups.items():
            val = group.loc[val_idx]
            train = group[group[cfg.fold_col] != fold_id]
            if train.empty or val.empty:
                continue

            distances = haversine_km(
                val[cfg.lat_col].to_numpy(dtype=float),
                val[cfg.lon_col].to_numpy(dtype=float),
                train[cfg.lat_col].to_numpy(dtype=float),
                train[cfg.lon_col].to_numpy(dtype=float),
            )
            out.loc[val.index, cfg.distance_col] = np.nanmin(distances, axis=1)

    return out


def fit_quantile_edges(
    target_df: pd.DataFrame,
    columns: Iterable[str],
    n_bins: int = 5,
) -> dict[str, np.ndarray]:
    """Fit quantile bin edges from the deployment/target distribution.

    Binning from the target distribution prevents each validation fold from
    defining its own task space. Degenerate variables are safely skipped by
    creating a tiny two-edge interval.
    """

    edges: dict[str, np.ndarray] = {}
    for col in columns:
        if col not in target_df.columns:
            continue

        values = pd.to_numeric(target_df[col], errors="coerce").dropna()
        if values.empty:
            continue

        q = np.linspace(0.0, 1.0, n_bins + 1)
        raw_edges = np.quantile(values.to_numpy(dtype=float), q)
        unique_edges = np.unique(raw_edges)

        if len(unique_edges) < 2:
            v = float(values.iloc[0])
            unique_edges = np.array([v - 1e-9, v + 1e-9])
        elif len(unique_edges) == 2 and unique_edges[0] == unique_edges[1]:
            v = float(unique_edges[0])
            unique_edges = np.array([v - 1e-9, v + 1e-9])

        unique_edges = unique_edges.astype(float)
        unique_edges[0] = -np.inf
        unique_edges[-1] = np.inf
        edges[col] = unique_edges

    return edges


#: Bin value for tasks that cannot be placed: NaN feature OR NaN deployment
#: distance (no observed/reference support). Kept as an explicit category rather
#: than dropped, so "no reference support" and missing features surface in the
#: distribution, JS, missing-bin, and weight logic instead of vanishing.
SENTINEL_BIN = -1


def apply_bins(
    df: pd.DataFrame,
    edges: dict[str, np.ndarray],
    suffix: str = "_bin",
) -> pd.DataFrame:
    """Apply precomputed bin edges and append categorical bin columns.

    Unbinnable values (NaN feature, or NaN nearest-distance meaning no reference
    support) are assigned ``SENTINEL_BIN`` rather than NA, so they remain a
    visible category downstream and cannot be silently dropped.
    """

    out = df.copy()
    for col, col_edges in edges.items():
        if col not in out.columns:
            continue
        bin_col = f"{col}{suffix}"
        binned = pd.cut(
            pd.to_numeric(out[col], errors="coerce"),
            bins=col_edges,
            labels=False,
            include_lowest=True,
        ).astype("Int64")
        out[bin_col] = binned.fillna(SENTINEL_BIN).astype("Int64")
    return out


def build_task_descriptors(
    features: pd.DataFrame,
    folds: pd.DataFrame,
    descriptor_cols: list[str],
    cfg: TaskDescriptorConfig = TaskDescriptorConfig(),
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build row-level validation task descriptors from features and folds."""

    join_cols = list(cfg.id_cols)
    require_columns(features, join_cols + [cfg.lat_col, cfg.lon_col] + descriptor_cols, "features")
    require_columns(folds, join_cols + [cfg.fold_col], "folds")

    merged = features.merge(
        folds[join_cols + [cfg.fold_col]].drop_duplicates(),
        on=join_cols,
        how="inner",
        validate="many_to_one",
    )

    keep_cols = join_cols + [cfg.fold_col, cfg.lat_col, cfg.lon_col] + descriptor_cols
    if group_cols:
        keep_cols += [c for c in group_cols if c in merged.columns and c not in keep_cols]

    task_df = merged[keep_cols].copy()
    task_df = add_nearest_training_distance(task_df, cfg=cfg, group_cols=group_cols)
    return task_df


def build_target_descriptors(
    features: pd.DataFrame,
    descriptor_cols: list[str],
    cfg: TaskDescriptorConfig = TaskDescriptorConfig(),
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build deployment/target task descriptors.

    In v0, target is all available rows in the feature table. For a real flood
    deployment domain, replace this with a scenario mask, forecast grid, or
    target ZCTA-event universe.
    """

    keep_cols = list(cfg.id_cols) + [cfg.lat_col, cfg.lon_col] + descriptor_cols
    if group_cols:
        keep_cols += [c for c in group_cols if c in features.columns and c not in keep_cols]
    require_columns(features, keep_cols, "features")
    return features[keep_cols].copy()
