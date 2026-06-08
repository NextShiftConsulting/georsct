"""Corrected deployment-target descriptors for deployment-aligned validation.

This module replaces the circular v0 ``build_target_descriptors``
(which returned all rows in the feature/sample table and merged validation
distances back onto them). It builds the deployment task distribution from an
explicit eligible universe and computes deployment prediction distance
natively as the distance from each deployment unit to the nearest
*observed* (sampled/training) location -- never from the validation folds.

Key inputs
----------
universe : full candidate table (features + centroids). Superset of
    ``observed``. For pluvial regimes this is the entire metro set.
reference : the rows whose locations are AVAILABLE TO THE MODEL AT PREDICTION
    TIME and serve as the reference set for deployment prediction distance. This
    is NOT automatically "all labeled rows": under leave-event-out / spatial
    transfer the held-out labels are unavailable at deployment time and MUST be
    excluded from `reference` by the caller. Passing held-out labels here
    understates prediction distance and corrupts the certificate.
domain : a RegimeDomain from ``deployment_domains`` selecting eligible rows.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .deployment_domains import RegimeDomain
from .task_descriptors import TaskDescriptorConfig, haversine_km, require_columns


def nearest_distance_to_reference(
    points: pd.DataFrame,
    reference: pd.DataFrame,
    cfg: TaskDescriptorConfig = TaskDescriptorConfig(),
    group_cols: Sequence[str] | None = None,
) -> np.ndarray:
    """Nearest-neighbour distance (km) from each point to the reference set.

    Computed within groups (e.g. per event) when ``group_cols`` is given, so a
    deployment point is never matched to an observed point from a different
    event. Points with no reference rows in their group receive NaN -- that is a
    genuine 'no observed support' signal, not an error to paper over.
    """

    require_columns(points, [cfg.lat_col, cfg.lon_col], "points")
    require_columns(reference, [cfg.lat_col, cfg.lon_col], "reference")
    out = np.full(len(points), np.nan, dtype=float)

    if group_cols:
        require_columns(points, group_cols, "points")
        require_columns(reference, group_cols, "reference")

        def _key(k):  # pandas yields scalar for 1-col, tuple for >1; normalize
            return k if isinstance(k, tuple) else (k,)

        ref_groups = {_key(k): g for k, g in reference.groupby(list(group_cols), dropna=False)}
        point_groups = points.groupby(list(group_cols), dropna=False).indices
        for key, positions in point_groups.items():
            ref = ref_groups.get(_key(key))
            if ref is None or ref.empty:
                continue
            pts = points.iloc[positions]
            d = haversine_km(
                pts[cfg.lat_col].to_numpy(float), pts[cfg.lon_col].to_numpy(float),
                ref[cfg.lat_col].to_numpy(float), ref[cfg.lon_col].to_numpy(float),
            )
            out[positions] = np.nanmin(d, axis=1)
        return out

    if reference.empty:
        return out
    d = haversine_km(
        points[cfg.lat_col].to_numpy(float), points[cfg.lon_col].to_numpy(float),
        reference[cfg.lat_col].to_numpy(float), reference[cfg.lon_col].to_numpy(float),
    )
    return np.nanmin(d, axis=1)


def build_deployment_target_descriptors(
    universe: pd.DataFrame,
    reference: pd.DataFrame,
    descriptor_cols: list[str],
    domain: RegimeDomain,
    cfg: TaskDescriptorConfig = TaskDescriptorConfig(),
    reference_group_cols: Sequence[str] | None = ("event_id",),
) -> pd.DataFrame:
    """Build deployment-task descriptors for one regime.

    The deployment domain is ``domain.resolve(universe)`` (eligible rows).
    ``cfg.distance_col`` is the distance from each deployment row to the nearest
    *prediction-time-available* reference location -- never a value borrowed from
    the validation folds.

    reference_group_cols is the geometry-specific grouping for the distance
    match; it is NOT hard-coded. Defaults to ("event_id",) for same-event
    mapping. For leave-event-out / spatial transfer, the caller must pass the
    grouping (or pre-filter `reference`) that reflects what is available at
    deployment time -- e.g. only prior events, or no grouping for pure spatial
    transfer. A deployment point with no reference in its group yields NaN
    distance, which is preserved downstream as a no-reference-support signal.
    """

    keep = list(cfg.id_cols) + [cfg.lat_col, cfg.lon_col] + descriptor_cols
    if reference_group_cols:
        keep += [c for c in reference_group_cols if c in universe.columns and c not in keep]
    require_columns(universe, list(cfg.id_cols) + [cfg.lat_col, cfg.lon_col] + descriptor_cols, "universe")
    require_columns(reference, [cfg.lat_col, cfg.lon_col], "reference")

    deployment = domain.resolve(universe)
    target = deployment[[c for c in keep if c in deployment.columns]].copy()

    grp = None
    if reference_group_cols:
        grp = [c for c in reference_group_cols if c in target.columns and c in reference.columns]
        grp = grp or None
    target[cfg.distance_col] = nearest_distance_to_reference(
        target, reference, cfg=cfg, group_cols=grp
    )
    return target


def build_target_descriptors(*args, **kwargs):
    """Deprecated. The v0 all-rows target was circular; use
    ``build_deployment_target_descriptors(universe, reference, ..., domain=...)``.
    """

    raise NotImplementedError(
        "build_target_descriptors (v0, target=all sample rows) was circular and "
        "is removed. Call build_deployment_target_descriptors(universe, reference, "
        "descriptor_cols, domain) with an explicit RegimeDomain from "
        "rsct.validation.deployment_domains."
    )
