"""Kappa computation from spatial geometry.

Pure functions -- no I/O, no S3, no SQL.
Source patterns: AGDS Ch.8 (Spatial Clustering -- Queen/KNN weights, sparse
connectivity, spatially-constrained clustering).
Existing impl:  compute_geometry_kappa.py (SageMaker job).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeometryKappa:
    """Pre-model geometry compatibility score."""

    spatial_connectivity: float     # fraction of units with >= 1 neighbor
    support_coverage: float         # feature non-null rate
    scale_stability: float          # between-county / total variance ratio
    admin_alignment: float          # 1 - fraction multi-county units
    kappa_geom: float               # mean of available terms

    @property
    def kappa_prior(self) -> float:
        """Q-007 canonical name: prior_geometry.

        Pre-model diagnostic from spatial geometry stats.
        NOT kappa_coupling (R*(1-N)) — cannot flow to gates (ADR-020 D8.6).
        """
        return self.kappa_geom


def build_weights_queen(gdf: "gpd.GeoDataFrame") -> "libpysal.weights.W":
    """Queen contiguity weights from a GeoDataFrame.

    Pattern: libpysal.weights.Queen.from_dataframe (AGDS Ch.8).
    """
    from libpysal.weights import Queen

    w = Queen.from_dataframe(gdf)
    w.transform = "R"
    return w


def build_weights_knn(
    gdf: "gpd.GeoDataFrame",
    k: int = 10,
) -> "libpysal.weights.W":
    """K-nearest-neighbor weights from a GeoDataFrame.

    Pattern: libpysal.weights.KNN.from_dataframe (AGDS Ch.8).
    """
    from libpysal.weights import KNN

    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "R"
    return w


def build_weights_from_adjacency(
    adj_df: pd.DataFrame,
    unit_ids: list[str],
) -> "libpysal.weights.W":
    """Build weights from an adjacency edge list.

    Existing impl: compute_residual_lisa.py::build_weights_from_adjacency().
    """
    from libpysal.weights import W

    id_set = set(unit_ids)
    cols = adj_df.columns.tolist()
    c1, c2 = cols[0], cols[1]

    neighbors: dict[str, list[str]] = {z: [] for z in unit_ids}
    for _, row in adj_df.iterrows():
        a, b = str(row[c1]), str(row[c2])
        if a in id_set and b in id_set:
            neighbors.setdefault(a, []).append(b)
            neighbors.setdefault(b, []).append(a)

    neighbors = {k: list(set(v)) for k, v in neighbors.items()}
    return W(neighbors, silence_warnings=True)


def compute_spatial_connectivity(
    adj_df: pd.DataFrame,
    unit_ids: set[str],
) -> float:
    """Fraction of spatial units with at least one neighbor.

    Existing impl: compute_geometry_kappa.py::compute_spatial_connectivity().
    """
    if adj_df is None or adj_df.empty or not unit_ids:
        return 0.0

    cols = adj_df.columns.tolist()
    src, dst = cols[0], cols[1]
    connected = set()
    for _, row in adj_df.iterrows():
        s, d = str(row[src]), str(row[dst])
        if s in unit_ids and d in unit_ids:
            connected.update([s, d])

    return len(connected) / len(unit_ids)


def compute_support_coverage(
    df: pd.DataFrame,
    target: str,
    features: list[str],
) -> float:
    """Feature non-null rate for rows where target is present.

    Existing impl: compute_geometry_kappa.py::compute_support_coverage().
    """
    if target not in df.columns:
        return 0.0
    sub = df.loc[df[target].notna(), [f for f in features if f in df.columns]]
    if sub.empty:
        return 0.0
    return float(sub.notna().mean().mean())


def compute_scale_stability(
    df: pd.DataFrame,
    crosswalk: pd.DataFrame,
    features: list[str],
) -> float:
    """Between-county / total variance ratio.

    Existing impl: compute_geometry_kappa.py::compute_scale_stability().
    """
    if crosswalk is None or crosswalk.empty:
        return 0.5

    df = df.copy()
    df["zcta_id"] = df["zcta_id"].astype(str)
    xwalk = crosswalk[["zcta_id", "county_fips"]].drop_duplicates().copy()
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)
    merged = df.merge(xwalk, on="zcta_id", how="left")

    ratios = []
    for col in features:
        if col not in merged.columns:
            continue
        vals = merged[[col, "county_fips"]].dropna()
        if len(vals) < 10:
            continue
        total_var = vals[col].var()
        if total_var < 1e-12:
            continue
        between_var = vals.groupby("county_fips")[col].mean().var()
        ratios.append(between_var / total_var)

    return float(np.clip(np.mean(ratios), 0.0, 1.0)) if ratios else 0.5


def compute_geometry_kappa(
    spatial_connectivity: float,
    support_coverage: float,
    scale_stability: float,
    admin_alignment: float,
) -> GeometryKappa:
    """Aggregate geometry terms into kappa_geom."""
    terms = [spatial_connectivity, support_coverage, scale_stability, admin_alignment]
    return GeometryKappa(
        spatial_connectivity=spatial_connectivity,
        support_coverage=support_coverage,
        scale_stability=scale_stability,
        admin_alignment=admin_alignment,
        kappa_geom=float(np.mean(terms)),
    )
