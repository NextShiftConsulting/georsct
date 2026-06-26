#!/usr/bin/env python3
"""
adversarial_reconstruct.py

End-to-end adversarial geography-permutation harness for GeoRSCT / FloodRSCT.

Purpose
-------
Wires kappa_reconstruct into the s035-style feature-ladder workflow by providing
a concrete refit_and_embed adapter:

    W or W_perm -> rebuild spatial-lag features -> refit HistGBDT on frozen folds
    -> produce forward score, kappa_spatial, and representation embeddings
    -> compute kappa_reconstruct -> evaluate whether kappa_reconstruct drops under
       adversarial geography permutation beyond what kappa_spatial explains.

Architecture
------------
Domain math (Gabriel graph, crossings, baseline correction, Gate 3B) lives in
georsct.domain.kappa_reconstruct.  This adapter provides:

  - CLI interface for local files (parquet, CSV) or S3 via --scenario shortcut
  - CSR-based W-matrix operations and spatial-lag recomputation
  - HistGBDT training with frozen folds
  - Moran's I convenience kappa_spatial (replace with yrsn in production)
  - Graded corruption ladder with discriminant tests (patches v1)

Design commitments
------------------
1. kappa_reconstruct is backward-recoverability / Gate 3B.  Not residual
   spatial autocorrelation.

2. The falsification test is over-and-above:
       delta_kappa_reconstruct ~ 1 + delta_kappa_spatial [+ level]
   Two test modes:
     intercept_test : full permutation only, tests beta_0 < 0.
     ladder_test    : graded corruption, tests gamma_level < 0 (recommended).

3. The old "mean of intercept-model OLS residuals" statistic is identically
   zero by the normal equations and is NOT used.

FloodRSCT spatial-reasoning commitment: flood risk is fundamentally spatial.
The geographic topology constrains the physically valid outcomes, so a
representation that cannot recover that topology cannot reason about the task
correctly.

Lineage: S018U backward recoverability -> geospatial topology instantiation.
Reference topology: load_huc_drainage_topology.py (pluggable).

CLI examples
------------
# Local files, graded ladder (recommended)
python adversarial_reconstruct.py \\
  --data-parquet processed/houston/r1_features.parquet \\
  --folds-csv processed/houston/folds_county_blocked.csv \\
  --w-edges-csv processed/houston/w_matrix_edges.csv \\
  --coords-csv processed/houston/zcta_centroids.csv \\
  --target nfip_loss --region-id zcta \\
  --feature-regex "^(acs_|svi_|fema_|hydro_|precip_|r1_)" \\
  --spatial-lag-cols nfip_loss precip_total impervious_pct elevation_mean \\
  --n-permutations 25 \\
  --corruption-levels 0.1 0.25 0.5 0.75 1.0 \\
  --out-dir artifacts/kappa_reconstruct_houston

# S3 shortcut (uses _coverage_common infrastructure)
python adversarial_reconstruct.py \\
  --scenario houston --n-permutations 25 --upload
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from scipy import sparse
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import t as student_t
except Exception as exc:  # pragma: no cover
    raise RuntimeError("requires scipy: pip install scipy") from exc

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise RuntimeError("requires scikit-learn: pip install scikit-learn") from exc

# ---------------------------------------------------------------------------
# Domain math from georsct
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from georsct.domain.kappa_reconstruct import (
    SpatialRecoverabilityResult,
    compute_kappa_reconstruct,
    count_crossings,
    gabriel_graph,
    gate_3b_decision,
)

# Optional: S3 infrastructure for --scenario shortcut
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from _coverage_common import BUCKET, SCENARIOS, get_s3_client, load_adjacency
    _HAS_S3 = True
except ImportError:
    _HAS_S3 = False
    SCENARIOS: list[str] = []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


# =========================================================================
# Data containers
# =========================================================================

@dataclass
class RefitResult:
    """Result of a single refit-and-embed pass."""

    label: str
    corruption_level: float
    forward_score: float
    kappa_spatial: float
    morans_i: float
    kappa_reconstruct: float
    n_crossings: int
    n_crossings_baseline: float
    n_implied_edges: int
    stress: float
    mantel_r: float
    coordinate_lift: float
    n_regions: int


@dataclass
class AdversarialSummary:
    """Summary of the adversarial permutation harness."""

    baseline: RefitResult
    n_trials: int
    test_mode: str
    intercept: Optional[float]
    intercept_t: Optional[float]
    intercept_p: Optional[float]
    gamma_level: Optional[float]
    gamma_ci: Optional[tuple[float, float]]
    gamma_p: Optional[float]
    gate_3b_baseline: str
    gate_eligible: bool
    verdict: str
    commitment: str


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adversarial W-matrix permutation test for kappa_reconstruct",
    )

    # --- S3 shortcut ---
    p.add_argument("--scenario", choices=SCENARIOS or None, default=None,
                   help="S3 scenario shortcut (uses _coverage_common)")
    p.add_argument("--upload", action="store_true",
                   help="Upload results to S3 (requires --scenario)")

    # --- Local file inputs ---
    p.add_argument("--data-parquet", type=Path, default=None)
    p.add_argument("--folds-csv", type=Path, default=None)
    p.add_argument("--w-edges-csv", type=Path, default=None)
    p.add_argument("--coords-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)

    # --- Column names ---
    p.add_argument("--target", default="obs_nfip_event_claims")
    p.add_argument("--region-id", default="zcta_id")
    p.add_argument("--event-id", default=None)
    p.add_argument("--fold-col", default="fold")
    p.add_argument("--task", choices=["regression", "binary_classification",
                                      "multiclass_classification"],
                   default="regression")

    # --- Feature selection ---
    p.add_argument("--feature-cols", nargs="*", default=None)
    p.add_argument("--feature-regex", default=None)
    p.add_argument("--exclude-cols", nargs="*", default=None)
    p.add_argument("--spatial-lag-cols", nargs="*", default=[])

    # --- Permutation ---
    p.add_argument("--n-permutations", type=int, default=5,
                   help="Replicate seeds per corruption level")
    p.add_argument("--corruption-levels", nargs="*", type=float,
                   default=[0.1, 0.25, 0.5, 0.75, 1.0],
                   help="Graded corruption fractions (0=baseline, 1=full scramble)")
    p.add_argument("--test-mode", choices=["ladder", "intercept"], default="ladder",
                   help="ladder = graded + gamma_level test; "
                        "intercept = full-permutation-only + beta_0 test")

    # --- Thresholds ---
    p.add_argument("--n-null-graphs", type=int, default=20)
    p.add_argument("--n-mantel-perms", type=int, default=99,
                   help="Mantel permutations for corroboration (baseline only)")
    p.add_argument("--forward-floor", type=float, default=None)
    p.add_argument("--reconstruct-floor", type=float, default=0.30)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# =========================================================================
# Utility functions
# =========================================================================

def ensure_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def select_feature_columns(
    df: pd.DataFrame,
    target: str,
    region_id: str,
    event_id: Optional[str],
    feature_cols: Optional[list[str]],
    feature_regex: Optional[str],
    exclude_cols: Optional[list[str]],
) -> list[str]:
    reserved = {target, region_id, "fold"}
    if event_id:
        reserved.add(event_id)
    if exclude_cols:
        reserved |= set(exclude_cols)

    if feature_cols:
        cols = list(feature_cols)
    elif feature_regex:
        pattern = re.compile(feature_regex)
        cols = [c for c in df.columns if pattern.search(c)]
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in numeric if c not in reserved]

    cols = [c for c in cols if c not in reserved]
    ensure_columns(df, cols, "data")
    if not cols:
        raise ValueError(
            "No feature columns selected. Pass --feature-cols or --feature-regex."
        )
    return cols


# =========================================================================
# CSR W-matrix operations
# =========================================================================

def load_edges_as_csr(
    path: Path,
    region_order: list[str],
    src_col: str = "src",
    dst_col: str = "dst",
    weight_col: str = "weight",
) -> sparse.csr_matrix:
    edges = pd.read_csv(path)
    ensure_columns(edges, [src_col, dst_col, weight_col], "w-edges-csv")
    return _edges_df_to_csr(edges, region_order, src_col, dst_col, weight_col)


def _edges_df_to_csr(
    edges: pd.DataFrame,
    region_order: list[str],
    src_col: str = "src",
    dst_col: str = "dst",
    weight_col: str = "weight",
) -> sparse.csr_matrix:
    idx = {str(r): i for i, r in enumerate(region_order)}
    rows, cols, vals = [], [], []
    for row in edges.itertuples(index=False):
        src = str(getattr(row, src_col))
        dst = str(getattr(row, dst_col))
        if src not in idx or dst not in idx:
            continue
        rows.append(idx[src])
        cols.append(idx[dst])
        vals.append(float(getattr(row, weight_col)))
    n = len(region_order)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return _row_normalize(W)


def _row_normalize(W: sparse.csr_matrix) -> sparse.csr_matrix:
    W = W.tocsr(copy=True)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sums, dtype=float)
    nz = row_sums > 0
    inv[nz] = 1.0 / row_sums[nz]
    return sparse.diags(inv) @ W


def _neighbor_dict_to_csr(
    neighbors: dict[str, list[str]],
    region_order: list[str],
) -> sparse.csr_matrix:
    """Convert string-keyed neighbor dict (from S3 adjacency) to CSR."""
    idx = {str(r): i for i, r in enumerate(region_order)}
    rows, cols, vals = [], [], []
    for z, nbrs in neighbors.items():
        if str(z) not in idx:
            continue
        i = idx[str(z)]
        for nb in nbrs:
            if str(nb) not in idx:
                continue
            j = idx[str(nb)]
            rows.append(i)
            cols.append(j)
            vals.append(1.0)
    n = len(region_order)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return _row_normalize(W)


def _subset_csr(
    W: sparse.csr_matrix,
    old_order: list[str],
    new_order: list[str],
) -> sparse.csr_matrix:
    """Extract submatrix of W for regions in new_order (subset of old_order)."""
    old_idx = {r: i for i, r in enumerate(old_order)}
    keep = [old_idx[r] for r in new_order if r in old_idx]
    W_sub = W[np.ix_(keep, keep)]
    return _row_normalize(sparse.csr_matrix(W_sub))


# =========================================================================
# Graded permutation (self-loop-safe)
# =========================================================================

def partial_permute_w(
    W: sparse.csr_matrix,
    level: float,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    """Relabel floor(level*n) neighbor columns, re-zero diagonal, renormalize.

    level=0.0 returns W unchanged; level=1.0 is a full scramble.
    Re-zeroing the diagonal prevents a node being permuted onto itself,
    which would inject a self-loop into the spatial-lag operator.
    """
    n = W.shape[0]
    perm = np.arange(n)
    k = int(np.floor(level * n))
    if k >= 2:
        chosen = rng.choice(n, size=k, replace=False)
        shuffled = chosen.copy()
        for _ in range(16):
            rng.shuffle(shuffled)
            if np.all(shuffled != chosen):
                break
        perm[chosen] = shuffled

    Wp = W[:, perm].tocsr()
    Wp.setdiag(0.0)
    Wp.eliminate_zeros()
    return _row_normalize(Wp)


def build_permutation_schedule(
    levels: Sequence[float],
    n_seeds: int,
) -> list[tuple[float, int]]:
    """Cross corruption levels with seed offsets.

    Level 0.0 is the baseline anchor, handled separately.
    """
    return [(float(lv), int(s)) for lv in levels for s in range(n_seeds)]


# =========================================================================
# Spatial-lag feature recomputation
# =========================================================================

def add_spatial_lags(
    df: pd.DataFrame,
    W: sparse.csr_matrix,
    region_order: list[str],
    region_id: str,
    lag_cols: list[str],
    event_id: Optional[str],
) -> pd.DataFrame:
    """Append W-lagged columns via CSR matrix multiplication.

    For region-event data, lags are computed per event slice.
    """
    if not lag_cols:
        return df.copy()
    ensure_columns(df, [region_id] + lag_cols, "data")
    out = df.copy()
    region_to_pos = {str(r): i for i, r in enumerate(region_order)}

    def _process_group(g: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=region_order, columns=lag_cols, dtype=float)
        for rid, rows in g.groupby(region_id):
            rid_s = str(rid)
            if rid_s in region_to_pos:
                X.loc[rid_s, :] = rows[lag_cols].mean(numeric_only=True).values
        means = X.mean(axis=0, skipna=True).fillna(0.0)
        X = X.fillna(means)

        lagged = W @ X.to_numpy(dtype=float)
        lagged_df = pd.DataFrame(
            lagged, index=region_order,
            columns=[f"wlag_{c}" for c in lag_cols],
        )
        g2 = g.copy()
        for c in lag_cols:
            new_c = f"wlag_{c}"
            g2[new_c] = g2[region_id].astype(str).map(lagged_df[new_c]).astype(float)
        return g2

    if event_id and event_id in out.columns:
        pieces = [_process_group(g) for _, g in out.groupby(event_id, sort=False)]
        return pd.concat(pieces, axis=0).loc[out.index]
    return _process_group(out)


# =========================================================================
# Modeling and embeddings
# =========================================================================

def fit_predict_folds(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    fold_col: str,
    task: str,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Train HistGBDT on frozen folds, return (oof_predictions, forward_score)."""
    y = df[target].to_numpy(dtype=float)
    X = (df[feature_cols]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0)
         .to_numpy(dtype=float))
    folds = sorted(pd.unique(df[fold_col]))
    pred = np.full(len(df), np.nan, dtype=float)

    for fold in folds:
        train_mask = df[fold_col].to_numpy() != fold
        test_mask = ~train_mask
        if train_mask.sum() < 20 or test_mask.sum() < 5:
            continue

        if task == "regression":
            model = HistGradientBoostingRegressor(
                loss="squared_error", max_iter=300, max_depth=6,
                min_samples_leaf=10, learning_rate=0.05, random_state=seed,
            )
            model.fit(X[train_mask], y[train_mask])
            pred[test_mask] = model.predict(X[test_mask])
        else:
            model = HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, min_samples_leaf=10,
                learning_rate=0.05, random_state=seed,
            )
            model.fit(X[train_mask], y[train_mask])
            if task == "binary_classification" and len(model.classes_) == 2:
                pred[test_mask] = model.predict_proba(X[test_mask])[:, 1]
            else:
                pred[test_mask] = model.predict(X[test_mask])

    valid = ~np.isnan(pred)
    if valid.sum() < 3:
        raise RuntimeError("Not enough out-of-fold predictions produced.")

    if task == "regression":
        score = float(r2_score(y[valid], pred[valid]))
    elif task == "binary_classification":
        y_v = y[valid]
        score = float(roc_auc_score(y_v, pred[valid])) if len(np.unique(y_v)) >= 2 else float("nan")
    else:
        score = float(accuracy_score(y[valid], pred[valid]))

    return pred, score


def aggregate_region_embedding(
    df: pd.DataFrame,
    feature_cols: list[str],
    region_id: str,
    region_order: list[str],
) -> np.ndarray:
    """Mean feature vector per region, z-scored. Shape (n_regions, n_features)."""
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    agg = X.assign(__region=df[region_id].astype(str)).groupby("__region")[feature_cols].mean()
    agg = agg.reindex(region_order).fillna(agg.mean(axis=0)).fillna(0.0)
    Z = agg.to_numpy(dtype=float)
    if Z.shape[0] <= 1:
        return Z
    return StandardScaler().fit_transform(Z)


# =========================================================================
# kappa_spatial convenience (Moran's I)
# =========================================================================

def _compute_morans_i(
    values_by_region: np.ndarray,
    W: sparse.csr_matrix,
) -> float:
    """Moran's I on a region-level vector. Replace with yrsn in production."""
    x = np.asarray(values_by_region, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 3:
        return float("nan")
    Wm = W[mask][:, mask].tocsr()
    x = x[mask]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    s0 = float(Wm.sum())
    if denom <= 0 or s0 <= 0:
        return 0.0
    n = len(x)
    num = float(x @ (Wm @ x))
    return (n / s0) * (num / denom)


def compute_kappa_spatial_from_residuals(
    df: pd.DataFrame,
    residuals: np.ndarray,
    region_id: str,
    region_order: list[str],
    W_geo: sparse.csr_matrix,
) -> tuple[float, float]:
    """Convenience kappa_spatial = 1 - |Moran's I|. Returns (kappa, morans_i)."""
    tmp = pd.DataFrame({
        "region": df[region_id].astype(str).to_numpy(),
        "resid": residuals,
    })
    resid_by_region = (
        tmp.groupby("region")["resid"]
        .mean()
        .reindex(region_order)
        .to_numpy()
    )
    I = _compute_morans_i(resid_by_region, W_geo)
    if not np.isfinite(I):
        return float("nan"), float("nan")
    kappa = float(np.clip(1.0 - abs(I), 0.0, 1.0))
    return kappa, float(I)


# =========================================================================
# refit_and_embed adapter
# =========================================================================

def refit_and_embed(
    *,
    label: str,
    corruption_level: float,
    df_base: pd.DataFrame,
    W_lag: sparse.csr_matrix,
    W_geo: sparse.csr_matrix,
    coords: np.ndarray,
    region_order: list[str],
    base_feature_cols: list[str],
    lag_cols: list[str],
    target: str,
    region_id: str,
    event_id: Optional[str],
    fold_col: str,
    task: str,
    seed: int,
    n_baseline_trials: int,
    n_mantel_perms: int,
) -> RefitResult:
    """Rebuild spatial lags, refit HistGBDT, compute both kappas."""
    df = add_spatial_lags(
        df_base, W=W_lag, region_order=region_order,
        region_id=region_id, lag_cols=lag_cols, event_id=event_id,
    )

    feature_cols = list(base_feature_cols)
    feature_cols += [f"wlag_{c}" for c in lag_cols if f"wlag_{c}" in df.columns]
    feature_cols = sorted(dict.fromkeys(feature_cols))

    pred, forward_score = fit_predict_folds(
        df=df, feature_cols=feature_cols, target=target,
        fold_col=fold_col, task=task, seed=seed,
    )

    residuals = df[target].to_numpy(dtype=float) - pred
    kappa_spatial, morans_i = compute_kappa_spatial_from_residuals(
        df=df, residuals=residuals, region_id=region_id,
        region_order=region_order, W_geo=W_geo,
    )

    Z = aggregate_region_embedding(
        df=df, feature_cols=feature_cols,
        region_id=region_id, region_order=region_order,
    )

    # Domain module: Gabriel graph + crossings + baseline + corroboration
    sr = compute_kappa_reconstruct(
        Z, coords,
        n_baseline_trials=n_baseline_trials,
        n_mantel_perms=n_mantel_perms,
    )

    return RefitResult(
        label=label,
        corruption_level=corruption_level,
        forward_score=float(forward_score),
        kappa_spatial=float(kappa_spatial),
        morans_i=float(morans_i),
        kappa_reconstruct=sr.kappa_reconstruct,
        n_crossings=sr.n_crossings,
        n_crossings_baseline=sr.n_crossings_baseline,
        n_implied_edges=sr.n_gabriel_edges,
        stress=sr.stress,
        mantel_r=sr.mantel_r,
        coordinate_lift=sr.coordinate_lift_score,
        n_regions=sr.n_regions,
    )


# =========================================================================
# Discriminant tests (corrected -- not mean-of-OLS-residuals)
# =========================================================================

def discriminant_intercept_test(
    delta_reconstruct: np.ndarray,
    delta_spatial: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Test the intercept of d_recon ~ 1 + d_spatial < 0.

    The intercept is the predicted kappa_reconstruct drop at d_spatial = 0,
    i.e. the drop through a channel kappa_spatial does not cover.
    """
    y = np.asarray(delta_reconstruct, float)
    x = np.asarray(delta_spatial, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size
    if n < 3:
        return {"intercept": np.nan, "slope_spatial": np.nan,
                "t_intercept": np.nan, "p_one_sided": np.nan,
                "n_obs": n, "verdict": "INSUFFICIENT"}

    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    sigma2 = float(resid @ resid) / dof if dof > 0 else np.nan
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_b0 = float(np.sqrt(max(cov[0, 0], 0.0)))
    if se_b0 > 0:
        t0 = float(beta[0] / se_b0)
    else:
        t0 = -np.inf if beta[0] < 0 else np.inf
    p = float(student_t.cdf(t0, df=dof))
    verdict = "EARNS_SLOT" if (beta[0] < 0 and p < alpha) else "INCONCLUSIVE"
    return {
        "intercept": float(beta[0]),
        "slope_spatial": float(beta[1]),
        "t_intercept": t0,
        "p_one_sided": p,
        "n_obs": n,
        "verdict": verdict,
    }


def discriminant_ladder_test(
    delta_reconstruct: np.ndarray,
    delta_spatial: np.ndarray,
    level: np.ndarray,
    fold_id: Optional[np.ndarray] = None,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Regress d_recon ~ 1 + d_spatial + level, test gamma_level < 0.

    Bootstrap resamples over folds (when fold_id given) for cross-fold
    generalization, else over trials. The gamma_level coefficient captures
    the kappa_reconstruct response to corruption through the channel NOT
    already covered by kappa_spatial (Frisch-Waugh-Lovell).
    """
    y = np.asarray(delta_reconstruct, float)
    x = np.asarray(delta_spatial, float)
    lv = np.asarray(level, float)
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(lv)
    y, x, lv = y[m], x[m], lv[m]
    fid = None if fold_id is None else np.asarray(fold_id)[m]
    n = y.size
    if n < 4:
        return {"gamma_level": np.nan, "slope_spatial": np.nan,
                "gamma_ci": (np.nan, np.nan), "p_one_sided": np.nan,
                "n_obs": n, "verdict": "INSUFFICIENT"}

    def _fit(idx: np.ndarray) -> tuple[float, float]:
        X = np.column_stack([np.ones(idx.size), x[idx], lv[idx]])
        beta, *_ = np.linalg.lstsq(X, y[idx], rcond=None)
        return float(beta[1]), float(beta[2])

    slope_spatial, gamma = _fit(np.arange(n))

    rng = np.random.default_rng(seed)
    if fid is not None:
        groups = [np.where(fid == g)[0] for g in np.unique(fid)]
    else:
        groups = [np.array([i]) for i in range(n)]

    gammas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[p] for p in pick])
        _, gammas[b] = _fit(idx)

    lo = (1 - ci) / 2
    gamma_ci = (float(np.quantile(gammas, lo)), float(np.quantile(gammas, 1 - lo)))
    p = float(np.mean(gammas >= 0.0))
    verdict = "EARNS_SLOT" if gamma_ci[1] < 0.0 else "INCONCLUSIVE"
    return {
        "gamma_level": gamma,
        "slope_spatial": slope_spatial,
        "gamma_ci": gamma_ci,
        "p_one_sided": p,
        "n_obs": n,
        "verdict": verdict,
    }


# =========================================================================
# S3 data loading (--scenario shortcut)
# =========================================================================

def _load_scenario_from_s3(scenario: str) -> tuple[
    pd.DataFrame, list[str], np.ndarray, sparse.csr_matrix
]:
    """Load assembled parquet, folds, centroids, adjacency from S3."""
    from _coverage_common import load_processed_parquet

    s3 = get_s3_client()
    df = load_processed_parquet(s3, scenario)

    folds_key = f"folds/{scenario}_folds.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=folds_key)
    folds = pd.read_parquet(io.BytesIO(resp["Body"].read()))

    fold_col_candidates = [
        "fold_spatial_blocked", "fold",
    ]
    fold_col = next((c for c in fold_col_candidates if c in folds.columns), None)
    if fold_col is None:
        raise ValueError(f"No fold column found in {folds.columns.tolist()}")

    df["zcta_id"] = df["zcta_id"].astype(str)
    folds["zcta_id"] = folds["zcta_id"].astype(str)
    df = df.merge(folds[["zcta_id", "event", fold_col]], on=["zcta_id", "event"],
                  how="left")
    if fold_col != "fold":
        df["fold"] = df[fold_col]

    region_order = sorted(df["zcta_id"].unique())

    # Centroids
    key = "raw/geocertdb2026/zcta_features_labels.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    static = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    zcta_col = next(c for c in static.columns if "zcta" in c.lower())
    lat_col = next(c for c in static.columns if "lat" in c.lower())
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    lookup = {}
    for _, row in static.iterrows():
        zid = str(row[zcta_col])
        if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
            lookup[zid] = (float(row[lat_col]), float(row[lon_col]))
    coords = np.zeros((len(region_order), 2), dtype=np.float64)
    for i, zid in enumerate(region_order):
        if zid in lookup:
            coords[i] = lookup[zid]

    # Adjacency -> CSR
    adj_df = load_adjacency(s3)
    cols = adj_df.columns.tolist()
    pair_candidates = [
        ("zcta_from", "zcta_to"), ("zcta_id_1", "zcta_id_2"),
        ("source", "target"), ("src", "dst"),
    ]
    pair = next((c for c in pair_candidates if c[0] in cols and c[1] in cols), None)
    if pair is None:
        pair = (cols[0], cols[1])
    neighbors: dict[str, list[str]] = {}
    for _, row in adj_df.iterrows():
        u, v = str(row[pair[0]]), str(row[pair[1]])
        neighbors.setdefault(u, []).append(v)
        neighbors.setdefault(v, []).append(u)
    neighbors = {k: sorted(set(vs)) for k, vs in neighbors.items()}
    W = _neighbor_dict_to_csr(neighbors, region_order)

    return df, region_order, coords, W


# =========================================================================
# Main harness
# =========================================================================

def run_adversarial_harness(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)

    # ── Load data ──
    if args.scenario:
        if not _HAS_S3:
            raise RuntimeError("--scenario requires _coverage_common (S3 infra)")
        df, region_order, coords, W_geo = _load_scenario_from_s3(args.scenario)
        W_lag = W_geo.copy()
        base_feature_cols = select_feature_columns(
            df=df, target=args.target, region_id=args.region_id,
            event_id=args.event_id, feature_cols=args.feature_cols,
            feature_regex=args.feature_regex, exclude_cols=args.exclude_cols,
        )
    else:
        if not all([args.data_parquet, args.folds_csv, args.w_edges_csv,
                     args.coords_csv]):
            raise ValueError(
                "Provide --scenario OR all of --data-parquet, --folds-csv, "
                "--w-edges-csv, --coords-csv"
            )
        df = pd.read_parquet(args.data_parquet)
        folds = pd.read_csv(args.folds_csv)
        coords_df = pd.read_csv(args.coords_csv)

        ensure_columns(df, [args.region_id, args.target], "data-parquet")
        ensure_columns(folds, [args.region_id, "fold"], "folds-csv")
        ensure_columns(coords_df, [args.region_id], "coords-csv")

        coord_cols = [c for c in ["x", "y"] if c in coords_df.columns]
        if len(coord_cols) != 2:
            coord_cols = [c for c in ["lon", "lat"] if c in coords_df.columns]
        if len(coord_cols) != 2:
            raise ValueError("coords-csv must contain [x,y] or [lon,lat].")

        df[args.region_id] = df[args.region_id].astype(str)
        folds[args.region_id] = folds[args.region_id].astype(str)
        coords_df[args.region_id] = coords_df[args.region_id].astype(str)

        df = df.merge(folds[[args.region_id, "fold"]], on=args.region_id, how="left")
        if df["fold"].isna().any():
            missing = df.loc[df["fold"].isna(), args.region_id].unique()[:10]
            raise ValueError(f"Rows missing fold assignment. Examples: {missing}")

        region_order = sorted(
            set(df[args.region_id].astype(str)) & set(coords_df[args.region_id].astype(str))
        )
        if len(region_order) < 4:
            raise ValueError("Need at least 4 regions with both data and coordinates.")

        coords = (
            coords_df.set_index(args.region_id)
            .reindex(region_order)[coord_cols]
            .astype(float)
            .to_numpy()
        )
        W_geo = load_edges_as_csr(args.w_edges_csv, region_order)
        W_lag = W_geo.copy()

        base_feature_cols = select_feature_columns(
            df=df, target=args.target, region_id=args.region_id,
            event_id=args.event_id, feature_cols=args.feature_cols,
            feature_regex=args.feature_regex, exclude_cols=args.exclude_cols,
        )

    ensure_columns(df, args.spatial_lag_cols, "spatial-lag-cols in data")

    # Drop rows with NaN target (degenerate scenarios like NYC/Riverside)
    nan_target = df[args.target].isna()
    if nan_target.any():
        n_drop = int(nan_target.sum())
        log.warning("Dropping %d/%d rows with NaN target '%s'",
                    n_drop, len(df), args.target)
        df = df[~nan_target].reset_index(drop=True)
        if len(df) < 20:
            raise RuntimeError(
                f"Only {len(df)} rows remain after dropping NaN target -- "
                f"scenario too degenerate for adversarial harness"
            )
        # Recompute region_order, coords, and W after dropping NaN rows
        old_region_order = region_order
        region_order = sorted(df[args.region_id].astype(str).unique())
        if len(region_order) < len(old_region_order):
            # Rebuild coords array for surviving regions
            old_idx = {r: i for i, r in enumerate(old_region_order)}
            new_coords = np.zeros((len(region_order), 2), dtype=np.float64)
            for i, r in enumerate(region_order):
                if r in old_idx:
                    new_coords[i] = coords[old_idx[r]]
            coords = new_coords
            # Rebuild W matrices with new region_order
            W_geo = _subset_csr(W_geo, old_region_order, region_order)
            W_lag = W_geo.copy()

    out_dir = args.out_dir or Path(f"artifacts/adversarial_{args.scenario or 'local'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Adversarial kappa_reconstruct ===")
    log.info("  regions: %d  features: %d  lag_cols: %s",
             len(region_order), len(base_feature_cols), args.spatial_lag_cols)

    # ── Baseline ──
    log.info("Computing baseline...")
    baseline = refit_and_embed(
        label="baseline", corruption_level=0.0,
        df_base=df, W_lag=W_lag, W_geo=W_geo, coords=coords,
        region_order=region_order, base_feature_cols=base_feature_cols,
        lag_cols=args.spatial_lag_cols, target=args.target,
        region_id=args.region_id, event_id=args.event_id,
        fold_col="fold", task=args.task, seed=args.seed,
        n_baseline_trials=args.n_null_graphs,
        n_mantel_perms=args.n_mantel_perms,
    )
    log.info("  forward=%.4f  kappa_reconstruct=%.4f  kappa_spatial=%.4f",
             baseline.forward_score, baseline.kappa_reconstruct, baseline.kappa_spatial)

    gate_3b = gate_3b_decision(
        baseline.forward_score, baseline.kappa_reconstruct,
        forward_floor=args.forward_floor or 0.0,
        reconstruct_floor=args.reconstruct_floor,
    )
    log.info("  Gate 3B baseline: %s", gate_3b)

    # ── Build permutation schedule ──
    if args.test_mode == "ladder":
        schedule = build_permutation_schedule(args.corruption_levels, args.n_permutations)
    else:
        schedule = [(1.0, s) for s in range(args.n_permutations)]

    log.info("Running %d trials (%s mode)...", len(schedule), args.test_mode)

    # ── Run trials ──
    rows = [asdict(baseline)]
    for trial_idx, (level, seed_offset) in enumerate(schedule):
        W_perm = partial_permute_w(W_lag, level, rng)
        result = refit_and_embed(
            label=f"perm_{trial_idx:04d}",
            corruption_level=level,
            df_base=df, W_lag=W_perm, W_geo=W_geo, coords=coords,
            region_order=region_order, base_feature_cols=base_feature_cols,
            lag_cols=args.spatial_lag_cols, target=args.target,
            region_id=args.region_id, event_id=args.event_id,
            fold_col="fold", task=args.task,
            seed=args.seed + seed_offset + 1,
            n_baseline_trials=max(args.n_null_graphs // 2, 5),
            n_mantel_perms=0,  # skip Mantel for permutation trials
        )
        rows.append(asdict(result))
        if (trial_idx + 1) % 5 == 0 or trial_idx == len(schedule) - 1:
            log.info("  trial %d/%d: level=%.2f forward=%.4f kr=%.4f ks=%.4f",
                     trial_idx + 1, len(schedule), level,
                     result.forward_score, result.kappa_reconstruct,
                     result.kappa_spatial)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "adversarial_refit_results.csv", index=False)

    # ── Compute deltas ──
    trial_df = results_df[results_df["label"] != "baseline"].copy()
    trial_df["delta_kappa_reconstruct"] = (
        trial_df["kappa_reconstruct"] - baseline.kappa_reconstruct
    )
    trial_df["delta_kappa_spatial"] = (
        trial_df["kappa_spatial"] - baseline.kappa_spatial
    )
    trial_df.to_csv(out_dir / "adversarial_delta_results.csv", index=False)

    # ── Discriminant test ──
    d_recon = trial_df["delta_kappa_reconstruct"].to_numpy()
    d_spatial = trial_df["delta_kappa_spatial"].to_numpy()
    levels = trial_df["corruption_level"].to_numpy()

    if args.test_mode == "ladder":
        test_result = discriminant_ladder_test(
            d_recon, d_spatial, levels,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            alpha=args.alpha,
        )
        test_mode = "ladder"
    else:
        test_result = discriminant_intercept_test(
            d_recon, d_spatial, alpha=args.alpha,
        )
        test_mode = "intercept"

    log.info("=== Discriminant test (%s) ===", test_mode)
    for k, v in test_result.items():
        log.info("  %s: %s", k, v)

    gate_eligible = test_result["verdict"] == "EARNS_SLOT"

    # ── Summary ──
    summary = AdversarialSummary(
        baseline=baseline,
        n_trials=len(schedule),
        test_mode=test_mode,
        intercept=test_result.get("intercept"),
        intercept_t=test_result.get("t_intercept"),
        intercept_p=test_result.get("p_one_sided") if test_mode == "intercept" else None,
        gamma_level=test_result.get("gamma_level"),
        gamma_ci=test_result.get("gamma_ci"),
        gamma_p=test_result.get("p_one_sided") if test_mode == "ladder" else None,
        gate_3b_baseline=gate_3b,
        gate_eligible=gate_eligible,
        verdict=test_result["verdict"],
        commitment=(
            "FloodRSCT tasks are spatial-reasoning tasks: the geographic "
            "topology constrains the physically valid outcomes, so a "
            "representation that cannot recover that topology cannot reason "
            "about the task correctly."
        ),
    )

    summary_dict = asdict(summary)
    summary_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(out_dir / "adversarial_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=str)

    print(json.dumps(summary_dict, indent=2, default=str))

    # ── Optional S3 upload ──
    if args.upload and args.scenario and _HAS_S3:
        from _s3_result import upload_json_result
        s3 = get_s3_client()
        key = f"results/s035/adversarial_reconstruct_{args.scenario}.json"
        upload_json_result(s3, BUCKET, key, summary_dict)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)

    return summary_dict


def main() -> None:
    args = parse_args()
    run_adversarial_harness(args)


if __name__ == "__main__":
    main()
