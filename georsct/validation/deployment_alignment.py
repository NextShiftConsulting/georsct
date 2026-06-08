"""Deployment-alignment diagnostics and weighting.

Implements three weighting strategies for deployment-aligned validation:

1. **TWCV-lite** (``marginal_ratio_weights``): product-of-marginals, fast
   screening diagnostic. Correct only when descriptor marginals are independent.
2. **TWCV** (``rake_weights``): full iterative proportional fitting (IPF)
   following Brenning & Suesse (2026). Converges to joint marginal consistency
   and is the registered primary estimator.
3. **IWCV** (``compute_iwcv_weights``): logistic density-ratio estimation
   following Brenning & Suesse (2026). A complementary comparison estimator.

Certificate signals (ESS, max relative weight, dropped fraction,
no-reference fraction, missing bins, distribution gap) are emitted via
``alignment_summary`` -> PASS/WARN/FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .task_descriptors import SENTINEL_BIN


@dataclass(frozen=True)
class AlignmentGates:
    """Pre-committed certificate gates for deployment alignment.

    Thresholds determine PASS/WARN/FAIL and so are registration parameters: fix
    them before computing results. Each is documented with its derivation.

    min_ess_fraction (PASS floor):
        ESS/n = 1 / DEFF (Kish design effect). 0.35 <-> DEFF ~= 2.86: weighting
        roughly triples estimator variance, the conventional "seriously
        degraded" line.
    hard_min_ess_fraction (FAIL floor):
        0.15 <-> DEFF ~= 6.7; below this the estimate is dominated by a few tasks
        and is not reportable in any form.
    max_weight_x_uniform:
        n * max(w_norm): largest single-task influence vs its uniform share.
        Decoupled from the per-descriptor ratio clip.
    max_missing_bins (PASS requires 0):
        A deployment bin with zero validation coverage is an uncloseable gap.
    max_dropped_fraction (FAIL above):
        Fraction of validation mass outside the deployment support; above this
        the folds barely overlap the deployment domain.
    max_no_reference_fraction (FAIL above):
        Fraction of the deployment domain with NO observed/reference support
        nearby (pure extrapolation). 0.10 = at most 10% of deployment may be
        unsupported before the certificate fails closed.
    max_mean_js (PASS ceiling):
        Screening default ONLY. Prefer js_null_threshold() at the SAME unit being
        certified (fold/group). JS uses natural log, range [0, ln 2].
    """

    min_ess_fraction: float = 0.35          # DEFF ~= 2.86
    hard_min_ess_fraction: float = 0.15     # DEFF ~= 6.7 (FAIL below)
    max_weight_x_uniform: float = 10.0
    max_missing_bins: int = 0
    max_dropped_fraction: float = 0.50
    max_no_reference_fraction: float = 0.10
    max_mean_js: float = 0.20               # screening default; prefer js_null_threshold
    registration_tag: str = "deploy_align_gates_v1"


def normalize_weights(w: np.ndarray, zero_policy: str = "uniform") -> np.ndarray:
    """Normalize nonnegative finite weights to sum to one.

    zero_policy controls the degenerate all-zero case:
      "uniform" -> return uniform weights (safe for generic math).
      "zeros"   -> return all zeros (correct for deployment alignment: a total
                   support failure must NOT masquerade as a representative
                   uniform sample).
    """

    w = np.asarray(w, dtype=float)
    if len(w) == 0:
        return w
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    total = float(w.sum())
    if total <= 0:
        if zero_policy == "zeros":
            return np.zeros_like(w, dtype=float)
        return np.ones_like(w, dtype=float) / len(w)
    return w / total


def _apply_shrinkage(w: np.ndarray, shrinkage: float) -> np.ndarray:
    """Shrink weights toward uniform-over-survivors.

    Follows Brenning & Suesse (2026) shrink_weights(): convex combination
    (1 - lambda) * w_norm + lambda * uniform_kept. Shrinkage cannot
    resurrect a dropped (zero-weight) row.
    """

    kept = w > 0
    if not kept.any() or shrinkage <= 0:
        return w
    uniform_kept = np.where(kept, 1.0 / int(kept.sum()), 0.0)
    w_norm = normalize_weights(w, zero_policy="zeros")
    return (1.0 - shrinkage) * w_norm + shrinkage * uniform_kept


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size for (already nonnegative) weights."""

    w = np.asarray(w, dtype=float)
    if len(w) == 0:
        return 0.0
    total = float(w.sum())
    if total <= 0:
        return 0.0
    w = w / total
    denom = float(np.sum(w**2))
    return 0.0 if denom <= 0 else float(1.0 / denom)


def distribution(df: pd.DataFrame, col: str) -> pd.Series:
    """Normalized discrete distribution for a binned descriptor column.

    SENTINEL_BIN is a real category here (no-reference / unbinnable) and is
    intentionally NOT dropped.
    """

    if col not in df.columns:
        return pd.Series(dtype=float)
    s = df[col].dropna()  # genuine NA only; SENTINEL_BIN survives
    if s.empty:
        return pd.Series(dtype=float)
    return s.value_counts(normalize=True).sort_index()


def js_divergence(p: pd.Series, q: pd.Series, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (natural log) between two discrete dists."""

    idx = p.index.union(q.index)
    if len(idx) == 0:
        return np.nan
    p_arr = p.reindex(idx, fill_value=0.0).to_numpy(dtype=float) + eps
    q_arr = q.reindex(idx, fill_value=0.0).to_numpy(dtype=float) + eps
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    m = 0.5 * (p_arr + q_arr)
    kl_pm = np.sum(p_arr * np.log(p_arr / m))
    kl_qm = np.sum(q_arr * np.log(q_arr / m))
    return float(0.5 * (kl_pm + kl_qm))


def no_reference_fraction(df: pd.DataFrame, support_col: str) -> float:
    """Fraction of rows flagged SENTINEL_BIN in the support (distance) column."""

    if support_col is None or support_col not in df.columns:
        return float("nan")
    s = df[support_col].dropna()
    if s.empty:
        return float("nan")
    return float((s == SENTINEL_BIN).mean())


def alignment_gaps(
    validation_df: pd.DataFrame,
    target_df: pd.DataFrame,
    bin_cols: Iterable[str],
) -> pd.DataFrame:
    """Descriptor-by-descriptor task-distribution mismatch summary."""

    rows: list[dict] = []
    for col in bin_cols:
        p_val = distribution(validation_df, col)
        p_tgt = distribution(target_df, col)
        target_bins = set(p_tgt.index.dropna())
        validation_bins = set(p_val.index.dropna())
        missing_bins = sorted(target_bins - validation_bins)
        rows.append(
            {
                "descriptor": col,
                "validation_bins": len(validation_bins),
                "target_bins": len(target_bins),
                "missing_bins": len(missing_bins),
                "missing_bin_values": ",".join(map(str, missing_bins)),
                "js_divergence": js_divergence(p_val, p_tgt),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Weighting strategy 1: TWCV-lite (product of marginals)
# ---------------------------------------------------------------------------

def marginal_ratio_weights(
    validation_df: pd.DataFrame,
    target_df: pd.DataFrame,
    bin_cols: Iterable[str],
    clip: float = 10.0,
    shrinkage: float = 0.20,
) -> np.ndarray:
    """Stable TWCV-lite marginal ratio weights.

    Per binned descriptor, factor = P_target(bin) / P_validation(bin):
      * bin in both        -> ratio, bounded to [1/clip, clip] for variance control
      * bin in validation only (P_target = 0) -> 0.0: task is NOT in the
        deployment domain, so it is DROPPED, not up-weighted.
      * bin in target only (P_validation = 0) -> coverage gap; surfaced by
        alignment_gaps/missing_bins, never as a weight here.
    Survivors are shrunk toward uniform-over-survivors (shrinkage cannot
    resurrect a dropped row). If every row is out of support, returns all zeros
    (zero_policy="zeros") so a total support failure does not become uniform.

    This is the one-iteration special case of ``rake_weights``. It is correct
    only when descriptor marginals are independent; use ``rake_weights`` as the
    registered primary estimator.
    """

    n = len(validation_df)
    if n == 0:
        return np.array([], dtype=float)

    w = np.ones(n, dtype=float)
    in_support = np.ones(n, dtype=bool)
    for col in bin_cols:
        if col not in validation_df.columns or col not in target_df.columns:
            continue
        p_val = distribution(validation_df, col)
        p_tgt = distribution(target_df, col)
        ratios: dict[object, float] = {}
        for bin_value in p_val.index:
            p_validation = float(p_val.get(bin_value, 0.0))
            p_target = float(p_tgt.get(bin_value, 0.0))
            ratios[bin_value] = (p_target / p_validation) if p_validation > 0 else 0.0
        factor = validation_df[col].map(ratios).to_numpy(dtype=float)
        factor = np.where(np.isfinite(factor), factor, 0.0)
        support = factor > 0
        if clip and clip > 1:
            factor = np.where(support, np.clip(factor, 1.0 / clip, clip), 0.0)
        in_support &= support
        w *= factor

    w = np.where(in_support, w, 0.0)
    if shrinkage > 0:
        w = _apply_shrinkage(w, shrinkage)

    return normalize_weights(w, zero_policy="zeros")


# ---------------------------------------------------------------------------
# Weighting strategy 2: Full IPF raking (Brenning & Suesse 2026)
# ---------------------------------------------------------------------------

def rake_weights(
    validation_df: pd.DataFrame,
    target_df: pd.DataFrame,
    bin_cols: Iterable[str],
    clip: float = 10.0,
    shrinkage: float = 0.20,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, bool, int]:
    """Full IPF raking matching validation marginals to target marginals.

    Translates Brenning & Suesse (2026) ``rake_weights()`` from R to Python.
    ``marginal_ratio_weights`` is the one-iteration special case; this iterates
    until all marginals are jointly consistent.

    Returns
    -------
    weights : np.ndarray
        Normalized weight vector summing to 1 (or all zeros on total failure).
    converged : bool
        Whether IPF converged within max_iter.
    iterations : int
        Number of iterations performed.
    """

    bin_cols = list(bin_cols)
    n = len(validation_df)
    if n == 0:
        return np.array([], dtype=float), True, 0

    target_dists = {col: distribution(target_df, col) for col in bin_cols
                    if col in target_df.columns}
    if not target_dists:
        return normalize_weights(np.ones(n), zero_policy="uniform"), True, 0

    w = np.ones(n, dtype=float)

    for iteration in range(1, max_iter + 1):
        w_old = w.copy()

        for col in bin_cols:
            if col not in validation_df.columns or col not in target_dists:
                continue
            p_target = target_dists[col]
            vals = validation_df[col].to_numpy()
            all_levels = sorted(set(vals) | set(p_target.index))
            total_w = w.sum()
            if total_w <= 0:
                break

            for lev in all_levels:
                mask = vals == lev
                current_total = float(w[mask].sum())
                target_total = total_w * float(p_target.get(lev, 0.0))

                if current_total > 0 and target_total > 0:
                    factor = target_total / current_total
                    if clip > 1:
                        factor = np.clip(factor, 1.0 / clip, clip)
                    w[mask] *= factor
                elif current_total > 0 and target_total <= 0:
                    w[mask] = 0.0

        rel_change = np.max(np.abs(w - w_old) / np.maximum(np.abs(w_old), 1e-12))
        if rel_change < tol:
            if shrinkage > 0:
                w = _apply_shrinkage(w, shrinkage)
            return normalize_weights(w, zero_policy="zeros"), True, iteration

    converged = rel_change < tol
    if shrinkage > 0:
        w = _apply_shrinkage(w, shrinkage)
    return normalize_weights(w, zero_policy="zeros"), converged, max_iter


# ---------------------------------------------------------------------------
# Weighting strategy 3: IWCV — logistic density-ratio (Brenning & Suesse 2026)
# ---------------------------------------------------------------------------

def compute_iwcv_weights(
    validation_df: pd.DataFrame,
    target_df: pd.DataFrame,
    iwcv_vars: Iterable[str],
    shrinkage: float = 0.0,
    clip_prob: tuple[float, float] = (0.001, 0.999),
    C: float = 1.0,
) -> np.ndarray:
    """Importance-weighted CV via logistic density-ratio estimation.

    Translates Brenning & Suesse (2026) ``compute_iwcv_weights()`` from R.
    Pools validation (label=0) and deployment (label=1) rows, fits a weighted
    L2-regularized logistic classifier, returns p/(1-p) as density-ratio
    weights.

    Parameters
    ----------
    validation_df, target_df:
        DataFrames with the iwcv_vars columns (continuous, not binned).
    iwcv_vars:
        Feature columns for the density-ratio model.
    shrinkage:
        Convex shrinkage toward uniform after normalization.
    clip_prob:
        (min, max) clipping bounds for predicted probabilities before
        converting to density ratios, following Brenning's min_prob/max_prob.
    C:
        Inverse regularization strength for LogisticRegression.

    Returns
    -------
    Normalized weight vector (sums to 1).
    """

    from sklearn.linear_model import LogisticRegression

    iwcv_vars = list(iwcv_vars)
    common = [v for v in iwcv_vars
              if v in validation_df.columns and v in target_df.columns]
    if not common:
        raise ValueError("No common IWCV variables available")

    X_val = validation_df[common].to_numpy(dtype=float)
    X_tgt = target_df[common].to_numpy(dtype=float)

    X = np.vstack([X_val, X_tgt])
    y = np.concatenate([np.zeros(len(X_val)), np.ones(len(X_tgt))])
    sample_weight = np.concatenate([
        np.full(len(X_val), len(X_tgt) / max(len(X_val), 1)),
        np.ones(len(X_tgt)),
    ])

    finite = np.all(np.isfinite(X), axis=1)
    if finite.sum() < 10:
        return normalize_weights(np.ones(len(X_val)), zero_policy="uniform")

    model = LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs")
    model.fit(X[finite], y[finite], sample_weight=sample_weight[finite])

    p = model.predict_proba(X_val)[:, 1]
    p = np.clip(p, clip_prob[0], clip_prob[1])
    w = p / (1.0 - p)

    w = normalize_weights(w, zero_policy="uniform")
    if shrinkage > 0:
        w = _apply_shrinkage(w, shrinkage)
        w = normalize_weights(w, zero_policy="uniform")

    return w


# ---------------------------------------------------------------------------
# Null-threshold calibration
# ---------------------------------------------------------------------------

def js_null_threshold(
    target_df: pd.DataFrame,
    n_validation: int,
    bin_cols: Iterable[str],
    n_perm: int = 200,
    quantile: float = 0.95,
    seed: int = 0,
) -> float:
    """Null threshold for mean JS under perfect alignment, at size n_validation.

    Draws n_validation rows (with replacement) FROM the target n_perm times and
    returns the given quantile of mean-JS. IMPORTANT: pass the size of the unit
    actually being certified (a single fold or scenario group), not the full
    validation table, or the gate will not match the decision unit.
    """

    bin_cols = list(bin_cols)
    rng = np.random.default_rng(seed)
    full = {c: distribution(target_df, c) for c in bin_cols}
    m = len(target_df)
    if m == 0 or n_validation <= 0:
        return float("nan")

    idx = np.arange(m)
    stats: list[float] = []
    for _ in range(n_perm):
        draw = target_df.iloc[rng.choice(idx, size=min(n_validation, m), replace=True)]
        js = [js_divergence(distribution(draw, c), full[c]) for c in bin_cols]
        js = [v for v in js if np.isfinite(v)]
        if js:
            stats.append(float(np.mean(js)))
    return float(np.quantile(stats, quantile)) if stats else float("nan")


# ---------------------------------------------------------------------------
# Certificate summary
# ---------------------------------------------------------------------------

def alignment_summary(
    validation_df: pd.DataFrame,
    target_df: pd.DataFrame,
    bin_cols: Iterable[str],
    weights: np.ndarray,
    gates: AlignmentGates = AlignmentGates(),
    support_col: str | None = None,
) -> dict:
    """Certificate summary for a validation-task distribution.

    Decision logic (explicit hard-stops, then soft gates):
      FAIL if any hard-stop: total support failure (all weights zero), a
           deployment coverage gap (missing_bins > 0), ESS below the hard floor,
           dropped mass above max_dropped_fraction, or deployment
           no-reference fraction above max_no_reference_fraction.
      PASS if every soft gate holds and no hard-stop fires.
      WARN otherwise (sensitivity check, not a corrected truth).

    support_col: the distance bin column (e.g. 'nearest_train_km_bin'); when
    given, the target's no-reference fraction is computed and gated.
    """

    bin_cols = list(bin_cols)
    gaps = alignment_gaps(validation_df, target_df, bin_cols)
    missing_bins = int(gaps["missing_bins"].sum()) if not gaps.empty else 0
    mean_js = float(gaps["js_divergence"].mean()) if not gaps.empty else float("nan")
    tgt_no_ref = no_reference_fraction(target_df, support_col) if support_col else float("nan")
    val_no_ref = no_reference_fraction(validation_df, support_col) if support_col else float("nan")

    w = normalize_weights(weights, zero_policy="zeros")
    total_support = float(np.sum(w))
    n = max(len(validation_df), 1)

    if total_support <= 0:
        return {
            "n_validation": int(len(validation_df)),
            "n_target": int(len(target_df)),
            "ess": 0.0,
            "ess_fraction": 0.0,
            "max_weight_x_uniform": float("nan"),
            "dropped_fraction": 1.0,
            "target_no_reference_fraction": tgt_no_ref,
            "validation_no_reference_fraction": val_no_ref,
            "missing_bins": missing_bins,
            "mean_js_divergence": mean_js,
            "gates_tag": gates.registration_tag,
            "alignment_decision": "FAIL",
        }

    ess = effective_sample_size(w)
    ess_fraction = ess / n
    max_weight_x_uniform = float(np.max(w) * len(w))
    dropped_fraction = float(np.mean(w <= 0))

    hard_fail = (
        missing_bins > gates.max_missing_bins
        or ess_fraction < gates.hard_min_ess_fraction
        or dropped_fraction > gates.max_dropped_fraction
        or (np.isfinite(tgt_no_ref) and tgt_no_ref > gates.max_no_reference_fraction)
    )
    soft_pass = (
        ess_fraction >= gates.min_ess_fraction
        and max_weight_x_uniform <= gates.max_weight_x_uniform
        and (np.isnan(mean_js) or mean_js <= gates.max_mean_js)
    )

    decision = "FAIL" if hard_fail else ("PASS" if soft_pass else "WARN")

    return {
        "n_validation": int(len(validation_df)),
        "n_target": int(len(target_df)),
        "ess": float(ess),
        "ess_fraction": float(ess_fraction),
        "max_weight_x_uniform": max_weight_x_uniform,
        "dropped_fraction": dropped_fraction,
        "target_no_reference_fraction": tgt_no_ref,
        "validation_no_reference_fraction": val_no_ref,
        "missing_bins": missing_bins,
        "mean_js_divergence": mean_js,
        "gates_tag": gates.registration_tag,
        "alignment_decision": decision,
    }
