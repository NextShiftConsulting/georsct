"""
georsct.analysis.variance_stack

Explicit three-layer variance-control decomposition for deployment-aligned CV.

The paper (Brenning & Suesse 2026) sells one mechanism: reweight validation
losses so the validation-task distribution matches deployment. To make richer
variants converge they lean on a stack of variance-control moves that sit
OUTSIDE the unbiasedness derivation. This module makes that stack explicit
as a named, ordered pipeline where each layer carries its own diagnostic,
its own bias-variance ledger, and an on/off switch so convergence is
attributable to each one separately.

    Layer 0  COVERAGE   (precondition / positivity gate)
        Buffered task generator must cover the deployment-task support.
        Weighting cannot reach tasks the validation set never visited.
        Bias traded:      none (gate, not estimator).
        Failure scalar:   max_uncovered_target_mass -> FAIL_COVERAGE

    Layer 1  MATCHING   (calibration weighting)
        Two estimators on identical bins:
          - "product": product of independent marginal ratios (TWCV-lite).
          - "raking":  IPF matching marginals jointly.
        The falsifiable signature is residual_margin_gap (TV distance
        between achieved and target margins). Product leaves it large
        under descriptor correlation; raking drives it to ~0.
        Failure scalar:   residual_margin_gap -> WARN_MATCH

    Layer 2  SHRINKAGE  (regularization toward uniform)
        w <- (1-lambda)*w + lambda*uniform. DELIBERATELY violates the
        calibration constraint to buy ESS. This is a bias-for-variance
        trade, not part of the unbiasedness story.
        Ledger:           shrinkage_delta_ess (>0 = variance bought)
                          shrinkage_delta_margin_gap (>0 = bias paid)
        Failure scalar:   ess_fraction / max_weight_x_uniform -> WARN_CONCENTRATION

Certificate mapping: Layer 1/2 instability fields (ess_fraction,
max_weight_x_uniform, residual_margin_gap, shrinkage ledger) are turbulence
(sigma) signals. max_uncovered_target_mass is a support / N_ceiling signal.
Report them as certificate fields, do not tune them silently.

This module delegates all weighting math to georsct.validation (the production
implementation). It adds diagnostic decomposition, not duplicate algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from georsct.validation.deployment_alignment import (
    alignment_gaps,
    distribution,
    effective_sample_size,
    marginal_ratio_weights,
    normalize_weights,
    rake_weights,
    _apply_shrinkage,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StackConfig:
    """One configuration of the variance-control stack.

    Flags act as ablation switches -- turn layers on/off to attribute
    convergence to each one separately.

    Args:
        use_coverage_gate: Whether Layer 0 gate is active.
        matching: Weighting strategy for Layer 1.
        shrinkage_lambda: Pull toward uniform (0.0 = off).
        clip: High-side cap for marginal ratio weights.
        rake_max_iter: Max IPF iterations for raking.
        rake_tol: Convergence tolerance for raking.
        coverage_mass_tol: Max deployment mass left uncovered before
            FAIL_COVERAGE.
        min_ess_fraction: ESS/n floor for PASS.
        hard_min_ess_fraction: ESS/n floor below which FAIL_ESS.
        max_residual_gap: TV gap above which matching is suspect.
        max_weight_x_uniform: Max single-task influence vs uniform share.
    """

    use_coverage_gate: bool = True
    matching: Literal["none", "product", "raking"] = "raking"
    shrinkage_lambda: float = 0.0
    clip: float = 10.0
    rake_max_iter: int = 500
    rake_tol: float = 1e-6
    coverage_mass_tol: float = 0.02
    min_ess_fraction: float = 0.35
    hard_min_ess_fraction: float = 0.15
    max_residual_gap: float = 0.05
    max_weight_x_uniform: float = 10.0


# ---------------------------------------------------------------------------
# Layer 0 -- COVERAGE (positivity gate)
# ---------------------------------------------------------------------------

def coverage_report(
    validation_df: pd.DataFrame,
    target_df: pd.DataFrame,
    bin_cols: list[str],
) -> pd.DataFrame:
    """Per-descriptor coverage diagnostics.

    Returns one row per descriptor with target bins, covered bins,
    uncovered bins, and uncovered target mass.
    """
    rows = []
    for col in bin_cols:
        p_tgt = distribution(target_df, col)
        p_val = distribution(validation_df, col)
        target_bins = set(p_tgt.index)
        val_bins = set(p_val.index)
        uncovered = {
            b: float(p_tgt[b]) for b in target_bins if b not in val_bins
        }
        rows.append({
            "descriptor": col,
            "target_bins": len(target_bins),
            "covered_bins": sum(1 for b in target_bins if b in val_bins),
            "uncovered_bins": len(uncovered),
            "uncovered_target_mass": float(sum(uncovered.values())),
        })
    return pd.DataFrame(rows)


def max_uncovered_target_mass(cov_df: pd.DataFrame) -> float:
    """Worst-case uncovered deployment mass across descriptors."""
    if cov_df.empty:
        return 0.0
    return float(cov_df["uncovered_target_mass"].max())


# ---------------------------------------------------------------------------
# Layer 1 diagnostic -- RESIDUAL MARGIN GAP
# ---------------------------------------------------------------------------

def residual_margin_gap(
    weights: np.ndarray,
    validation_df: pd.DataFrame,
    target_df: pd.DataFrame,
    bin_cols: list[str],
) -> float:
    """Worst-case TV distance between achieved and target margins.

    0.0 means the weighted validation margins reproduce the deployment
    margins exactly. This is the falsifiable signature of Layer 1:
    'product' leaves it large under descriptor correlation; 'raking'
    drives it to ~0.
    """
    if len(validation_df) == 0:
        return float("nan")

    w = normalize_weights(weights)
    worst = 0.0

    for col in bin_cols:
        p_tgt = distribution(target_df, col)
        vals = validation_df[col]

        all_levels = sorted(set(vals.dropna()) | set(p_tgt.index))

        # Achieved marginals under current weights
        achieved = {}
        for lev in all_levels:
            mask = (vals == lev).to_numpy()
            achieved[lev] = float(w[mask].sum())

        # TV distance for this descriptor
        tv = 0.5 * sum(
            abs(achieved.get(lev, 0.0) - float(p_tgt.get(lev, 0.0)))
            for lev in all_levels
        )
        worst = max(worst, tv)

    return float(worst)


# ---------------------------------------------------------------------------
# Layer 2 diagnostic -- SHRINKAGE LEDGER
# ---------------------------------------------------------------------------

def shrinkage_ledger(
    w_pre: np.ndarray,
    w_post: np.ndarray,
    validation_df: pd.DataFrame,
    target_df: pd.DataFrame,
    bin_cols: list[str],
) -> dict:
    """Track what shrinkage bought (ESS) and what it cost (margin gap).

    Args:
        w_pre: Weights before shrinkage.
        w_post: Weights after shrinkage.
        validation_df: Binned validation dataframe.
        target_df: Binned target dataframe.
        bin_cols: Descriptor bin columns.

    Returns:
        Dict with shrinkage_delta_ess (>0 = variance bought) and
        shrinkage_delta_margin_gap (>0 = bias paid).
    """
    ess_pre = effective_sample_size(w_pre)
    ess_post = effective_sample_size(w_post)
    gap_pre = residual_margin_gap(w_pre, validation_df, target_df, bin_cols)
    gap_post = residual_margin_gap(w_post, validation_df, target_df, bin_cols)
    return {
        "shrinkage_delta_ess": float(ess_post - ess_pre),
        "shrinkage_delta_margin_gap": float(gap_post - gap_pre),
    }


# ---------------------------------------------------------------------------
# Stack orchestrator
# ---------------------------------------------------------------------------

def _stack_status(cert: dict, cfg: StackConfig) -> str:
    """Determine typed status from certificate fields."""
    if (cfg.use_coverage_gate
            and cert["max_uncovered_target_mass"] > cfg.coverage_mass_tol):
        return "FAIL_COVERAGE"

    if cert.get("rake_degenerate", False):
        return "FAIL_DEGENERATE"

    ess_frac = cert["ess_fraction"]

    if np.isfinite(ess_frac) and ess_frac < cfg.hard_min_ess_fraction:
        return "FAIL_ESS"

    gap = cert["residual_margin_gap"]
    if np.isfinite(gap) and gap > cfg.max_residual_gap:
        return "WARN_MATCH"

    if np.isfinite(ess_frac) and ess_frac < cfg.min_ess_fraction:
        return "WARN_CONCENTRATION"

    mwxu = cert["max_weight_x_uniform"]
    if np.isfinite(mwxu) and mwxu > cfg.max_weight_x_uniform:
        return "WARN_CONCENTRATION"

    return "PASS"


def build_weights_layered(
    validation_df: pd.DataFrame,
    target_df: pd.DataFrame,
    bin_cols: list[str],
    cfg: StackConfig = StackConfig(),
) -> tuple[np.ndarray, dict]:
    """Build weights by applying the stack in order, recording each layer.

    Returns (weights, certificate). The certificate is self-describing:
    it records which layers were active and the failure scalar for each.

    Args:
        validation_df: Binned validation dataframe.
        target_df: Binned target/deployment dataframe.
        bin_cols: Binned descriptor columns present in both dataframes.
        cfg: Stack configuration (ablation switches).

    Returns:
        Tuple of (normalized weight vector, certificate dict).
    """
    n = len(validation_df)
    cert: dict = {
        "n_validation": int(n),
        "layer0_coverage_gate": cfg.use_coverage_gate,
        "layer1_matching": cfg.matching,
        "layer2_shrinkage_lambda": float(cfg.shrinkage_lambda),
    }

    # Layer 0 -- coverage (diagnostic/gate only; does not alter weights)
    cov = coverage_report(validation_df, target_df, bin_cols)
    cert["max_uncovered_target_mass"] = max_uncovered_target_mass(cov)

    # Layer 1 -- matching (delegates to existing deployment_alignment)
    if cfg.matching == "none":
        w = normalize_weights(np.ones(n)) if n else np.array([], dtype=float)

    elif cfg.matching == "product":
        w = marginal_ratio_weights(
            validation_df, target_df, bin_cols,
            clip=cfg.clip, shrinkage=0.0,
        )

    elif cfg.matching == "raking":
        w, converged, iterations = rake_weights(
            validation_df, target_df, bin_cols,
            clip=cfg.clip, shrinkage=0.0,
            max_iter=cfg.rake_max_iter, tol=cfg.rake_tol,
        )
        cert["rake_iterations"] = iterations
        cert["rake_converged"] = converged
        # Detect degenerate raking (all weights went to zero)
        cert["rake_degenerate"] = bool(np.sum(w) <= 0) if n else False

    else:
        raise ValueError(f"unknown matching: {cfg.matching!r}")

    cert["residual_margin_gap_pre_shrinkage"] = (
        residual_margin_gap(w, validation_df, target_df, bin_cols) if n else float("nan")
    )

    # Layer 2 -- shrinkage (uses existing _apply_shrinkage)
    if cfg.shrinkage_lambda > 0 and n > 0:
        w_pre = w.copy()
        w = _apply_shrinkage(w, cfg.shrinkage_lambda)
        w = normalize_weights(w, zero_policy="zeros")
        cert.update(
            shrinkage_ledger(w_pre, w, validation_df, target_df, bin_cols),
        )
    else:
        cert["shrinkage_delta_ess"] = 0.0
        cert["shrinkage_delta_margin_gap"] = 0.0

    # Final diagnostics
    cert["ess"] = effective_sample_size(w) if n else float("nan")
    cert["ess_fraction"] = (cert["ess"] / n) if n else float("nan")
    cert["max_weight_x_uniform"] = (
        float(np.max(w) * n) if n else float("nan")
    )
    cert["residual_margin_gap"] = (
        residual_margin_gap(w, validation_df, target_df, bin_cols) if n else float("nan")
    )
    cert["stack_status"] = _stack_status(cert, cfg)

    return w, cert


# ---------------------------------------------------------------------------
# Cumulative ladder -- attribute convergence layer by layer
# ---------------------------------------------------------------------------

#: Default ladder: each row turns on exactly one more layer.
DEFAULT_LADDER: list[tuple[str, StackConfig]] = [
    ("0_unweighted", StackConfig(
        use_coverage_gate=False, matching="none", shrinkage_lambda=0.0,
    )),
    ("1_coverage_gate", StackConfig(
        use_coverage_gate=True, matching="none", shrinkage_lambda=0.0,
    )),
    ("2_marginal_product", StackConfig(
        use_coverage_gate=True, matching="product", shrinkage_lambda=0.0,
    )),
    ("3_marginal_raking", StackConfig(
        use_coverage_gate=True, matching="raking", shrinkage_lambda=0.0,
    )),
    ("4_raking_shrunk", StackConfig(
        use_coverage_gate=True, matching="raking", shrinkage_lambda=0.2,
    )),
]


def weighted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute weighted RMSE, MAE, and weighted mean for certificates.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        weights: Normalized weight vector (or None for uniform).

    Returns:
        Dict with rmse, mae, weighted_mean_true, weighted_mean_pred.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if weights is None:
        w = np.full(len(y_true), 1.0 / max(len(y_true), 1))
    else:
        w = normalize_weights(weights)

    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.sum(w * residuals**2)))
    mae = float(np.sum(w * np.abs(residuals)))
    wmean_true = float(np.sum(w * y_true))
    wmean_pred = float(np.sum(w * y_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "weighted_mean_true": wmean_true,
        "weighted_mean_pred": wmean_pred,
    }


def decompose_stack(
    validation_df: pd.DataFrame,
    target_df: pd.DataFrame,
    bin_cols: list[str],
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    ladder: list[tuple[str, StackConfig]] | None = None,
    identity: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Run the cumulative ladder and return one row per configuration.

    delta_* columns attribute how much each layer moved the estimate.
    The diagnostic columns say WHETHER each move was legitimate alignment
    (residual_margin_gap dropped) or regularization artifact (ESS rose,
    gap rose).

    Args:
        validation_df: Binned validation dataframe.
        target_df: Binned target/deployment dataframe.
        bin_cols: Binned descriptor columns.
        y_true: Ground truth (optional; enables metric columns).
        y_pred: Predictions (optional; enables metric columns).
        ladder: Cumulative stack configs. Defaults to DEFAULT_LADDER.
        identity: Optional benchmark identity columns prepended to every
            row (e.g. {"scenario": "Houston", "target": "obs_nfip_event_claims"}).
            The driver stamps the identity; decompose_stack just carries it.

    Returns:
        DataFrame with one row per ladder step, diagnostic columns,
        and delta columns for attribution.
    """
    ladder = ladder or DEFAULT_LADDER
    has_metrics = y_true is not None and y_pred is not None

    if has_metrics:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

    rows = []
    for name, cfg in ladder:
        w, cert = build_weights_layered(
            validation_df, target_df, bin_cols, cfg,
        )

        row: dict = {}
        if identity:
            row.update(identity)
        row["config"] = name

        if has_metrics:
            use_w = None if (cfg.matching == "none"
                             and cfg.shrinkage_lambda <= 0) else w
            m = weighted_metrics(y_true, y_pred, use_w)
            row.update(m)

        row.update(cert)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Attribution: how much did each layer move the estimate?
    if has_metrics:
        df["delta_rmse"] = df["rmse"].diff()
        df["delta_mae"] = df["mae"].diff()

    df["delta_ess"] = df["ess"].diff()
    df["delta_margin_gap"] = df["residual_margin_gap"].diff()

    return df
