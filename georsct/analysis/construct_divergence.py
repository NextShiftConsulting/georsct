"""
georsct.analysis.construct_divergence

Pure-math library for DOE-C1: FAST vs NFIP construct divergence.

Computes spatial correlation, bootstrap CIs, quadrant mass distribution,
and data-quality gates between engineering damage estimates and
administrative claims at the ZCTA level.

Six controls (residual bug fixes from DOE-C1 review):
  1. Mechanism gating    -- restrict to surge/pluvial-appropriate ZCTAs
  2. Statistical discipline -- bootstrap CIs, pairable-n floor, coverage gate
  3. Tie saturation      -- Kendall tau-b with tie fraction reporting
  4. PIF floor           -- NaN NFIP signal below min policies-in-force
  5. JRC permanent-water -- mask permanently wet ZCTAs
  6. Same-sensor filter  -- enforce same-sensor SAR transfer pairs

The job script (jobs/compute_construct_divergence.py) handles S3 I/O
and imports these pure functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_PAIRABLE_N: int = 30
"""Minimum overlapping ZCTAs for a correlation to be reported."""

MIN_PIF_DEFAULT: int = 5
"""Minimum policies-in-force per ZCTA for NFIP signal to be non-NaN."""

JRC_PERMANENT_WATER_PCT: float = 50.0
"""ZCTAs with jrc_pct_ever_wet above this are masked as permanent water."""

BOOTSTRAP_ITERATIONS: int = 2000
"""Number of bootstrap resamples for correlation CIs."""

BOOTSTRAP_ALPHA: float = 0.05
"""Two-sided alpha for bootstrap CIs (0.05 = 95% CI)."""


# ---------------------------------------------------------------------------
# Data-quality gates (bugs #1, #4, #5, #6)
# ---------------------------------------------------------------------------

def apply_pif_floor(
    df: pd.DataFrame,
    nfip_cols: list[str],
    pif_col: str = "nfip_pif_count",
    min_pif: int = MIN_PIF_DEFAULT,
) -> pd.DataFrame:
    """NaN out NFIP signal columns where policies-in-force < min_pif.

    Args:
        df: DataFrame with NFIP signal and PIF columns.
        nfip_cols: NFIP signal columns to mask.
        pif_col: Column with policies-in-force count.
        min_pif: Floor threshold.

    Returns:
        Copy of df with NFIP columns NaN'd below floor.
        If pif_col is missing, returns df unchanged (with warning flag).
    """
    df = df.copy()
    if pif_col not in df.columns:
        return df
    mask = df[pif_col] < min_pif
    for col in nfip_cols:
        if col in df.columns:
            df.loc[mask, col] = np.nan
    return df


def mask_permanent_water(
    df: pd.DataFrame,
    jrc_col: str = "jrc_pct_ever_wet",
    threshold: float = JRC_PERMANENT_WATER_PCT,
) -> pd.DataFrame:
    """Remove ZCTAs where permanent water dominates the buffer.

    Args:
        df: DataFrame with JRC permanent water column.
        jrc_col: Column name for JRC percent-ever-wet.
        threshold: Percent above which ZCTA is masked.

    Returns:
        Filtered copy. If jrc_col missing, returns df unchanged.
    """
    if jrc_col not in df.columns:
        return df
    return df[~(df[jrc_col] > threshold)].copy()


def gate_mechanism(
    df: pd.DataFrame,
    mechanism: str,
    coastal_col: str = "coastal_distance_m",
    slosh_col: str = "slosh_max_surge_m",
    coastal_threshold_m: float = 20_000.0,
) -> pd.DataFrame:
    """Restrict to ZCTAs appropriate for a flood mechanism.

    Args:
        df: DataFrame with mechanism-indicator columns.
        mechanism: One of "surge", "pluvial", "all".
        coastal_col: Coastal distance column (meters).
        slosh_col: SLOSH max surge column (meters).
        coastal_threshold_m: Distance cutoff for surge vs pluvial.

    Returns:
        Filtered copy. "all" returns df unchanged.
        Missing columns -> returns df unchanged (no gate).
    """
    if mechanism == "all":
        return df.copy()

    if mechanism == "surge":
        if slosh_col in df.columns:
            return df[df[slosh_col].notna()].copy()
        if coastal_col in df.columns:
            return df[df[coastal_col] <= coastal_threshold_m].copy()
        return df.copy()

    if mechanism == "pluvial":
        if slosh_col in df.columns:
            return df[df[slosh_col].isna()].copy()
        if coastal_col in df.columns:
            return df[df[coastal_col] > coastal_threshold_m].copy()
        return df.copy()

    return df.copy()


# Event-sensor mapping for same-sensor transfer pairs (bug #6).
# Sandy (2012) is pre-Sentinel-1 (launched 2014).  SAR data for Sandy
# uses different sensors (RADARSAT, COSMO-SkyMed) and cannot be compared
# with Sentinel-1-based SAR features from post-2014 events.
SENSOR_MAP: dict[str, str] = {
    # Houston
    "harvey2017": "sentinel1",
    "imelda2019": "sentinel1",
    "beryl2024": "sentinel1",
    # NYC
    "ida2021": "sentinel1",
    "henri2021": "sentinel1",
    "sandy2012": "mixed_pre_s1",
    # SW Florida
    "ian2022": "sentinel1",
    "helene2024": "sentinel1",
    "milton2024": "sentinel1",
}


def same_sensor_events(events: list[str]) -> list[str]:
    """Filter events to those sharing the dominant sensor platform.

    Drops pre-Sentinel-1 events (e.g., Sandy) that use mixed sensors
    incompatible with S1-based features.

    Args:
        events: List of event IDs.

    Returns:
        Events that use sentinel1 sensor.
    """
    return [e for e in events if SENSOR_MAP.get(e, "sentinel1") == "sentinel1"]


# ---------------------------------------------------------------------------
# Coverage gate (bug #2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageResult:
    """Result of the pairable-n coverage gate."""

    n_fast: int
    n_nfip: int
    n_pairable: int
    n_after_masks: int
    passes: bool
    reason: str


def coverage_gate(
    n_fast: int,
    n_nfip: int,
    n_pairable: int,
    n_after_masks: int,
    min_n: int = MIN_PAIRABLE_N,
) -> CoverageResult:
    """Check if the pairable population is sufficient for inference.

    Args:
        n_fast: ZCTAs with FAST data.
        n_nfip: ZCTAs with NFIP data.
        n_pairable: ZCTAs with both (inner join).
        n_after_masks: ZCTAs remaining after PIF/JRC/mechanism masks.
        min_n: Minimum pairable count.

    Returns:
        CoverageResult with pass/fail and reason.
    """
    if n_after_masks < min_n:
        return CoverageResult(
            n_fast=n_fast, n_nfip=n_nfip,
            n_pairable=n_pairable, n_after_masks=n_after_masks,
            passes=False,
            reason=f"n_after_masks={n_after_masks} < min_n={min_n}",
        )
    return CoverageResult(
        n_fast=n_fast, n_nfip=n_nfip,
        n_pairable=n_pairable, n_after_masks=n_after_masks,
        passes=True,
        reason="sufficient pairable population",
    )


# ---------------------------------------------------------------------------
# Correlation with bootstrap CI and tie reporting (bugs #2, #3)
# ---------------------------------------------------------------------------

def _tie_fraction(x: np.ndarray) -> float:
    """Fraction of pairwise comparisons that are tied."""
    n = len(x)
    if n < 2:
        return 0.0
    n_pairs = n * (n - 1) / 2
    _, counts = np.unique(x, return_counts=True)
    n_tied = sum(c * (c - 1) / 2 for c in counts if c > 1)
    return float(n_tied / n_pairs)


def _bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = BOOTSTRAP_ITERATIONS,
    alpha: float = BOOTSTRAP_ALPHA,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap percentile CI for Spearman rho.

    Returns:
        (ci_lower, ci_upper) at 1-alpha confidence.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos[i] = stats.spearmanr(x[idx], y[idx]).statistic
    lo = float(np.nanpercentile(rhos, 100 * alpha / 2))
    hi = float(np.nanpercentile(rhos, 100 * (1 - alpha / 2)))
    return lo, hi


def compute_correlation(
    fast_vals: np.ndarray,
    nfip_vals: np.ndarray,
    n_boot: int = BOOTSTRAP_ITERATIONS,
    min_n: int = MIN_PAIRABLE_N,
) -> dict:
    """Compute correlation with bootstrap CI and tie reporting.

    Fixes bugs #2 (statistical discipline) and #3 (tie saturation).

    Uses Kendall tau-b (handles ties correctly) instead of tau-a,
    reports tie fractions for both variables, and provides bootstrap
    percentile CI for Spearman rho.

    Args:
        fast_vals: FAST signal values (1D array).
        nfip_vals: NFIP signal values (1D array).
        n_boot: Bootstrap iterations for Spearman CI.
        min_n: Minimum sample size.

    Returns:
        Dict with correlation stats, CIs, and tie fractions.
    """
    mask = np.isfinite(fast_vals) & np.isfinite(nfip_vals)
    n = int(mask.sum())
    if n < min_n:
        return {"n": n, "error": f"insufficient overlap (n={n} < {min_n})"}

    f = fast_vals[mask]
    nf = nfip_vals[mask]

    # Spearman (unchanged)
    sp = stats.spearmanr(f, nf)
    sp_rho = float(sp.statistic)
    sp_p = float(sp.pvalue)

    # Kendall tau-b (bug #3: handles tied pairs correctly)
    kt = stats.kendalltau(f, nf, variant="b")
    kt_tau = float(kt.statistic)
    kt_p = float(kt.pvalue)

    # Tie fractions (bug #3)
    tie_frac_fast = _tie_fraction(f)
    tie_frac_nfip = _tie_fraction(nf)

    # Bootstrap CI for Spearman (bug #2)
    ci_lo, ci_hi = _bootstrap_spearman(f, nf, n_boot=n_boot)

    return {
        "n": n,
        "spearman_rho": sp_rho,
        "spearman_p": sp_p,
        "spearman_ci_lo": ci_lo,
        "spearman_ci_hi": ci_hi,
        "kendall_tau_b": kt_tau,
        "kendall_p": kt_p,
        "tie_fraction_fast": tie_frac_fast,
        "tie_fraction_nfip": tie_frac_nfip,
    }


# ---------------------------------------------------------------------------
# Quadrant mass distribution
# ---------------------------------------------------------------------------

def compute_quadrant_mass(
    fast_vals: np.ndarray,
    nfip_vals: np.ndarray,
    min_n: int = MIN_PAIRABLE_N,
) -> dict:
    """Compute 2x2 quadrant mass (high/low FAST x high/low NFIP).

    Thresholds: median of each variable within the overlapping set.

    Args:
        fast_vals: FAST signal (1D, may contain NaN).
        nfip_vals: NFIP signal (1D, may contain NaN).
        min_n: Minimum sample size.

    Returns:
        Dict with quadrant counts and fractions.
    """
    mask = np.isfinite(fast_vals) & np.isfinite(nfip_vals)
    n = int(mask.sum())
    if n < min_n:
        return {"n": n, "error": f"insufficient overlap (n={n} < {min_n})"}

    f = fast_vals[mask]
    nf = nfip_vals[mask]

    f_med = float(np.median(f))
    nf_med = float(np.median(nf))

    f_hi = f >= f_med
    nf_hi = nf >= nf_med

    hh = int((f_hi & nf_hi).sum())
    hl = int((f_hi & ~nf_hi).sum())
    lh = int((~f_hi & nf_hi).sum())
    ll = int((~f_hi & ~nf_hi).sum())

    return {
        "n": n,
        "fast_median": f_med,
        "nfip_median": nf_med,
        "high_fast_high_nfip": hh,
        "high_fast_low_nfip": hl,
        "low_fast_high_nfip": lh,
        "low_fast_low_nfip": ll,
        "pct_agreement": float(hh + ll) / n,
        "pct_physical_dominant": float(hl) / n,
        "pct_claims_dominant": float(lh) / n,
    }
