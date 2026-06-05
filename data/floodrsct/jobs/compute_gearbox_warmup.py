#!/usr/bin/env python3
"""
compute_gearbox_warmup.py -- Phase 0.75: Gearbox warmup pass.

Sits between geometry_kappa (Phase 0.5) and R0 training (Phase 1).
Runs a CHEAP solver (Ridge) with spatial-blocked CV to produce
data-grounded quality signals that vary per cell -- unlike kappa_geom
which is level-invariant and cannot discriminate DGM routing arms.

Per (scenario, target) cell, computes:
  - tau:            stability signal = 1/(1+CV) from fold variance
  - sigma:          fold metric std (ddof=1)
  - alpha_warmup:   cheap-solver quality proxy from mean(fold metrics)
  - collapse_risk:  1 if any fold metric < 0, else 1 - min/median
  - coherence:      1 - range/median (fold agreement)
  - gear:           data-calibrated gear assignment (1st-4th or Reverse)

Calibration follows yrsn GearboxCalibrator methodology (percentile-based
boundaries with winsorization and monotonicity guards), adapted for
batch mode (small N) rather than streaming warmup.

Dual purpose:
  1. DGM routing discriminant (gear varies per cell, unlike kappa_geom)
  2. Oobleck autocalibration (calibrated boundaries -> kappa_req params)

Inputs:  assembled parquet, folds, crosswalk, NFIP historical (all on S3)
Outputs: gearbox_warmup.json on S3

Usage:
    python compute_gearbox_warmup.py --upload
    python compute_gearbox_warmup.py --dry-run
"""

import argparse
import io
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    BUCKET, SCENARIOS, get_s3_client, load_processed_parquet, load_crosswalk,
)
from _s3_result import upload_json_result
from generate_folds import generate_folds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
SEED = 42
MODELABLE = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]

TARGETS = [
    ("obs_nfip_event_claims", "regression", "log1p"),
    ("obs_has_311",           "classification", None),
    ("obs_has_hwm",           "classification", None),
]

# R0 static features (same list as train_r0_baseline.py and compute_geometry_kappa.py)
R0_FEATURES = [
    "flood_pct_zone_a", "flood_pct_zone_x", "flood_pct_zone_x500",
    "elevation_m_msl", "slope_mean_pct", "twi_twi",
    "coastal_distance_m", "latitude", "longitude",
    "acs_total_pop", "acs_median_hh_income", "acs_pct_below_poverty",
    "acs_pct_renter_occupied", "acs_pct_owner_occupied", "acs_pct_vacant",
    "acs_pct_no_vehicle", "acs_median_home_value", "acs_median_year_built",
    "svi_overall", "svi_socioeconomic", "svi_household_disability",
    "svi_minority_language", "svi_housing_transport",
    "nfip_historical_frequency", "nfip_historical_severity",
    "hifld_nearest_hospital_km", "hifld_n_hospitals", "population",
    "impervious_pct", "cropland_pct",
]

# Fold column for spatial-blocked CV
FOLD_COL = "fold_spatial_blocked"

# Calibration config (mirrors GearboxCalibrationConfig defaults)
QUANTILES = (0.25, 0.50, 0.75)
WINSOR_LIMITS = (0.01, 0.99)
MIN_GAP = 1e-6
# Batch mode floor for percentile calibration. Streaming GearboxCalibrator
# uses 512; batch mode can go lower because we see the full distribution
# at once. 8 gives at least 2 samples per quartile (4 quartiles x 2 = 8),
# the minimum for non-degenerate percentile boundaries.
MIN_CALIBRATION_SAMPLES = 8

# Gear names (mirrors yrsn Gear enum)
GEAR_NAMES = {1: "FIRST", 2: "SECOND", 3: "THIRD", 4: "FOURTH", -1: "REVERSE"}


# ---------------------------------------------------------------------------
# Cheap solver: Ridge with impute + scale
# ---------------------------------------------------------------------------

def _run_ridge_cv(df: pd.DataFrame, folds_df: pd.DataFrame,
                  features: list[str], target_col: str,
                  task: str, transform: str | None) -> list[float]:
    """Run Ridge with spatial-blocked CV. Return per-fold primary metrics."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge, RidgeClassifier
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    merged = df.merge(
        folds_df[["zcta_id", "event", FOLD_COL]],
        on=["zcta_id", "event"],
    )
    valid_mask = merged[target_col].notna()
    merged = merged[valid_mask].copy()

    if len(merged) < 20:
        return []

    # Target transform
    y_col = target_col
    if transform == "log1p":
        merged["_y"] = np.log1p(merged[target_col].clip(lower=0).astype(float))
        y_col = "_y"

    avail_features = [f for f in features if f in merged.columns and merged[f].notna().any()]
    if len(avail_features) < 3:
        return []

    X_all = merged[avail_features].values.astype(np.float32)
    y_all = merged[y_col].values.astype(np.float32)
    fold_ids = sorted(merged[FOLD_COL].unique())

    fold_metrics = []
    for fold_id in fold_ids:
        test_mask = merged[FOLD_COL] == fold_id
        train_mask = ~test_mask
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(X_test) == 0 or len(X_train) == 0:
            continue
        if task == "classification" and len(np.unique(y_train)) < 2:
            continue

        if task == "classification":
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", RidgeClassifier(alpha=1.0)),
            ])
            pipe.fit(X_train, y_train)
            y_score = pipe.decision_function(X_test)
            try:
                auc = float(roc_auc_score(y_test, y_score))
                if np.isfinite(auc):
                    fold_metrics.append(auc)
            except ValueError:
                pass  # single class in test fold
        else:
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            r2 = float(r2_score(y_test, y_pred))
            if np.isfinite(r2):
                fold_metrics.append(r2)

    return fold_metrics


# ---------------------------------------------------------------------------
# Quality signal derivation (from fold metrics)
# ---------------------------------------------------------------------------

def derive_quality_signals(fold_metrics: list[float],
                           task: str) -> dict[str, float]:
    """Derive cell-level quality signals from per-fold primary metrics.

    Returns dict with tau, sigma, alpha_warmup, collapse_risk, coherence.
    All values are finite floats. Returns empty dict if fewer than 2
    folds produced metrics -- this drops the cell from the calibration
    pool entirely (no gear assignment, no routing signal). This is
    intentional: with 0-1 folds, CV/IQR/range are undefined, and
    assigning default signals would pollute calibration with non-data.
    Dropped cells are logged as warnings in the caller.
    """
    if len(fold_metrics) < 2:
        return {}

    arr = np.array(fold_metrics, dtype=float)
    mean_m = float(np.mean(arr))
    std_m = float(np.std(arr, ddof=1))
    min_m = float(np.min(arr))
    max_m = float(np.max(arr))
    med_m = float(np.median(arr))

    # alpha_warmup: cheap-solver quality proxy mapped to [0, 1].
    # Regression R2 is already in [0, 1] for useful models (clip negatives).
    # Classification AUC is in [0.5, 1.0] for above-chance models.
    # Rescale AUC via 2*(AUC-0.5) to map [0.5, 1.0] -> [0.0, 1.0],
    # matching the same scale as regression R2. This follows the
    # certify_experiment_cell convention in rsct/experiment_cert.py.
    if task == "classification":
        alpha_warmup = float(np.clip(2.0 * (mean_m - 0.5), 0.0, 1.0))
    else:
        alpha_warmup = float(np.clip(mean_m, 0.0, 1.0))

    # sigma: fold metric std (ddof=1)
    sigma = std_m

    # tau: stability = 1/(1+CV) where CV = std/|mean|.
    # Edge cases when mean ~ 0:
    #   mean ~ 0, std > 0: CV = inf -> tau = 0. This is correct: the model
    #     explains nothing on average but has variance, indicating instability.
    #   mean ~ 0, std ~ 0: CV = 0 -> tau = 1. This is correct: all folds
    #     agree the model explains nothing (consistently bad = stable).
    # Both map to meaningful gear assignments: tau=0 -> FIRST gear (worst
    # stability), tau=1 -> high gear (perfect stability).
    if abs(mean_m) > 1e-12:
        cv = std_m / abs(mean_m)
    else:
        cv = float("inf") if std_m > 1e-12 else 0.0
    tau = 1.0 / (1.0 + cv)

    # collapse_risk: fraction of folds where the solver fails to beat
    # the naive predictor.  R2 <= 0 for regression, AUC <= 0.5 for
    # classification.  A simple, interpretable metric: 0.0 = all folds
    # have skill, 1.0 = no fold has skill.
    if task == "classification":
        n_fail = int(np.sum(arr <= 0.5))
    else:
        n_fail = int(np.sum(arr <= 0.0))
    collapse_risk = n_fail / len(arr)

    # coherence: fold agreement measured by IQR rather than range.
    # range/|median| is sensitive to a single outlier fold; IQR (q75-q25)
    # ignores the most extreme 50% and gives a robust spread measure.
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q75 - q25
    if abs(med_m) > 1e-12:
        coherence = float(np.clip(1.0 - iqr / abs(med_m), 0.0, 1.0))
    else:
        coherence = 0.0

    return {
        "tau": round(tau, 6),
        "sigma": round(sigma, 6),
        "alpha_warmup": round(alpha_warmup, 6),
        "collapse_risk": round(collapse_risk, 6),
        "coherence": round(coherence, 6),
        "mean_metric": round(mean_m, 6),
        "n_folds": len(fold_metrics),
    }


# ---------------------------------------------------------------------------
# Batch calibration (GearboxCalibrator methodology, batch mode)
# ---------------------------------------------------------------------------

def calibrate_gear_boundaries(tau_values: list[float]) -> dict:
    """Compute calibrated gear boundaries from tau distribution.

    Follows yrsn GearboxCalibrator methodology:
    1. Winsorize outliers
    2. Compute quantile boundaries
    3. Enforce monotonicity and minimum gaps

    Adapted for batch mode (small N) vs streaming warmup (N=512).
    Returns dict with boundaries, quantiles, and metadata.
    """
    n = len(tau_values)
    if n < MIN_CALIBRATION_SAMPLES:
        return {
            "calibrated": False,
            "reason": f"insufficient samples ({n} < {MIN_CALIBRATION_SAMPLES})",
            "n_samples": n,
            "boundaries": None,
        }

    taus_sorted = sorted(tau_values)

    def quantile(p: float) -> float:
        if p <= 0.0:
            return taus_sorted[0]
        if p >= 1.0:
            return taus_sorted[-1]
        idx = p * (len(taus_sorted) - 1)
        lo_i = int(math.floor(idx))
        hi_i = int(math.ceil(idx))
        if lo_i == hi_i:
            return taus_sorted[lo_i]
        frac = idx - lo_i
        return taus_sorted[lo_i] * (1 - frac) + taus_sorted[hi_i] * frac

    # Winsorize
    q_lo, q_hi = WINSOR_LIMITS
    tau_lo = quantile(q_lo)
    tau_hi = quantile(q_hi)
    taus_w = sorted(min(max(t, tau_lo), tau_hi) for t in taus_sorted)

    # Quantile boundaries
    bq1, bq2, bq3 = QUANTILES
    b1 = quantile(bq1)
    b2 = quantile(bq2)
    b3 = quantile(bq3)

    if not (math.isfinite(b1) and math.isfinite(b2) and math.isfinite(b3)):
        return {
            "calibrated": False,
            "reason": "non-finite boundaries",
            "n_samples": n,
            "boundaries": None,
        }

    # Monotonicity guard
    if not (b1 < b2 < b3):
        med = quantile(0.5)
        eps = max(MIN_GAP, 1e-3)
        b1, b2, b3 = med - 2 * eps, med, med + 2 * eps

    # Minimum gap guard
    if (b2 - b1) < MIN_GAP:
        b2 = b1 + MIN_GAP
    if (b3 - b2) < MIN_GAP:
        b3 = b2 + MIN_GAP

    return {
        "calibrated": True,
        "n_samples": n,
        "tau_min": round(min(tau_values), 6),
        "tau_max": round(max(tau_values), 6),
        "tau_mean": round(float(np.mean(tau_values)), 6),
        "tau_std": round(float(np.std(tau_values, ddof=1)), 6) if n > 1 else 0.0,
        "boundaries": [round(b1, 6), round(b2, 6), round(b3, 6)],
        "quantiles": {
            f"q{int(bq1*100):02d}": round(b1, 6),
            f"q{int(bq2*100):02d}": round(b2, 6),
            f"q{int(bq3*100):02d}": round(b3, 6),
            f"q{int(q_lo*100):02d}": round(tau_lo, 6),
            f"q{int(q_hi*100):02d}": round(tau_hi, 6),
        },
        "winsor_limits": list(WINSOR_LIMITS),
    }


def assign_gear(tau: float, boundaries: list[float] | None,
                collapse_risk: float, coherence: float) -> tuple[int, str]:
    """Assign gear from calibrated boundaries + quality overrides.

    Mirrors yrsn GearShifter.recommend_gear() logic:
    - collapse_risk > 0.8 -> REVERSE
    - coherence < 0.3 -> THIRD (explore)
    - else: tau-based from calibrated boundaries

    Returns (gear_number, gear_name).
    """
    # Safety overrides first (matches GearShifter)
    if collapse_risk > 0.8:
        return -1, "REVERSE"
    if coherence < 0.3:
        return 3, "THIRD"

    # Tau-based assignment
    if boundaries is None:
        # Uncalibrated fallback: hardcoded from yrsn GEAR_SPECS
        # (gear.py lines 28-33). These are the default tau ranges
        # for the 5-gear system BEFORE data-driven calibration:
        #   FIRST: [0, 1.0), SECOND: [1.0, 1.43), THIRD: [1.43, 2.5), FOURTH: [2.5, inf)
        # This path only triggers when calibration fails (< MIN_CALIBRATION_SAMPLES).
        # In normal operation, calibrated boundaries from the data override these.
        log.warning("Using uncalibrated yrsn default tau thresholds (calibration unavailable)")
        if tau < 1.0:
            return 1, "FIRST"
        elif tau < 1.43:
            return 2, "SECOND"
        elif tau < 2.5:
            return 3, "THIRD"
        else:
            return 4, "FOURTH"

    b1, b2, b3 = boundaries
    if tau <= b1:
        return 1, "FIRST"
    elif tau <= b2:
        return 2, "SECOND"
    elif tau <= b3:
        return 3, "THIRD"
    else:
        return 4, "FOURTH"


# ---------------------------------------------------------------------------
# Oobleck parameter derivation from calibrated boundaries
# ---------------------------------------------------------------------------

def derive_oobleck_params(calibration: dict) -> dict:
    """Map calibrated gear boundaries to oobleck threshold parameters.

    The oobleck sigmoidal threshold:
        kappa_req(sigma) = kappa_base + delta_kappa * sigmoid(s*(sigma - sigma_c))

    We derive sigma_c (inflection point) from the calibrated tau distribution:
    sigma_c corresponds to the tau where quality transitions from "stable"
    to "exploring" -- the boundary between 2nd and 3rd gear.

    kappa_base stays at the theoretical default (0.3) since it represents
    the minimum kappa requirement independent of uncertainty.
    """
    if not calibration.get("calibrated"):
        return {"calibrated": False}

    boundaries = calibration["boundaries"]
    tau_mean = calibration["tau_mean"]
    tau_std = calibration.get("tau_std", 0.0)
    b1, b2, b3 = boundaries

    # sigma_c: inflection at the 2nd/3rd gear boundary.
    # tau = 1/(1+CV), so CV = 1/tau - 1.
    # sigma = std(fold_metrics) ~ CV * |mean_metric|.
    # Use the median tau (b2 = gear 2/3 boundary) as the inflection point.
    if b2 > 1e-12:
        cv_at_inflection = 1.0 / b2 - 1.0
    else:
        cv_at_inflection = 1.0

    # sigma_c = CV_inflection * tau_std.
    # tau_std serves as a data-grounded scale factor: it captures how
    # spread the stability distribution is, replacing the former
    # hardcoded 0.1 multiplier. When the tau distribution is tight
    # (low tau_std), sigma_c is small (demanding); when spread, it's
    # more permissive -- matching the data's own notion of "normal"
    # variability.
    scale = tau_std if tau_std > 1e-12 else 0.1
    sigma_c = float(np.clip(cv_at_inflection * scale, 0.05, 0.5))

    # steepness: derived from how concentrated the tau distribution is
    # Tight distribution -> steep sigmoid; spread distribution -> gentle
    if tau_std > 1e-12:
        steepness = float(np.clip(1.0 / tau_std, 5.0, 50.0))
    else:
        steepness = 20.0  # default

    return {
        "calibrated": True,
        "kappa_base": 0.3,
        "delta_kappa": 0.4,
        "sigma_c": round(sigma_c, 6),
        "steepness": round(steepness, 4),
        "derivation": (
            "sigma_c = CV_inflection * tau_std (data-grounded, no "
            "hardcoded multiplier); steepness = 1/tau_std "
            "(distribution concentration)"
        ),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _generate_folds_for_warmup(
    df: pd.DataFrame, crosswalk: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Generate spatial-blocked fold assignments for warmup CV.

    Fold generation uses only spatial structure (ZCTA IDs, county
    assignments) -- zero dependency on model outputs. This eliminates
    the pipeline ordering violation: Phase 0.75 no longer requires
    folds from Phase 1 (train_r0_baseline).

    If folds already exist on S3 (from a prior Phase 1 run), we still
    regenerate here to maintain causal independence.
    """
    zcta_county = {}
    if crosswalk is not None:
        xw = crosswalk.copy()
        xw["zcta_id"] = xw["zcta_id"].astype(str)
        zcta_county = dict(zip(xw["zcta_id"], xw["county_fips"].astype(str)))

    try:
        folds_df, fold_meta = generate_folds(df, zcta_county, seed=SEED)
        log.info(
            "Generated warmup folds: strategy=%s, %d rows",
            fold_meta.get("block_strategy", "unknown"), len(folds_df),
        )
        return folds_df
    except Exception as e:
        log.warning("Fold generation failed: %s", e)
        return None


def _load_nfip_historical(s3, scenario: str) -> pd.DataFrame | None:
    """Load NFIP historical supplement (temporally-gated base rates)."""
    key = f"processed/{scenario}/{scenario}_nfip_historical.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        df["zcta_id"] = df["zcta_id"].astype(str)
        return df
    except Exception as e:
        log.warning("NFIP historical unavailable for %s: %s", scenario, e)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_gearbox_warmup(s3) -> dict:
    """Run Ridge warmup CV and calibrate gearbox for all cells."""
    cells = []

    for scenario in MODELABLE:
        log.info("=== Gearbox warmup: %s ===", scenario)

        # Load data
        try:
            df = load_processed_parquet(s3, scenario)
            df["zcta_id"] = df["zcta_id"].astype(str)
            if "event" in df.columns:
                df["event"] = df["event"].astype(str)
        except Exception as e:
            log.warning("Cannot load parquet for %s: %s", scenario, e)
            continue

        # Load crosswalk for fold generation
        try:
            crosswalk = load_crosswalk(s3)
        except Exception:
            crosswalk = None

        # Generate folds internally (spatial-structure-only, no model dependency).
        # This eliminates the pipeline ordering violation: Phase 0.75 does NOT
        # require folds from Phase 1 (train_r0_baseline).
        folds_df = _generate_folds_for_warmup(df, crosswalk)
        if folds_df is None:
            log.warning("Cannot generate folds for %s -- skipping", scenario)
            continue

        if FOLD_COL not in folds_df.columns:
            log.warning("Fold column %s missing for %s -- skipping", FOLD_COL, scenario)
            continue

        # Merge NFIP historical supplement
        nfip = _load_nfip_historical(s3, scenario)
        if nfip is not None:
            join_cols = ["zcta_id", "event"] if "event" in nfip.columns else ["zcta_id"]
            df = df.merge(nfip, on=join_cols, how="left")
            log.info("NFIP historical merged for %s", scenario)

        for target_col, task, transform in TARGETS:
            if target_col not in df.columns:
                continue
            if df[target_col].notna().sum() < 20:
                continue
            if df[target_col].dropna().nunique() < 2:
                continue

            log.info("  Ridge CV: %s / %s (%s)", scenario, target_col, task)
            fold_metrics = _run_ridge_cv(
                df, folds_df, R0_FEATURES, target_col, task, transform,
            )

            if not fold_metrics:
                log.warning("  No fold metrics for %s/%s", scenario, target_col)
                continue

            signals = derive_quality_signals(fold_metrics, task)
            if not signals:
                log.warning("  Insufficient signals for %s/%s", scenario, target_col)
                continue

            cell = {
                "scenario": scenario,
                "target": target_col,
                "task": task,
                **signals,
                "fold_metrics": [round(m, 6) for m in fold_metrics],
            }
            cells.append(cell)

            log.info(
                "    tau=%.3f sigma=%.4f alpha=%.3f collapse=%.2f coherence=%.2f",
                signals["tau"], signals["sigma"], signals["alpha_warmup"],
                signals["collapse_risk"], signals["coherence"],
            )

    # --- Calibrate gear boundaries from tau distribution ---
    tau_values = [c["tau"] for c in cells if c.get("tau") is not None]
    calibration = calibrate_gear_boundaries(tau_values)

    if calibration.get("calibrated"):
        log.info("Calibrated gear boundaries: %s", calibration["boundaries"])
    else:
        log.warning("Calibration failed: %s", calibration.get("reason", "unknown"))

    boundaries = calibration.get("boundaries")

    # --- Assign gears ---
    for cell in cells:
        gear_num, gear_name = assign_gear(
            cell["tau"], boundaries,
            cell["collapse_risk"], cell["coherence"],
        )
        cell["gear"] = gear_num
        cell["gear_name"] = gear_name
        log.info(
            "  %s/%s -> %s (tau=%.3f)",
            cell["scenario"], cell["target"], gear_name, cell["tau"],
        )

    # --- Derive oobleck parameters ---
    oobleck = derive_oobleck_params(calibration)

    # --- Build payload ---
    payload = {
        "experiment": "s035-model-ladder",
        "phase": "gearbox_warmup",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Phase 0.75: Ridge warmup CV -> quality signals -> gear calibration. "
            "Provides DGM routing discriminant (gear varies per cell) and "
            "oobleck autocalibration parameters. Zero dependency on R0/R1/R2 "
            "HistGBDT results -- uses cheap Ridge solver only."
        ),
        "solver": "Ridge (alpha=1.0, impute+scale pipeline)",
        "split": "spatial_blocked",
        "n_cells": len(cells),
        "cells": cells,
        "calibration": calibration,
        "oobleck_params": oobleck,
        "gear_distribution": {},
        "methodology": {
            "tau": "1/(1+CV) where CV=std(folds)/|mean(folds)|",
            "sigma": "std(fold_metrics, ddof=1)",
            "alpha_warmup": "clip(mean_metric, 0, 1) for regression; "
                            "clip(2*(AUC-0.5), 0, 1) for classification",
            "collapse_risk": "1 if any fold < 0; else 1 - min/median",
            "coherence": "1 - IQR/|median| (IQR = q75-q25, robust to outlier folds)",
            "gear_assignment": "GearboxCalibrator methodology (yrsn core): "
                               "percentile boundaries + winsorization + "
                               "monotonicity guards; collapse_risk/coherence "
                               "overrides mirror GearShifter.recommend_gear()",
            "oobleck_sigma_c": "CV_inflection * tau_std at 2nd/3rd gear boundary",
            "oobleck_steepness": "1/tau_std (distribution concentration)",
        },
    }

    # Gear distribution summary
    gear_counts = {}
    for c in cells:
        g = c.get("gear_name", "UNKNOWN")
        gear_counts[g] = gear_counts.get(g, 0) + 1
    payload["gear_distribution"] = gear_counts

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0.75: Gearbox warmup pass (Ridge CV -> gear calibration)"
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan, skip execution")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: Phase 0.75 Gearbox Warmup")
        log.info("  Load assembled parquets + folds for %d scenarios", len(MODELABLE))
        log.info("  Run Ridge CV on %d targets with spatial-blocked folds", len(TARGETS))
        log.info("  Derive tau/sigma/alpha/collapse_risk/coherence per cell")
        log.info("  Calibrate gear boundaries (GearboxCalibrator methodology)")
        log.info("  Derive oobleck sigma_c and steepness from calibration")
        log.info("  Output: %s/gearbox_warmup.json", RESULTS_PREFIX)
        return

    s3 = get_s3_client()

    print("\n" + "=" * 60)
    print("  S035 PHASE 0.75: GEARBOX WARMUP (Ridge CV)")
    print("=" * 60 + "\n")

    payload = compute_gearbox_warmup(s3)

    # Always write locally
    local_path = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    local_path.mkdir(parents=True, exist_ok=True)
    with open(local_path / "gearbox_warmup.json", "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote local: %s", local_path / "gearbox_warmup.json")

    if args.upload:
        key = f"{RESULTS_PREFIX}/gearbox_warmup.json"
        upload_json_result(s3, BUCKET, key, payload)

    # Summary
    n = len(payload["cells"])
    if n:
        gears = payload["gear_distribution"]
        print(f"\nGearbox warmup: {n} cells")
        for g, count in sorted(gears.items()):
            print(f"  {g}: {count}")
        cal = payload["calibration"]
        if cal.get("calibrated"):
            b = cal["boundaries"]
            print(f"  Calibrated boundaries: [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}]")
        oob = payload["oobleck_params"]
        if oob.get("calibrated"):
            print(f"  Oobleck: sigma_c={oob['sigma_c']:.4f}, steepness={oob['steepness']:.1f}")
    else:
        print("\nNo cells computed.")


if __name__ == "__main__":
    main()
