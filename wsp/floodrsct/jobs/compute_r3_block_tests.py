#!/usr/bin/env python3
"""
compute_r3_block_tests.py -- Phase R3_1a: Per-block test battery.

For each candidate block B, runs the DOE-mandated test battery under
identical folds, solvers, targets, and splits:

  1. Add-block:      R2 + B  (marginal value over current strongest baseline)
  2. Drop-block:     R2_all - B  (necessity once candidates interact)
  3. Block-only:     R0 + B  (independent signal)
  4. Leakage-stress: Random vs spatial-blocked gap
  5. Transfer-stress: Leave-event-out vs spatial-blocked gap
  6. Solver-compat:  HistGBDT vs Ridge agreement

Produces per-block certificates (R, S_sup, N, kappa, sigma) from the
spatial-blocked fold metrics.

Usage:
    python compute_r3_block_tests.py --scenario houston --upload
    python compute_r3_block_tests.py --scenario houston --dry-run
"""

import argparse
import io
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, level_prefix
from _s3_result import upload_json_result
from _validate_contract import check_causal_boundary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)
RESULTS_PREFIX = "results/s035"

# ---------------------------------------------------------------------------
# Feature lists (identical to train_r2_temporal.py)
# ---------------------------------------------------------------------------
R0_FEATURES = [
    "flood_pct_zone_a", "flood_pct_zone_x", "flood_pct_zone_x500",
    "elevation_m_msl", "slope_mean_pct", "twi_twi", "coastal_distance_m",
    "latitude", "longitude",
    "acs_total_pop", "acs_median_hh_income", "acs_pct_below_poverty",
    "acs_pct_renter_occupied", "acs_pct_owner_occupied", "acs_pct_vacant",
    "acs_pct_no_vehicle", "acs_median_home_value", "acs_median_year_built",
    "svi_overall", "svi_socioeconomic", "svi_household_disability",
    "svi_minority_language", "svi_housing_transport",
    "nfip_historical_frequency", "nfip_historical_severity",
    "hifld_nearest_hospital_km", "hifld_n_hospitals",
    "population", "impervious_pct", "cropland_pct",
]

R1_UNIVERSAL = [
    "nhd_catchment_area_km2", "slope_basin_slope", "slope_stream_slope",
    "twi_acc_twi", "twi_tot_twi",
    "drive_min_to_county_centroid", "drive_min_to_county_seat",
    "drive_min_to_nearest_hospital", "hifld_n_hospital_beds",
    "hifld_n_pharmacies", "hifld_nearest_pharmacy_km",
    "hifld_nearest_trauma_center_km",
]

R1_SCENARIO_SPECIFIC = [
    "upstream_catchment_km2", "hcfcd_drainage_district",
    "levee_nearest_km", "levee_condition_rating",
    "sewershed_name", "slosh_max_surge_m",
]

R1_WMATRIX = [
    "zcta_degree", "zcta_mean_neighbor_dist_km",
    "wlag_flood_zone_pct", "wlag_population_density", "wlag_median_income",
    "wlag_impervious_pct", "wlag_cropland_pct", "wlag_rainfall_mm",
    "wlag_nfip_claims",
]

R2_TEMPORAL = [
    "peak_1h_mm", "peak_3h_mm", "peak_6h_mm", "storm_duration_h",
    "time_to_peak_h", "rainfall_intensity_cv", "tide_peak_m",
    "surge_rain_lag_h", "storm_min_dist_km", "storm_landfall_category",
]

R1_HYDRO = R1_UNIVERSAL + R1_SCENARIO_SPECIFIC
R2_FEATURES = R0_FEATURES + R1_HYDRO + R1_WMATRIX + R2_TEMPORAL

# Block-to-feature mapping (must match build_r3_feature_registry.py)
BLOCK_FEATURES = {
    "hydrology": [
        "nhd_catchment_area_km2", "slope_basin_slope", "slope_stream_slope",
        "twi_acc_twi", "twi_tot_twi", "upstream_catchment_km2",
        "hcfcd_drainage_district", "levee_nearest_km", "levee_condition_rating",
        "sewershed_name",
    ],
    "spatial_relation": [
        "zcta_degree", "zcta_mean_neighbor_dist_km",
        "wlag_flood_zone_pct", "wlag_population_density", "wlag_median_income",
        "wlag_impervious_pct", "wlag_cropland_pct", "wlag_rainfall_mm",
        "wlag_nfip_claims",
    ],
    "temporal": [
        "peak_1h_mm", "peak_3h_mm", "peak_6h_mm", "storm_duration_h",
        "time_to_peak_h", "rainfall_intensity_cv", "tide_peak_m",
        "surge_rain_lag_h", "storm_min_dist_km", "storm_landfall_category",
    ],
    "infrastructure": [
        "hifld_nearest_hospital_km", "hifld_n_hospitals", "hifld_n_hospital_beds",
        "hifld_n_pharmacies", "hifld_nearest_pharmacy_km",
        "hifld_nearest_trauma_center_km",
        "drive_min_to_county_centroid", "drive_min_to_county_seat",
        "drive_min_to_nearest_hospital",
    ],
    "socioeconomic": [
        "acs_total_pop", "acs_median_hh_income", "acs_pct_below_poverty",
        "acs_pct_renter_occupied", "acs_pct_owner_occupied", "acs_pct_vacant",
        "acs_pct_no_vehicle", "acs_median_home_value", "acs_median_year_built",
        "population", "svi_overall", "svi_socioeconomic", "svi_household_disability",
        "svi_minority_language", "svi_housing_transport",
        "nfip_historical_frequency", "nfip_historical_severity",
    ],
    "terrain": [
        "flood_pct_zone_a", "flood_pct_zone_x", "flood_pct_zone_x500",
        "flood_sfha", "elevation_m_msl", "slope_mean_pct", "twi_twi",
        "coastal_distance_m", "latitude", "longitude",
        "impervious_pct", "cropland_pct",
    ],
}

TARGETS = [
    ("obs_nfip_event_claims", "regression", "log1p"),
    ("obs_has_311",           "classification", None),
    ("obs_has_hwm",           "classification", None),
]

PRIMARY_METRIC = {
    "regression": "r2",
    "classification": "roc_auc",
}

SPLITS = {
    "random": "fold_random",
    "spatial_blocked": "fold_spatial_blocked",
    "leave_event_out": "fold_leave_event_out",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_parquet(s3, key: str) -> pd.DataFrame:
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


def _load_json(s3, key: str):
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception:
        return None


def _load_supplement(s3, scenario: str, level: str) -> pd.DataFrame:
    key = f"processed/{scenario}/{scenario}_{level}_supplement.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        log.info("%s supplement: %d rows x %d cols", level.upper(), len(df), len(df.columns))
        return df
    except Exception as exc:
        raise RuntimeError(
            f"{level.upper()} supplement missing: s3://{BUCKET}/{key}. "
            f"Run build_{level}_features.py --scenario {scenario} first."
        ) from exc


# ---------------------------------------------------------------------------
# Solvers (identical to train_r2_temporal.py)
# ---------------------------------------------------------------------------

def _train_histgbdt(X_train, y_train, X_test, y_test, task: str) -> tuple:
    from sklearn.ensemble import (
        HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    )
    n_bins = min(255, max(2, len(X_train) - 1))
    try:
        if task == "classification":
            model = HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, learning_rate=0.1,
                max_bins=n_bins, random_state=SEED,
            )
            model.fit(X_train, y_train)
            y_score = model.predict_proba(X_test)[:, 1]
            return y_score, _classification_metrics(y_test, model.predict(X_test), y_score)
        else:
            model = HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=0.1,
                max_bins=n_bins, random_state=SEED,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return y_pred, _regression_metrics(y_test, y_pred)
    except ValueError as exc:
        log.error("[DATA_QUALITY] HistGBDT fit FAILED n_train=%d: %s", len(X_train), exc)
        null_metrics = {"r2": None, "rmse": None, "roc_auc": None, "f1": None}
        return np.full(len(X_test), np.nan), null_metrics


def _train_ridge(X_train, y_train, X_test, y_test, task: str) -> tuple:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge, RidgeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if task == "classification":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", RidgeClassifier(alpha=1.0)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_score = pipe.decision_function(X_test)
        return y_score, _classification_metrics(y_test, y_pred, y_score)
    else:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        return y_pred, _regression_metrics(y_test, y_pred)


def _nan_to_none(v: float):
    return None if (np.isnan(v) or np.isinf(v)) else v


def _regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "rmse": _nan_to_none(float(np.sqrt(mean_squared_error(y_true, y_pred)))),
        "mae": _nan_to_none(float(mean_absolute_error(y_true, y_pred))),
        "r2": _nan_to_none(float(r2_score(y_true, y_pred))),
    }


def _classification_metrics(y_true, y_pred, y_score) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        auc = float(roc_auc_score(y_true, y_score))
        m["roc_auc"] = None if np.isnan(auc) else auc
    except ValueError:
        m["roc_auc"] = None
    return m


SOLVERS = {"histgbdt": _train_histgbdt, "ridge": _train_ridge}


# ---------------------------------------------------------------------------
# Core: run one feature set across folds
# ---------------------------------------------------------------------------

MIN_FOLD_SAMPLES = 10  # sklearn HistGBDT binning needs >= ~10 rows


def _run_folds(
    merged: pd.DataFrame,
    features: list[str],
    target_col: str,
    y_col: str,
    task: str,
    solver_name: str,
    fold_col: str,
) -> list[dict]:
    """Run solver across all folds for a given feature set. Returns per-fold metrics."""
    solver_fn = SOLVERS[solver_name]
    # Drop rows where target is NaN
    valid_mask = merged[y_col].notna().values
    n_dropped = int((~valid_mask).sum())
    df_valid = merged[valid_mask]
    if n_dropped > 0:
        log.error("[DATA_QUALITY] target=%s dropped %d/%d NaN rows (%.1f%%)",
                  y_col, n_dropped, len(merged), 100 * n_dropped / len(merged))
    X_all = df_valid[features].values.astype(np.float32)
    y_all = df_valid[y_col].values.astype(np.float32)
    fold_ids = sorted(df_valid[fold_col].unique())
    results = []

    for fold_id in fold_ids:
        test_mask = (df_valid[fold_col] == fold_id).values
        train_mask = ~test_mask
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(X_train) < MIN_FOLD_SAMPLES or len(X_test) == 0:
            log.error("[DATA_QUALITY] fold=%s SKIPPED: n_train=%d n_test=%d (min=%d)",
                      fold_id, len(X_train), len(X_test), MIN_FOLD_SAMPLES)
            continue
        if len(np.unique(y_train)) < 2 and task == "classification":
            log.error("[DATA_QUALITY] fold=%s SKIPPED: single-class y_train (n=%d)",
                      fold_id, len(y_train))
            continue

        _, metrics = solver_fn(X_train, y_train, X_test, y_test, task)
        metric_name = PRIMARY_METRIC[task]
        val = metrics.get(metric_name)
        results.append({
            "fold": str(fold_id),
            "metric_name": metric_name,
            "metric_value": val,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        })

    return results


def _mean_metric(fold_results: list[dict]) -> float:
    vals = [r["metric_value"] for r in fold_results if r["metric_value"] is not None]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Block test battery
# ---------------------------------------------------------------------------

def run_block_tests(
    block_name: str,
    block_features: list[str],
    merged: pd.DataFrame,
    available_r2: list[str],
    target_col: str,
    y_col: str,
    task: str,
    scenario: str,
) -> dict:
    """Run the 6-test battery for a single block."""
    log.info("  Testing block: %s (%d features)", block_name, len(block_features))

    # Filter to features actually present in the dataframe
    present = [f for f in block_features if f in merged.columns]
    if not present:
        log.warning("    No features from block %s found in data, skipping", block_name)
        return {"block": block_name, "status": "NO_FEATURES", "tests": {}}

    # Feature sets for each test
    r2_base = [f for f in available_r2 if f in merged.columns]
    r0_base = [f for f in R0_FEATURES if f in merged.columns]

    r2_set = set(r2_base)
    block_set = set(present)

    # Add-block: R2 + B (only features not already in R2)
    effective_added = [f for f in present if f not in r2_set]
    add_features = r2_base + effective_added

    # Drop-block: R2 - B
    effective_dropped = [f for f in r2_base if f in block_set]
    drop_features = [f for f in r2_base if f not in block_set]

    # Block-only: R0 + B
    block_only_features = r0_base + [f for f in present if f not in set(r0_base)]

    # Feature-set audit: prove each contrast is structurally valid
    feature_audit = {
        "r2_base_n": len(r2_base),
        "r0_base_n": len(r0_base),
        "block_n": len(present),
        "effective_added_n": len(effective_added),
        "effective_added": effective_added,
        "effective_dropped_n": len(effective_dropped),
        "effective_dropped": effective_dropped,
        "block_only_n": len(block_only_features),
        "add_block_valid": len(effective_added) > 0,
        "drop_block_valid": len(effective_dropped) > 0,
        "block_only_valid": len(present) > 0,
    }

    if not feature_audit["add_block_valid"]:
        log.warning(
            "    %s: add_block is a no-op (B subset of R2, 0 effective additions)",
            block_name,
        )

    tests = {}

    for solver_name in SOLVERS:
        for split_name, fold_col in SPLITS.items():
            if fold_col not in merged.columns:
                continue

            prefix = f"{solver_name}_{split_name}"

            # 1. Add-block
            add_results = _run_folds(
                merged, add_features, target_col, y_col, task, solver_name, fold_col,
            )
            tests[f"add_block_{prefix}"] = {
                "feature_set": "R2 + B",
                "n_features": len(add_features),
                "folds": add_results,
                "mean_metric": _mean_metric(add_results),
            }

            # 2. Drop-block
            if drop_features:
                drop_results = _run_folds(
                    merged, drop_features, target_col, y_col, task, solver_name, fold_col,
                )
                tests[f"drop_block_{prefix}"] = {
                    "feature_set": "R2 - B",
                    "n_features": len(drop_features),
                    "folds": drop_results,
                    "mean_metric": _mean_metric(drop_results),
                }

            # 3. Block-only
            block_only_results = _run_folds(
                merged, block_only_features, target_col, y_col, task, solver_name, fold_col,
            )
            tests[f"block_only_{prefix}"] = {
                "feature_set": "R0 + B",
                "n_features": len(block_only_features),
                "folds": block_only_results,
                "mean_metric": _mean_metric(block_only_results),
            }

    # R2 baseline (for delta computation)
    r2_baseline = {}
    for solver_name in SOLVERS:
        for split_name, fold_col in SPLITS.items():
            if fold_col not in merged.columns:
                continue
            prefix = f"{solver_name}_{split_name}"
            r2_results = _run_folds(
                merged, r2_base, target_col, y_col, task, solver_name, fold_col,
            )
            r2_baseline[prefix] = _mean_metric(r2_results)

    # Compute deltas and stress tests
    deltas = {}
    for solver_name in SOLVERS:
        sb_key = f"{solver_name}_spatial_blocked"
        rnd_key = f"{solver_name}_random"

        add_sb = tests.get(f"add_block_{sb_key}", {}).get("mean_metric")
        add_rnd = tests.get(f"add_block_{rnd_key}", {}).get("mean_metric")
        r2_sb = r2_baseline.get(sb_key)
        r2_rnd = r2_baseline.get(rnd_key)

        # Spatial skill: delta_spatial = add_block(spatial) - R2(spatial)
        if add_sb is not None and r2_sb is not None:
            deltas[f"delta_spatial_{solver_name}"] = add_sb - r2_sb

        # Leakage stress: (add_block_random - add_block_spatial) - (R2_random - R2_spatial)
        if all(v is not None for v in [add_sb, add_rnd, r2_sb, r2_rnd]):
            add_gap = add_rnd - add_sb
            r2_gap = r2_rnd - r2_sb
            deltas[f"delta_leakage_{solver_name}"] = add_gap - r2_gap

        # Solver compatibility: sign agreement between histgbdt and ridge
        hgb_delta = deltas.get("delta_spatial_histgbdt")
        ridge_delta = deltas.get("delta_spatial_ridge")
        if hgb_delta is not None and ridge_delta is not None:
            deltas["solver_agreement"] = (
                (hgb_delta > 0) == (ridge_delta > 0)
            )

    # Fold stability: std of per-fold delta_spatial for histgbdt spatial_blocked
    add_sb_folds = tests.get("add_block_histgbdt_spatial_blocked", {}).get("folds", [])
    r2_sb_folds_data = _run_folds(
        merged, r2_base, target_col, y_col, task, "histgbdt", "fold_spatial_blocked",
    ) if "fold_spatial_blocked" in merged.columns else []

    if add_sb_folds and r2_sb_folds_data:
        add_vals = [f["metric_value"] for f in add_sb_folds if f["metric_value"] is not None]
        r2_vals = [f["metric_value"] for f in r2_sb_folds_data if f["metric_value"] is not None]
        if len(add_vals) == len(r2_vals) and len(add_vals) > 1:
            fold_deltas = [a - b for a, b in zip(add_vals, r2_vals)]
            deltas["fold_stability_std"] = float(np.std(fold_deltas))
            deltas["fold_stability_all_positive"] = all(d > 0 for d in fold_deltas)

    return {
        "block": block_name,
        "scenario": scenario,
        "target": target_col,
        "n_block_features": len(present),
        "status": "COMPLETE",
        "tests": tests,
        "r2_baseline": r2_baseline,
        "deltas": deltas,
        "feature_audit": feature_audit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Block certificate computation
# ---------------------------------------------------------------------------

_SOLVER_PRIORITY = ("histgbdt", "ridge")


def _pick_solver_metric(
    tests: dict,
    test_prefix: str,
    split: str = "spatial_blocked",
) -> dict:
    """Try histgbdt first, then ridge fallback. Returns full solver trace.

    Returns dict with:
      metric: winning metric value (or None)
      solver_used: which solver produced the metric (or None)
      primary_solver: always "histgbdt"
      fallback_solver: "ridge" if fallback was triggered
      fallback_triggered: bool
      primary_failure_reason: why primary was skipped (or None)
    """
    primary = _SOLVER_PRIORITY[0]
    primary_key = f"{test_prefix}_{primary}_{split}"
    primary_entry = tests.get(primary_key, {})
    primary_val = primary_entry.get("mean_metric")
    primary_ok = primary_val is not None and not (
        isinstance(primary_val, float) and np.isnan(primary_val)
    )

    if primary_ok:
        return {
            "metric": primary_val,
            "solver_used": primary,
            "primary_solver": primary,
            "fallback_solver": None,
            "fallback_triggered": False,
            "primary_failure_reason": None,
        }

    # Primary failed — diagnose why
    if primary_key not in tests:
        primary_reason = "no_test_entry"
    elif primary_val is None:
        primary_reason = "null_metric"
    else:
        primary_reason = "nan_metric"

    # Try fallback
    for fallback in _SOLVER_PRIORITY[1:]:
        fb_key = f"{test_prefix}_{fallback}_{split}"
        fb_entry = tests.get(fb_key, {})
        fb_val = fb_entry.get("mean_metric")
        if fb_val is not None and not (isinstance(fb_val, float) and np.isnan(fb_val)):
            return {
                "metric": fb_val,
                "solver_used": fallback,
                "primary_solver": primary,
                "fallback_solver": fallback,
                "fallback_triggered": True,
                "primary_failure_reason": primary_reason,
            }

    # All solvers failed
    return {
        "metric": None,
        "solver_used": None,
        "primary_solver": primary,
        "fallback_solver": _SOLVER_PRIORITY[-1],
        "fallback_triggered": True,
        "primary_failure_reason": primary_reason,
    }


def compute_block_certificate(block_result: dict) -> dict:
    """Compute an RSCT-style certificate from block test results.

    Evidence structure:
      - independent_signal: from block_only (R0 + B), measures B's standalone value.
      - marginal_signal: full_R2 minus drop_block (R2 - B) ablation delta,
        measures B's marginal contribution within R2.
      - feature_audit: structural validity of each contrast.

    Certificate R/S/N is derived from independent + marginal signals.
    Solver fallback: histgbdt primary, ridge secondary, with lineage.
    certificate_status = INVALID when all required contrasts are unavailable.
    """
    tests = block_result.get("tests", {})
    deltas = block_result.get("deltas", {})
    audit = block_result.get("feature_audit", {})

    # --- Independent signal: block_only (R0 + B) ---
    indep_trace = _pick_solver_metric(tests, "block_only")
    indep_metric = indep_trace["metric"]
    independent_signal = {
        "source": "block_only",
        **indep_trace,
    }

    # --- Marginal signal: full_R2 - drop_block ablation delta ---
    # full_R2 metric: use _pick_solver_metric against r2_baseline keys
    # (r2_baseline is keyed as "{solver}_{split}", so we build synthetic test dict)
    r2_baseline = block_result.get("r2_baseline", {})
    r2_as_tests = {}
    for bkey, bval in r2_baseline.items():
        r2_as_tests[f"full_R2_{bkey}"] = {"mean_metric": bval}
    full_trace = _pick_solver_metric(r2_as_tests, "full_R2")
    full_metric = full_trace["metric"]

    # drop_block metric
    drop_valid = audit.get("drop_block_valid", False)
    drop_trace = {"metric": None, "solver_used": None, "primary_solver": "histgbdt",
                  "fallback_solver": None, "fallback_triggered": False,
                  "primary_failure_reason": "contrast_invalid"}
    if drop_valid:
        drop_trace = _pick_solver_metric(tests, "drop_block")
    drop_metric = drop_trace["metric"]

    # Ablation delta: full_R2 - drop_block
    # Positive = useful (removing block hurts), zero = redundant, negative = harmful
    ablation_delta = None
    if full_metric is not None and drop_metric is not None:
        ablation_delta = full_metric - drop_metric

    marginal_signal = {
        "source": "full_R2_minus_drop_block",
        "full_R2": {**full_trace},
        "drop_block": {**drop_trace},
        "delta": ablation_delta,
    }

    # --- Feature audit summary ---
    add_valid = audit.get("add_block_valid", False)
    feature_audit_summary = {
        "add_effective_features": audit.get("effective_added", []),
        "drop_effective_features": audit.get("effective_dropped", []),
        "add_status": "VALID" if add_valid else "NO_OP_B_SUBSET_R2",
        "drop_status": "VALID" if drop_valid else "NO_FEATURES_TO_DROP",
        "block_only_valid": audit.get("block_only_valid", False),
    }

    # --- Certificate R/S/N derivation ---
    # R comes from independent signal (block_only spatial_blocked performance)
    # Ablation delta informs marginal contribution but doesn't override R
    certificate_status = "VALID"

    if indep_metric is None:
        # No independent signal at all -- certificate is invalid
        certificate_status = "INVALID"
        R, S_sup, N = float("nan"), float("nan"), float("nan")
    else:
        # Block-only spatial_blocked vs block-only random to get R/S/N
        indep_rnd_trace = _pick_solver_metric(tests, "block_only", "random")
        indep_rnd_metric = indep_rnd_trace["metric"]

        if indep_rnd_metric is not None:
            denom = max(abs(indep_rnd_metric), abs(indep_metric), 0.001)
            R = max(0.0, min(1.0, indep_metric / denom)) if indep_metric > 0 else 0.0
            S_sup = max(0.0, min(1.0, (indep_rnd_metric - indep_metric) / denom)) if indep_rnd_metric > indep_metric else 0.0
            N = max(0.0, 1.0 - R - S_sup)
        else:
            # Only spatial_blocked available, no random contrast
            R = max(0.0, min(1.0, indep_metric)) if indep_metric > 0 else 0.0
            S_sup = 0.0
            N = max(0.0, 1.0 - R)

    # Sigma: fold-level instability (from block_only folds, not add_block)
    bo_sb_folds = tests.get("block_only_histgbdt_spatial_blocked", {}).get("folds", [])
    if not bo_sb_folds:
        bo_sb_folds = tests.get("block_only_ridge_spatial_blocked", {}).get("folds", [])
    fold_vals = [f["metric_value"] for f in bo_sb_folds if f.get("metric_value") is not None]
    sigma = float(np.std(fold_vals)) if len(fold_vals) > 1 else float("nan")

    # Alpha
    alpha = R / (R + N) if (R + N) > 0 else float("nan")

    # Kappa: use independent signal metric as proxy
    kappa = max(0.0, indep_metric) if indep_metric is not None else float("nan")

    # Delta leakage (from existing deltas)
    delta_leakage = deltas.get("delta_leakage_histgbdt", float("nan"))

    # Protect: negative ablation delta = harmful block (flag but don't break)
    ablation_interpretation = None
    if ablation_delta is not None:
        if ablation_delta > 0.001:
            ablation_interpretation = "useful"
        elif ablation_delta < -0.001:
            ablation_interpretation = "harmful_or_superfluous"
        else:
            ablation_interpretation = "redundant"

    return {
        "block": block_result["block"],
        "scenario": block_result["scenario"],
        "target": block_result["target"],
        "certificate_status": certificate_status,
        "R": R,
        "S_sup": S_sup,
        "N": N,
        "alpha": alpha,
        "kappa": kappa,
        "sigma": sigma,
        "evidence": {
            "independent_signal": independent_signal,
            "marginal_signal": marginal_signal,
            "feature_audit": feature_audit_summary,
            "ablation_interpretation": ablation_interpretation,
        },
        "delta_leakage": delta_leakage,
        "solver_agreement": deltas.get("solver_agreement"),
        "fold_stability_std": sigma,
        "fold_stability_all_positive": deltas.get("fold_stability_all_positive"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase R3_1a: Per-block test battery"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would run block tests for %s", args.scenario)
        log.info("  %d blocks, 6 tests each, across targets and folds", len(BLOCK_FEATURES))
        return 0

    s3 = get_s3_client()
    scenario = args.scenario

    print(f"\n{'='*60}")
    print(f"  S035 PHASE R3_1a: BLOCK TESTS -- {scenario}")
    print(f"{'='*60}\n")

    # Load feature registry
    registry = _load_json(s3, f"{RESULTS_PREFIX}/r3_feature_registry.json")
    if not registry:
        log.error("Feature registry not found. Run build_r3_feature_registry.py first.")
        return 1

    # Load assembled parquet
    from _coverage_common import OUTPUT_KEYS
    df = _load_parquet(s3, OUTPUT_KEYS[scenario])
    log.info("Loaded %s: %d rows x %d cols", scenario, len(df), len(df.columns))

    # Load supplements
    r1_supp = _load_supplement(s3, scenario, "r1")
    r2_supp = _load_supplement(s3, scenario, "r2")

    # Merge
    all_merge_keys = ["zcta_id", "event"]
    merged = df.copy()
    for supp, label in [(r1_supp, "R1"), (r2_supp, "R2")]:
        mk = [k for k in all_merge_keys if k in supp.columns]
        supp_cols = [c for c in supp.columns if c not in merged.columns or c in mk]
        if supp_cols:
            merged = merged.merge(supp[supp_cols], on=mk, how="left")
            log.info("  Merged %s supplement on %s: +%d cols", label, mk, len(supp_cols) - len(mk))

    # Load folds
    folds_key = f"folds/{scenario}_folds.parquet"
    folds_df = _load_parquet(s3, folds_key)
    merged = merged.merge(folds_df, on=all_merge_keys, how="left")

    # Available R2 features
    available_r2 = [f for f in R2_FEATURES if f in merged.columns]
    log.info("R2 features available: %d/%d", len(available_r2), len(R2_FEATURES))

    # Encode categoricals
    for col in merged.columns:
        if merged[col].dtype == object:
            codes, _ = pd.factorize(merged[col])
            merged[col] = codes.astype(np.float32)
            merged.loc[merged[col] < 0, col] = np.nan

    # Run block tests per target
    all_block_results = []
    all_block_certs = []

    for target_col, task, transform in TARGETS:
        if target_col not in merged.columns:
            log.warning("Target %s not in data, skipping", target_col)
            continue
        valid = merged[target_col].dropna()
        if len(valid) < 20 or valid.nunique() < 2:
            log.warning("Target %s has insufficient data, skipping", target_col)
            continue

        y_col = target_col
        if transform == "log1p":
            merged["_y"] = np.log1p(merged[target_col].clip(lower=0).astype(float))
            y_col = "_y"

        log.info("\n--- Target: %s (%s) ---", target_col, task)

        for block_name, block_feats in BLOCK_FEATURES.items():
            result = run_block_tests(
                block_name=block_name,
                block_features=block_feats,
                merged=merged,
                available_r2=available_r2,
                target_col=target_col,
                y_col=y_col,
                task=task,
                scenario=scenario,
            )
            all_block_results.append(result)

            cert = compute_block_certificate(result)
            all_block_certs.append(cert)

    # Assemble output
    output = {
        "phase": "R3_1a_block_tests",
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_blocks": len(BLOCK_FEATURES),
        "n_targets": sum(1 for t, _, _ in TARGETS if t in merged.columns),
        "block_results": all_block_results,
    }

    cert_output = {
        "phase": "R3_1a_block_certificates",
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "certificates": all_block_certs,
    }

    # Write local
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"r3_block_tests_{scenario}.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    with open(out_dir / f"r3_block_certificates_{scenario}.json", "w") as f:
        json.dump(cert_output, f, indent=2, default=str)

    if args.upload:
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/r3_block_tests_{scenario}.json", output)
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/r3_block_certificates_{scenario}.json", cert_output)
        log.info("Uploaded to S3")

    # Summary
    log.info("\n=== R3_1a Block Tests Summary (%s) ===", scenario)
    for cert in all_block_certs:
        status = cert.get("certificate_status", "?")
        ev = cert.get("evidence", {})
        indep = ev.get("independent_signal", {})
        marg = ev.get("marginal_signal", {})
        solver_tag = indep.get("solver_used") or "none"
        if indep.get("fallback_triggered"):
            solver_tag += "*"  # asterisk marks fallback
        log.info(
            "  %s/%s [%s]: R=%.3f S=%.3f N=%.3f "
            "indep=%.4f(%s) delta=%.4f(%s)",
            cert["block"], cert["target"], status,
            cert.get("R", 0) or 0, cert.get("S_sup", 0) or 0,
            cert.get("N", 0) or 0,
            indep.get("metric") or 0, solver_tag,
            marg.get("delta") or 0, ev.get("ablation_interpretation") or "n/a",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
