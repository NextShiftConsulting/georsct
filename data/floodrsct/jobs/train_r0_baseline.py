#!/usr/bin/env python3
"""
train_r0_baseline.py -- Phase 1: Fold generation + R0 tabular baseline.

Single SageMaker job that:
  1. Loads assembled parquet + crosswalk
  2. Generates deterministic fold assignments (random, spatial-blocked, leave-event-out)
  3. Trains HistGBDT + Ridge on R0 static features across 3 targets x 3 splits
  4. Uploads folds, metrics, and predictions to S3

R0 features are STATIC only (ZCTA-level, invariant across events).
Event-level features (rainfall, storm track) are R2.

Targets with zero variance or all-null in a scenario are skipped.

Usage:
    python train_r0_baseline.py --scenario houston --upload
    python train_r0_baseline.py --scenario southwest_florida --upload
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

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    BUCKET, SCENARIOS, get_s3_client, load_processed_parquet, load_crosswalk,
)
from generate_folds import generate_folds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

SEED = 42
RESULTS_PREFIX = "results/s035"

# ---------------------------------------------------------------------------
# R0 feature list: STATIC ZCTA features only.
# These do NOT vary by event. Order doesn't matter; missing columns are dropped.
# See MODELS.md for rationale.
# ---------------------------------------------------------------------------
R0_FEATURES = [
    # Flood zone exposure
    "flood_pct_zone_a",
    "flood_pct_zone_x",
    "flood_pct_zone_x500",
    # Historical flood record
    "flood_event_count",
    "flood_event_count_5y",
    "flood_events_per_year",
    "flood_property_damage_k",
    "flood_crop_damage_k",
    # Terrain
    "elevation_m_msl",
    "slope_mean_pct",
    "twi_twi",
    # Coastal / geographic
    "coastal_distance_m",
    "latitude",
    "longitude",
    # Demographics (ACS)
    "acs_total_pop",
    "acs_median_hh_income",
    "acs_pct_below_poverty",
    "acs_pct_renter_occupied",
    "acs_pct_owner_occupied",
    "acs_pct_vacant",
    "acs_pct_no_vehicle",
    "acs_median_home_value",
    "acs_median_year_built",
    # Social vulnerability (SVI)
    "svi_overall",
    "svi_socioeconomic",
    "svi_household_disability",
    "svi_minority_language",
    "svi_housing_transport",
    # NFIP insurance (temporally-gated historical base rate).
    # Built by build_nfip_historical.py: claims with dateOfLoss < event start.
    # Enforces IBNR temporal boundary -- same-event claims excluded.
    "nfip_historical_frequency",
    "nfip_historical_severity",
    # Infrastructure
    "hifld_nearest_hospital_km",
    "hifld_n_hospitals",
    "population",
    # NOTE: storm_min_dist_km and storm_landfall_category moved to R2
    # (event-level features, not static). See DOE_R2_temporal.md.
]

# Targets: (column_name, task_type, transform)
TARGETS = [
    ("obs_nfip_event_claims", "regression", "log1p"),
    ("obs_has_311",           "classification", None),
    ("obs_has_hwm",           "classification", None),
]

# Split column names in the folds parquet
SPLITS = {
    "random": "fold_random",
    "spatial_blocked": "fold_spatial_blocked",
    "leave_event_out": "fold_leave_event_out",
}


@dataclass
class RunResult:
    scenario: str
    target: str
    task: str
    solver: str
    split: str
    fold: str
    n_train: int
    n_test: int
    metrics: dict
    naive_baseline: dict
    features_used: int
    timestamp: str


def _available_features(df: pd.DataFrame) -> list[str]:
    """Return R0 features that exist in df and have at least some non-null values."""
    available = []
    for f in R0_FEATURES:
        if f in df.columns and df[f].notna().any():
            available.append(f)
    return available


def _check_target(df: pd.DataFrame, col: str, task: str) -> bool:
    """Return True if target is usable (exists, has variation, not all-null)."""
    if col not in df.columns:
        log.warning("Target %s not in columns, skipping", col)
        return False
    valid = df[col].dropna()
    if len(valid) < 20:
        log.warning("Target %s has only %d non-null values, skipping", col, len(valid))
        return False
    if valid.nunique() < 2:
        log.warning("Target %s has no variation (all=%s), skipping", col, valid.iloc[0])
        return False
    return True


def _train_histgbdt(X_train, y_train, X_test, y_test, task: str) -> tuple:
    """Train HistGradientBoosting and return (predictions, metrics)."""
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        HistGradientBoostingClassifier,
    )

    if task == "classification":
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=SEED,
        )
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = _classification_metrics(y_test, y_pred, y_pred_proba)
        return y_pred_proba, metrics
    else:
        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=SEED,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = _regression_metrics(y_test, y_pred)
        return y_pred, metrics


def _train_ridge(X_train, y_train, X_test, y_test, task: str) -> tuple:
    """Train Ridge pipeline (impute + scale + model) and return (predictions, metrics)."""
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
        # RidgeClassifier has decision_function, not predict_proba
        y_score = pipe.decision_function(X_test)
        metrics = _classification_metrics(y_test, y_pred, y_score)
        return y_score, metrics
    else:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = _regression_metrics(y_test, y_pred)
        return y_pred, metrics


def _regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _classification_metrics(y_true, y_pred, y_score) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        m["roc_auc"] = None  # single class in test fold
    return m


def _naive_baseline(y_train, y_test, task: str) -> dict:
    """Compute naive baseline: mean predictor (regression) or majority class (classification)."""
    if task == "classification":
        majority = int(np.round(y_train.mean()))
        y_naive = np.full_like(y_test, majority)
        return _classification_metrics(y_test, y_naive, np.full_like(y_test, y_train.mean(), dtype=float))
    else:
        mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)
        return _regression_metrics(y_test, mean_pred)


SOLVERS = {
    "histgbdt": _train_histgbdt,
    "ridge": _train_ridge,
}


def run_split(
    df: pd.DataFrame,
    folds_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    task: str,
    transform: str | None,
    solver_name: str,
    split_name: str,
    scenario: str,
    prediction_rows: list[dict] | None = None,
) -> list[RunResult]:
    """Run one (target, solver, split) combination across all folds.

    If prediction_rows is not None and split_name == "spatial_blocked",
    appends per-row predictions for kappa diagnostics (Moran's I).
    """
    ts = datetime.now(timezone.utc).isoformat()
    fold_col = SPLITS[split_name]
    solver_fn = SOLVERS[solver_name]

    # Merge folds with data
    merged = df.merge(folds_df[["zcta_id", "event", fold_col]], on=["zcta_id", "event"])

    # Drop rows with null target
    valid_mask = merged[target_col].notna()
    merged = merged[valid_mask].copy()

    # Apply target transform
    y_col = target_col
    if transform == "log1p":
        merged["_y"] = np.log1p(merged[target_col].clip(lower=0).astype(float))
        y_col = "_y"

    X_all = merged[features].values.astype(np.float32)
    y_all = merged[y_col].values.astype(np.float32)

    # Get fold IDs
    fold_ids = sorted(merged[fold_col].unique())
    results = []

    for fold_id in fold_ids:
        test_mask = merged[fold_col] == fold_id
        train_mask = ~test_mask

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(X_test) == 0 or len(X_train) == 0:
            log.warning("Empty fold %s in split %s, skipping", fold_id, split_name)
            continue

        # Check target variation in train
        if len(np.unique(y_train)) < 2 and task == "classification":
            log.warning("No target variation in train fold %s, skipping", fold_id)
            continue

        y_pred, metrics = solver_fn(X_train, y_train, X_test, y_test, task)
        naive = _naive_baseline(y_train, y_test, task)

        results.append(RunResult(
            scenario=scenario,
            target=target_col,
            task=task,
            solver=solver_name,
            split=split_name,
            fold=str(fold_id),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            metrics=metrics,
            naive_baseline=naive,
            features_used=len(features),
            timestamp=ts,
        ))

        # Save per-row predictions for spatial_blocked (kappa diagnostics)
        if prediction_rows is not None and split_name == "spatial_blocked":
            test_idx = merged.index[test_mask]
            for i, idx in enumerate(test_idx):
                prediction_rows.append({
                    "zcta_id": str(merged.at[idx, "zcta_id"]),
                    "event": str(merged.at[idx, "event"]),
                    "target": target_col,
                    "solver": solver_name,
                    "split": split_name,
                    "fold": str(fold_id),
                    "y_true": float(y_all[merged.index.get_loc(idx)]),
                    "y_pred": float(y_pred[i]),
                })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate folds + train R0 baseline"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()
    scenario = args.scenario

    print(f"\n{'='*60}")
    print(f"  S035 PHASE 1: R0 BASELINE -- {scenario}")
    print(f"{'='*60}\n")

    # --- Load data ---
    df = load_processed_parquet(s3, scenario)
    df["zcta_id"] = df["zcta_id"].astype(str)
    if "event" in df.columns:
        df["event"] = df["event"].astype(str)

    log.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    # --- Load crosswalk ---
    try:
        xwalk = load_crosswalk(s3)
        xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)
        zcta_county = dict(zip(xwalk["zcta_id"], xwalk["county_fips"].astype(str)))
    except Exception as e:
        log.warning("Crosswalk unavailable (%s), blocking on ZIP3 only", e)
        zcta_county = {}

    # --- Generate folds ---
    folds_df, fold_meta = generate_folds(df, zcta_county, args.seed)
    log.info("Folds: strategy=%s, %d rows", fold_meta["block_strategy"], len(folds_df))

    # Upload folds
    if args.upload:
        buf = io.BytesIO()
        folds_df.to_parquet(buf, index=False)
        buf.seek(0)
        fold_key = f"folds/{scenario}_folds.parquet"
        s3.put_object(Bucket=BUCKET, Key=fold_key, Body=buf.read())
        s3.put_object(
            Bucket=BUCKET,
            Key=f"folds/{scenario}_folds_meta.json",
            Body=json.dumps(fold_meta, indent=2).encode(),
            ContentType="application/json",
        )
        log.info("Uploaded folds to s3://%s/%s", BUCKET, fold_key)

    # --- Load NFIP historical supplement (temporally-gated) ---
    nfip_key = f"processed/{scenario}/{scenario}_nfip_historical.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=nfip_key)
        nfip_hist = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        nfip_hist["zcta_id"] = nfip_hist["zcta_id"].astype(str)
        join_cols = ["zcta_id", "event"] if "event" in nfip_hist.columns else ["zcta_id"]
        df = df.merge(nfip_hist, on=join_cols, how="left")
        log.info("NFIP historical supplement merged: %d rows from %s", len(nfip_hist), nfip_key)
    except Exception as exc:
        raise RuntimeError(
            f"NFIP historical supplement missing: {nfip_key}. "
            f"Run build_nfip_historical.py --scenario {scenario} --upload first."
        ) from exc

    # --- Identify usable features ---
    features = _available_features(df)
    log.info("R0 features: %d / %d available", len(features), len(R0_FEATURES))
    log.info("  Available: %s", features)
    missing = [f for f in R0_FEATURES if f not in features]
    if missing:
        log.info("  Missing: %s", missing)

    # --- Train all combinations ---
    all_results: list[RunResult] = []
    prediction_rows: list[dict] = []  # Per-row predictions for kappa diagnostics

    for target_col, task, transform in TARGETS:
        if not _check_target(df, target_col, task):
            continue

        log.info("\n--- Target: %s (%s, transform=%s) ---", target_col, task, transform)

        for solver_name in SOLVERS:
            for split_name in SPLITS:
                log.info("  %s / %s", solver_name, split_name)
                try:
                    results = run_split(
                        df, folds_df, features,
                        target_col, task, transform,
                        solver_name, split_name, scenario,
                        prediction_rows=prediction_rows,
                    )
                    all_results.extend(results)

                    # Summarize
                    if results:
                        primary = "roc_auc" if task == "classification" else "rmse"
                        vals = [r.metrics.get(primary) for r in results if r.metrics.get(primary) is not None]
                        if vals:
                            log.info("    %s: mean=%.4f (n_folds=%d)", primary, np.mean(vals), len(vals))
                except Exception as e:
                    log.error("    FAILED: %s", e)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  R0 SUMMARY: {scenario}")
    print(f"  Total runs: {len(all_results)}")
    print(f"{'='*60}\n")

    # Group by target+solver+split for summary table
    summary_rows = []
    for r in all_results:
        primary = "roc_auc" if r.task == "classification" else "rmse"
        naive_primary = r.naive_baseline.get(primary)
        model_primary = r.metrics.get(primary)

        if r.task == "regression" and naive_primary and naive_primary > 0:
            skill_ratio = model_primary / naive_primary
        elif r.task == "classification" and naive_primary is not None:
            skill_ratio = model_primary  # AUC is absolute
        else:
            skill_ratio = None

        summary_rows.append({
            "target": r.target,
            "solver": r.solver,
            "split": r.split,
            "fold": r.fold,
            "metric": primary,
            "model_value": model_primary,
            "naive_value": naive_primary,
            "skill_ratio": skill_ratio,
            "n_train": r.n_train,
            "n_test": r.n_test,
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))

    # --- Upload results ---
    results_payload = {
        "experiment": "s035-model-ladder",
        "phase": "r0_baseline",
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "representation": "R0",
        "features_used": features,
        "features_missing": missing,
        "fold_metadata": fold_meta,
        "runs": [asdict(r) for r in all_results],
    }

    results_json = json.dumps(results_payload, indent=2, default=str)

    if args.upload:
        key = f"{RESULTS_PREFIX}/r0_{scenario}.json"
        s3.put_object(
            Bucket=BUCKET, Key=key,
            Body=results_json.encode(),
            ContentType="application/json",
        )
        log.info("Uploaded results to s3://%s/%s", BUCKET, key)

        # Upload predictions parquet (spatial_blocked only, for kappa diagnostics)
        if prediction_rows:
            pred_df = pd.DataFrame(prediction_rows)
            buf = io.BytesIO()
            pred_df.to_parquet(buf, index=False)
            buf.seek(0)
            pred_key = f"{RESULTS_PREFIX}/r0_{scenario}_predictions.parquet"
            s3.put_object(Bucket=BUCKET, Key=pred_key, Body=buf.getvalue())
            log.info("Uploaded %d prediction rows to s3://%s/%s",
                     len(pred_df), BUCKET, pred_key)
    else:
        local = f"/tmp/r0_{scenario}.json"
        Path(local).write_text(results_json)
        log.info("Wrote %s", local)

    print(results_json)


if __name__ == "__main__":
    main()
