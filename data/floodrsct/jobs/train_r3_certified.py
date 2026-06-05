#!/usr/bin/env python3
"""
train_r3_certified.py -- Phase R3_2: Certified representation training.

Trains models using ONLY blocks that received DGM EXECUTE decisions
in R3_1c. Three variants are trained for comparison:

  1. R3_2 headline:    EXECUTE @ FIRST/SECOND gear only
  2. R3_2 full:        All EXECUTE blocks (any gear) + post-morph survivors
  3. R3_2 stabilized:  Headline + THIRD gear diagnostic-stabilizers

All variants use the same folds, targets, solvers, and split protocols
as R0/R1/R2 for direct comparison.

Usage:
    python train_r3_certified.py --scenario houston --upload
    python train_r3_certified.py --scenario houston --dry-run
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
from _coverage_common import BUCKET, SCENARIOS, get_s3_client
from _s3_result import upload_json_result

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

# R0 baseline features (always included -- R3 = R0 + admitted blocks)
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

# Block-to-feature mapping (must match compute_r3_block_tests.py)
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

SPLITS = {
    "random": "fold_random",
    "spatial_blocked": "fold_spatial_blocked",
    "leave_event_out": "fold_leave_event_out",
}

R3_VARIANTS = {
    "headline": {
        "tiers": {"headline"},
        "description": "EXECUTE @ FIRST/SECOND gear only",
    },
    "full": {
        "tiers": {"headline", "diagnostic-stabilizer", "marginal"},
        "description": "All EXECUTE blocks (any gear)",
    },
    "stabilized": {
        "tiers": {"headline", "diagnostic-stabilizer"},
        "description": "Headline + THIRD gear diagnostic-stabilizers",
    },
}


@dataclass
class RunResult:
    scenario: str
    target: str
    task: str
    solver: str
    split: str
    fold: str
    variant: str
    n_train: int
    n_test: int
    metrics: dict
    naive_baseline: dict
    features_used: int
    admitted_blocks: list
    timestamp: str


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
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


# ---------------------------------------------------------------------------
# Solvers (identical to R0/R1/R2)
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
        log.warning("HistGBDT fit failed (n_train=%d): %s", len(X_train), exc)
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


def _naive_baseline(y_train, y_test, task: str) -> dict:
    if task == "classification":
        majority = int(np.round(y_train.mean()))
        y_naive = np.full_like(y_test, majority)
        return _classification_metrics(
            y_test, y_naive,
            np.full_like(y_test, y_train.mean(), dtype=float),
        )
    else:
        mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)
        return _regression_metrics(y_test, mean_pred)


SOLVERS = {"histgbdt": _train_histgbdt, "ridge": _train_ridge}


# ---------------------------------------------------------------------------
# Feature set construction from admission table
# ---------------------------------------------------------------------------

def build_variant_features(
    admission_table: list[dict],
    variant_name: str,
    scenario: str,
    target: str,
    available_cols: set[str],
) -> tuple[list[str], list[str]]:
    """Build the feature list for a given R3 variant.

    Returns (features, admitted_block_names).
    """
    variant_def = R3_VARIANTS[variant_name]
    admitted_tiers = variant_def["tiers"]

    # Filter admission table for this scenario/target
    admitted_blocks = []
    for entry in admission_table:
        if entry.get("scenario") != scenario:
            continue
        if entry.get("target") != target:
            continue
        if entry.get("admission_tier") in admitted_tiers:
            block_name = entry.get("block")
            if block_name and block_name not in admitted_blocks:
                admitted_blocks.append(block_name)

    # Build feature set: R0 + admitted block features
    features = [f for f in R0_FEATURES if f in available_cols]
    for block_name in admitted_blocks:
        block_feats = BLOCK_FEATURES.get(block_name, [])
        for f in block_feats:
            if f in available_cols and f not in features:
                features.append(f)

    return features, admitted_blocks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase R3_2: Certified representation training"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would train R3 certified models for %s", args.scenario)
        log.info("  Variants: %s", list(R3_VARIANTS.keys()))
        return 0

    s3 = get_s3_client()
    scenario = args.scenario

    print(f"\n{'='*60}")
    print(f"  S035 PHASE R3_2: CERTIFIED TRAINING -- {scenario}")
    print(f"{'='*60}\n")

    # Load admission table
    admission_data = _load_json(s3, f"{RESULTS_PREFIX}/r3_block_admission_table.json")
    if not admission_data:
        log.error("Admission table not found. Run compute_r3_block_admission.py first.")
        return 1
    admission_table = admission_data.get("admission_table", [])
    log.info("Loaded admission table: %d entries", len(admission_table))

    # Load data
    from _coverage_common import OUTPUT_KEYS
    df = _load_parquet(s3, OUTPUT_KEYS[scenario])
    r1_supp = _load_supplement(s3, scenario, "r1")
    r2_supp = _load_supplement(s3, scenario, "r2")

    all_merge_keys = ["zcta_id", "event"]
    merged = df.copy()
    for supp in [r1_supp, r2_supp]:
        mk = [k for k in all_merge_keys if k in supp.columns]
        supp_cols = [c for c in supp.columns if c not in merged.columns or c in mk]
        if supp_cols:
            merged = merged.merge(supp[supp_cols], on=mk, how="left")

    folds_df = _load_parquet(s3, f"folds/{scenario}_folds.parquet")
    merged = merged.merge(folds_df, on=all_merge_keys, how="left")

    # Encode categoricals
    for col in merged.columns:
        if merged[col].dtype == object:
            codes, _ = pd.factorize(merged[col])
            merged[col] = codes.astype(np.float32)
            merged.loc[merged[col] < 0, col] = np.nan

    available_cols = set(merged.columns)
    all_results = []
    prediction_rows = []

    for target_col, task, transform in TARGETS:
        if target_col not in merged.columns:
            continue
        valid = merged[target_col].dropna()
        if len(valid) < 20 or valid.nunique() < 2:
            continue

        y_col = target_col
        if transform == "log1p":
            merged["_y"] = np.log1p(merged[target_col].clip(lower=0).astype(float))
            y_col = "_y"

        log.info("\n--- Target: %s (%s) ---", target_col, task)

        for variant_name in R3_VARIANTS:
            features, admitted_blocks = build_variant_features(
                admission_table, variant_name, scenario, target_col, available_cols,
            )
            log.info("  Variant %s: %d features from blocks %s",
                     variant_name, len(features), admitted_blocks)

            if len(features) == 0:
                log.warning("  No features for variant %s, skipping", variant_name)
                continue

            # Drop rows where target is NaN
            valid_mask = merged[y_col].notna().values
            df_valid = merged[valid_mask]
            X_all = df_valid[features].values.astype(np.float32)
            y_all = df_valid[y_col].values.astype(np.float32)

            MIN_FOLD_SAMPLES = 10

            for solver_name, solver_fn in SOLVERS.items():
                for split_name, fold_col in SPLITS.items():
                    if fold_col not in df_valid.columns:
                        continue

                    fold_ids = sorted(df_valid[fold_col].unique())
                    ts = datetime.now(timezone.utc).isoformat()

                    for fold_id in fold_ids:
                        test_mask = (df_valid[fold_col] == fold_id).values
                        train_mask = ~test_mask
                        X_train = X_all[train_mask]
                        y_train = y_all[train_mask]
                        X_test = X_all[test_mask]
                        y_test = y_all[test_mask]

                        if len(X_train) < MIN_FOLD_SAMPLES or len(X_test) == 0:
                            continue
                        if len(np.unique(y_train)) < 2 and task == "classification":
                            continue

                        y_pred, metrics = solver_fn(X_train, y_train, X_test, y_test, task)
                        naive = _naive_baseline(y_train, y_test, task)

                        all_results.append(RunResult(
                            scenario=scenario,
                            target=target_col,
                            task=task,
                            solver=solver_name,
                            split=split_name,
                            fold=str(fold_id),
                            variant=variant_name,
                            n_train=int(train_mask.sum()),
                            n_test=int(test_mask.sum()),
                            metrics=metrics,
                            naive_baseline=naive,
                            features_used=len(features),
                            admitted_blocks=admitted_blocks,
                            timestamp=ts,
                        ))

                        # Collect spatial-blocked predictions for diagnostics
                        if "blocked" in split_name and solver_name == "histgbdt":
                            test_idx = merged.index[test_mask]
                            for i, idx in enumerate(test_idx):
                                prediction_rows.append({
                                    "zcta_id": str(merged.at[idx, "zcta_id"]),
                                    "event": str(merged.at[idx, "event"]),
                                    "target": target_col,
                                    "solver": solver_name,
                                    "split": split_name,
                                    "fold": str(fold_id),
                                    "variant": variant_name,
                                    "y_true": float(y_all[merged.index.get_loc(idx)]),
                                    "y_pred": float(y_pred[i]),
                                })

    # Assemble output
    output = {
        "phase": "R3_2_certified_training",
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "representation": "R3",
        "variants": {k: v["description"] for k, v in R3_VARIANTS.items()},
        "runs": [asdict(r) for r in all_results],
    }

    # Write local
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"r3_{scenario}.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    if prediction_rows:
        pred_df = pd.DataFrame(prediction_rows)
        pred_path = out_dir / f"r3_{scenario}_predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
        log.info("Written predictions: %d rows", len(pred_df))

    if args.upload:
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/r3_{scenario}.json", output)
        if prediction_rows:
            pred_df = pd.DataFrame(prediction_rows)
            buf = io.BytesIO()
            pred_df.to_parquet(buf, index=False)
            buf.seek(0)
            s3.put_object(
                Bucket=BUCKET,
                Key=f"{RESULTS_PREFIX}/r3_{scenario}_predictions.parquet",
                Body=buf.getvalue(),
            )
        log.info("Uploaded to S3")

    # Summary
    log.info("\n=== R3_2 Training Summary (%s) ===", scenario)
    from collections import Counter
    variant_counts = Counter(r.variant for r in all_results)
    for v, count in sorted(variant_counts.items()):
        log.info("  %s: %d runs", v, count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
