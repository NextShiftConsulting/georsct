#!/usr/bin/env python3
"""
train_r2_temporal.py -- Phase 3: R2 representation (R1 + temporal event dynamics).

Loads R0 folds (from Phase 1) + R1 supplement + R2 supplement parquets.
Trains the same solvers, targets, and splits as R0/R1 -- only the feature set changes.

R2 adds event-dynamic columns from MRMS hourly rainfall and tide gauges:
  - peak_1h_mm, peak_3h_mm, peak_6h_mm: rolling rainfall maxima
  - storm_duration_h: hours with rainfall > 1mm
  - time_to_peak_h: hours from first rain to peak
  - rainfall_intensity_cv: coefficient of variation of hourly rainfall
  - tide_peak_m: peak water level from nearest tide station
  - surge_rain_lag_h: hours between peak rainfall and peak surge

DOE constraint: same folds, same solver hyperparameters, same targets as R0/R1.

Usage:
    python train_r2_temporal.py --scenario houston --upload
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
from _s3_result import upload_json_result
from _validate_contract import check_causal_boundary
from _coverage_common import (
    BUCKET, SCENARIOS, get_s3_client, load_processed_parquet,
    load_adjacency,
)

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
# R0 features (identical to train_r0_baseline.py)
# ---------------------------------------------------------------------------
R0_FEATURES = [
    "flood_pct_zone_a",
    "flood_pct_zone_x",
    "flood_pct_zone_x500",
    # QUARANTINED: county-level constants, zero within-scenario variance,
    # not temporally gated. Replaced by nfip_historical_frequency/severity.
    # "flood_event_count",
    # "flood_event_count_5y",
    # "flood_events_per_year",
    # "flood_property_damage_k",
    # "flood_crop_damage_k",
    "elevation_m_msl",
    "slope_mean_pct",
    "twi_twi",
    "coastal_distance_m",
    "latitude",
    "longitude",
    "acs_total_pop",
    "acs_median_hh_income",
    "acs_pct_below_poverty",
    "acs_pct_renter_occupied",
    "acs_pct_owner_occupied",
    "acs_pct_vacant",
    "acs_pct_no_vehicle",
    "acs_median_home_value",
    "acs_median_year_built",
    "svi_overall",
    "svi_socioeconomic",
    "svi_household_disability",
    "svi_minority_language",
    "svi_housing_transport",
    "nfip_historical_frequency",
    "nfip_historical_severity",
    "hifld_nearest_hospital_km",
    "hifld_n_hospitals",
    "population",
    # Land cover (NLCD 2021)
    "impervious_pct",
    "cropland_pct",
]

# ---------------------------------------------------------------------------
# R1 features (identical to train_r1_hydrology.py)
# ---------------------------------------------------------------------------
R1_UNIVERSAL = [
    "nhd_catchment_area_km2",
    "slope_basin_slope",
    "slope_stream_slope",
    "twi_acc_twi",
    "twi_tot_twi",
    "drive_min_to_county_centroid",
    "drive_min_to_county_seat",
    "drive_min_to_nearest_hospital",
    "hifld_n_hospital_beds",
    "hifld_n_pharmacies",
    "hifld_nearest_pharmacy_km",
    "hifld_nearest_trauma_center_km",
    # QUARANTINED: county-level Storm Events totals, not temporally gated,
    # outcome-from-outcome leakage risk. No ZCTA-level replacement exists.
    # "flood_deaths",
    # "flood_injuries",
]

R1_SCENARIO_SPECIFIC = [
    "upstream_catchment_km2",
    "hcfcd_drainage_district",
    "levee_nearest_km",
    "levee_condition_rating",
    "sewershed_name",
    "slosh_max_surge_m",
]

# ---------------------------------------------------------------------------
# W-matrix spatial features (identical to train_r1_hydrology.py)
# DOE invariant: R2 = R1 + additions. Dropping W-matrix would silently
# redefine R2 as (R0 + hydro + temporal), confounding the R1->R2 delta.
# ---------------------------------------------------------------------------
R1_WMATRIX = [
    "zcta_degree",
    "zcta_mean_neighbor_dist_km",
    "wlag_flood_zone_pct",
    "wlag_population_density",
    "wlag_median_income",
    "wlag_impervious_pct",
    "wlag_cropland_pct",
    "wlag_rainfall_mm",
    "wlag_nfip_claims",       # TARGET LAG -- per-fold recompute required
]

R1_HYDRO = R1_UNIVERSAL + R1_SCENARIO_SPECIFIC

# ---------------------------------------------------------------------------
# R2 temporal features (from build_r2_features.py)
# ---------------------------------------------------------------------------
R2_TEMPORAL = [
    "peak_1h_mm",
    "peak_3h_mm",
    "peak_6h_mm",
    "storm_duration_h",
    "time_to_peak_h",
    "rainfall_intensity_cv",
    "tide_peak_m",
    "surge_rain_lag_h",
    # Storm track features (event-level, moved from R0 where they were mislabeled static)
    "storm_min_dist_km",
    "storm_landfall_category",
]

# R2 = R1 + temporal = R0 + hydro + W-matrix + temporal (DOE invariant)
R2_FEATURES = R0_FEATURES + R1_HYDRO + R1_WMATRIX + R2_TEMPORAL

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
    features_from_r1_supplement: int
    features_from_r2_supplement: int
    wlag_nan_rates: dict  # {wlag_col: frac_nan_in_test_rows} per recomputed wlag
    timestamp: str


def _load_supplement(s3, scenario: str, level: str) -> pd.DataFrame:
    """Load R1 or R2 supplement parquet from S3. Hard failure if missing."""
    key = f"processed/{scenario}/{scenario}_{level}_supplement.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        log.info("%s supplement: %d rows x %d cols from %s",
                 level.upper(), len(df), len(df.columns), key)
        return df
    except Exception as exc:
        raise RuntimeError(
            f"{level.upper()} supplement missing: s3://{BUCKET}/{key}. "
            f"Run build_{level}_features.py --scenario {scenario} --upload first. "
            f"{level.upper()} arm is meaningless without its supplement features."
        ) from exc


def _load_folds(s3, scenario: str) -> pd.DataFrame:
    key = f"folds/{scenario}_folds.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    log.info("Folds: %d rows from %s", len(df), key)
    return df


def _encode_categoricals(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for col in features:
        if col in df.columns and df[col].dtype == object:
            codes, _ = pd.factorize(df[col])
            df[col] = codes.astype(np.float32)
            df.loc[df[col] < 0, col] = np.nan
    return df


def _build_neighbor_dict(adjacency) -> dict[str, list[str]]:
    """Build {zcta_id: [neighbor,...]} from adjacency (dict or edge-list DataFrame).
    Ported from train_r1_hydrology.py -- identical logic."""
    if adjacency is None:
        return {}
    if isinstance(adjacency, dict):
        return {str(k): [str(v) for v in vs] for k, vs in adjacency.items()}
    df = adjacency
    cols = list(df.columns)
    candidates = [
        ("zcta_id", "neighbor_zcta_id"), ("zcta_a", "zcta_b"),
        ("src", "dst"), ("src_zcta", "dst_zcta"), ("from", "to"),
        ("zcta_id_1", "zcta_id_2"),
    ]
    pair = next((c for c in candidates if c[0] in cols and c[1] in cols), None)
    if pair is None:
        if len(cols) >= 2:
            pair = (cols[0], cols[1])
        else:
            raise RuntimeError(f"Cannot identify edge columns in adjacency: {cols}")
    a, b = pair
    neigh: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        u, v = str(row[a]), str(row[b])
        neigh.setdefault(u, []).append(v)
        neigh.setdefault(v, []).append(u)
    return {k: sorted(set(vs)) for k, vs in neigh.items()}


def _recompute_wlag_per_fold(
    merged: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    neighbors: dict[str, list[str]],
    source_col: str,
) -> pd.Series:
    """Recompute spatial lag using ONLY training ZCTAs' values.
    Ported from train_r1_hydrology.py -- identical logic.

    Test ZCTAs receive lag from training neighbors only (NaN if all
    neighbors are in test fold).
    """
    train_rows = merged[train_mask]
    zcta_vals = train_rows.groupby("zcta_id")[source_col].mean().to_dict()

    wlag_vals = pd.Series(np.nan, index=merged.index, dtype=np.float64)
    for idx, row in merged.iterrows():
        zid = str(row["zcta_id"])
        nbrs = neighbors.get(zid, [])
        if not nbrs:
            continue
        vals = []
        for nb in nbrs:
            v = zcta_vals.get(nb)
            if v is not None and not np.isnan(v):
                vals.append(v)
        if vals:
            wlag_vals.at[idx] = float(np.mean(vals))

    return wlag_vals


def _available_features(df: pd.DataFrame) -> tuple[list[str], int, int]:
    """Return R2 features present in df. Track R1 and R2 supplement counts."""
    available = []
    r1_supp_names = {"nhd_catchment_area_km2", "levee_nearest_km",
                     "levee_condition_rating", "sewershed_name"}
    r2_supp_names = set(R2_TEMPORAL)
    r1_count = 0
    r2_count = 0
    for f in R2_FEATURES:
        if f in df.columns and df[f].notna().any():
            available.append(f)
            if f in r1_supp_names:
                r1_count += 1
            if f in r2_supp_names:
                r2_count += 1
    return available, r1_count, r2_count


def _check_target(df: pd.DataFrame, col: str, task: str) -> bool:
    if col not in df.columns:
        log.warning("Target %s not in columns, skipping", col)
        return False
    valid = df[col].dropna()
    if len(valid) < 20:
        log.warning("Target %s has only %d non-null values, skipping", col, len(valid))
        return False
    if valid.nunique() < 2:
        log.warning("Target %s has no variation, skipping", col)
        return False
    return True


# ---------------------------------------------------------------------------
# Solvers: IDENTICAL to R0/R1 (same hyperparams, same code)
# ---------------------------------------------------------------------------

def _train_histgbdt(X_train, y_train, X_test, y_test, task: str) -> tuple:
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


def _nan_to_none(v: float) -> float | None:
    """Convert NaN/Inf to None for JSON-safe serialization."""
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
        # Newer sklearn returns nan instead of raising ValueError
        # when only one class is present in y_true
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


SOLVERS = {
    "histgbdt": _train_histgbdt,
    "ridge": _train_ridge,
}


def run_split(
    df: pd.DataFrame,
    folds_df: pd.DataFrame,
    features: list[str],
    r1_supp_count: int,
    r2_supp_count: int,
    target_col: str,
    task: str,
    transform: str | None,
    solver_name: str,
    split_name: str,
    scenario: str,
    fold_col: str,
    prediction_rows: list[dict] | None = None,
    neighbors: dict[str, list[str]] | None = None,
) -> list[RunResult]:
    ts = datetime.now(timezone.utc).isoformat()
    solver_fn = SOLVERS[solver_name]

    merged = df.merge(folds_df[["zcta_id", "event", fold_col]], on=["zcta_id", "event"])
    valid_mask = merged[target_col].notna()
    merged = merged[valid_mask].copy()

    y_col = target_col
    if transform == "log1p":
        merged["_y"] = np.log1p(merged[target_col].clip(lower=0).astype(float))
        y_col = "_y"

    # ── Per-fold wlag recomputation specs (ported from R1) ──────────────
    perfold_wlag_specs: list[tuple[str, str]] = []

    if "wlag_nfip_claims" in features and target_col == "obs_nfip_event_claims":
        if neighbors is None:
            raise RuntimeError(
                "wlag_nfip_claims is in the feature set and the target is "
                "obs_nfip_event_claims, but no adjacency/neighbors were provided. "
                "The pre-computed wlag leaks test-fold target values. Refusing to run."
            )
        perfold_wlag_specs.append(("wlag_nfip_claims", "obs_nfip_event_claims"))

    if "wlag_rainfall_mm" in features:
        if neighbors is None:
            raise RuntimeError(
                "wlag_rainfall_mm is in the feature set but no adjacency/neighbors "
                "were provided. rainfall_total_mm is event-concurrent. Refusing to run."
            )
        perfold_wlag_specs.append(("wlag_rainfall_mm", "rainfall_total_mm"))

    perfold_wlag_indices: list[tuple[int, str, str]] = []
    for wlag_col, source_col in perfold_wlag_specs:
        col_idx = features.index(wlag_col)
        perfold_wlag_indices.append((col_idx, wlag_col, source_col))
        log.info("    Per-fold %s recomputation enabled (leakage mitigation)", wlag_col)

    X_all = merged[features].values.astype(np.float32)
    y_all = merged[y_col].values.astype(np.float32)
    fold_ids = sorted(merged[fold_col].unique())
    results = []

    for fold_id in fold_ids:
        test_mask = merged[fold_col] == fold_id
        train_mask = ~test_mask

        # Per-fold wlag recomputation (DOE leakage protocol)
        fold_nan_rates: dict[str, float] = {}
        for col_idx, wlag_col, source_col in perfold_wlag_indices:
            wlag_safe = _recompute_wlag_per_fold(
                merged, train_mask, test_mask, neighbors, source_col,
            )
            test_vals = wlag_safe[test_mask].values
            test_n = len(test_vals)
            test_nan = int(np.isnan(test_vals).sum())
            nan_frac = test_nan / test_n if test_n > 0 else 0.0
            fold_nan_rates[wlag_col] = round(nan_frac, 4)
            if test_nan > 0:
                log.info("    fold %s %s: %d/%d test rows NaN (%.1f%%)",
                         fold_id, wlag_col, test_nan, test_n, 100.0 * nan_frac)
            X_all[:, col_idx] = wlag_safe.values.astype(np.float32)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(X_test) == 0 or len(X_train) == 0:
            log.warning("Empty fold %s in split %s, skipping", fold_id, split_name)
            continue
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
            features_from_r1_supplement=r1_supp_count,
            features_from_r2_supplement=r2_supp_count,
            wlag_nan_rates=fold_nan_rates,
            timestamp=ts,
        ))

        if prediction_rows is not None and "blocked" in split_name:
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
        description="Phase 3: R2 temporal event dynamics training"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--folds-key",
        help="S3 key for external fold assignments (overrides default "
             "folds/{scenario}_folds.parquet). Read-only consumption.",
    )
    parser.add_argument(
        "--fold-col",
        help="Single fold column to run (e.g. fold_region_blocked). "
             "When omitted, iterates all SPLITS columns present in the folds.",
    )
    parser.add_argument(
        "--output-prefix", default=RESULTS_PREFIX,
        help=f"S3 output prefix (default: {RESULTS_PREFIX}). "
             "Redirect to sidecar namespace for robustness runs.",
    )
    args = parser.parse_args()

    s3 = get_s3_client()
    scenario = args.scenario

    # Hard gate: reject any feature that violates the causal boundary.
    # wlag_nfip_claims is post_event but DOE-approved with per-fold
    # recomputation (Change 13). The leakage gate below enforces this.
    check_causal_boundary(R2_FEATURES, exempt={"wlag_nfip_claims"})

    print(f"\n{'='*60}")
    print(f"  S035 PHASE 3: R2 TEMPORAL -- {scenario}")
    print(f"{'='*60}\n")

    # --- Load assembled parquet ---
    df = load_processed_parquet(s3, scenario)
    df["zcta_id"] = df["zcta_id"].astype(str)
    if "event" in df.columns:
        df["event"] = df["event"].astype(str)
    log.info("Assembled: %d rows x %d cols", len(df), len(df.columns))

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

    # --- Load R1 supplement and join ---
    r1_supp = _load_supplement(s3, scenario, "r1")
    r1_supp["zcta_id"] = r1_supp["zcta_id"].astype(str)
    assert r1_supp["zcta_id"].is_unique, (
        f"R1 supplement has {r1_supp['zcta_id'].duplicated().sum()} duplicate zcta_ids"
    )
    pre_rows = len(df)
    df = df.merge(r1_supp, on="zcta_id", how="left")
    assert len(df) == pre_rows, f"R1 join changed row count: {pre_rows} -> {len(df)}"
    log.info("After R1 join: %d cols", len(df.columns))

    # --- Load R2 supplement and join ---
    r2_supp = _load_supplement(s3, scenario, "r2")
    r2_supp["zcta_id"] = r2_supp["zcta_id"].astype(str)
    # R2 supplement is event-level (one row per zcta_id x event)
    join_cols = ["zcta_id"]
    if "event" in r2_supp.columns:
        r2_supp["event"] = r2_supp["event"].astype(str)
        join_cols.append("event")
    assert r2_supp[join_cols].duplicated().sum() == 0, (
        f"R2 supplement has duplicates on {join_cols}"
    )
    pre_rows = len(df)
    pre_cols = set(df.columns)
    df = df.merge(r2_supp, on=join_cols, how="left")
    assert len(df) == pre_rows, f"R2 join changed row count: {pre_rows} -> {len(df)}"
    new_cols = set(df.columns) - pre_cols
    log.info("R2 supplement added %d columns: %s", len(new_cols), sorted(new_cols))

    # --- Encode categoricals ---
    df = _encode_categoricals(df, R1_SCENARIO_SPECIFIC)

    # --- Load folds: external (sidecar) or Phase 1 default ---
    if args.folds_key:
        resp = s3.get_object(Bucket=BUCKET, Key=args.folds_key)
        folds_df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        log.info("External folds from %s: %d rows, columns=%s",
                 args.folds_key, len(folds_df), list(folds_df.columns))
    else:
        folds_df = _load_folds(s3, scenario)
    folds_df["zcta_id"] = folds_df["zcta_id"].astype(str)
    if "event" in folds_df.columns:
        folds_df["event"] = folds_df["event"].astype(str)

    # Build active splits: custom fold-col or all SPLITS (skip-missing-column)
    if args.fold_col:
        split_label = args.fold_col.removeprefix("fold_")
        active_splits = {split_label: args.fold_col}
    else:
        active_splits = SPLITS

    # --- Load adjacency for per-fold wlag recomputation (leakage mitigation) ---
    neighbors: dict[str, list[str]] | None = None
    try:
        adjacency = load_adjacency(s3)
        neighbors = _build_neighbor_dict(adjacency)
        log.info("Adjacency loaded: %d ZCTAs with neighbors", len(neighbors))
    except Exception as exc:
        log.warning("Adjacency unavailable (%s); per-fold wlag recompute disabled", exc)
        neighbors = None

    # --- Identify usable features ---
    features, r1_supp_count, r2_supp_count = _available_features(df)
    r0_count = sum(1 for f in features if f in R0_FEATURES)
    wlag_count = sum(1 for f in features if f in R1_WMATRIX)
    r1_count = sum(1 for f in features if f in R1_UNIVERSAL + R1_SCENARIO_SPECIFIC)
    log.info("R2 features: %d total (%d R0 + %d R1-hydro + %d W-matrix + %d R2-temporal)",
             len(features), r0_count, r1_count, wlag_count, r2_supp_count)

    # --- Hard gate: refuse leaky variants without adjacency ---
    if "wlag_nfip_claims" in features and not neighbors:
        raise RuntimeError(
            "R2 includes wlag_nfip_claims (target spatial lag) from R1's W-matrix, "
            f"but adjacency/neighbors are unavailable for {scenario}. The pre-computed "
            "wlag leaks test-fold target values into training. Refusing to run. "
            "Provide adjacency to enable per-fold recomputation."
        )

    missing = [f for f in R2_FEATURES if f not in features]
    if missing:
        log.info("Missing R2 features: %s", missing)

    # --- Train all combinations ---
    all_results: list[RunResult] = []
    prediction_rows: list[dict] = []

    for target_col, task, transform in TARGETS:
        if not _check_target(df, target_col, task):
            continue
        log.info("\n--- Target: %s (%s, transform=%s) ---", target_col, task, transform)

        for solver_name in SOLVERS:
            for split_name, fold_col in active_splits.items():
                if fold_col not in folds_df.columns:
                    log.info("  Skipping %s/%s: column %s not in folds",
                             solver_name, split_name, fold_col)
                    continue
                log.info("  %s / %s", solver_name, split_name)
                try:
                    results = run_split(
                        df, folds_df, features,
                        r1_supp_count, r2_supp_count,
                        target_col, task, transform,
                        solver_name, split_name, scenario, fold_col,
                        prediction_rows=prediction_rows,
                        neighbors=neighbors,
                    )
                    all_results.extend(results)
                    if results:
                        primary = "roc_auc" if task == "classification" else "rmse"
                        vals = [r.metrics.get(primary) for r in results
                                if r.metrics.get(primary) is not None]
                        if vals:
                            log.info("    %s: mean=%.4f (n_folds=%d)",
                                     primary, np.mean(vals), len(vals))
                except Exception as e:
                    log.error("    FAILED: %s", e)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  R2 SUMMARY: {scenario}")
    print(f"  Total runs: {len(all_results)}")
    print(f"  Features: {len(features)} ({r0_count} R0 + {r1_count} R1-hydro + {wlag_count} W-matrix + {r2_supp_count} R2-temporal)")
    print(f"{'='*60}\n")

    # --- Upload results ---
    output_prefix = args.output_prefix
    results_payload = {
        "experiment": "s035-model-ladder",
        "phase": "r2_temporal",
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "representation": "R2",
        "features_used": features,
        "features_missing": missing,
        "r0_feature_count": r0_count,
        "r1_hydro_feature_count": r1_count,
        "wmatrix_feature_count": wlag_count,
        "r2_temporal_feature_count": r2_supp_count,
        "output_prefix": output_prefix,
        "runs": [asdict(r) for r in all_results],
    }

    results_json = json.dumps(results_payload, indent=2, default=str)

    if args.upload:
        key = f"{output_prefix}/r2_{scenario}.json"
        upload_json_result(s3, BUCKET, key, results_payload)

        if prediction_rows:
            pred_df = pd.DataFrame(prediction_rows)
            buf = io.BytesIO()
            pred_df.to_parquet(buf, index=False)
            buf.seek(0)
            pred_key = f"{output_prefix}/r2_{scenario}_predictions.parquet"
            s3.put_object(Bucket=BUCKET, Key=pred_key, Body=buf.getvalue())
            log.info("Uploaded %d prediction rows to s3://%s/%s",
                     len(pred_df), BUCKET, pred_key)
    else:
        local = f"/tmp/r2_{scenario}.json"
        Path(local).write_text(results_json)
        log.info("Wrote %s", local)

    print(results_json)


if __name__ == "__main__":
    main()
