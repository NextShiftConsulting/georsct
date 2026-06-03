#!/usr/bin/env python3
"""
train_r1_hydrology.py -- Phase 2: R1 representation (R0 + hydrology/infrastructure).

Loads R0 folds (from Phase 1) + R1 supplement parquet (from build_r1_features.py).
Trains the same solvers, targets, and splits as R0 — only the feature set changes.

R1 adds:
  - Universal: nhd_catchment_area_km2, basin/stream slopes, TWI variants,
    accessibility, infrastructure detail, historical flood impact, detailed NFIP
  - Scenario-specific: upstream_catchment_km2, hcfcd_drainage_district,
    levee_nearest_km, levee_condition_rating, sewershed_name, slosh_max_surge_m
  - W-matrix (spatial structure): zcta_degree, zcta_mean_neighbor_dist_km,
    wlag_flood_zone_pct, wlag_population_density, wlag_median_income,
    wlag_impervious_pct, wlag_rainfall_mm, wlag_nfip_claims

DOE constraint: same folds, same solver hyperparameters, same targets as R0.

Ablation variants (DOE_R1_spatial.md):
  --ablation full           R0 + hydro + W-matrix (default)
  --ablation no-wlag        R0 + hydro only (no spatial structure)
  --ablation no-target-lag  R0 + hydro + W-matrix minus wlag_nfip_claims
  --ablation wlag-only      R0 + W-matrix only (no hydro features)

Usage:
    python train_r1_hydrology.py --scenario houston --upload
    python train_r1_hydrology.py --scenario houston --ablation no-wlag --upload
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
    BUCKET, SCENARIOS, get_s3_client, load_processed_parquet, load_crosswalk,
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
# R0 features (identical to train_r0_baseline.py — DO NOT diverge)
# ---------------------------------------------------------------------------
R0_FEATURES = [
    "flood_pct_zone_a",
    "flood_pct_zone_x",
    "flood_pct_zone_x500",
    "flood_event_count",
    "flood_event_count_5y",
    "flood_events_per_year",
    "flood_property_damage_k",
    "flood_crop_damage_k",
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
]

# ---------------------------------------------------------------------------
# R1 supplement features — added on top of R0
# Universal (all scenarios): from assembled parquet or R1 supplement
# ---------------------------------------------------------------------------
R1_UNIVERSAL = [
    # From R1 supplement (build_r1_features.py)
    "nhd_catchment_area_km2",
    # From assembled parquet (already present but not in R0 feature list)
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
    # QUARANTINED: flood_deaths and flood_injuries are county-level Storm
    # Events totals from geocertdb2026 -- constant across all ZCTAs within
    # a scenario, not temporally gated, and likely include target-event
    # outcomes. Zero predictive variance within scenario; outcome-from-
    # outcome leakage risk across scenarios. Removed pending audit.
    # "flood_deaths",
    # "flood_injuries",
]

# Scenario-specific R1 features (present only in some scenarios)
R1_SCENARIO_SPECIFIC = [
    "upstream_catchment_km2",       # Houston, Riverside
    "hcfcd_drainage_district",      # Houston only (categorical -> will encode)
    "levee_nearest_km",             # NOLA, NYC
    "levee_condition_rating",       # NOLA, NYC
    "sewershed_name",               # NYC only (categorical -> will encode)
    "slosh_max_surge_m",            # SW Florida
]

# ---------------------------------------------------------------------------
# W-matrix spatial features (from build_event_dataset.py)
# These are in the assembled parquet. DOE_R1_spatial.md Change 13.
# ---------------------------------------------------------------------------
R1_WMATRIX = [
    "zcta_degree",
    "zcta_mean_neighbor_dist_km",
    "wlag_flood_zone_pct",
    "wlag_population_density",
    "wlag_median_income",
    "wlag_impervious_pct",
    "wlag_rainfall_mm",
    "wlag_nfip_claims",       # TARGET LAG — highest leakage risk, see DOE
]

# Full R1 feature set
R1_HYDRO = R1_UNIVERSAL + R1_SCENARIO_SPECIFIC
R1_FEATURES = R0_FEATURES + R1_HYDRO + R1_WMATRIX

# ---------------------------------------------------------------------------
# Ablation variants (DOE_R1_spatial.md "Ablations (mandatory)")
#
# full:           R0 + hydro + W-matrix (default, all R1 features)
# no-wlag:        R0 + hydro, remove all 8 W-matrix features
# no-target-lag:  R0 + hydro + W-matrix minus wlag_nfip_claims
# wlag-only:      R0 + 8 W-matrix features only (no hydro universal/specific)
# ---------------------------------------------------------------------------
ABLATION_MODES = {
    "full":           lambda: R0_FEATURES + R1_HYDRO + R1_WMATRIX,
    "no-wlag":        lambda: R0_FEATURES + R1_HYDRO,
    "no-target-lag":  lambda: R0_FEATURES + R1_HYDRO + [f for f in R1_WMATRIX if f != "wlag_nfip_claims"],
    "wlag-only":      lambda: R0_FEATURES + R1_WMATRIX,
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
    timestamp: str


def _load_r1_supplement(s3, scenario: str) -> pd.DataFrame:
    """Load R1 supplement parquet from S3. Hard failure if missing."""
    key = f"processed/{scenario}/{scenario}_r1_supplement.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        log.info("R1 supplement: %d rows x %d cols from %s", len(df), len(df.columns), key)
        return df
    except Exception as exc:
        raise RuntimeError(
            f"R1 supplement missing: s3://{BUCKET}/{key}. "
            f"Run build_r1_features.py --scenario {scenario} --upload first. "
            f"R1 arm is meaningless without its supplement features."
        ) from exc


def _build_neighbor_dict(adjacency) -> dict[str, list[str]]:
    """Build an undirected {zcta_id: [neighbor_zcta_id,...]} dict from
    whatever load_adjacency returns (a dict, or an edge-list DataFrame).
    Keys/values coerced to str. Edges added in both directions."""
    if adjacency is None:
        return {}
    if isinstance(adjacency, dict):
        return {str(k): [str(v) for v in vs] for k, vs in adjacency.items()}
    # assume an edge-list DataFrame; detect the two ZCTA id columns
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
        neigh.setdefault(v, []).append(u)  # undirected
    # dedupe
    return {k: sorted(set(vs)) for k, vs in neigh.items()}


def _load_folds(s3, scenario: str) -> pd.DataFrame:
    """Load fold assignments from Phase 1."""
    key = f"folds/{scenario}_folds.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    log.info("Folds: %d rows from %s", len(df), key)
    return df


def _encode_categoricals(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Label-encode string columns so sklearn can consume them."""
    for col in features:
        if col in df.columns and df[col].dtype == object:
            codes, _ = pd.factorize(df[col])
            df[col] = codes.astype(np.float32)
            # factorize assigns -1 to NaN, convert to NaN for sklearn
            df.loc[df[col] < 0, col] = np.nan
    return df


def _available_features(df: pd.DataFrame, candidate_features: list[str]) -> tuple[list[str], int]:
    """Return features from *candidate_features* present in df.

    Track how many came from the R1 supplement.
    """
    available = []
    r1_supp_count = 0
    r1_supp_names = {"nhd_catchment_area_km2", "levee_nearest_km",
                     "levee_condition_rating", "sewershed_name"}
    for f in candidate_features:
        if f in df.columns and df[f].notna().any():
            available.append(f)
            if f in r1_supp_names:
                r1_supp_count += 1
    return available, r1_supp_count


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
# Solvers: IDENTICAL to R0 (same hyperparams, same code)
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


def _recompute_wlag_per_fold(
    merged: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    neighbors: dict[str, list[str]],
    target_col: str,
) -> pd.Series:
    """Recompute wlag of target column using ONLY training ZCTAs' values.

    DOE_R1_spatial.md leakage protocol: test ZCTAs receive lag from
    training neighbors only (NaN if all neighbors are in test fold).

    Returns wlag as pd.Series aligned to merged index.
    """
    # Build lookup: zcta_id -> target value, training ZCTAs only
    train_rows = merged[train_mask]
    # Multiple events per ZCTA: use mean target across events in training fold
    zcta_target = train_rows.groupby("zcta_id")[target_col].mean().to_dict()

    wlag_vals = pd.Series(np.nan, index=merged.index, dtype=np.float64)
    for idx, row in merged.iterrows():
        zid = str(row["zcta_id"])
        nbrs = neighbors.get(zid, [])
        if not nbrs:
            continue
        # Row-standardized lag from training neighbors only
        vals = []
        for nb in nbrs:
            v = zcta_target.get(nb)
            if v is not None and not np.isnan(v):
                vals.append(v)
        if vals:
            wlag_vals.at[idx] = float(np.mean(vals))

    return wlag_vals


def run_split(
    df: pd.DataFrame,
    folds_df: pd.DataFrame,
    features: list[str],
    r1_supp_count: int,
    target_col: str,
    task: str,
    transform: str | None,
    solver_name: str,
    split_name: str,
    scenario: str,
    prediction_rows: list[dict] | None = None,
    neighbors: dict[str, list[str]] | None = None,
) -> list[RunResult]:
    ts = datetime.now(timezone.utc).isoformat()
    fold_col = SPLITS[split_name]
    solver_fn = SOLVERS[solver_name]

    merged = df.merge(folds_df[["zcta_id", "event", fold_col]], on=["zcta_id", "event"])
    valid_mask = merged[target_col].notna()
    merged = merged[valid_mask].copy()

    y_col = target_col
    if transform == "log1p":
        merged["_y"] = np.log1p(merged[target_col].clip(lower=0).astype(float))
        y_col = "_y"

    # Leakage guard for wlag_nfip_claims (the spatial lag of the target).
    # When target IS nfip_event_claims, the pre-computed wlag in the parquet
    # includes test-fold target values -> leakage. It MUST be recomputed per
    # fold from training ZCTAs only. If we cannot (no neighbors), we REFUSE
    # rather than silently train on the leaked column.
    wlag_col_idx = None
    leaky_wlag_active = (
        "wlag_nfip_claims" in features
        and target_col == "obs_nfip_event_claims"
    )
    if leaky_wlag_active and neighbors is None:
        raise RuntimeError(
            "wlag_nfip_claims is in the feature set and the target is "
            "obs_nfip_event_claims, but no adjacency/neighbors were provided. "
            "The pre-computed wlag leaks test-fold target values. Refusing to "
            "run. Load adjacency and pass neighbors= to enable per-fold "
            "recomputation, or use --ablation no-wlag / no-target-lag."
        )
    needs_perfold_wlag = leaky_wlag_active and neighbors is not None
    if needs_perfold_wlag:
        wlag_col_idx = features.index("wlag_nfip_claims")
        log.info("    Per-fold wlag_nfip_claims recomputation enabled (leakage mitigation)")

    X_all = merged[features].values.astype(np.float32)
    y_all = merged[y_col].values.astype(np.float32)
    fold_ids = sorted(merged[fold_col].unique())
    results = []

    for fold_id in fold_ids:
        test_mask = merged[fold_col] == fold_id
        train_mask = ~test_mask

        # Per-fold wlag recomputation (DOE leakage protocol)
        if needs_perfold_wlag:
            wlag_safe = _recompute_wlag_per_fold(
                merged, train_mask, test_mask, neighbors,
                "obs_nfip_event_claims",
            )
            X_all[:, wlag_col_idx] = wlag_safe.values.astype(np.float32)

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
        description="Phase 2: R1 hydrology/infrastructure training"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--ablation", default="full", choices=list(ABLATION_MODES.keys()),
        help="Feature ablation variant (default: full). "
             "no-wlag: remove W-matrix. no-target-lag: remove wlag_nfip_claims. "
             "wlag-only: R0 + W-matrix only (no hydro)."
    )
    args = parser.parse_args()
    ablation = args.ablation

    s3 = get_s3_client()
    scenario = args.scenario

    # Resolve ablation feature list
    ablation_features = ABLATION_MODES[ablation]()
    phase_label = f"r1_hydrology" if ablation == "full" else f"r1_{ablation.replace('-', '_')}"

    # Hard gate: reject any feature that violates the causal boundary.
    # wlag_nfip_claims is post_event but DOE-approved with per-fold
    # recomputation (Change 13). The leakage gate below enforces this.
    check_causal_boundary(ablation_features, exempt={"wlag_nfip_claims"})

    print(f"\n{'='*60}")
    print(f"  S035 PHASE 2: R1 HYDROLOGY -- {scenario}")
    if ablation != "full":
        print(f"  ABLATION: {ablation}")
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
    r1_supp = _load_r1_supplement(s3, scenario)
    r1_supp["zcta_id"] = r1_supp["zcta_id"].astype(str)
    assert r1_supp["zcta_id"].is_unique, (
        f"R1 supplement has {r1_supp['zcta_id'].duplicated().sum()} duplicate zcta_ids "
        f"-- LEFT join would multiply rows and break paired ablation"
    )
    pre_rows = len(df)
    pre_cols = set(df.columns)
    df = df.merge(r1_supp, on="zcta_id", how="left")
    assert len(df) == pre_rows, (
        f"R1 join changed row count: {pre_rows} -> {len(df)}. "
        f"Grain must stay (zcta_id, event) for paired comparison with R0."
    )
    new_cols = set(df.columns) - pre_cols
    log.info("R1 supplement added %d columns: %s", len(new_cols), sorted(new_cols))

    # --- Encode categoricals ---
    df = _encode_categoricals(df, R1_SCENARIO_SPECIFIC)

    # --- Load folds from Phase 1 ---
    folds_df = _load_folds(s3, scenario)
    folds_df["zcta_id"] = folds_df["zcta_id"].astype(str)
    if "event" in folds_df.columns:
        folds_df["event"] = folds_df["event"].astype(str)

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
    features, r1_supp_count = _available_features(df, ablation_features)
    r0_count = sum(1 for f in features if f in R0_FEATURES)
    wlag_count = sum(1 for f in features if f in R1_WMATRIX)
    hydro_count = len(features) - r0_count - wlag_count
    log.info("R1 features [%s]: %d total (%d R0 + %d hydro + %d W-matrix), %d from supplement",
             ablation, len(features), r0_count, hydro_count, wlag_count, r1_supp_count)

    # --- Hard gate: refuse leaky variants without adjacency ---
    # full and wlag-only contain wlag_nfip_claims (the target spatial lag).
    # Running them off the pre-computed parquet leaks test-fold target values.
    # no-wlag and no-target-lag do not contain the column and run freely.
    if "wlag_nfip_claims" in features and not neighbors:
        raise RuntimeError(
            f"Ablation '{ablation}' includes wlag_nfip_claims (target spatial lag), "
            f"but adjacency/neighbors are unavailable for {scenario}. The pre-computed "
            f"wlag leaks test-fold target values into training. Refusing to run. "
            f"Provide adjacency, or use --ablation no-wlag / no-target-lag "
            f"(both clean and safe to run now; no-target-lag is the headline "
            f"relational verdict)."
        )

    missing = [f for f in ablation_features if f not in features]
    if missing:
        log.info("Missing R1 features (expected for some scenarios): %s", missing)

    # --- Train all combinations ---
    all_results: list[RunResult] = []
    prediction_rows: list[dict] = []

    for target_col, task, transform in TARGETS:
        if not _check_target(df, target_col, task):
            continue
        log.info("\n--- Target: %s (%s, transform=%s) ---", target_col, task, transform)

        for solver_name in SOLVERS:
            for split_name in SPLITS:
                log.info("  %s / %s", solver_name, split_name)
                try:
                    results = run_split(
                        df, folds_df, features, r1_supp_count,
                        target_col, task, transform,
                        solver_name, split_name, scenario,
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
    print(f"  R1 SUMMARY: {scenario} [ablation={ablation}]")
    print(f"  Total runs: {len(all_results)}")
    print(f"  Features: {len(features)} ({r0_count} R0 + {hydro_count} hydro + {wlag_count} W-matrix)")
    print(f"{'='*60}\n")

    # --- Upload results ---
    results_payload = {
        "experiment": "s035-model-ladder",
        "phase": phase_label,
        "ablation": ablation,
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "representation": "R1" if ablation == "full" else f"R1_{ablation}",
        "features_used": features,
        "features_missing": missing,
        "r0_feature_count": r0_count,
        "hydro_feature_count": hydro_count,
        "wmatrix_feature_count": wlag_count,
        "r1_supplement_feature_count": r1_supp_count,
        "runs": [asdict(r) for r in all_results],
    }

    results_json = json.dumps(results_payload, indent=2, default=str)

    if args.upload:
        key = f"{RESULTS_PREFIX}/{phase_label}_{scenario}.json"
        upload_json_result(s3, BUCKET, key, results_payload)

        if prediction_rows:
            pred_df = pd.DataFrame(prediction_rows)
            buf = io.BytesIO()
            pred_df.to_parquet(buf, index=False)
            buf.seek(0)
            pred_key = f"{RESULTS_PREFIX}/{phase_label}_{scenario}_predictions.parquet"
            s3.put_object(Bucket=BUCKET, Key=pred_key, Body=buf.getvalue())
            log.info("Uploaded %d prediction rows to s3://%s/%s",
                     len(pred_df), BUCKET, pred_key)
    else:
        local = f"/tmp/{phase_label}_{scenario}.json"
        Path(local).write_text(results_json)
        log.info("Wrote %s", local)

    print(results_json)


if __name__ == "__main__":
    main()
