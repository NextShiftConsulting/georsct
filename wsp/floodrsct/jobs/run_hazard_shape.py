#!/usr/bin/env python3
"""
run_hazard_shape.py -- DOE-H1: Hazard Shape ablation test.

3-arm ablation asking "Is the floodplain a scalar?"
  - h1_depth_scalar:  R0 + deltares_depth_ft_rp100 (single tail-risk scalar)
  - h1_depth_shape:   R0 + depth_level + depth_gradient (derived shape)
  - h1_depth_vector:  R0 + full RP vector + max_depth + inundation_pct

Model: HistGradientBoostingRegressor (max_bins=32, min_samples_leaf=20)
CV: Spatial blocked folds from existing fold files
Target: obs_nfip_event_claims (log1p-transformed regression)

Deltares features come from processed/shared/zcta_deltares_depth.parquet.
If the cache is missing or has <10% coverage for the scenario, the job
exits with FAIL_DATA_READY status.

Output:
  s3://swarm-floodrsct-data/results/s035/story/hazard_shape_{scenario}.json

Usage:
    python run_hazard_shape.py --scenario houston --upload
    python run_hazard_shape.py --scenario houston
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, load_processed_parquet
from _s3_result import upload_json_result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
MAX_BINS = 32
MIN_SAMPLES_LEAF = 20
FOLD_FLOOR = 2 * MIN_SAMPLES_LEAF  # 40

DELTARES_CACHE_KEY = "processed/shared/zcta_deltares_depth.parquet"
MIN_DELTARES_COVERAGE = 0.10  # At least 10% of scenario ZCTAs must have data

# Deltares feature columns expected in the cache
DELTARES_ALL_FEATURES = [
    "deltares_depth_ft_rp10",
    "deltares_depth_ft_rp50",
    "deltares_depth_ft_rp100",
    "deltares_max_depth_ft_rp10",
    "deltares_max_depth_ft_rp50",
    "deltares_max_depth_ft_rp100",
    "deltares_inundation_pct_rp10",
    "deltares_inundation_pct_rp50",
    "deltares_inundation_pct_rp100",
]

# R0 features -- same as train_r0_baseline.py canonical list.
R0_FEATURES = [
    "flood_pct_zone_a",
    "flood_pct_zone_x",
    "flood_pct_zone_x500",
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
    "impervious_pct",
    "cropland_pct",
    # Hydrology (OWP HAND + 3DEP zonal stats)
    "hand_mean_m",
    "twi_mean",
    "gfi_mean",
    "spi_mean",
]

TARGET_COL = "obs_nfip_event_claims"
RESULTS_PREFIX = "results/s035/story"

N_BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_read_parquet(s3, key: str) -> pd.DataFrame | None:
    """Read parquet from S3; return None if missing."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        log.warning("Could not read s3://%s/%s: %s", BUCKET, key, e)
        return None


def load_folds(s3, scenario: str) -> pd.DataFrame:
    """Load spatial blocked fold assignments."""
    key = f"folds/{scenario}_folds.parquet"
    log.info("Loading folds: s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


def get_git_hash() -> str:
    """Return current git hash or 'unknown'."""
    gh = os.environ.get("S035_GIT_HASH")
    if gh:
        return gh
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Data loading + Deltares join
# ---------------------------------------------------------------------------

def load_and_join(
    s3, scenario: str
) -> tuple[pd.DataFrame, list[str], int]:
    """Load features + folds + Deltares, join, and return merged DataFrame.

    Returns:
        (merged_df, r0_features_available, n_folds)

    Raises:
        RuntimeError: If Deltares cache is missing or coverage is too low.
    """
    # 1. Load scenario features
    features = load_processed_parquet(s3, scenario)
    features["zcta_id"] = features["zcta_id"].astype(str)
    if "event" in features.columns:
        features["event"] = features["event"].astype(str)
    log.info("Features: %d rows x %d cols", len(features), len(features.columns))

    # 2. Load NFIP historical supplement (required for R0 features)
    nfip_key = f"processed/{scenario}/{scenario}_nfip_historical.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=nfip_key)
        nfip_hist = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        nfip_hist["zcta_id"] = nfip_hist["zcta_id"].astype(str)
        join_cols = ["zcta_id", "event"] if "event" in nfip_hist.columns else ["zcta_id"]
        features = features.merge(nfip_hist, on=join_cols, how="left")
        log.info("NFIP historical merged: %d rows", len(nfip_hist))
    except Exception as exc:
        raise RuntimeError(
            f"NFIP historical supplement missing: {nfip_key}. "
            f"Run build_nfip_historical.py --scenario {scenario} --upload first."
        ) from exc

    # 3. Load Deltares depth cache
    deltares = s3_read_parquet(s3, DELTARES_CACHE_KEY)
    if deltares is None:
        raise RuntimeError(
            f"FAIL_DATA_READY: Deltares cache not found at "
            f"s3://{BUCKET}/{DELTARES_CACHE_KEY}. "
            "Run fetch_jrc_deltares.py --deltares-only first."
        )
    deltares["zcta_id"] = deltares["zcta_id"].astype(str)
    log.info("Deltares cache: %d rows, columns: %s",
             len(deltares), list(deltares.columns))

    # Check required columns exist
    missing_cols = [c for c in DELTARES_ALL_FEATURES if c not in deltares.columns]
    if missing_cols:
        raise RuntimeError(
            f"FAIL_DATA_READY: Deltares cache missing columns: {missing_cols}. "
            f"Available: {list(deltares.columns)}"
        )

    # 4. Check coverage for this scenario
    scenario_zctas = set(features["zcta_id"].unique())
    deltares_non_null = deltares.dropna(subset=["deltares_depth_ft_rp100"])
    deltares_zctas = set(deltares_non_null["zcta_id"].unique())
    matched = scenario_zctas & deltares_zctas
    coverage = len(matched) / len(scenario_zctas) if scenario_zctas else 0.0
    log.info(
        "Deltares coverage for %s: %d / %d ZCTAs (%.1f%%)",
        scenario, len(matched), len(scenario_zctas), coverage * 100,
    )
    if coverage < MIN_DELTARES_COVERAGE:
        raise RuntimeError(
            f"FAIL_DATA_READY: Deltares coverage {coverage:.1%} < "
            f"{MIN_DELTARES_COVERAGE:.0%} threshold for {scenario}. "
            f"Only {len(matched)} of {len(scenario_zctas)} ZCTAs have Deltares data. "
            f"Re-run fetch_jrc_deltares.py --deltares-only."
        )

    # 5. Join Deltares onto features (left join -- NaN for missing ZCTAs is 0 depth)
    deltares_cols = ["zcta_id"] + DELTARES_ALL_FEATURES
    features = features.merge(deltares[deltares_cols], on="zcta_id", how="left")

    # Fill NaN depth with 0 (no modeled flood = no depth). This is semantically
    # correct for Deltares: if the model doesn't predict inundation, depth IS 0.
    for col in DELTARES_ALL_FEATURES:
        pct_null = features[col].isna().mean()
        log.info("  %s: %.1f%% null after join", col, pct_null * 100)
        features[col] = features[col].fillna(0.0)

    # 6. Derive shape features for h1_depth_shape arm
    features["deltares_depth_level"] = features[
        ["deltares_depth_ft_rp10", "deltares_depth_ft_rp50", "deltares_depth_ft_rp100"]
    ].mean(axis=1)
    denom = features["deltares_depth_ft_rp100"].replace(0.0, np.nan)
    features["deltares_depth_gradient"] = (
        (features["deltares_depth_ft_rp100"] - features["deltares_depth_ft_rp10"]) / denom
    ).fillna(0.0)

    # 7. Derive max and inundation aggregates for h1_depth_vector arm
    features["deltares_max_depth_ft"] = features[
        ["deltares_max_depth_ft_rp10", "deltares_max_depth_ft_rp50",
         "deltares_max_depth_ft_rp100"]
    ].max(axis=1)
    features["deltares_inundation_pct"] = features[
        ["deltares_inundation_pct_rp10", "deltares_inundation_pct_rp50",
         "deltares_inundation_pct_rp100"]
    ].mean(axis=1)

    # 8. Load folds
    folds_df = load_folds(s3, scenario)
    folds_df["zcta_id"] = folds_df["zcta_id"].astype(str)
    if "event" in folds_df.columns:
        folds_df["event"] = folds_df["event"].astype(str)

    # 9. Merge folds
    merged = features.merge(
        folds_df[["zcta_id", "event", "fold_spatial_blocked"]],
        on=["zcta_id", "event"],
        how="inner",
    )
    log.info("Merged with folds: %d rows", len(merged))

    # 10. Identify available R0 features
    r0_available = [f for f in R0_FEATURES
                    if f in merged.columns and merged[f].notna().any()]
    log.info("R0 features available: %d / %d", len(r0_available), len(R0_FEATURES))

    n_folds = merged["fold_spatial_blocked"].nunique()
    return merged, r0_available, n_folds


# ---------------------------------------------------------------------------
# Arm execution
# ---------------------------------------------------------------------------

def run_arm(
    merged: pd.DataFrame,
    feature_cols: list[str],
    arm_name: str,
) -> list[dict]:
    """Run spatial-blocked CV for one arm. Returns per-fold metric dicts."""
    log.info("--- Arm: %s (%d features) ---", arm_name, len(feature_cols))

    usable = merged[TARGET_COL].notna()
    for col in feature_cols:
        usable = usable & merged[col].notna()

    df = merged[usable].copy()
    df["_y"] = np.log1p(df[TARGET_COL].clip(lower=0).astype(float))
    log.info("  Usable rows: %d / %d", len(df), len(merged))

    folds = sorted(df["fold_spatial_blocked"].unique())
    fold_results = []

    for fold_id in folds:
        test_mask = df["fold_spatial_blocked"] == fold_id
        train_mask = ~test_mask

        X_train = df.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
        y_train = df.loc[train_mask, "_y"].to_numpy(dtype=np.float32)
        X_test = df.loc[test_mask, feature_cols].to_numpy(dtype=np.float32)
        y_test = df.loc[test_mask, "_y"].to_numpy(dtype=np.float32)

        n_train = len(X_train)
        n_test = len(X_test)

        if n_train < FOLD_FLOOR:
            log.warning(
                "  Fold %s: n_train=%d < fold_floor=%d, skipping",
                fold_id, n_train, FOLD_FLOOR,
            )
            continue
        if n_test == 0:
            log.warning("  Fold %s: empty test set, skipping", fold_id)
            continue

        model = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=6,
            learning_rate=0.1,
            max_bins=MAX_BINS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=SEED,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        fold_results.append({
            "fold": str(fold_id),
            "n_train": n_train,
            "n_test": n_test,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        })

        log.info(
            "  Fold %s: n_train=%d, n_test=%d, RMSE=%.4f, MAE=%.4f, R2=%.4f",
            fold_id, n_train, n_test, rmse, mae, r2,
        )

    return fold_results


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_bootstrap_delta(
    baseline_rmses: list[float],
    comparison_rmses: list[float],
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Compute paired bootstrap CI on delta RMSE (baseline - comparison).

    For hazard shape, baseline is the scalar arm. A positive delta means
    scalar has HIGHER RMSE (worse) than the comparison.
    """
    rng = np.random.default_rng(seed)
    deltas = np.array(baseline_rmses) - np.array(comparison_rmses)
    observed_mean = float(np.mean(deltas))

    boot_means = np.empty(n_boot)
    n = len(deltas)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(deltas[idx])

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    if observed_mean > 0:
        p_value = float(np.mean(boot_means <= 0))
    elif observed_mean < 0:
        p_value = float(np.mean(boot_means >= 0))
    else:
        p_value = 1.0

    return {
        "mean": observed_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_bootstrap": n_boot,
    }


def compute_fold_stability(
    baseline_rmses: list[float],
    comparison_rmses: list[float],
) -> dict:
    """Check if all folds agree on direction of delta."""
    deltas = np.array(baseline_rmses) - np.array(comparison_rmses)
    n_positive = int(np.sum(deltas > 0))  # scalar worse
    n_negative = int(np.sum(deltas < 0))  # scalar better
    n_zero = int(np.sum(deltas == 0))
    all_agree = (n_positive == len(deltas)) or (n_negative == len(deltas))
    return {
        "n_folds": len(deltas),
        "n_scalar_worse": n_positive,
        "n_scalar_better": n_negative,
        "n_tied": n_zero,
        "all_folds_agree": all_agree,
    }


def compute_kendall_tau_ranking(arm_summaries: dict) -> dict:
    """Compute Kendall tau-b ranking consistency across arms.

    Ranks arms by mean RMSE and checks if the ranking is stable across folds.
    """
    arm_names = list(arm_summaries.keys())
    arm_mean_rmse = {
        name: arm_summaries[name]["mean_rmse"]
        for name in arm_names
        if arm_summaries[name]["mean_rmse"] is not None
    }
    if len(arm_mean_rmse) < 2:
        return {"tau_b": None, "ranking": []}

    ranked = sorted(arm_mean_rmse.items(), key=lambda x: x[1])
    ranking = [{"rank": i + 1, "arm": name, "mean_rmse": rmse}
               for i, (name, rmse) in enumerate(ranked)]

    # Per-fold ranking stability: count how many folds preserve the overall ranking
    n_folds = min(
        arm_summaries[name]["n_folds"]
        for name in arm_mean_rmse
    )
    if n_folds < 2:
        return {"tau_b": None, "ranking": ranking}

    # Build fold-level rank arrays
    fold_ranks = {}
    for name in arm_mean_rmse:
        fold_rmses = [f["rmse"] for f in arm_summaries[name]["folds"]]
        fold_ranks[name] = fold_rmses

    # Kendall tau between overall ranking and per-fold rankings
    overall_order = [name for name, _ in ranked]
    fold_agreements = 0
    for fold_idx in range(n_folds):
        fold_rmse_at_fold = {name: fold_ranks[name][fold_idx] for name in arm_mean_rmse}
        fold_order = [name for name, _ in sorted(fold_rmse_at_fold.items(), key=lambda x: x[1])]
        if fold_order == overall_order:
            fold_agreements += 1

    return {
        "tau_b": fold_agreements / n_folds if n_folds > 0 else None,
        "fold_ranking_agreement": fold_agreements,
        "n_folds": n_folds,
        "ranking": ranking,
    }


# ---------------------------------------------------------------------------
# Verdict logic (from pre-registration)
# ---------------------------------------------------------------------------

def determine_verdict(
    n_folds: int,
    delta_scalar_vs_shape: dict,
    delta_scalar_vs_vector: dict,
    stability_vs_shape: dict,
    stability_vs_vector: dict,
) -> str:
    """Apply pre-registered verdict logic.

    SHAPE_MATTERS:
        Vector or level+gradient outperforms scalar. CI excludes zero.
        i.e., delta mean > 0 (scalar RMSE higher) AND CI_lower > 0.

    SCALAR_SUFFICIENT:
        Neither shape nor vector significantly outperforms scalar.
        CI includes zero for both.

    INCONCLUSIVE:
        Fewer than 3 folds completed.
    """
    if n_folds < 3:
        return "INCONCLUSIVE"

    # Check if either shape or vector significantly outperforms scalar
    shape_better = (
        delta_scalar_vs_shape.get("mean", 0) > 0
        and delta_scalar_vs_shape.get("ci_lower", 0) > 0
    )
    vector_better = (
        delta_scalar_vs_vector.get("mean", 0) > 0
        and delta_scalar_vs_vector.get("ci_lower", 0) > 0
    )

    if shape_better or vector_better:
        return "SHAPE_MATTERS"
    else:
        return "SCALAR_SUFFICIENT"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scenario: str, upload: bool) -> None:
    t0 = time.time()
    s3 = get_s3_client()
    git_hash = get_git_hash()

    print("=" * 60)
    print(f"  DOE-H1: HAZARD SHAPE -- {scenario}")
    print("=" * 60)

    # Load and join all data
    merged, r0_available, n_fold_groups = load_and_join(s3, scenario)

    # Check target
    if TARGET_COL not in merged.columns:
        raise RuntimeError(f"Target column {TARGET_COL} not in merged data")

    n_valid_target = int(merged[TARGET_COL].notna().sum())
    if n_valid_target == 0:
        raise RuntimeError(f"Zero non-null values for {TARGET_COL}")

    log.info("Target %s: %d non-null values", TARGET_COL, n_valid_target)

    # Build feature sets for each arm

    # Arm 1: h1_depth_scalar = R0 + deltares_depth_ft_rp100
    scalar_features = r0_available + ["deltares_depth_ft_rp100"]

    # Arm 2: h1_depth_shape = R0 + derived level + gradient
    shape_features = r0_available + [
        "deltares_depth_level",
        "deltares_depth_gradient",
    ]

    # Arm 3: h1_depth_vector = R0 + full RP vector + max + inundation
    vector_features = r0_available + [
        "deltares_depth_ft_rp10",
        "deltares_depth_ft_rp50",
        "deltares_depth_ft_rp100",
        "deltares_max_depth_ft",
        "deltares_inundation_pct",
    ]

    log.info("Feature counts: scalar=%d, shape=%d, vector=%d",
             len(scalar_features), len(shape_features), len(vector_features))

    # Run all 3 arms
    arms = {}

    log.info("\n=== ARM 1: h1_depth_scalar ===")
    scalar_folds = run_arm(merged, scalar_features, "h1_depth_scalar")
    arms["h1_depth_scalar"] = scalar_folds

    log.info("\n=== ARM 2: h1_depth_shape ===")
    shape_folds = run_arm(merged, shape_features, "h1_depth_shape")
    arms["h1_depth_shape"] = shape_folds

    log.info("\n=== ARM 3: h1_depth_vector ===")
    vector_folds = run_arm(merged, vector_features, "h1_depth_vector")
    arms["h1_depth_vector"] = vector_folds

    # Compute arm summaries
    arm_summaries = {}
    for arm_name, fold_list in arms.items():
        if fold_list:
            rmses = [f["rmse"] for f in fold_list]
            maes = [f["mae"] for f in fold_list]
            r2s = [f["r2"] for f in fold_list]
            arm_summaries[arm_name] = {
                "folds": fold_list,
                "mean_rmse": float(np.mean(rmses)),
                "std_rmse": float(np.std(rmses)),
                "mean_mae": float(np.mean(maes)),
                "std_mae": float(np.std(maes)),
                "mean_r2": float(np.mean(r2s)),
                "std_r2": float(np.std(r2s)),
                "n_folds": len(fold_list),
            }
        else:
            arm_summaries[arm_name] = {
                "folds": [],
                "mean_rmse": None,
                "std_rmse": None,
                "mean_mae": None,
                "std_mae": None,
                "mean_r2": None,
                "std_r2": None,
                "n_folds": 0,
            }

    # Paired comparisons -- scalar is baseline, shape/vector are treatments
    def align_folds(a_folds, b_folds):
        a_map = {f["fold"]: f["rmse"] for f in a_folds}
        b_map = {f["fold"]: f["rmse"] for f in b_folds}
        common = sorted(set(a_map.keys()) & set(b_map.keys()))
        return [a_map[k] for k in common], [b_map[k] for k in common], len(common)

    # scalar vs shape
    sc_rmse, sh_rmse, n_common_sh = align_folds(scalar_folds, shape_folds)
    # scalar vs vector
    sc_rmse_v, vec_rmse, n_common_vec = align_folds(scalar_folds, vector_folds)

    delta_rmse = {}

    if n_common_sh >= 2:
        delta_vs_shape = paired_bootstrap_delta(sc_rmse, sh_rmse)
        stability_vs_shape = compute_fold_stability(sc_rmse, sh_rmse)
        delta_rmse["scalar_vs_shape"] = {
            **delta_vs_shape,
            "fold_stability": stability_vs_shape,
        }
    else:
        delta_vs_shape = {"mean": None, "ci_lower": None, "ci_upper": None, "p_value": None}
        stability_vs_shape = {"n_folds": 0, "all_folds_agree": False}
        delta_rmse["scalar_vs_shape"] = delta_vs_shape

    if n_common_vec >= 2:
        delta_vs_vector = paired_bootstrap_delta(sc_rmse_v, vec_rmse)
        stability_vs_vector = compute_fold_stability(sc_rmse_v, vec_rmse)
        delta_rmse["scalar_vs_vector"] = {
            **delta_vs_vector,
            "fold_stability": stability_vs_vector,
        }
    else:
        delta_vs_vector = {"mean": None, "ci_lower": None, "ci_upper": None, "p_value": None}
        stability_vs_vector = {"n_folds": 0, "all_folds_agree": False}
        delta_rmse["scalar_vs_vector"] = delta_vs_vector

    # Kendall ranking
    kendall_ranking = compute_kendall_tau_ranking(arm_summaries)

    # Verdict
    n_completed_folds = min(n_common_sh, n_common_vec)
    verdict = determine_verdict(
        n_completed_folds, delta_vs_shape, delta_vs_vector,
        stability_vs_shape, stability_vs_vector,
    )

    elapsed = time.time() - t0

    # Build output payload
    payload = {
        "scenario": scenario,
        "doe_id": "DOE-H1",
        "experiment": "s035-model-ladder",
        "phase": "hazard_shape",
        "arms": arm_summaries,
        "delta_rmse": delta_rmse,
        "kendall_ranking": kendall_ranking,
        "verdict": verdict,
        "n_folds": n_completed_folds,
        "n_samples": len(merged),
        "n_valid_target": n_valid_target,
        "deltares_coverage": {
            "cache_key": DELTARES_CACHE_KEY,
            "min_coverage_threshold": MIN_DELTARES_COVERAGE,
        },
        "feature_sets": {
            "scalar": scalar_features,
            "shape": shape_features,
            "vector": vector_features,
        },
        "target": TARGET_COL,
        "model": {
            "name": "HistGradientBoostingRegressor",
            "max_bins": MAX_BINS,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_iter": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "seed": SEED,
        },
        "gates": {
            "max_bins": MAX_BINS,
            "fold_floor": FOLD_FLOOR,
            "min_deltares_coverage": MIN_DELTARES_COVERAGE,
        },
        "git_hash": git_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 1),
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"  DOE-H1 RESULTS: {scenario}")
    print("=" * 60)

    for arm_name, summary in arm_summaries.items():
        if summary["mean_rmse"] is not None:
            print(
                f"  {arm_name:20s}  RMSE={summary['mean_rmse']:.4f} "
                f"+/- {summary['std_rmse']:.4f}  "
                f"R2={summary['mean_r2']:.4f}  "
                f"({summary['n_folds']} folds)"
            )
        else:
            print(f"  {arm_name:20s}  NO RESULTS")

    if delta_rmse.get("scalar_vs_shape", {}).get("mean") is not None:
        d = delta_rmse["scalar_vs_shape"]
        print(
            f"\n  delta RMSE (scalar - shape):  "
            f"{d['mean']:+.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"
        )
        if "p_value" in d and d["p_value"] is not None:
            print(f"  p-value: {d['p_value']:.4f}")

    if delta_rmse.get("scalar_vs_vector", {}).get("mean") is not None:
        d = delta_rmse["scalar_vs_vector"]
        print(
            f"  delta RMSE (scalar - vector): "
            f"{d['mean']:+.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"
        )
        if "p_value" in d and d["p_value"] is not None:
            print(f"  p-value: {d['p_value']:.4f}")

    print(f"\n  VERDICT: {verdict}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 60)

    # Upload or write locally
    if upload:
        key = f"{RESULTS_PREFIX}/hazard_shape_{scenario}.json"
        upload_json_result(s3, BUCKET, key, payload, git_hash=git_hash)
    else:
        local = Path(f"/tmp/hazard_shape_{scenario}.json")
        local.write_text(json.dumps(payload, indent=2, default=str))
        log.info("Wrote %s", local)

    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOE-H1: Hazard Shape ablation test"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS,
                        help="Scenario to process (one per job)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload results to S3")
    args = parser.parse_args()

    run(args.scenario, args.upload)


if __name__ == "__main__":
    main()
