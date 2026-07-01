#!/usr/bin/env python3
"""
run_topology_necessity.py -- DOE-P1: Topology Necessity ablation test.

3-arm ablation asking "Is topology load-bearing?"
  - p1_baseline:          R0 features + HAND + GFI + TWI + SPI
  - p1_no_topology:       R0 features only (topology removed)
  - p1_shuffled_topology: R0 features + shuffled(HAND/GFI/TWI/SPI), seed=42

Model: HistGradientBoostingRegressor (max_bins=32, min_samples_leaf=20)
CV: Spatial blocked folds from existing fold files
Target: obs_nfip_event_claims (log1p-transformed regression)

Topology features come from processed/shared/zcta_hydrology.parquet.
If the hydrology cache is missing or has insufficient coverage for the
scenario, the job exits with FAIL_DATA_READY status.

Output:
  s3://swarm-floodrsct-data/results/s035/story/topology_necessity_{scenario}.json

Usage:
    python run_topology_necessity.py --scenario houston --upload
    python run_topology_necessity.py --scenario houston
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

TOPOLOGY_FEATURES = ["hand_mean_m", "gfi_mean", "twi_mean", "spi_mean"]
HYDROLOGY_KEY_TEMPLATE = "processed/shared/zcta_hydrology_{scenario}.parquet"
# Legacy shared cache (Houston wrote here before per-scenario refactor)
HYDROLOGY_LEGACY_KEY = "processed/shared/zcta_hydrology.parquet"
MIN_HYDROLOGY_COVERAGE = 0.50  # At least 50% of scenario ZCTAs must have hydrology

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
# Data loading + hydrology join
# ---------------------------------------------------------------------------

def load_and_join(
    s3, scenario: str
) -> tuple[pd.DataFrame, list[str], int]:
    """Load features + folds + hydrology, join, and return merged DataFrame.

    Returns:
        (merged_df, r0_features_available, n_folds)

    Raises:
        RuntimeError: If hydrology cache is missing or coverage is too low.
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

    # 3. Load hydrology -- per-scenario file first, then legacy shared cache
    scenario_key = HYDROLOGY_KEY_TEMPLATE.format(scenario=scenario)
    hydro = s3_read_parquet(s3, scenario_key)
    hydro_source = scenario_key
    if hydro is None:
        log.info("Per-scenario hydrology not found, trying legacy shared cache")
        hydro = s3_read_parquet(s3, HYDROLOGY_LEGACY_KEY)
        hydro_source = HYDROLOGY_LEGACY_KEY
    if hydro is None:
        raise RuntimeError(
            "FAIL_DATA_READY: Hydrology cache not found at "
            f"s3://{BUCKET}/{scenario_key} or {HYDROLOGY_LEGACY_KEY}. "
            "Run the hydrology fetch job first."
        )
    hydro["zcta_id"] = hydro["zcta_id"].astype(str)
    log.info("Hydrology cache: %d rows from %s", len(hydro), hydro_source)

    # Check that all 4 topology features exist in the cache
    missing_cols = [f for f in TOPOLOGY_FEATURES if f not in hydro.columns]
    if missing_cols:
        raise RuntimeError(
            f"FAIL_DATA_READY: Hydrology cache missing columns: {missing_cols}. "
            f"Available: {list(hydro.columns)}"
        )

    # 4. Check coverage for this scenario
    scenario_zctas = set(features["zcta_id"].unique())
    hydro_zctas = set(hydro["zcta_id"].unique())
    matched = scenario_zctas & hydro_zctas
    coverage = len(matched) / len(scenario_zctas) if scenario_zctas else 0.0
    log.info(
        "Hydrology coverage for %s: %d / %d ZCTAs (%.1f%%)",
        scenario, len(matched), len(scenario_zctas), coverage * 100,
    )
    if coverage < MIN_HYDROLOGY_COVERAGE:
        raise RuntimeError(
            f"FAIL_DATA_READY: Hydrology coverage {coverage:.1%} < "
            f"{MIN_HYDROLOGY_COVERAGE:.0%} threshold for {scenario}. "
            f"Only {len(matched)} of {len(scenario_zctas)} ZCTAs have hydrology data."
        )

    # 5. Join hydrology onto features (left join -- NaN for missing ZCTAs)
    hydro_cols = ["zcta_id"] + TOPOLOGY_FEATURES
    features = features.merge(hydro[hydro_cols], on="zcta_id", how="left")

    # Check for NaN in topology features after join
    for col in TOPOLOGY_FEATURES:
        pct_null = features[col].isna().mean()
        log.info("  %s: %.1f%% null after join", col, pct_null * 100)

    # 6. Load folds
    folds_df = load_folds(s3, scenario)
    folds_df["zcta_id"] = folds_df["zcta_id"].astype(str)
    if "event" in folds_df.columns:
        folds_df["event"] = folds_df["event"].astype(str)

    # 7. Merge folds
    merged = features.merge(
        folds_df[["zcta_id", "event", "fold_spatial_blocked"]],
        on=["zcta_id", "event"],
        how="inner",
    )
    log.info("Merged with folds: %d rows", len(merged))

    # 8. Identify available R0 features
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
    """Run spatial-blocked CV for one arm. Returns per-fold metric dicts.

    Each fold dict contains: fold, n_train, n_test, rmse, mae, r2.
    """
    log.info("--- Arm: %s (%d features) ---", arm_name, len(feature_cols))

    # Filter to rows with non-null target and non-null features
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


def build_shuffled_features(
    merged: pd.DataFrame, seed: int = SEED
) -> pd.DataFrame:
    """Create shuffled versions of topology features.

    Shuffles within the merged DataFrame (breaking the spatial association
    between ZCTA and topology value) while preserving marginal distributions.
    """
    rng = np.random.default_rng(seed)
    df = merged.copy()
    for col in TOPOLOGY_FEATURES:
        shuffled_col = f"{col}_shuffled"
        vals = df[col].values.copy()
        # Shuffle only non-null values, keep NaN positions
        non_null_mask = ~np.isnan(vals.astype(float))
        non_null_vals = vals[non_null_mask].copy()
        rng.shuffle(non_null_vals)
        vals[non_null_mask] = non_null_vals
        df[shuffled_col] = vals
    return df


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

    A negative delta means baseline has LOWER RMSE (better).
    A positive delta means baseline has HIGHER RMSE (worse).

    For topology to be load-bearing, we need:
      baseline_vs_no_topology: negative (removing topology hurts)
      => delta = baseline_rmse - no_topology_rmse < 0

    Returns dict with mean, ci_lower, ci_upper, p_value.
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

    # Two-sided p-value: fraction of bootstrap samples on opposite side of 0
    if observed_mean < 0:
        p_value = float(np.mean(boot_means >= 0))
    elif observed_mean > 0:
        p_value = float(np.mean(boot_means <= 0))
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
    n_negative = int(np.sum(deltas < 0))
    n_positive = int(np.sum(deltas > 0))
    n_zero = int(np.sum(deltas == 0))
    all_agree = (n_negative == len(deltas)) or (n_positive == len(deltas))
    return {
        "n_folds": len(deltas),
        "n_baseline_better": n_negative,
        "n_comparison_better": n_positive,
        "n_tied": n_zero,
        "all_folds_agree": all_agree,
    }


# ---------------------------------------------------------------------------
# Verdict logic (from pre-registration)
# ---------------------------------------------------------------------------

def determine_verdict(
    n_folds: int,
    delta_no_topo: dict,
    delta_shuffled: dict,
    stability_no_topo: dict,
) -> str:
    """Apply pre-registered verdict logic.

    LOAD_BEARING:
        delta RMSE CI excludes 0 AND baseline < no_topology (topology helps).
        i.e., delta_no_topo mean < 0 AND CI_upper < 0.

    NOT_LOAD_BEARING:
        delta RMSE CI includes 0 OR baseline >= no_topology.

    INCONCLUSIVE:
        fewer than 3 folds completed.
    """
    if n_folds < 3:
        return "INCONCLUSIVE"

    ci_excludes_zero = (
        delta_no_topo["ci_upper"] < 0 or delta_no_topo["ci_lower"] > 0
    )
    baseline_better = delta_no_topo["mean"] < 0

    if ci_excludes_zero and baseline_better:
        return "LOAD_BEARING"
    else:
        return "NOT_LOAD_BEARING"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scenario: str, upload: bool) -> None:
    t0 = time.time()
    s3 = get_s3_client()
    git_hash = get_git_hash()

    print("=" * 60)
    print(f"  DOE-P1: TOPOLOGY NECESSITY -- {scenario}")
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
    topo_available = [f for f in TOPOLOGY_FEATURES
                      if f in merged.columns and merged[f].notna().any()]
    if len(topo_available) < len(TOPOLOGY_FEATURES):
        missing = set(TOPOLOGY_FEATURES) - set(topo_available)
        raise RuntimeError(
            f"FAIL_DATA_READY: Topology features missing after join: {missing}"
        )

    # Arm 1: p1_baseline = R0 + topology
    baseline_features = r0_available + TOPOLOGY_FEATURES

    # Arm 2: p1_no_topology = R0 only
    no_topo_features = r0_available.copy()

    # Arm 3: p1_shuffled_topology = R0 + shuffled topology
    merged = build_shuffled_features(merged)
    shuffled_cols = [f"{f}_shuffled" for f in TOPOLOGY_FEATURES]
    shuffled_features = r0_available + shuffled_cols

    log.info("Feature counts: baseline=%d, no_topology=%d, shuffled=%d",
             len(baseline_features), len(no_topo_features), len(shuffled_features))

    # Run all 3 arms
    arms = {}

    log.info("\n=== ARM 1: p1_baseline ===")
    baseline_folds = run_arm(merged, baseline_features, "p1_baseline")
    arms["p1_baseline"] = baseline_folds

    log.info("\n=== ARM 2: p1_no_topology ===")
    no_topo_folds = run_arm(merged, no_topo_features, "p1_no_topology")
    arms["p1_no_topology"] = no_topo_folds

    log.info("\n=== ARM 3: p1_shuffled_topology ===")
    shuffled_folds = run_arm(merged, shuffled_features, "p1_shuffled_topology")
    arms["p1_shuffled_topology"] = shuffled_folds

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

    # Paired comparisons (only if we have matching folds)
    # Align folds by fold ID
    def align_folds(a_folds, b_folds):
        a_map = {f["fold"]: f["rmse"] for f in a_folds}
        b_map = {f["fold"]: f["rmse"] for f in b_folds}
        common = sorted(set(a_map.keys()) & set(b_map.keys()))
        return [a_map[k] for k in common], [b_map[k] for k in common], len(common)

    # baseline vs no_topology
    bl_rmse, nt_rmse, n_common_nt = align_folds(baseline_folds, no_topo_folds)
    # baseline vs shuffled
    bl_rmse_s, sh_rmse, n_common_sh = align_folds(baseline_folds, shuffled_folds)

    delta_rmse = {}
    if n_common_nt >= 2:
        delta_no_topo = paired_bootstrap_delta(bl_rmse, nt_rmse)
        stability_no_topo = compute_fold_stability(bl_rmse, nt_rmse)
        delta_rmse["baseline_vs_no_topology"] = {
            **delta_no_topo,
            "fold_stability": stability_no_topo,
        }
    else:
        delta_no_topo = {"mean": None, "ci_lower": None, "ci_upper": None, "p_value": None}
        stability_no_topo = {"n_folds": 0, "all_folds_agree": False}
        delta_rmse["baseline_vs_no_topology"] = delta_no_topo

    if n_common_sh >= 2:
        delta_shuffled = paired_bootstrap_delta(bl_rmse_s, sh_rmse)
        stability_shuffled = compute_fold_stability(bl_rmse_s, sh_rmse)
        delta_rmse["baseline_vs_shuffled"] = {
            **delta_shuffled,
            "fold_stability": stability_shuffled,
        }
    else:
        delta_shuffled = {"mean": None, "ci_lower": None, "ci_upper": None}
        delta_rmse["baseline_vs_shuffled"] = delta_shuffled

    # Verdict
    n_completed_folds = min(n_common_nt, n_common_sh)
    verdict = determine_verdict(
        n_completed_folds, delta_no_topo, delta_shuffled, stability_no_topo,
    )

    elapsed = time.time() - t0

    # Build output payload
    payload = {
        "scenario": scenario,
        "doe_id": "DOE-P1",
        "experiment": "s035-model-ladder",
        "phase": "topology_necessity",
        "arms": arm_summaries,
        "delta_rmse": delta_rmse,
        "verdict": verdict,
        "n_folds": n_completed_folds,
        "n_samples": len(merged),
        "n_valid_target": n_valid_target,
        "topology_features": TOPOLOGY_FEATURES,
        "r0_features_used": r0_available,
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
            "min_hydrology_coverage": MIN_HYDROLOGY_COVERAGE,
        },
        "git_hash": git_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 1),
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"  DOE-P1 RESULTS: {scenario}")
    print("=" * 60)

    for arm_name, summary in arm_summaries.items():
        if summary["mean_rmse"] is not None:
            print(
                f"  {arm_name:25s}  RMSE={summary['mean_rmse']:.4f} "
                f"+/- {summary['std_rmse']:.4f}  "
                f"R2={summary['mean_r2']:.4f}  "
                f"({summary['n_folds']} folds)"
            )
        else:
            print(f"  {arm_name:25s}  NO RESULTS")

    if delta_rmse.get("baseline_vs_no_topology", {}).get("mean") is not None:
        d = delta_rmse["baseline_vs_no_topology"]
        print(
            f"\n  delta RMSE (baseline - no_topology): "
            f"{d['mean']:+.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"
        )
        if "p_value" in d and d["p_value"] is not None:
            print(f"  p-value: {d['p_value']:.4f}")

    if delta_rmse.get("baseline_vs_shuffled", {}).get("mean") is not None:
        d = delta_rmse["baseline_vs_shuffled"]
        print(
            f"  delta RMSE (baseline - shuffled):    "
            f"{d['mean']:+.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"
        )

    print(f"\n  VERDICT: {verdict}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 60)

    # Upload or write locally
    if upload:
        key = f"{RESULTS_PREFIX}/topology_necessity_{scenario}.json"
        upload_json_result(s3, BUCKET, key, payload, git_hash=git_hash)
    else:
        local = Path(f"/tmp/topology_necessity_{scenario}.json")
        local.write_text(json.dumps(payload, indent=2, default=str))
        log.info("Wrote %s", local)

    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOE-P1: Topology Necessity ablation test"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS,
                        help="Scenario to process (one per job)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload results to S3")
    args = parser.parse_args()

    run(args.scenario, args.upload)


if __name__ == "__main__":
    main()
