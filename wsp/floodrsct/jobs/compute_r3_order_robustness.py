#!/usr/bin/env python3
"""
compute_r3_order_robustness.py -- Phase R3_1b: Order robustness test.

Tests whether block admission order affects the R3 outcome. For each
target, runs the block addition in multiple orderings:
  - Forward:  terrain -> hydrology -> spatial -> temporal -> infra -> socioeconomic
  - Reverse:  socioeconomic -> infra -> temporal -> spatial -> hydrology -> terrain
  - Random permutations (20 seeds)

A block is order-robust if its contribution (positive/negative/neutral)
is consistent across orderings. H7 requires >= 80% concordance.

Usage:
    python compute_r3_order_robustness.py --scenario houston --upload
    python compute_r3_order_robustness.py --scenario houston --dry-run
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import permutations
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
RESULTS_PREFIX = "results/s035"
N_RANDOM_ORDERINGS = 20

# Block definitions (must match compute_r3_block_tests.py)
BLOCK_NAMES = [
    "terrain", "hydrology", "spatial_relation",
    "temporal", "infrastructure", "socioeconomic",
]

FORWARD_ORDER = [
    "terrain", "hydrology", "spatial_relation",
    "temporal", "infrastructure", "socioeconomic",
]

REVERSE_ORDER = list(reversed(FORWARD_ORDER))

# Feature lists (imported logic from compute_r3_block_tests.py)
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
    # Hydrology (OWP HAND + 3DEP zonal stats)
    "hand_mean_m", "twi_mean", "gfi_mean", "spi_mean",
]

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

PRIMARY_METRIC = {"regression": "r2", "classification": "roc_auc"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_parquet(s3, key: str) -> pd.DataFrame:
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


def _load_supplement(s3, scenario: str, level: str) -> pd.DataFrame:
    key = f"processed/{scenario}/{scenario}_{level}_supplement.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


# ---------------------------------------------------------------------------
# Solver (histgbdt only for order test -- ridge is G5 solver compat)
# ---------------------------------------------------------------------------

def _train_histgbdt(X_train, y_train, X_test, y_test, task: str) -> float:
    from sklearn.ensemble import (
        HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    )
    from sklearn.metrics import r2_score, roc_auc_score

    n_bins = min(255, max(2, len(X_train) - 1))
    try:
        if task == "classification":
            model = HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, learning_rate=0.1,
                max_bins=n_bins, random_state=SEED,
            )
            model.fit(X_train, y_train)
            try:
                return float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            except ValueError:
                return float("nan")
        else:
            model = HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=0.1,
                max_bins=n_bins, random_state=SEED,
            )
            model.fit(X_train, y_train)
            return float(r2_score(y_test, model.predict(X_test)))
    except ValueError as exc:
        log.error("[DATA_QUALITY] HistGBDT fit FAILED n_train=%d: %s", len(X_train), exc)
        return float("nan")


MIN_FOLD_SAMPLES = 10  # sklearn HistGBDT binning needs >= ~10 rows


def _mean_fold_metric(
    merged: pd.DataFrame,
    features: list[str],
    y_col: str,
    task: str,
    fold_col: str,
) -> float:
    """Mean primary metric across folds for spatial_blocked histgbdt."""
    # Drop rows where target is NaN to avoid sklearn ValueError
    valid_mask = merged[y_col].notna().values
    df_valid = merged[valid_mask]
    X_all = df_valid[features].values.astype(np.float32)
    y_all = df_valid[y_col].values.astype(np.float32)
    fold_ids = sorted(df_valid[fold_col].unique())
    vals = []

    for fold_id in fold_ids:
        test_mask = (df_valid[fold_col] == fold_id).values
        train_mask = ~test_mask
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if len(X_train) < MIN_FOLD_SAMPLES or len(X_test) == 0:
            continue
        if len(np.unique(y_train)) < 2 and task == "classification":
            continue

        val = _train_histgbdt(X_train, y_train, X_test, y_test, task)
        if not np.isnan(val):
            vals.append(val)

    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Incremental ordering test
# ---------------------------------------------------------------------------

def run_ordering(
    ordering: list[str],
    merged: pd.DataFrame,
    target_col: str,
    y_col: str,
    task: str,
    fold_col: str,
) -> dict:
    """Run incremental block addition in the given order.

    Returns per-block marginal contribution (delta from adding that block).
    """
    # Start with R0 baseline
    current_features = [f for f in R0_FEATURES if f in merged.columns]
    baseline = _mean_fold_metric(merged, current_features, y_col, task, fold_col)

    steps = []
    prev_metric = baseline

    for block_name in ordering:
        block_feats = BLOCK_FEATURES.get(block_name, [])
        new_feats = [f for f in block_feats if f in merged.columns and f not in current_features]
        current_features = current_features + new_feats

        if not new_feats:
            steps.append({
                "block": block_name,
                "n_new_features": 0,
                "metric_after": prev_metric,
                "delta": 0.0,
                "contribution_sign": "neutral",
            })
            continue

        metric_after = _mean_fold_metric(merged, current_features, y_col, task, fold_col)
        delta = metric_after - prev_metric

        steps.append({
            "block": block_name,
            "n_new_features": len(new_feats),
            "metric_after": metric_after,
            "delta": delta,
            "contribution_sign": "positive" if delta > 0.001 else ("negative" if delta < -0.001 else "neutral"),
        })
        prev_metric = metric_after

    return {
        "ordering": ordering,
        "baseline_r0": baseline,
        "final_metric": prev_metric,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Concordance analysis
# ---------------------------------------------------------------------------

def compute_concordance(ordering_results: list[dict]) -> dict:
    """Compute per-block concordance rate across orderings.

    A block is order-robust if its contribution sign (positive/negative/neutral)
    is the same in >= 80% of orderings.
    """
    # Collect signs per block
    block_signs: dict[str, list[str]] = {b: [] for b in BLOCK_NAMES}
    for result in ordering_results:
        for step in result["steps"]:
            block_signs[step["block"]].append(step["contribution_sign"])

    concordance = {}
    for block_name, signs in block_signs.items():
        if not signs:
            concordance[block_name] = {
                "majority_sign": "unknown",
                "concordance_rate": 0.0,
                "n_orderings": 0,
                "order_robust": False,
            }
            continue

        from collections import Counter
        counts = Counter(signs)
        majority_sign, majority_count = counts.most_common(1)[0]
        rate = majority_count / len(signs)

        concordance[block_name] = {
            "majority_sign": majority_sign,
            "concordance_rate": round(rate, 4),
            "n_orderings": len(signs),
            "sign_distribution": dict(counts),
            "order_robust": rate >= 0.80,
        }

    n_robust = sum(1 for v in concordance.values() if v["order_robust"])
    overall_rate = n_robust / len(concordance) if concordance else 0.0

    return {
        "per_block": concordance,
        "n_robust": n_robust,
        "n_blocks": len(concordance),
        "overall_robustness_rate": round(overall_rate, 4),
        "h7_pass": overall_rate >= 0.80,
        "h7_threshold": 0.80,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase R3_1b: Order robustness permutation test"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--n-random", type=int, default=N_RANDOM_ORDERINGS,
                        help="Number of random orderings (default: 20)")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        n_total = 2 + args.n_random  # forward + reverse + random
        log.info("DRY RUN: would run %d orderings for %s", n_total, args.scenario)
        log.info("  Forward + reverse + %d random permutations", args.n_random)
        return 0

    s3 = get_s3_client()
    scenario = args.scenario

    print(f"\n{'='*60}")
    print(f"  S035 PHASE R3_1b: ORDER ROBUSTNESS -- {scenario}")
    print(f"{'='*60}\n")

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

    # Generate random orderings
    rng = np.random.default_rng(SEED)
    random_orderings = []
    for _ in range(args.n_random):
        order = list(BLOCK_NAMES)
        rng.shuffle(order)
        random_orderings.append(order)

    all_orderings = [
        ("forward", FORWARD_ORDER),
        ("reverse", REVERSE_ORDER),
    ] + [(f"random_{i}", order) for i, order in enumerate(random_orderings)]

    # Run per target
    all_results = []
    fold_col = "fold_spatial_blocked"

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

        if fold_col not in merged.columns:
            log.warning("Fold column %s not found, skipping target %s", fold_col, target_col)
            continue

        log.info("\n--- Target: %s (%s) ---", target_col, task)

        ordering_results = []
        for label, ordering in all_orderings:
            log.info("  Running ordering: %s", label)
            result = run_ordering(
                ordering=ordering,
                merged=merged,
                target_col=target_col,
                y_col=y_col,
                task=task,
                fold_col=fold_col,
            )
            result["label"] = label
            ordering_results.append(result)

        concordance = compute_concordance(ordering_results)

        all_results.append({
            "target": target_col,
            "task": task,
            "n_orderings": len(ordering_results),
            "orderings": ordering_results,
            "concordance": concordance,
        })

        log.info("  H7 concordance: %.1f%% (%s)",
                 concordance["overall_robustness_rate"] * 100,
                 "PASS" if concordance["h7_pass"] else "FAIL")

    # Assemble output
    output = {
        "phase": "R3_1b_order_robustness",
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_orderings_per_target": 2 + args.n_random,
        "results": all_results,
    }

    # Write local
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"r3_order_robustness_{scenario}.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    if args.upload:
        upload_json_result(
            s3, BUCKET,
            f"{RESULTS_PREFIX}/r3_order_robustness_{scenario}.json",
            output,
        )
        log.info("Uploaded to S3")

    return 0


if __name__ == "__main__":
    sys.exit(main())
