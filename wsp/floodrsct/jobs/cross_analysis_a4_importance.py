#!/usr/bin/env python3
"""
cross_analysis_a4_importance.py -- A4: Feature Importance Stability Across Scenarios.

Retrain R0/R1/R2 HistGBDT per scenario with spatial-blocked 5-fold CV,
extract feature_importances_, compute pairwise Kendall's tau, identify
universal vs scenario-specific features.

Usage:
    python cross_analysis_a4_importance.py --upload
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, load_processed_parquet
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
RESULTS_PREFIX = "results/s035/cross_analysis"

# R0 feature list (must match train_r0_baseline.py)
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

TARGET_COL = "obs_nfip_event_claims"
FOLD_COL = "fold_spatial_blocked"


def _load_scenario_data(s3, scenario: str) -> pd.DataFrame:
    """Load event features + NFIP historical for a scenario."""
    df = load_processed_parquet(s3, scenario)
    df["zcta_id"] = df["zcta_id"].astype(str)
    if "event" in df.columns:
        df["event"] = df["event"].astype(str)

    # Merge NFIP historical supplement
    nfip_key = f"processed/{scenario}/{scenario}_nfip_historical.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=nfip_key)
    nfip = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    nfip["zcta_id"] = nfip["zcta_id"].astype(str)
    join_cols = ["zcta_id", "event"] if "event" in nfip.columns else ["zcta_id"]
    df = df.merge(nfip, on=join_cols, how="left")
    return df


def _load_folds(s3, scenario: str) -> pd.DataFrame:
    """Load frozen fold assignments from S3."""
    key = f"folds/{scenario}_folds.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    folds = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    folds["zcta_id"] = folds["zcta_id"].astype(str)
    if "event" in folds.columns:
        folds["event"] = folds["event"].astype(str)
    return folds


def _available_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """Return features present and non-null in df."""
    return [f for f in feature_list if f in df.columns and df[f].notna().any()]


def _extract_importances(
    df: pd.DataFrame,
    folds_df: pd.DataFrame,
    features: list[str],
    scenario: str,
) -> tuple[np.ndarray, list[dict]]:
    """Train HistGBDT across folds, return (mean_importances, fold_metrics)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    merged = df.merge(folds_df[["zcta_id", "event", FOLD_COL]], on=["zcta_id", "event"])
    mask = merged[TARGET_COL].notna()
    merged = merged[mask].copy()
    merged["_y"] = np.log1p(merged[TARGET_COL].clip(lower=0).astype(float))

    fold_ids = sorted(merged[FOLD_COL].unique())
    importances_per_fold = []
    fold_metrics = []

    for fold_id in fold_ids:
        test_mask = merged[FOLD_COL] == fold_id
        train_mask = ~test_mask

        X_train = merged.loc[train_mask, features].values.astype(np.float32)
        y_train = merged.loc[train_mask, "_y"].values
        X_test = merged.loc[test_mask, features].values.astype(np.float32)
        y_test = merged.loc[test_mask, "_y"].values

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=SEED,
        )
        model.fit(X_train, y_train)

        # sklearn HistGBDT: feature_importances_ may not exist on older versions.
        # Use permutation_importance as a robust fallback.
        if hasattr(model, "feature_importances_"):
            importances_per_fold.append(model.feature_importances_)
        else:
            from sklearn.inspection import permutation_importance
            perm = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=SEED, n_jobs=-1,
            )
            importances_per_fold.append(perm.importances_mean)

        from sklearn.metrics import r2_score, mean_squared_error
        y_pred = model.predict(X_test)
        fold_metrics.append({
            "fold": str(fold_id),
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        })

    mean_imp = np.mean(importances_per_fold, axis=0)
    # Normalize to sum=1
    total = mean_imp.sum()
    if total > 0:
        mean_imp = mean_imp / total

    return mean_imp, fold_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="A4: Feature Importance Stability")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    print("\n" + "=" * 60)
    print("  A4: FEATURE IMPORTANCE STABILITY ACROSS SCENARIOS")
    print("=" * 60 + "\n")

    # --- Load all scenarios and extract importances ---
    all_importances = {}  # scenario -> {feature: importance}
    all_metrics = {}
    sanity_r2 = {}

    # Load existing R0 results for sanity check
    for sc in SCENARIOS:
        try:
            r0_key = f"results/s035/r0_{sc}.json"
            resp = s3.get_object(Bucket=BUCKET, Key=r0_key)
            r0_data = json.loads(resp["Body"].read().decode())
            # Extract spatial_blocked HistGBDT R2 for primary target
            for run in r0_data.get("runs", []):
                if (run.get("solver") == "histgbdt" and
                    run.get("split") == "spatial_blocked" and
                    run.get("target") == TARGET_COL):
                    r2_val = run.get("metrics", {}).get("r2")
                    if r2_val is not None:
                        sanity_r2.setdefault(sc, []).append(r2_val)
        except Exception as e:
            log.warning("Could not load R0 results for %s: %s", sc, e)

    # Compute shared feature set across scenarios first
    datasets = {}
    folds = {}
    for sc in SCENARIOS:
        log.info("Loading %s...", sc)
        datasets[sc] = _load_scenario_data(s3, sc)
        folds[sc] = _load_folds(s3, sc)

    shared_features = _available_features(
        pd.concat(datasets.values(), ignore_index=True), R0_FEATURES
    )
    # Stricter: only features available in ALL scenarios
    shared_features = [
        f for f in R0_FEATURES
        if all(f in datasets[sc].columns and datasets[sc][f].notna().any() for sc in SCENARIOS)
    ]
    log.info("Shared R0 features: %d", len(shared_features))

    for sc in SCENARIOS:
        log.info("Extracting importances for %s...", sc)
        mean_imp, fold_mets = _extract_importances(
            datasets[sc], folds[sc], shared_features, sc
        )
        all_importances[sc] = dict(zip(shared_features, mean_imp.tolist()))
        all_metrics[sc] = fold_mets

        mean_r2 = np.mean([m["r2"] for m in fold_mets])
        log.info("  %s: mean R2=%.4f across %d folds", sc, mean_r2, len(fold_mets))

        # Sanity check against original R0
        if sc in sanity_r2:
            orig_mean = np.mean(sanity_r2[sc])
            delta = abs(mean_r2 - orig_mean)
            status = "PASS" if delta <= 0.01 else "FAIL"
            log.info("  Sanity check: retrained=%.4f, original=%.4f, delta=%.4f [%s]",
                     mean_r2, orig_mean, delta, status)

    # --- Pairwise Kendall's tau ---
    from scipy.stats import kendalltau

    pairwise_tau = {}
    scenario_pairs = list(combinations(SCENARIOS, 2))

    for s1, s2 in scenario_pairs:
        imp1 = np.array([all_importances[s1][f] for f in shared_features])
        imp2 = np.array([all_importances[s2][f] for f in shared_features])
        tau, p_val = kendalltau(imp1, imp2)
        pair_key = f"{s1}__{s2}"
        pairwise_tau[pair_key] = {
            "kendall_tau": float(tau) if not np.isnan(tau) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
        }
        log.info("  %s vs %s: tau=%.4f, p=%.4f", s1, s2, tau, p_val)

    # AC-A4-1: all pairwise tau > 0.40
    tau_values = [v["kendall_tau"] for v in pairwise_tau.values() if v["kendall_tau"] is not None]
    ac_a4_1_pass = all(t > 0.40 for t in tau_values) if tau_values else False

    # --- Top features analysis ---
    top_k = 5
    top_per_scenario = {}
    for sc in SCENARIOS:
        sorted_feats = sorted(all_importances[sc].items(), key=lambda x: x[1], reverse=True)
        top_per_scenario[sc] = [f[0] for f in sorted_feats[:top_k]]

    # Universal features: top-5 in >= 4 scenarios
    from collections import Counter
    all_top5 = [f for feats in top_per_scenario.values() for f in feats]
    top5_counts = Counter(all_top5)
    universal_features = [f for f, c in top5_counts.items() if c >= 4]
    scenario_specific = [f for f, c in top5_counts.items() if c == 1]

    # AC-A4-2: top-3 features same in >= 3 scenarios
    top3_per_scenario = {sc: set(feats[:3]) for sc, feats in top_per_scenario.items()}
    # Find features in top-3 of >= 3 scenarios
    all_top3 = [f for feats in top3_per_scenario.values() for f in feats]
    top3_counts = Counter(all_top3)
    top3_in_majority = [f for f, c in top3_counts.items() if c >= 3]
    ac_a4_2_pass = len(top3_in_majority) >= 3  # at least 3 shared top-3 features

    # --- Build importance heatmap data ---
    heatmap = {sc: all_importances[sc] for sc in SCENARIOS}

    # --- Assemble payload ---
    payload = {
        "experiment": "s035-model-ladder",
        "analysis": "A4_feature_importance_stability",
        "version": "2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shared_features": shared_features,
        "n_shared_features": len(shared_features),
        "importances": all_importances,
        "fold_metrics": all_metrics,
        "pairwise_kendall_tau": pairwise_tau,
        "top_5_per_scenario": top_per_scenario,
        "universal_features_gte_4_scenarios": universal_features,
        "scenario_specific_features": scenario_specific,
        "top3_in_majority": top3_in_majority,
        "importance_heatmap": heatmap,
        "hypotheses": {
            "AC_A4_1": {
                "description": "Kendall tau > 0.40 for all pairwise scenario comparisons",
                "pass": ac_a4_1_pass,
                "tau_values": tau_values,
                "min_tau": min(tau_values) if tau_values else None,
                "threshold": 0.40,
            },
            "AC_A4_2": {
                "description": "Top-3 features same in >= 3/5 scenarios",
                "pass": ac_a4_2_pass,
                "shared_top3": top3_in_majority,
            },
        },
        "quality_gates": {
            "shared_features_gte_25": len(shared_features) >= 25,
            "sanity_r2_check": {
                sc: {
                    "retrained_mean_r2": float(np.mean([m["r2"] for m in all_metrics[sc]])),
                    "original_mean_r2": float(np.mean(sanity_r2.get(sc, [0]))),
                    "delta": float(abs(
                        np.mean([m["r2"] for m in all_metrics[sc]]) -
                        np.mean(sanity_r2.get(sc, [0]))
                    )),
                    "pass": abs(
                        np.mean([m["r2"] for m in all_metrics[sc]]) -
                        np.mean(sanity_r2.get(sc, [0]))
                    ) <= 0.01 if sc in sanity_r2 else None,
                }
                for sc in SCENARIOS
            },
        },
    }

    results_json = json.dumps(payload, indent=2, default=str)
    print("\n" + results_json)

    if args.upload:
        key = f"{RESULTS_PREFIX}/a4_feature_importance_stability.json"
        upload_json_result(s3, BUCKET, key, payload)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)
    else:
        local = "/tmp/a4_feature_importance_stability.json"
        Path(local).write_text(results_json)
        log.info("Wrote %s", local)


if __name__ == "__main__":
    main()
