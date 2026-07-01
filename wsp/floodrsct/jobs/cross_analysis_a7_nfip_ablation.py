#!/usr/bin/env python3
"""
cross_analysis_a7_nfip_ablation.py -- A7: NFIP Feature Ablation.

Retrain R0 HistGBDT WITHOUT nfip_historical_frequency and
nfip_historical_severity. Tests whether removing the circular predictor:
  1. Stabilizes feature importance across scenarios (A4 replication)
  2. Improves cross-scenario transfer (A3 replication)
  3. Changes within-scenario R2 (measures NFIP dependence)

Usage:
    python cross_analysis_a7_nfip_ablation.py --upload
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

# Full R0 feature list
R0_FEATURES_FULL = [
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
    # Hydrology (OWP HAND + 3DEP zonal stats)
    "hand_mean_m", "twi_mean", "gfi_mean", "spi_mean",
]

# Ablated list: remove NFIP circular predictors
NFIP_ABLATED = ["nfip_historical_frequency", "nfip_historical_severity"]
R0_FEATURES_ABLATED = [f for f in R0_FEATURES_FULL if f not in NFIP_ABLATED]

TARGET_COL = "obs_nfip_event_claims"
FOLD_COL = "fold_spatial_blocked"


def _load_scenario_data(s3, scenario: str) -> pd.DataFrame:
    """Load event features + NFIP historical for a scenario."""
    df = load_processed_parquet(s3, scenario)
    df["zcta_id"] = df["zcta_id"].astype(str)
    if "event" in df.columns:
        df["event"] = df["event"].astype(str)

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


def _get_shared_features(datasets, feature_list):
    """Return features present and non-null in ALL scenarios."""
    return [
        f for f in feature_list
        if all(f in df.columns and df[f].notna().any() for df in datasets.values())
    ]


def _prepare_xy(df, features):
    """Prepare X, y arrays with log1p target transform."""
    mask = df[TARGET_COL].notna()
    df_valid = df[mask].copy()
    X = df_valid[features].values.astype(np.float32)
    y = np.log1p(df_valid[TARGET_COL].clip(lower=0).astype(float).values)
    return X, y


def _train_model(X_train, y_train):
    """Train HistGBDT."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=200, max_depth=6, learning_rate=0.1, random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model


def _r2(y_true, y_pred):
    from sklearn.metrics import r2_score
    v = float(r2_score(y_true, y_pred))
    return None if np.isnan(v) or np.isinf(v) else v


def _extract_importances(model, X_test, y_test, n_features):
    """Extract feature importances (split-based or permutation fallback)."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=SEED, n_jobs=-1,
    )
    return perm.importances_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="A7: NFIP Feature Ablation")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    print("\n" + "=" * 60)
    print("  A7: NFIP FEATURE ABLATION")
    print("  Testing: does removing NFIP history stabilize importance")
    print("  and improve cross-scenario transfer?")
    print("=" * 60 + "\n")

    # --- Load all scenarios ---
    datasets = {}
    folds = {}
    for sc in SCENARIOS:
        log.info("Loading %s...", sc)
        datasets[sc] = _load_scenario_data(s3, sc)
        folds[sc] = _load_folds(s3, sc)

    # --- Shared features for BOTH arms ---
    shared_full = _get_shared_features(datasets, R0_FEATURES_FULL)
    shared_ablated = _get_shared_features(datasets, R0_FEATURES_ABLATED)
    log.info("Shared features: full=%d, ablated=%d", len(shared_full), len(shared_ablated))
    log.info("Removed: %s", NFIP_ABLATED)

    # =========================================================
    # PART 1: Within-scenario R2 comparison (full vs ablated)
    # Uses spatial-blocked 5-fold CV like original R0
    # =========================================================
    log.info("\n--- PART 1: Within-Scenario R2 (Full vs Ablated) ---")

    within_results = {}
    importance_full = {}
    importance_ablated = {}

    for sc in SCENARIOS:
        log.info("Within-scenario CV for %s...", sc)
        df = datasets[sc]
        folds_df = folds[sc]

        merged = df.merge(
            folds_df[["zcta_id", "event", FOLD_COL]], on=["zcta_id", "event"],
        )
        mask = merged[TARGET_COL].notna()
        merged = merged[mask].copy()
        merged["_y"] = np.log1p(merged[TARGET_COL].clip(lower=0).astype(float))

        fold_ids = sorted(merged[FOLD_COL].unique())
        r2_full_folds = []
        r2_ablated_folds = []
        imp_full_folds = []
        imp_ablated_folds = []

        for fold_id in fold_ids:
            test_mask = merged[FOLD_COL] == fold_id
            train_mask = ~test_mask

            y_train = merged.loc[train_mask, "_y"].values
            y_test = merged.loc[test_mask, "_y"].values

            if len(y_test) == 0 or len(y_train) == 0:
                continue

            # Full model
            X_train_f = merged.loc[train_mask, shared_full].values.astype(np.float32)
            X_test_f = merged.loc[test_mask, shared_full].values.astype(np.float32)
            model_f = _train_model(X_train_f, y_train)
            r2_f = _r2(y_test, model_f.predict(X_test_f))
            r2_full_folds.append(r2_f)
            imp_full_folds.append(
                _extract_importances(model_f, X_test_f, y_test, len(shared_full))
            )

            # Ablated model
            X_train_a = merged.loc[train_mask, shared_ablated].values.astype(np.float32)
            X_test_a = merged.loc[test_mask, shared_ablated].values.astype(np.float32)
            model_a = _train_model(X_train_a, y_train)
            r2_a = _r2(y_test, model_a.predict(X_test_a))
            r2_ablated_folds.append(r2_a)
            imp_ablated_folds.append(
                _extract_importances(model_a, X_test_a, y_test, len(shared_ablated))
            )

        mean_r2_full = float(np.mean([r for r in r2_full_folds if r is not None]))
        mean_r2_ablated = float(np.mean([r for r in r2_ablated_folds if r is not None]))
        delta = mean_r2_ablated - mean_r2_full

        # Paired t-test on fold R2
        from scipy.stats import ttest_rel
        if len(r2_full_folds) == len(r2_ablated_folds) and len(r2_full_folds) >= 3:
            t_stat, p_val = ttest_rel(r2_ablated_folds, r2_full_folds)
            t_stat = float(t_stat)
            p_val = float(p_val)
        else:
            t_stat, p_val = None, None

        within_results[sc] = {
            "r2_full": mean_r2_full,
            "r2_ablated": mean_r2_ablated,
            "delta": float(delta),
            "t_stat": t_stat,
            "p_value": p_val,
            "n_folds": len(r2_full_folds),
            "r2_full_per_fold": [float(r) for r in r2_full_folds],
            "r2_ablated_per_fold": [float(r) for r in r2_ablated_folds],
        }

        log.info("  %s: R2_full=%.4f, R2_ablated=%.4f, delta=%.4f, p=%s",
                 sc, mean_r2_full, mean_r2_ablated, delta,
                 f"{p_val:.4f}" if p_val is not None else "N/A")

        # Store mean importances
        mean_imp_f = np.mean(imp_full_folds, axis=0)
        total_f = mean_imp_f.sum()
        if total_f > 0:
            mean_imp_f = mean_imp_f / total_f
        importance_full[sc] = dict(zip(shared_full, [float(v) for v in mean_imp_f]))

        mean_imp_a = np.mean(imp_ablated_folds, axis=0)
        total_a = mean_imp_a.sum()
        if total_a > 0:
            mean_imp_a = mean_imp_a / total_a
        importance_ablated[sc] = dict(zip(shared_ablated, [float(v) for v in mean_imp_a]))

    # =========================================================
    # PART 2: Feature importance stability (ablated)
    # Pairwise Kendall's tau on ablated importances
    # =========================================================
    log.info("\n--- PART 2: Feature Importance Stability (Ablated) ---")

    from scipy.stats import kendalltau

    tau_full = {}
    tau_ablated = {}

    for s1, s2 in combinations(SCENARIOS, 2):
        pair_key = f"{s1}__{s2}"

        # Full
        imp1_f = np.array([importance_full[s1][f] for f in shared_full])
        imp2_f = np.array([importance_full[s2][f] for f in shared_full])
        t_f, p_f = kendalltau(imp1_f, imp2_f)
        tau_full[pair_key] = {
            "kendall_tau": float(t_f) if not np.isnan(t_f) else None,
            "p_value": float(p_f) if not np.isnan(p_f) else None,
        }

        # Ablated
        imp1_a = np.array([importance_ablated[s1][f] for f in shared_ablated])
        imp2_a = np.array([importance_ablated[s2][f] for f in shared_ablated])
        t_a, p_a = kendalltau(imp1_a, imp2_a)
        tau_ablated[pair_key] = {
            "kendall_tau": float(t_a) if not np.isnan(t_a) else None,
            "p_value": float(p_a) if not np.isnan(p_a) else None,
        }

        log.info("  %s vs %s: tau_full=%.3f, tau_ablated=%.3f",
                 s1, s2,
                 t_f if not np.isnan(t_f) else 0,
                 t_a if not np.isnan(t_a) else 0)

    # Summary statistics
    taus_full = [v["kendall_tau"] for v in tau_full.values() if v["kendall_tau"] is not None]
    taus_ablated = [v["kendall_tau"] for v in tau_ablated.values() if v["kendall_tau"] is not None]

    stability_improved = bool(np.mean(taus_ablated) > np.mean(taus_full)) if taus_full and taus_ablated else None

    log.info("Mean tau: full=%.3f, ablated=%.3f, improved=%s",
             np.mean(taus_full) if taus_full else 0,
             np.mean(taus_ablated) if taus_ablated else 0,
             stability_improved)

    # =========================================================
    # PART 3: Cross-scenario transfer (ablated)
    # =========================================================
    log.info("\n--- PART 3: Cross-Scenario Transfer (Ablated) ---")

    # Prepare full and ablated data for transfer
    prepared_full = {}
    prepared_ablated = {}
    for sc in SCENARIOS:
        prepared_full[sc] = _prepare_xy(datasets[sc], shared_full)
        prepared_ablated[sc] = _prepare_xy(datasets[sc], shared_ablated)

    transfer_full = {}
    transfer_ablated = {}

    for source in SCENARIOS:
        transfer_full[source] = {}
        transfer_ablated[source] = {}
        for target in SCENARIOS:
            if source == target:
                transfer_full[source][target] = None
                transfer_ablated[source][target] = None
                continue

            # Full
            X_src_f, y_src_f = prepared_full[source]
            X_tgt_f, y_tgt_f = prepared_full[target]
            model_f = _train_model(X_src_f, y_src_f)
            r2_f = _r2(y_tgt_f, model_f.predict(X_tgt_f))
            transfer_full[source][target] = r2_f

            # Ablated
            X_src_a, y_src_a = prepared_ablated[source]
            X_tgt_a, y_tgt_a = prepared_ablated[target]
            model_a = _train_model(X_src_a, y_src_a)
            r2_a = _r2(y_tgt_a, model_a.predict(X_tgt_a))
            transfer_ablated[source][target] = r2_a

            delta = (r2_a or 0) - (r2_f or 0)
            log.info("  %s -> %s: full=%.4f, ablated=%.4f, delta=%+.4f",
                     source, target, r2_f or 0, r2_a or 0, delta)

    # Count positive transfer pairs
    n_pos_full = sum(
        1 for s in SCENARIOS for t in SCENARIOS
        if s != t and transfer_full[s][t] is not None and transfer_full[s][t] > 0
    )
    n_pos_ablated = sum(
        1 for s in SCENARIOS for t in SCENARIOS
        if s != t and transfer_ablated[s][t] is not None and transfer_ablated[s][t] > 0
    )

    # Mean off-diagonal R2
    off_diag_full = [
        transfer_full[s][t] for s in SCENARIOS for t in SCENARIOS
        if s != t and transfer_full[s][t] is not None
    ]
    off_diag_ablated = [
        transfer_ablated[s][t] for s in SCENARIOS for t in SCENARIOS
        if s != t and transfer_ablated[s][t] is not None
    ]
    mean_transfer_full = float(np.mean(off_diag_full)) if off_diag_full else None
    mean_transfer_ablated = float(np.mean(off_diag_ablated)) if off_diag_ablated else None

    log.info("Positive transfer pairs: full=%d/20, ablated=%d/20", n_pos_full, n_pos_ablated)
    log.info("Mean transfer R2: full=%.4f, ablated=%.4f",
             mean_transfer_full or 0, mean_transfer_ablated or 0)

    # =========================================================
    # PART 4: Top features (ablated)
    # =========================================================
    log.info("\n--- PART 4: Top Features (Ablated) ---")

    top5_ablated = {}
    for sc in SCENARIOS:
        sorted_feats = sorted(importance_ablated[sc].items(), key=lambda x: x[1], reverse=True)
        top5_ablated[sc] = [f[0] for f in sorted_feats[:5]]
        log.info("  %s: %s", sc, top5_ablated[sc])

    from collections import Counter
    all_top5 = [f for feats in top5_ablated.values() for f in feats]
    top5_counts = Counter(all_top5)
    universal_ablated = [f for f, c in top5_counts.items() if c >= 4]

    # =========================================================
    # Assemble payload
    # =========================================================
    payload = {
        "experiment": "s035-model-ladder",
        "analysis": "A7_nfip_ablation",
        "version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ablated_features": NFIP_ABLATED,
        "shared_features_full": shared_full,
        "shared_features_ablated": shared_ablated,
        "n_features_full": len(shared_full),
        "n_features_ablated": len(shared_ablated),
        "within_scenario_comparison": within_results,
        "importance_stability": {
            "tau_full": tau_full,
            "tau_ablated": tau_ablated,
            "mean_tau_full": float(np.mean(taus_full)) if taus_full else None,
            "mean_tau_ablated": float(np.mean(taus_ablated)) if taus_ablated else None,
            "stability_improved": stability_improved,
        },
        "transfer_comparison": {
            "transfer_r2_full": transfer_full,
            "transfer_r2_ablated": transfer_ablated,
            "n_positive_full": n_pos_full,
            "n_positive_ablated": n_pos_ablated,
            "mean_transfer_r2_full": mean_transfer_full,
            "mean_transfer_r2_ablated": mean_transfer_ablated,
        },
        "top_5_features_ablated": top5_ablated,
        "universal_features_ablated": universal_ablated,
        "importances_full": importance_full,
        "importances_ablated": importance_ablated,
        "hypotheses": {
            "H_A7_1": {
                "description": "Removing NFIP features reduces within-scenario R2 (measures NFIP dependence)",
                "mean_r2_delta": float(np.mean([v["delta"] for v in within_results.values()])),
            },
            "H_A7_2": {
                "description": "Removing NFIP features increases mean Kendall tau (importance stability improves)",
                "stability_improved": stability_improved,
                "mean_tau_full": float(np.mean(taus_full)) if taus_full else None,
                "mean_tau_ablated": float(np.mean(taus_ablated)) if taus_ablated else None,
            },
            "H_A7_3": {
                "description": "Removing NFIP features increases positive transfer pairs",
                "n_positive_full": n_pos_full,
                "n_positive_ablated": n_pos_ablated,
                "transfer_improved": bool(n_pos_ablated > n_pos_full),
            },
        },
    }

    results_json = json.dumps(payload, indent=2, default=str)
    print("\n" + results_json)

    if args.upload:
        key = f"{RESULTS_PREFIX}/a7_nfip_ablation.json"
        upload_json_result(s3, BUCKET, key, payload)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)
    else:
        local = "/tmp/a7_nfip_ablation.json"
        Path(local).write_text(results_json)
        log.info("Wrote %s", local)


if __name__ == "__main__":
    main()
