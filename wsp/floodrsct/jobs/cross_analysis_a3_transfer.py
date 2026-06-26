#!/usr/bin/env python3
"""
cross_analysis_a3_transfer.py -- A3: Cross-Scenario Transfer Matrix.

Train R0 HistGBDT on each source scenario, predict on all target scenarios.
Build 5x5 transfer matrix. Compute partial Spearman correlation between
transfer R2 and Wasserstein distance controlling for source n_events.

Usage:
    python cross_analysis_a3_transfer.py --upload
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

# R0 feature list (must match train_r0_baseline.py exactly)
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
TRANSFORM = "log1p"


def _load_scenario_data(s3, scenario: str) -> pd.DataFrame:
    """Load event features + NFIP historical for a scenario."""
    df = load_processed_parquet(s3, scenario)
    df["zcta_id"] = df["zcta_id"].astype(str)

    # Merge NFIP historical supplement
    nfip_key = f"processed/{scenario}/{scenario}_nfip_historical.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=nfip_key)
    nfip = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    nfip["zcta_id"] = nfip["zcta_id"].astype(str)
    join_cols = ["zcta_id", "event"] if "event" in nfip.columns else ["zcta_id"]
    df = df.merge(nfip, on=join_cols, how="left")

    return df


def _get_shared_features(datasets: dict[str, pd.DataFrame]) -> list[str]:
    """Return R0 features present and non-null in ALL scenarios."""
    shared = []
    for f in R0_FEATURES:
        if all(f in df.columns and df[f].notna().any() for df in datasets.values()):
            shared.append(f)
    return shared


def _prepare_xy(df: pd.DataFrame, features: list[str]):
    """Prepare X, y arrays with log1p target transform. Drop null-target rows."""
    mask = df[TARGET_COL].notna()
    df_valid = df[mask].copy()
    X = df_valid[features].values.astype(np.float32)
    y = np.log1p(df_valid[TARGET_COL].clip(lower=0).astype(float).values)
    return X, y


def _train_and_predict(X_train, y_train, X_test):
    """Train HistGBDT on source, predict on target."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=200, max_depth=6, learning_rate=0.1, random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _r2(y_true, y_pred) -> float:
    from sklearn.metrics import r2_score
    v = float(r2_score(y_true, y_pred))
    return None if np.isnan(v) or np.isinf(v) else v


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _ks_test(a, b) -> dict:
    """Two-sample KS test between arrays a and b."""
    from scipy.stats import ks_2samp
    stat, p = ks_2samp(a, b)
    return {"statistic": float(stat), "p_value": float(p)}


def main() -> None:
    parser = argparse.ArgumentParser(description="A3: Cross-Scenario Transfer Matrix")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    print("\n" + "=" * 60)
    print("  A3: CROSS-SCENARIO TRANSFER MATRIX")
    print("=" * 60 + "\n")

    # --- Load all scenarios ---
    datasets = {}
    for sc in SCENARIOS:
        log.info("Loading %s...", sc)
        datasets[sc] = _load_scenario_data(s3, sc)
        log.info("  %s: %d rows", sc, len(datasets[sc]))

    # --- Shared feature intersection ---
    shared_features = _get_shared_features(datasets)
    log.info("Shared features: %d / %d", len(shared_features), len(R0_FEATURES))
    if len(shared_features) < 25:
        log.warning("QUALITY GATE FAIL: only %d shared features (need >= 25)", len(shared_features))

    # Report missing features per scenario
    feature_report = {}
    for sc, df in datasets.items():
        available = [f for f in R0_FEATURES if f in df.columns and df[f].notna().any()]
        missing = [f for f in R0_FEATURES if f not in available]
        feature_report[sc] = {"available": len(available), "missing": missing}
        log.info("  %s: %d available, missing=%s", sc, len(available), missing)

    # --- Prepare X, y for all scenarios ---
    prepared = {}
    for sc, df in datasets.items():
        X, y = _prepare_xy(df, shared_features)
        prepared[sc] = (X, y)
        log.info("  %s: %d samples after target filter", sc, len(y))
        if len(y) < 50:
            log.warning("QUALITY GATE: %s has only %d rows (need >= 50)", sc, len(y))

    # --- Build transfer matrix ---
    transfer_r2 = {}
    transfer_rmse = {}
    transfer_details = {}

    for source in SCENARIOS:
        X_src, y_src = prepared[source]
        transfer_r2[source] = {}
        transfer_rmse[source] = {}
        transfer_details[source] = {}

        for target in SCENARIOS:
            if source == target:
                transfer_r2[source][target] = None  # diagonal left empty
                transfer_rmse[source][target] = None
                continue

            X_tgt, y_tgt = prepared[target]
            log.info("Transfer: %s -> %s (train=%d, test=%d)",
                     source, target, len(y_src), len(y_tgt))

            y_pred = _train_and_predict(X_src, y_src, X_tgt)
            r2_val = _r2(y_tgt, y_pred)
            rmse_val = _rmse(y_tgt, y_pred)

            # Mean-prediction baseline
            y_mean = np.full_like(y_tgt, y_src.mean())
            r2_mean = _r2(y_tgt, y_mean)
            rmse_mean = _rmse(y_tgt, y_mean)

            # Random-feature baseline
            rng = np.random.default_rng(SEED)
            X_src_rand = rng.standard_normal(X_src.shape).astype(np.float32)
            X_tgt_rand = rng.standard_normal(X_tgt.shape).astype(np.float32)
            y_pred_rand = _train_and_predict(X_src_rand, y_src, X_tgt_rand)
            r2_rand = _r2(y_tgt, y_pred_rand)

            transfer_r2[source][target] = r2_val
            transfer_rmse[source][target] = rmse_val

            # Per-feature KS test for distribution shift
            ks_results = {}
            for i, feat in enumerate(shared_features):
                ks_results[feat] = _ks_test(X_src[:, i], X_tgt[:, i])

            n_shifted = sum(1 for v in ks_results.values() if v["p_value"] < 0.05)

            transfer_details[source][target] = {
                "r2": r2_val,
                "rmse": rmse_val,
                "r2_mean_baseline": r2_mean,
                "rmse_mean_baseline": rmse_mean,
                "r2_random_baseline": r2_rand,
                "n_train": len(y_src),
                "n_test": len(y_tgt),
                "n_features_shifted": n_shifted,
                "n_features_total": len(shared_features),
            }

            log.info("  R2=%.4f (mean_baseline=%.4f, random=%.4f), RMSE=%.4f, shifted=%d/%d",
                     r2_val or 0, r2_mean or 0, r2_rand or 0, rmse_val,
                     n_shifted, len(shared_features))

    # --- Load event distance matrix ---
    log.info("Loading event distance matrix...")
    edm_key = "results/s035/sidecar/event_distance_matrix.json"
    resp = s3.get_object(Bucket=BUCKET, Key=edm_key)
    edm = json.loads(resp["Body"].read().decode())

    # --- Partial Spearman correlation: R2 vs Wasserstein, controlling for n_events ---
    from scipy.stats import spearmanr

    off_diag_r2 = []
    off_diag_dist = []
    off_diag_n_src = []
    pair_labels = []

    for source in SCENARIOS:
        for target in SCENARIOS:
            if source == target:
                continue
            r2_val = transfer_r2[source][target]
            if r2_val is None:
                continue

            # Extract Wasserstein distance from EDM
            # EDM structure may vary; try common key patterns
            dist = None
            for key_pattern in [
                f"{source}__{target}",
                f"{target}__{source}",
            ]:
                if isinstance(edm, dict):
                    if "wasserstein" in edm:
                        dist = edm["wasserstein"].get(key_pattern)
                    elif "distances" in edm:
                        for d in edm["distances"]:
                            if (d.get("source") == source and d.get("target") == target) or \
                               (d.get("source") == target and d.get("target") == source):
                                dist = d.get("wasserstein")
                                break
                    elif key_pattern in edm:
                        dist = edm[key_pattern].get("wasserstein")

            if dist is not None:
                off_diag_r2.append(r2_val)
                off_diag_dist.append(dist)
                off_diag_n_src.append(len(prepared[source][1]))
                pair_labels.append(f"{source}->{target}")

    # Partial correlation: residualize both R2 and distance on n_events
    partial_corr = None
    if len(off_diag_r2) >= 5:
        from scipy.stats import spearmanr as _sr

        r2_arr = np.array(off_diag_r2)
        dist_arr = np.array(off_diag_dist)
        n_arr = np.array(off_diag_n_src)

        # Simple partial correlation via residuals
        # Regress R2 on n, get residuals; regress dist on n, get residuals; correlate
        from numpy.polynomial.polynomial import polyfit, polyval
        r2_fit = polyfit(n_arr, r2_arr, 1)
        r2_resid = r2_arr - polyval(n_arr, r2_fit)
        dist_fit = polyfit(n_arr, dist_arr, 1)
        dist_resid = dist_arr - polyval(n_arr, dist_fit)

        rho, p_val = spearmanr(r2_resid, dist_resid)
        partial_corr = {
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
            "n_pairs": len(off_diag_r2),
            "method": "partial_spearman_controlling_n_events",
            "pairs": pair_labels,
        }
        log.info("Partial Spearman (R2 vs Wasserstein | n_events): rho=%.4f, p=%.4f, n=%d",
                 rho, p_val, len(off_diag_r2))

        # Raw (unconditional) for comparison
        rho_raw, p_raw = spearmanr(off_diag_r2, off_diag_dist)
        partial_corr["raw_spearman_rho"] = float(rho_raw) if not np.isnan(rho_raw) else None
        partial_corr["raw_p_value"] = float(p_raw) if not np.isnan(p_raw) else None
    else:
        log.warning("Insufficient pairs for partial correlation: %d (need >= 5)", len(off_diag_r2))

    # --- Check AC-A3-2: positive transfer exists ---
    positive_transfer_count = 0
    for source in SCENARIOS:
        targets_above = sum(
            1 for t in SCENARIOS
            if t != source and transfer_r2[source].get(t) is not None
            and transfer_r2[source][t] > 0.10
        )
        if targets_above >= 2:
            positive_transfer_count += 1
    ac_a3_2_pass = positive_transfer_count >= 1

    # --- Assemble payload ---
    payload = {
        "experiment": "s035-model-ladder",
        "analysis": "A3_cross_scenario_transfer",
        "version": "2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shared_features": shared_features,
        "n_shared_features": len(shared_features),
        "feature_report": feature_report,
        "target": TARGET_COL,
        "transform": TRANSFORM,
        "transfer_r2_matrix": transfer_r2,
        "transfer_rmse_matrix": transfer_rmse,
        "transfer_details": transfer_details,
        "scenario_n_events": {sc: len(prepared[sc][1]) for sc in SCENARIOS},
        "partial_correlation": partial_corr,
        "hypotheses": {
            "AC_A3_1": {
                "description": "Transfer R2 correlates negatively with Wasserstein distance (partial, controlling for n_events)",
                "result": partial_corr,
            },
            "AC_A3_2": {
                "description": "At least one source achieves transfer R2 > 0.10 on >= 2 targets",
                "pass": ac_a3_2_pass,
                "positive_transfer_count": positive_transfer_count,
            },
        },
        "quality_gates": {
            "shared_features_gte_25": len(shared_features) >= 25,
            "all_scenarios_gte_50_rows": all(len(prepared[sc][1]) >= 50 for sc in SCENARIOS),
        },
    }

    results_json = json.dumps(payload, indent=2, default=str)
    print("\n" + results_json)

    if args.upload:
        key = f"{RESULTS_PREFIX}/a3_transfer_matrix.json"
        upload_json_result(s3, BUCKET, key, payload)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)
    else:
        local = "/tmp/a3_transfer_matrix.json"
        Path(local).write_text(results_json)
        log.info("Wrote %s", local)


if __name__ == "__main__":
    main()
