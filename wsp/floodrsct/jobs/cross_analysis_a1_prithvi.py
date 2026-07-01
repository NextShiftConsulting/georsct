#!/usr/bin/env python3
"""
cross_analysis_a1_prithvi.py -- A1: Prithvi vs Tabular Predictive Utility Test.

PCA-reduce Prithvi-EO-2.0 embeddings, train R0-only vs R0+Prithvi HistGBDT
with spatial-blocked 5-fold CV, compute per-fold R2 delta with paired t-test.
Descriptive CCA between Prithvi PCA and R0 features.

Usage:
    python cross_analysis_a1_prithvi.py --upload
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
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
    # Hydrology (OWP HAND + 3DEP zonal stats)
    "hand_mean_m", "twi_mean", "gfi_mean", "spi_mean",
]

TARGET_COL = "obs_nfip_event_claims"
FOLD_COL = "fold_spatial_blocked"


def _load_scenario_data(s3, scenario: str) -> pd.DataFrame:
    """Load event features + NFIP historical."""
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
    key = f"folds/{scenario}_folds.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    folds = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    folds["zcta_id"] = folds["zcta_id"].astype(str)
    if "event" in folds.columns:
        folds["event"] = folds["event"].astype(str)
    return folds


def _load_prithvi(s3, scenario: str) -> pd.DataFrame:
    """Load Prithvi embeddings, filter to HLS source only, drop NaN rows."""
    key = f"results/s035/prithvi_embeddings/{scenario}_prithvi_embeddings.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    prithvi = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    # Prithvi parquet uses "zcta" not "zcta_id"
    if "zcta" in prithvi.columns and "zcta_id" not in prithvi.columns:
        prithvi = prithvi.rename(columns={"zcta": "zcta_id"})
    prithvi["zcta_id"] = prithvi["zcta_id"].astype(str)

    # Filter: HLS source only (drop fallback_no_data rows which have NaN embeddings)
    if "source" in prithvi.columns:
        n_before = len(prithvi)
        prithvi = prithvi[prithvi["source"] == "hls"].copy()
        log.info("  Prithvi %s: %d -> %d after HLS filter", scenario, n_before, len(prithvi))

    # Identify embedding columns (emb_0, emb_1, ..., emb_1023)
    emb_cols = [c for c in prithvi.columns if c.startswith("prithvi_emb_") or c.startswith("emb_")]
    if not emb_cols:
        raise ValueError(f"No embedding columns found in Prithvi parquet for {scenario}")

    # Drop rows with any NaN in embeddings
    mask = prithvi[emb_cols].notna().all(axis=1)
    prithvi = prithvi[mask].copy()

    return prithvi, emb_cols


def _pca_on_unique_zctas(prithvi: pd.DataFrame, emb_cols: list[str], var_threshold: float = 0.90):
    """Fit PCA on unique ZCTAs to avoid inflating variance from duplicated rows."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    unique_zctas = prithvi.drop_duplicates(subset=["zcta_id"])
    X_unique = unique_zctas[emb_cols].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unique)

    # Find k components for >= var_threshold cumulative variance
    pca_full = PCA(n_components=min(100, X_scaled.shape[0], X_scaled.shape[1]), random_state=SEED)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, var_threshold) + 1)
    k = max(k, 5)  # at least 5 components

    pca = PCA(n_components=k, random_state=SEED)
    pca.fit(X_scaled)

    log.info("  PCA: %d components explain %.1f%% variance (threshold=%.0f%%)",
             k, cumvar[k - 1] * 100, var_threshold * 100)

    return scaler, pca, k


def main() -> None:
    parser = argparse.ArgumentParser(description="A1: Prithvi vs Tabular Utility Test")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    print("\n" + "=" * 60)
    print("  A1: PRITHVI vs TABULAR -- PREDICTIVE UTILITY TEST")
    print("=" * 60 + "\n")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    from scipy.stats import ttest_rel

    scenario_results = {}
    scenarios_with_uplift = 0

    for scenario in SCENARIOS:
        log.info("\n--- %s ---", scenario)

        # Load data
        df = _load_scenario_data(s3, scenario)
        folds_df = _load_folds(s3, scenario)
        prithvi, emb_cols = _load_prithvi(s3, scenario)

        # Available R0 features
        r0_feats = [f for f in R0_FEATURES if f in df.columns and df[f].notna().any()]

        # Join events with Prithvi on zcta_id (many-to-one: events to ZCTA embedding)
        n_before = len(df)
        df_joined = df.merge(prithvi[["zcta_id"] + emb_cols], on="zcta_id", how="inner")
        n_after = len(df_joined)
        join_pct = n_after / n_before * 100 if n_before > 0 else 0
        log.info("  Join: %d -> %d events (%.1f%% retained)", n_before, n_after, join_pct)

        if join_pct < 80:
            log.warning("  QUALITY GATE: join coverage %.1f%% < 80%%", join_pct)

        # PCA on unique ZCTAs
        scaler, pca, k = _pca_on_unique_zctas(prithvi, emb_cols)

        # Transform all joined rows (duplicated ZCTAs get same PCA values)
        X_emb_scaled = scaler.transform(df_joined[emb_cols].values.astype(np.float32))
        X_pca = pca.transform(X_emb_scaled)
        pca_cols = [f"prithvi_pc{i}" for i in range(k)]
        for i, col in enumerate(pca_cols):
            df_joined[col] = X_pca[:, i]

        # Merge folds
        merged = df_joined.merge(folds_df[["zcta_id", "event", FOLD_COL]], on=["zcta_id", "event"])
        mask = merged[TARGET_COL].notna()
        merged = merged[mask].copy()
        merged["_y"] = np.log1p(merged[TARGET_COL].clip(lower=0).astype(float))

        if len(merged) < 100:
            log.warning("  QUALITY GATE: only %d rows after join+filter (need >= 100)", len(merged))

        fold_ids = sorted(merged[FOLD_COL].unique())
        r2_r0_only = []
        r2_r0_prithvi = []

        for fold_id in fold_ids:
            test_mask = merged[FOLD_COL] == fold_id
            train_mask = ~test_mask

            y_train = merged.loc[train_mask, "_y"].values
            y_test = merged.loc[test_mask, "_y"].values

            if len(y_test) == 0 or len(y_train) == 0:
                continue

            # R0-only arm
            X_train_r0 = merged.loc[train_mask, r0_feats].values.astype(np.float32)
            X_test_r0 = merged.loc[test_mask, r0_feats].values.astype(np.float32)
            model_r0 = HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=0.1, random_state=SEED,
            )
            model_r0.fit(X_train_r0, y_train)
            r2_r0 = float(r2_score(y_test, model_r0.predict(X_test_r0)))

            # R0+Prithvi arm
            r0_prithvi_feats = r0_feats + pca_cols
            X_train_rp = merged.loc[train_mask, r0_prithvi_feats].values.astype(np.float32)
            X_test_rp = merged.loc[test_mask, r0_prithvi_feats].values.astype(np.float32)
            model_rp = HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=0.1, random_state=SEED,
            )
            model_rp.fit(X_train_rp, y_train)
            r2_rp = float(r2_score(y_test, model_rp.predict(X_test_rp)))

            r2_r0_only.append(r2_r0)
            r2_r0_prithvi.append(r2_rp)

            log.info("  Fold %s: R0=%.4f, R0+Prithvi=%.4f, delta=%.4f",
                     fold_id, r2_r0, r2_rp, r2_rp - r2_r0)

        # Paired t-test
        r2_delta = [rp - r0 for r0, rp in zip(r2_r0_only, r2_r0_prithvi)]
        mean_delta = float(np.mean(r2_delta))
        if len(r2_delta) >= 2:
            t_stat, p_val = ttest_rel(r2_r0_prithvi, r2_r0_only)
            # Cohen's d for paired samples
            d_diff = np.array(r2_delta)
            cohens_d = float(np.mean(d_diff) / np.std(d_diff, ddof=1)) if np.std(d_diff, ddof=1) > 0 else 0.0
        else:
            t_stat, p_val, cohens_d = None, None, None

        if mean_delta >= 0.02 and p_val is not None and p_val < 0.05:
            scenarios_with_uplift += 1

        # Descriptive CCA on unique ZCTAs
        cca_result = None
        try:
            from sklearn.cross_decomposition import CCA
            unique_merged = merged.drop_duplicates(subset=["zcta_id"])
            X_r0_cca = unique_merged[r0_feats].fillna(0).values.astype(np.float64)
            X_pca_cca = unique_merged[pca_cols].values.astype(np.float64)

            n_components = min(5, len(r0_feats), k, len(unique_merged) - 1)
            cca = CCA(n_components=n_components)
            cca.fit(X_r0_cca, X_pca_cca)
            X_c, Y_c = cca.transform(X_r0_cca, X_pca_cca)

            canon_corrs = []
            for i in range(n_components):
                cc = float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1])
                canon_corrs.append(cc if not np.isnan(cc) else None)

            cca_result = {
                "n_components": n_components,
                "canonical_correlations": canon_corrs,
                "n_zctas": len(unique_merged),
                "note": "descriptive only, no PASS/FAIL threshold",
            }
            log.info("  CCA canonical correlations: %s", [f"{c:.3f}" if c else "NaN" for c in canon_corrs])
        except Exception as e:
            log.warning("  CCA failed: %s", e)
            cca_result = {"error": str(e)}

        scenario_results[scenario] = {
            "n_events": len(merged),
            "n_zctas_matched": int(df_joined["zcta_id"].nunique()),
            "join_coverage_pct": round(join_pct, 1),
            "n_prithvi_pca_components": k,
            "pca_variance_explained": round(float(np.sum(pca.explained_variance_ratio_)) * 100, 1),
            "n_r0_features": len(r0_feats),
            "per_fold_r2_r0_only": r2_r0_only,
            "per_fold_r2_r0_prithvi": r2_r0_prithvi,
            "per_fold_r2_delta": r2_delta,
            "mean_r2_r0_only": float(np.mean(r2_r0_only)),
            "mean_r2_r0_prithvi": float(np.mean(r2_r0_prithvi)),
            "mean_r2_delta": mean_delta,
            "paired_ttest": {
                "t_statistic": float(t_stat) if t_stat is not None else None,
                "p_value": float(p_val) if p_val is not None else None,
                "cohens_d": cohens_d,
            },
            "cca": cca_result,
        }

        log.info("  %s summary: mean delta R2 = %.4f, t=%.3f, p=%.4f",
                 scenario, mean_delta,
                 t_stat if t_stat is not None else 0,
                 p_val if p_val is not None else 1)

    # --- AC-A1-1: uplift >= 0.02 in >= 3/5 scenarios ---
    ac_a1_1_pass = scenarios_with_uplift >= 3

    payload = {
        "experiment": "s035-model-ladder",
        "analysis": "A1_prithvi_utility",
        "version": "2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": TARGET_COL,
        "transform": "log1p",
        "scenarios": scenario_results,
        "hypotheses": {
            "AC_A1_1": {
                "description": "Adding Prithvi PCA to R0 improves held-out R2 by >= 0.02 in >= 3/5 scenarios",
                "pass": ac_a1_1_pass,
                "scenarios_with_uplift": scenarios_with_uplift,
                "threshold_r2_delta": 0.02,
                "threshold_n_scenarios": 3,
            },
            "AC_A1_2": {
                "description": "Canonical correlation between Prithvi PCA and R0 (descriptive)",
                "note": "No PASS/FAIL threshold; see per-scenario CCA results",
            },
        },
        "quality_gates": {
            "join_coverage_gte_80pct": {
                sc: r["join_coverage_pct"] >= 80
                for sc, r in scenario_results.items()
            },
            "n_events_gte_100": {
                sc: r["n_events"] >= 100
                for sc, r in scenario_results.items()
            },
        },
    }

    results_json = json.dumps(payload, indent=2, default=str)
    print("\n" + results_json)

    if args.upload:
        key = f"{RESULTS_PREFIX}/a1_prithvi_utility.json"
        upload_json_result(s3, BUCKET, key, payload)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)
    else:
        local = "/tmp/a1_prithvi_utility.json"
        Path(local).write_text(results_json)
        log.info("Wrote %s", local)


if __name__ == "__main__":
    main()
