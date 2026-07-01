#!/usr/bin/env python3
"""run_vertical_slice.py -- V8 vertical slice: 3 arms x 5 geometries x 4 gates.

Single-event end-to-end run producing the paper's core evidence table.
Runs Houston (Harvey/Imelda/Beryl) through:

Arms:
    R0-baseline:  raw R0 features -> HistGBDT/Ridge
    R3-MAE:       per-fold MAE encoder -> embeddings -> HistGBDT/Ridge
    R3-TJEPA:     per-fold TJEPA encoder -> embeddings -> HistGBDT/Ridge

Geometries (per arm):
    Prediction:   R2 on spatial-blocked folds (regression target)
    Ranking:      Spearman rho on top-quartile risk ZCTAs
    Clustering:   Moran's I of prediction residuals
    Allocation:   decision-theoretic top-k precision
    Stability:    cross-fold sigma (std of per-fold R2)

Gates (per arm):
    Gate 1: integrity (N, alpha)
    Gate 2: consensus (auto-pass under GEOSPATIAL_CONUS27)
    Gate 3: admissibility (kappa_compat vs oobleck threshold)
    Gate 4-GC: grounding (non-collapse + proxy checkability)

Sidecar measurements:
    Empirical variogram on flood target
    HUC8 watershed blocking comparison (if NHDPlus available)

Resource: ml.m5.2xlarge (8 vCPU, 32 GB). TJEPA pretraining needs more
memory than R0 baseline. ~60 min estimated.

S3 output:
    results/s035/vertical_slice/vertical_slice_houston.json
    results/s035/vertical_slice/vertical_slice_houston_predictions.parquet
"""

import argparse
import io
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    BUCKET, SCENARIOS, get_s3_client, load_processed_parquet, load_crosswalk,
)
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
RESULTS_PREFIX = "results/s035/vertical_slice"

# R0 features (same as train_r0_baseline.py)
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

# Physical proxies for Gate 4-GC (exogenous to prediction target)
PHYSICAL_PROXIES = [
    "elevation_m_msl",
    "impervious_pct",
    "coastal_distance_m",
    "twi_twi",
]

# Primary regression target
TARGET_COL = "obs_nfip_event_claims"
TARGET_TRANSFORM = "log1p"

# GEOSPATIAL_CONUS27 gate thresholds (frozen from presets.py)
GATE_THRESHOLDS = {
    "N_thr": 0.50,
    "alpha_min": 0.60,
    "kappa_base": 0.22,
    "lambda_turbulence": 0.0,
    "delta_kappa": 0.28,
    "steepness": 6.0,
    "sigma_c": 0.15,
    "kappa_L_min": 0.15,
}


# =====================================================================
# Solver functions
# =====================================================================

def _train_eval_histgbdt(X_train, y_train, X_test, y_test):
    """Train HistGBDT, return (y_pred, metrics_dict)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error

    model = HistGradientBoostingRegressor(
        max_iter=200, max_depth=6, learning_rate=0.1, random_state=SEED,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return y_pred, {"r2": r2, "rmse": rmse}


def _train_eval_ridge(X_train, y_train, X_test, y_test):
    """Train Ridge pipeline, return (y_pred, metrics_dict)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return y_pred, {"r2": r2, "rmse": rmse}


# =====================================================================
# Encoder pretraining (per-fold)
# =====================================================================

def _pretrain_tjepa(X_train, n_features, seed):
    """Pretrain TabularJEPA on train fold features. Returns fitted encoder."""
    from georsct.encoders.tjepa import TabularJEPA, TJEPAConfig

    config = TJEPAConfig(
        n_features=n_features, embed_dim=128, hidden_dim=256,
        n_layers=2, mask_ratio=0.3, ema_decay=0.996,
        learning_rate=1e-3, weight_decay=1e-4,
        n_epochs=100, batch_size=256, seed=seed,
    )
    encoder = TabularJEPA(config)
    encoder.fit(X_train)
    return encoder


def _pretrain_mae(X_train, n_features, seed):
    """Pretrain MaskedTabularAE on train fold features. Returns fitted encoder."""
    from georsct.encoders.masked_ae import MaskedTabularAE, MAEConfig

    config = MAEConfig(
        n_features=n_features, embed_dim=128, hidden_dim=256,
        n_layers=2, mask_ratio=0.3,
        learning_rate=1e-3, weight_decay=1e-4,
        n_epochs=100, batch_size=256, seed=seed,
    )
    encoder = MaskedTabularAE(config)
    encoder.fit(X_train)
    return encoder


# =====================================================================
# Geometry computations
# =====================================================================

def compute_ranking_geometry(y_true, y_pred, top_frac=0.25):
    """Spearman rank correlation restricted to top-quartile true values."""
    from scipy.stats import spearmanr

    n = len(y_true)
    if n < 10:
        return {"spearman_rho": None, "n_top": 0}

    threshold = np.percentile(y_true, 100 * (1 - top_frac))
    top_mask = y_true >= threshold
    n_top = int(top_mask.sum())

    if n_top < 5:
        return {"spearman_rho": None, "n_top": n_top}

    rho, p = spearmanr(y_true[top_mask], y_pred[top_mask])
    return {
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "spearman_p": float(p) if np.isfinite(p) else None,
        "n_top": n_top,
    }


def compute_clustering_geometry(residuals, lat, lon):
    """Moran's I of prediction residuals (spatial autocorrelation)."""
    try:
        from scipy.spatial import cKDTree

        n = len(residuals)
        if n < 20:
            return {"morans_i": None, "reason": "too_few_units"}

        coords = np.column_stack([lat, lon])
        tree = cKDTree(coords)
        # k nearest neighbors for spatial weights
        k = min(8, n - 1)
        dists, indices = tree.query(coords, k=k + 1)

        # Row-standardized spatial weights
        z = residuals - np.mean(residuals)
        denominator = np.sum(z ** 2)
        if denominator <= 0:
            return {"morans_i": None, "reason": "zero_variance_residuals"}

        numerator = 0.0
        W = 0.0
        for i in range(n):
            for j_idx in range(1, k + 1):  # skip self
                j = indices[i, j_idx]
                w = 1.0 / max(dists[i, j_idx], 1e-6)
                numerator += w * z[i] * z[j]
                W += w

        morans_i = float((n / W) * (numerator / denominator))
        return {"morans_i": morans_i, "n_units": n, "k_neighbors": k}
    except Exception as e:
        return {"morans_i": None, "reason": str(e)}


def compute_allocation_geometry(y_true, y_pred, top_k_frac=0.20):
    """Top-k precision: fraction of predicted top-k that are truly top-k."""
    n = len(y_true)
    k = max(1, int(n * top_k_frac))

    true_top_k = set(np.argsort(y_true)[-k:])
    pred_top_k = set(np.argsort(y_pred)[-k:])

    precision = len(true_top_k & pred_top_k) / k
    return {"top_k_precision": float(precision), "k": k}


# =====================================================================
# Gate evaluation
# =====================================================================

def derive_simplex(spatial_r2s, random_r2s):
    """Derive R, S_sup, N from per-fold R2 values."""
    spatial_metric = float(np.mean(spatial_r2s))
    random_metric = float(np.mean(random_r2s))

    R = float(np.clip(spatial_metric, 0.0, 1.0))

    if abs(random_metric) < 1e-12:
        S_sup = 0.0
    else:
        S_sup = float(np.clip((random_metric - spatial_metric) / abs(random_metric), 0.0, 1.0))

    N = float(np.clip(1.0 - R - S_sup, 0.0, 1.0))

    # Renormalize to simplex
    total = R + S_sup + N
    if total > 0:
        R, S_sup, N = R / total, S_sup / total, N / total

    return R, S_sup, N


def evaluate_gates(R, S_sup, N, sigma):
    """Evaluate Gates 1-3 under GEOSPATIAL_CONUS27."""
    alpha = R / (R + N) if (R + N) > 0 else 0.0
    kappa_compat = R * (1 - N)
    omega = 1 - S_sup

    # Gate 1: integrity
    noise_ok = N < GATE_THRESHOLDS["N_thr"]
    alpha_ok = alpha >= GATE_THRESHOLDS["alpha_min"]
    gate1_pass = noise_ok or alpha_ok

    # Gate 2: auto-pass (consensus disabled under CONUS27)
    gate2_pass = True

    # Gate 3: admissibility (oobleck threshold, flat under CONUS27)
    lam = GATE_THRESHOLDS["lambda_turbulence"]
    kappa_base = GATE_THRESHOLDS["kappa_base"]
    delta_k = GATE_THRESHOLDS["delta_kappa"]
    steep = GATE_THRESHOLDS["steepness"]
    sigma_c = GATE_THRESHOLDS["sigma_c"]

    if lam > 0:
        sigma_eff = lam * sigma
    else:
        sigma_eff = 0.0

    kappa_req = kappa_base + delta_k / (1.0 + np.exp(-steep * (sigma_eff - sigma_c)))
    gate3_pass = kappa_compat >= kappa_req

    if gate1_pass and gate2_pass and gate3_pass:
        decision = "EXECUTE"
        gate_reached = "GATE_3_ADMISSIBILITY"
    elif not gate1_pass:
        decision = "REJECT"
        gate_reached = "GATE_1_INTEGRITY"
    elif not gate3_pass:
        decision = "REJECT"
        gate_reached = "GATE_3_ADMISSIBILITY"
    else:
        decision = "REJECT"
        gate_reached = "GATE_2_CONSENSUS"

    return {
        "decision": decision,
        "gate_reached": gate_reached,
        "R": R, "S_sup": S_sup, "N": N,
        "alpha": alpha, "kappa_compat": kappa_compat,
        "omega": omega, "sigma": sigma,
        "kappa_req": float(kappa_req),
        "gate1_pass": gate1_pass,
        "gate2_pass": gate2_pass,
        "gate3_pass": gate3_pass,
    }


# =====================================================================
# Variogram
# =====================================================================

def compute_empirical_variogram(values, lat, lon, n_bins=15):
    """Compute empirical variogram on spatial data.

    Returns bin edges (km), semivariance, and estimated range.
    """
    from scipy.spatial.distance import pdist, squareform

    n = len(values)
    if n < 20:
        return {"range_km": None, "reason": "too_few_units"}

    # Approximate distance in km using lat/lon
    coords_rad = np.column_stack([np.radians(lat), np.radians(lon)])
    # Haversine approximation via Euclidean on radians * R_earth
    R_earth = 6371.0
    coords_km = np.column_stack([
        lat * 111.0,  # 1 degree lat ~ 111 km
        lon * 111.0 * np.cos(np.radians(np.mean(lat))),  # adjust for latitude
    ])

    dists = pdist(coords_km)
    max_dist = np.percentile(dists, 50)  # use half the max distance
    bins = np.linspace(0, max_dist, n_bins + 1)

    z = np.asarray(values, dtype=float)
    sq_diffs = pdist(z.reshape(-1, 1), metric="sqeuclidean")

    bin_centers = []
    semivariance = []

    for i in range(n_bins):
        mask = (dists >= bins[i]) & (dists < bins[i + 1])
        if mask.sum() < 10:
            continue
        gamma = float(0.5 * np.mean(sq_diffs[mask]))
        center = float(0.5 * (bins[i] + bins[i + 1]))
        bin_centers.append(center)
        semivariance.append(gamma)

    if len(bin_centers) < 3:
        return {"range_km": None, "reason": "insufficient_bins"}

    bin_centers = np.array(bin_centers)
    semivariance = np.array(semivariance)

    # Estimate range: distance at which semivariance reaches 95% of sill
    sill = np.max(semivariance)
    threshold = 0.95 * sill
    above_threshold = np.where(semivariance >= threshold)[0]

    if len(above_threshold) > 0:
        range_km = float(bin_centers[above_threshold[0]])
    else:
        range_km = float(bin_centers[-1])

    return {
        "range_km": range_km,
        "sill": float(sill),
        "n_bins_used": len(bin_centers),
        "max_lag_km": float(max_dist),
        "bin_centers_km": [float(x) for x in bin_centers],
        "semivariance": [float(x) for x in semivariance],
    }


# =====================================================================
# Main vertical slice
# =====================================================================

def run_arm(
    arm_name, X_features, df, folds_df, features_list,
    fold_col="fold_spatial_blocked",
    also_random=True,
):
    """Run one arm through spatial-blocked CV (and optionally random).

    Returns per-fold results for both splits.
    """
    split_cols = [fold_col]
    if also_random and "fold_random" in folds_df.columns:
        split_cols.append("fold_random")

    all_fold_results = []
    all_predictions = []

    for current_col in split_cols:
        folds, preds = _run_arm_single_split(
            arm_name, X_features, df, folds_df, current_col,
        )
        all_fold_results.extend(folds)
        # Only save predictions from spatial_blocked
        if current_col == fold_col:
            all_predictions.extend(preds)

    return all_fold_results, all_predictions


def _run_arm_single_split(arm_name, X_features, df, folds_df, fold_col):
    """Run one arm through one CV split."""
    merged = df.merge(
        folds_df[["zcta_id", "event", fold_col]],
        on=["zcta_id", "event"],
    )

    valid_mask = merged[TARGET_COL].notna()
    merged = merged[valid_mask].copy()

    # Apply transform
    merged["_y"] = np.log1p(merged[TARGET_COL].clip(lower=0).astype(float))

    fold_ids = sorted(merged[fold_col].unique())
    per_fold_results = []
    all_predictions = []

    for fold_id in fold_ids:
        test_mask = merged[fold_col] == fold_id
        train_mask = ~test_mask

        X_train = X_features[valid_mask.values][train_mask.values]
        X_test = X_features[valid_mask.values][test_mask.values]
        y_train = merged.loc[train_mask, "_y"].values.astype(np.float32)
        y_test = merged.loc[test_mask, "_y"].values.astype(np.float32)

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        # Train both solvers
        for solver_name, solver_fn in [("histgbdt", _train_eval_histgbdt), ("ridge", _train_eval_ridge)]:
            y_pred, metrics = solver_fn(X_train, y_train, X_test, y_test)

            per_fold_results.append({
                "arm": arm_name,
                "solver": solver_name,
                "fold": str(fold_id),
                "split": fold_col,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                **metrics,
            })

            # Save per-row predictions for geometry computation
            if solver_name == "histgbdt":
                test_idx = merged.index[test_mask]
                for i, idx in enumerate(test_idx):
                    all_predictions.append({
                        "zcta_id": str(merged.at[idx, "zcta_id"]),
                        "event": str(merged.at[idx, "event"]),
                        "arm": arm_name,
                        "fold": str(fold_id),
                        "y_true": float(y_test[i]),
                        "y_pred": float(y_pred[i]),
                        "lat": float(merged.at[idx, "latitude"]) if "latitude" in merged.columns else None,
                        "lon": float(merged.at[idx, "longitude"]) if "longitude" in merged.columns else None,
                    })

    return per_fold_results, all_predictions


def run_vertical_slice(scenario, s3, upload=False):
    """Run the full vertical slice for one scenario."""
    t0 = time.time()
    log.info("=== VERTICAL SLICE: %s ===", scenario.upper())

    # Load data
    log.info("Loading data...")
    df = load_processed_parquet(s3, scenario)
    df["zcta_id"] = df["zcta_id"].astype(str)

    # Load folds from S3
    log.info("Loading folds...")
    resp = s3.get_object(
        Bucket=BUCKET,
        Key=f"folds/{scenario}_folds.parquet",
    )
    folds_df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    folds_df["zcta_id"] = folds_df["zcta_id"].astype(str)

    # Available R0 features
    features = [f for f in R0_FEATURES if f in df.columns and df[f].notna().any()]
    log.info("Available features: %d / %d", len(features), len(R0_FEATURES))

    # Prepare feature matrix (with NaN imputation for encoders)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X_raw = df[features].values.astype(np.float32)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Also need fold column aligned with df rows
    fold_col = "fold_spatial_blocked"

    # =========================================================
    # ARM 1: R0-baseline (raw features)
    # =========================================================
    log.info("--- ARM: R0-baseline ---")
    r0_folds, r0_preds = run_arm("R0-baseline", X_imputed, df, folds_df, features)
    log.info("R0: %d fold results, %d predictions", len(r0_folds), len(r0_preds))

    # =========================================================
    # ARM 2: R3-MAE (per-fold pretrained MAE encoder)
    # =========================================================
    log.info("--- ARM: R3-MAE ---")
    try:
        mae_embeddings = _encode_per_fold(
            "mae", X_scaled, df, folds_df, fold_col, len(features),
        )
        mae_folds, mae_preds = run_arm("R3-MAE", mae_embeddings, df, folds_df, features)
        log.info("MAE: %d fold results, %d predictions", len(mae_folds), len(mae_preds))
    except Exception as e:
        log.error("R3-MAE failed: %s", e)
        mae_folds, mae_preds = [], []
        mae_embeddings = None

    # =========================================================
    # ARM 3: R3-TJEPA (per-fold pretrained TJEPA encoder)
    # =========================================================
    log.info("--- ARM: R3-TJEPA ---")
    try:
        tjepa_embeddings = _encode_per_fold(
            "tjepa", X_scaled, df, folds_df, fold_col, len(features),
        )
        tjepa_folds, tjepa_preds = run_arm("R3-TJEPA", tjepa_embeddings, df, folds_df, features)
        log.info("TJEPA: %d fold results, %d predictions", len(tjepa_folds), len(tjepa_preds))
    except Exception as e:
        log.error("R3-TJEPA failed: %s", e)
        tjepa_folds, tjepa_preds = [], []
        tjepa_embeddings = None

    # =========================================================
    # Aggregate per-arm results
    # =========================================================
    all_folds = r0_folds + mae_folds + tjepa_folds
    all_preds = r0_preds + mae_preds + tjepa_preds

    arm_summaries = {}
    for arm_name, arm_folds, arm_preds, arm_Z in [
        ("R0-baseline", r0_folds, r0_preds, X_imputed),
        ("R3-MAE", mae_folds, mae_preds, mae_embeddings),
        ("R3-TJEPA", tjepa_folds, tjepa_preds, tjepa_embeddings),
    ]:
        if not arm_folds:
            arm_summaries[arm_name] = {"error": "no results"}
            continue

        summary = _compute_arm_summary(
            arm_name, arm_folds, arm_preds, arm_Z, df, folds_df,
        )
        arm_summaries[arm_name] = summary
        log.info(
            "%s: decision=%s, R=%.3f, S_sup=%.3f, N=%.3f, Gate4=%s",
            arm_name,
            summary.get("gates_1_3", {}).get("decision"),
            summary.get("gates_1_3", {}).get("R", 0),
            summary.get("gates_1_3", {}).get("S_sup", 0),
            summary.get("gates_1_3", {}).get("N", 0),
            summary.get("gate4_gc", {}).get("verdict"),
        )

    # =========================================================
    # Variogram
    # =========================================================
    log.info("--- VARIOGRAM ---")
    variogram = _compute_variogram(df)

    # =========================================================
    # Assemble output
    # =========================================================
    elapsed = time.time() - t0
    result = {
        "experiment": "s035-vertical-slice",
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_features": len(features),
        "n_zctas": int(df["zcta_id"].nunique()),
        "n_rows": len(df),
        "target": TARGET_COL,
        "arms": arm_summaries,
        "variogram": variogram,
        "gate_thresholds": GATE_THRESHOLDS,
        "per_fold_results": all_folds,
    }

    log.info("=== VERTICAL SLICE COMPLETE: %.1f seconds ===", elapsed)

    if upload:
        # Upload JSON result
        key = f"{RESULTS_PREFIX}/vertical_slice_{scenario}.json"
        body = json.dumps(result, indent=2, default=str).encode("utf-8")
        s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/json")
        log.info("Uploaded: s3://%s/%s", BUCKET, key)

        # Upload predictions parquet
        if all_preds:
            preds_df = pd.DataFrame(all_preds)
            buf = io.BytesIO()
            preds_df.to_parquet(buf, index=False)
            buf.seek(0)
            pred_key = f"{RESULTS_PREFIX}/vertical_slice_{scenario}_predictions.parquet"
            s3.put_object(Bucket=BUCKET, Key=pred_key, Body=buf.read())
            log.info("Uploaded: s3://%s/%s", BUCKET, pred_key)
    else:
        out_path = f"/tmp/vertical_slice_{scenario}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info("Wrote: %s", out_path)

    return result


def _encode_per_fold(encoder_type, X_scaled, df, folds_df, fold_col, n_features):
    """Per-fold pretraining: train encoder on train fold, encode all rows.

    Returns embedding matrix (n_rows, embed_dim) where each row's embedding
    comes from an encoder that never saw that row during pretraining.
    """
    merged = df.merge(
        folds_df[["zcta_id", "event", fold_col]],
        on=["zcta_id", "event"],
    )
    fold_ids = sorted(merged[fold_col].unique())

    embed_dim = 128
    embeddings = np.full((len(df), embed_dim), np.nan, dtype=np.float32)

    for fold_id in fold_ids:
        test_mask = (merged[fold_col] == fold_id).values
        train_mask = ~test_mask

        X_train = X_scaled[train_mask]

        log.info("  %s fold %s: pretrain on %d, encode %d + %d",
                 encoder_type, fold_id, train_mask.sum(), train_mask.sum(), test_mask.sum())

        if encoder_type == "tjepa":
            encoder = _pretrain_tjepa(X_train, n_features, SEED + int(fold_id))
        elif encoder_type == "mae":
            encoder = _pretrain_mae(X_train, n_features, SEED + int(fold_id))
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")

        # Encode ALL rows using this fold's encoder
        # (test rows get their OOF embedding; train rows get in-fold embedding)
        all_emb = encoder.encode(X_scaled)
        # Only assign test rows from this fold's encoder (OOF)
        embeddings[test_mask] = all_emb[test_mask]

    # For rows that are NaN (shouldn't happen with proper folds), fill with mean
    nan_rows = np.any(np.isnan(embeddings), axis=1)
    if nan_rows.sum() > 0:
        log.warning("  %d rows with NaN embeddings, filling with mean", nan_rows.sum())
        col_means = np.nanmean(embeddings, axis=0)
        embeddings[nan_rows] = col_means

    return embeddings


def _compute_arm_summary(arm_name, arm_folds, arm_preds, arm_Z, df, folds_df):
    """Compute geometry verdicts and gate evaluations for one arm."""
    # Per-fold R2 for HistGBDT, separated by split type
    histgbdt_folds = [f for f in arm_folds if f["solver"] == "histgbdt"]

    # Spatial-blocked R2 (fold column contains int fold IDs)
    spatial_folds = [f for f in histgbdt_folds if f.get("split", "").startswith("fold_spatial")]
    spatial_r2s = [f["r2"] for f in spatial_folds if f["r2"] is not None]

    # Random R2 (fold column is "train"/"test")
    random_folds = [f for f in histgbdt_folds if f.get("split", "").startswith("fold_random")]
    random_r2s = [f["r2"] for f in random_folds if f["r2"] is not None]

    if not spatial_r2s:
        return {"error": "no valid spatial R2"}

    mean_r2 = float(np.mean(spatial_r2s))
    sigma = float(np.std(spatial_r2s, ddof=1)) if len(spatial_r2s) > 1 else 0.0

    # Full simplex derivation
    if random_r2s:
        random_metric = float(np.mean(random_r2s))
        R, S_sup, N = derive_simplex(spatial_r2s, [random_metric])
    else:
        R = float(np.clip(mean_r2, 0.0, 1.0))
        S_sup = 0.0
        N = float(np.clip(1.0 - R, 0.0, 1.0))

    # Gates 1-3
    gates = evaluate_gates(R, S_sup, N, sigma)

    # Geometry: prediction
    prediction = {
        "mean_r2": mean_r2,
        "per_fold_r2": spatial_r2s,
        "sigma": sigma,
    }

    # Geometry: ranking
    if arm_preds:
        preds_arr = np.array([(p["y_true"], p["y_pred"]) for p in arm_preds])
        ranking = compute_ranking_geometry(preds_arr[:, 0], preds_arr[:, 1])
    else:
        ranking = {"spearman_rho": None}

    # Geometry: clustering (Moran's I)
    if arm_preds and arm_preds[0].get("lat") is not None:
        residuals = np.array([p["y_true"] - p["y_pred"] for p in arm_preds])
        lats = np.array([p["lat"] for p in arm_preds])
        lons = np.array([p["lon"] for p in arm_preds])
        valid = np.isfinite(residuals) & np.isfinite(lats) & np.isfinite(lons)
        if valid.sum() >= 20:
            clustering = compute_clustering_geometry(residuals[valid], lats[valid], lons[valid])
        else:
            clustering = {"morans_i": None, "reason": "insufficient_valid_rows"}
    else:
        clustering = {"morans_i": None, "reason": "no_coordinates"}

    # Geometry: allocation
    if arm_preds:
        y_true = np.array([p["y_true"] for p in arm_preds])
        y_pred = np.array([p["y_pred"] for p in arm_preds])
        allocation = compute_allocation_geometry(y_true, y_pred)
    else:
        allocation = {"top_k_precision": None}

    # Gate 4-GC
    gate4 = _evaluate_gate4_for_arm(arm_Z, df, folds_df)

    return {
        "arm": arm_name,
        "n_folds": len(spatial_r2s),
        "prediction": prediction,
        "ranking": ranking,
        "clustering": clustering,
        "allocation": allocation,
        "stability": {"sigma": sigma},
        "gates_1_3": gates,
        "gate4_gc": gate4,
    }


def _evaluate_gate4_for_arm(Z, df, folds_df):
    """Run Gate 4-GC on an arm's representation matrix."""
    if Z is None:
        return {"verdict": "INSUFFICIENT", "reason": "no_embeddings"}

    try:
        from georsct.healthcheck.layers.gate4_grounding_gc import (
            evaluate_gate4, ProxyCheckSpec, Gate4Config, gate4_result_to_dict,
        )

        # Physical proxies from the data
        proxy_specs = []
        for proxy_col in PHYSICAL_PROXIES:
            if proxy_col in df.columns:
                vals = df[proxy_col].values.astype(float)
                if np.isfinite(vals).sum() >= 20:
                    proxy_specs.append(ProxyCheckSpec(
                        name=proxy_col,
                        y_true_proxy=vals,
                        required=True,
                    ))

        if not proxy_specs:
            return {"verdict": "INSUFFICIENT", "reason": "no_physical_proxies"}

        # Block IDs from spatial folds
        merged = df.merge(
            folds_df[["zcta_id", "event", "fold_spatial_blocked"]],
            on=["zcta_id", "event"],
        )
        block_id = merged["fold_spatial_blocked"].values

        config = Gate4Config(
            alpha=0.05, n_perm=999, base_seed=42,
            min_block_size=3, min_admissible_units=10,
        )

        result = evaluate_gate4(
            Z=Z,
            proxy_specs=proxy_specs,
            block_id=block_id,
            required_proxy_names=[s.name for s in proxy_specs],
            config=config,
        )

        return gate4_result_to_dict(result)

    except Exception as e:
        log.error("Gate 4-GC failed: %s", e)
        return {"verdict": "INSUFFICIENT", "reason": str(e)}


def _compute_variogram(df):
    """Compute empirical variogram on the flood target."""
    if TARGET_COL not in df.columns:
        return {"range_km": None, "reason": "target_missing"}

    if "latitude" not in df.columns or "longitude" not in df.columns:
        return {"range_km": None, "reason": "no_coordinates"}

    valid = (
        df[TARGET_COL].notna()
        & df["latitude"].notna()
        & df["longitude"].notna()
    )

    if valid.sum() < 30:
        return {"range_km": None, "reason": "too_few_valid_rows"}

    values = np.log1p(df.loc[valid, TARGET_COL].clip(lower=0).values.astype(float))
    lat = df.loc[valid, "latitude"].values.astype(float)
    lon = df.loc[valid, "longitude"].values.astype(float)

    return compute_empirical_variogram(values, lat, lon)


# =====================================================================
# Entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V8 vertical slice: 3 arms x 5 geometries x 4 gates"
    )
    parser.add_argument("--scenario", default="houston", choices=SCENARIOS)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()
    run_vertical_slice(args.scenario, s3, upload=args.upload)


if __name__ == "__main__":
    main()
