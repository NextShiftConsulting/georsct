#!/usr/bin/env python3
"""run_c6_svi_overlay.py -- C6 artifact: SVI-disagreement overlay.

Re-runs DOE-C1 fit_predict for each construct (deterministic: frozen
hyperparameters, frozen folds, same seed) and saves per-ZCTA predictions.
Then computes per-ZCTA disagreement and joins with SVI + HIFLD columns
already present in the event_features parquet.

Output:
  results/s035/doe_c1/c6_svi_overlay_{scenario}.json     (correlations + summary)
  results/s035/doe_c1/c6_svi_overlay_{scenario}.parquet   (per-ZCTA table)
  results/s035/doe_c1/c6_svi_overlay_all.json             (cross-scenario summary)

Claims backed:
  C6: High-disagreement zones overlap with socially vulnerable areas.
  C8: Certificate decomposition captures structure raw comparison misses.

Usage:
    python run_c6_svi_overlay.py --scenario houston --upload
    python run_c6_svi_overlay.py --all --upload
    python run_c6_svi_overlay.py --dry-run
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
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

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

RESULTS_PREFIX = "results/s035/doe_c1"

# --- Frozen hyperparameters (identical to run_five_construct_divergence.py) ---
HGBDT_PARAMS = dict(
    loss="squared_error",
    max_iter=300,
    max_depth=6,
    min_samples_leaf=10,
    learning_rate=0.05,
)

# --- Construct definitions (from georsct domain) ---
CONSTRUCT_ORDER = ["jrc", "deltares", "fema", "nfip", "fast"]
CONSTRUCT_TARGET_COLUMNS = {
    "jrc": "jrc_occurrence_mean",
    "deltares": "deltares_depth_ft_rp100",
    "fema": "flood_pct_zone_a",
    "nfip": "obs_nfip_event_claims",
    "fast": "fast_total_loss_usd",
}

# FAST return-period mapping (same as DOE-C1)
FAST_RP_MAP = {
    "houston": "1000yr",
    "southwest_florida": "cat4",
    "nyc": "100yr",
    "riverside_coachella": "",
    "new_orleans": "",
}

# SVI columns in event_features
SVI_COLUMNS = [
    "svi_overall",
    "svi_socioeconomic",
    "svi_household_disability",
    "svi_minority_language",
    "svi_housing_transport",
]

# HIFLD columns in event_features
HIFLD_COLUMNS = [
    "hifld_n_hospitals",
    "hifld_n_hospital_beds",
    "hifld_n_pharmacies",
    "hifld_nearest_hospital_km",
    "hifld_nearest_pharmacy_km",
    "hifld_nearest_trauma_center_km",
]


# =========================================================================
# Data loading (reuses DOE-C1 pattern exactly)
# =========================================================================

def _load_event_features(s3, scenario: str) -> pd.DataFrame:
    from _coverage_common import OUTPUT_KEYS
    key = OUTPUT_KEYS[scenario]
    log.info("Loading s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    df["zcta_id"] = df["zcta_id"].astype(str)
    log.info("Loaded %d rows x %d cols", len(df), len(df.columns))
    return df


def _merge_shared_layers(s3, df: pd.DataFrame) -> pd.DataFrame:
    shared_layers = {
        "processed/shared/zcta_jrc_water_occurrence_pct.parquet": [
            "jrc_occurrence_mean",
        ],
        "processed/shared/zcta_deltares_depth.parquet": [
            "deltares_depth_ft_rp100",
        ],
    }
    for key, cols in shared_layers.items():
        if all(c in df.columns for c in cols):
            continue
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            layer = pd.read_parquet(io.BytesIO(resp["Body"].read()))
            layer["zcta_id"] = layer["zcta_id"].astype(str)
            merge_cols = ["zcta_id"] + [c for c in cols if c in layer.columns]
            if len(merge_cols) > 1:
                df = df.merge(layer[merge_cols], on="zcta_id", how="left")
                log.info("Merged %d cols from %s", len(merge_cols) - 1, key)
        except Exception as exc:
            log.warning("Could not load shared layer %s: %s", key, exc)
    return df


def _load_folds(s3, scenario: str) -> pd.DataFrame:
    key = f"folds/{scenario}_folds.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        if "fold" not in df.columns and "fold_spatial_blocked" in df.columns:
            df["fold"] = df["fold_spatial_blocked"]
        elif "fold" not in df.columns and "fold_random" in df.columns:
            df["fold"] = df["fold_random"]
        return df
    except Exception:
        log.warning("Folds not found at %s, will use hash-based folds", key)
        return pd.DataFrame()


def _load_fast_target(s3, event_df: pd.DataFrame, scenario: str) -> pd.Series:
    """Load FAST target and align to event_df rows. Returns NaN-filled Series."""
    rp = FAST_RP_MAP.get(scenario, "")
    if not rp:
        return pd.Series(np.nan, index=event_df.index, name="fast_total_loss_usd")
    key = f"processed/{scenario}/{scenario}_fast_zcta_{rp}.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        fast_df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception:
        return pd.Series(np.nan, index=event_df.index, name="fast_total_loss_usd")

    if fast_df.index.name == "zcta":
        fast_df = fast_df.reset_index().rename(columns={"zcta": "zcta_id"})
    elif "zcta" in fast_df.columns:
        fast_df = fast_df.rename(columns={"zcta": "zcta_id"})
    fast_df["zcta_id"] = fast_df["zcta_id"].astype(str)

    target_col = "fast_total_loss_usd"
    if target_col not in fast_df.columns:
        return pd.Series(np.nan, index=event_df.index, name=target_col)

    fast_lookup = fast_df[["zcta_id", target_col]].drop_duplicates(subset="zcta_id")
    merged = pd.merge(
        event_df[["zcta_id"]].reset_index(),
        fast_lookup,
        on="zcta_id",
        how="left",
    ).set_index("index")
    return merged[target_col]


def _select_features(df: pd.DataFrame) -> list[str]:
    """Select numeric feature columns, excluding all construct targets."""
    construct_targets = set(CONSTRUCT_TARGET_COLUMNS.values())
    reserved = construct_targets | {
        "zcta_id", "event", "fold", "lat", "lon",
        "nfip_event_claim_count", "nfip_event_total_loss",
        "obs_has_311", "obs_has_hwm",
    }
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in reserved and not c.startswith("_fs_")
    ]


# =========================================================================
# Per-construct fit-predict (deterministic replay of DOE-C1)
# =========================================================================

def _fit_predict_construct(
    features: np.ndarray,
    target: np.ndarray,
    fold_ids: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Out-of-fold predictions for a single construct. Returns (n_obs,) array."""
    X = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    y = target.astype(float)
    folds = sorted(set(fold_ids))
    pred = np.full(len(y), np.nan, dtype=float)
    target_valid = np.isfinite(y)

    for fold in folds:
        train = fold_ids != fold
        test = ~train
        train_valid = train & target_valid
        test_valid = test & target_valid
        if train_valid.sum() < 20 or test_valid.sum() < 5:
            continue
        model = HistGradientBoostingRegressor(
            random_state=seed, **HGBDT_PARAMS,
        )
        model.fit(X[train_valid], y[train_valid])
        pred[test_valid] = model.predict(X[test_valid])

    return pred


# =========================================================================
# Per-ZCTA disagreement computation
# =========================================================================

def _percentile_rank(arr: np.ndarray) -> np.ndarray:
    """Rank values to [0,1] percentiles. NaN stays NaN."""
    result = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr)
    if valid.sum() < 2:
        return result
    ranked = stats.rankdata(arr[valid])
    result[valid] = (ranked - 1) / (len(ranked) - 1)
    return result


def compute_per_zcta_disagreement(
    zcta_ids: np.ndarray,
    construct_predictions: dict[str, np.ndarray],
    construct_targets: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute per-ZCTA disagreement from per-construct predictions.

    For each ZCTA:
    1. Average predictions across events (same ZCTA appears 3x for 3 events)
    2. Percentile-rank each construct's ZCTA-level predictions
    3. max_pairwise_divergence = max |rank_A - rank_B| across construct pairs
    4. mean_pairwise_divergence = mean |rank_A - rank_B|
    5. Also compute raw-score baseline (C8): max |target_A_rank - target_B_rank|

    Returns DataFrame indexed by zcta_id.
    """
    unique_zctas = sorted(set(zcta_ids))
    zcta_to_idx = {z: i for i, z in enumerate(unique_zctas)}
    n_zctas = len(unique_zctas)

    # Aggregate predictions to ZCTA level (mean across events)
    zcta_preds = {}
    zcta_targets = {}
    for construct, pred in construct_predictions.items():
        agg = np.full(n_zctas, np.nan, dtype=float)
        counts = np.zeros(n_zctas, dtype=float)
        for k, z in enumerate(zcta_ids):
            i = zcta_to_idx[z]
            if np.isfinite(pred[k]):
                if np.isnan(agg[i]):
                    agg[i] = 0.0
                agg[i] += pred[k]
                counts[i] += 1
        nonzero = counts > 0
        agg[nonzero] /= counts[nonzero]
        zcta_preds[construct] = agg

    for construct, tgt in construct_targets.items():
        agg = np.full(n_zctas, np.nan, dtype=float)
        counts = np.zeros(n_zctas, dtype=float)
        for k, z in enumerate(zcta_ids):
            i = zcta_to_idx[z]
            if np.isfinite(tgt[k]):
                if np.isnan(agg[i]):
                    agg[i] = 0.0
                agg[i] += tgt[k]
                counts[i] += 1
        nonzero = counts > 0
        agg[nonzero] /= counts[nonzero]
        zcta_targets[construct] = agg

    # Percentile-rank predictions per construct
    ranked_preds = {c: _percentile_rank(v) for c, v in zcta_preds.items()}
    ranked_targets = {c: _percentile_rank(v) for c, v in zcta_targets.items()}

    available = [c for c in CONSTRUCT_ORDER if c in ranked_preds]
    n_constructs = len(available)
    log.info("Computing disagreement across %d constructs: %s", n_constructs, available)

    # Per-ZCTA max/mean pairwise divergence (certificate-based)
    max_div = np.full(n_zctas, np.nan, dtype=float)
    mean_div = np.full(n_zctas, np.nan, dtype=float)
    # Per-ZCTA raw target divergence (C8 baseline)
    max_raw_div = np.full(n_zctas, np.nan, dtype=float)
    mean_raw_div = np.full(n_zctas, np.nan, dtype=float)

    pairs = list(combinations(available, 2))
    for i in range(n_zctas):
        pw_dists = []
        raw_dists = []
        for ca, cb in pairs:
            ra, rb = ranked_preds[ca][i], ranked_preds[cb][i]
            if np.isfinite(ra) and np.isfinite(rb):
                pw_dists.append(abs(ra - rb))
            ta, tb = ranked_targets[ca][i], ranked_targets[cb][i]
            if np.isfinite(ta) and np.isfinite(tb):
                raw_dists.append(abs(ta - tb))

        if pw_dists:
            max_div[i] = max(pw_dists)
            mean_div[i] = float(np.mean(pw_dists))
        if raw_dists:
            max_raw_div[i] = max(raw_dists)
            mean_raw_div[i] = float(np.mean(raw_dists))

    result = pd.DataFrame({
        "zcta_id": unique_zctas,
        "max_pairwise_divergence": max_div,
        "mean_pairwise_divergence": mean_div,
        "max_raw_divergence": max_raw_div,
        "mean_raw_divergence": mean_raw_div,
        "n_constructs_available": [
            sum(1 for c in available if np.isfinite(ranked_preds[c][i]))
            for i in range(n_zctas)
        ],
    })

    # Add per-construct ranked predictions for downstream analysis
    for c in available:
        result[f"pred_rank_{c}"] = ranked_preds[c]
        result[f"target_rank_{c}"] = ranked_targets[c]

    return result.set_index("zcta_id")


# =========================================================================
# Correlation + summary
# =========================================================================

def compute_correlations(
    zcta_df: pd.DataFrame,
    svi_cols: list[str],
    hifld_cols: list[str],
) -> dict:
    """Spearman correlations between disagreement and SVI/HIFLD."""
    results = {"svi": {}, "hifld": {}, "c8_baseline": {}}
    div_col = "max_pairwise_divergence"
    raw_col = "max_raw_divergence"

    for col in svi_cols:
        if col not in zcta_df.columns:
            continue
        valid = zcta_df[[div_col, col]].dropna()
        if len(valid) < 10:
            continue
        rho, pval = stats.spearmanr(valid[div_col], valid[col])
        results["svi"][col] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": float(pval),
            "n": len(valid),
            "significant_005": bool(pval < 0.05),
        }

    for col in hifld_cols:
        if col not in zcta_df.columns:
            continue
        valid = zcta_df[[div_col, col]].dropna()
        if len(valid) < 10:
            continue
        rho, pval = stats.spearmanr(valid[div_col], valid[col])
        results["hifld"][col] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": float(pval),
            "n": len(valid),
            "significant_005": bool(pval < 0.05),
        }

    # C8 baseline: correlation between certificate divergence and raw divergence
    valid = zcta_df[[div_col, raw_col]].dropna()
    if len(valid) >= 10:
        rho, pval = stats.spearmanr(valid[div_col], valid[raw_col])
        results["c8_baseline"]["cert_vs_raw_divergence"] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": float(pval),
            "n": len(valid),
        }
        # Identify ZCTAs where certificate divergence reveals structure raw misses
        # (high cert divergence, low raw divergence)
        med_cert = valid[div_col].median()
        med_raw = valid[raw_col].median()
        high_cert_low_raw = (
            (valid[div_col] > med_cert) & (valid[raw_col] <= med_raw)
        ).sum()
        results["c8_baseline"]["n_high_cert_low_raw"] = int(high_cert_low_raw)
        results["c8_baseline"]["pct_high_cert_low_raw"] = round(
            100 * high_cert_low_raw / len(valid), 1
        )

    return results


# =========================================================================
# Single scenario runner
# =========================================================================

def run_scenario(s3, scenario: str, seed: int, upload: bool) -> dict:
    """Run C6 overlay for one scenario. Returns summary dict."""
    log.info("=" * 60)
    log.info("C6 SVI overlay: %s", scenario)
    log.info("=" * 60)

    # Load data (identical to DOE-C1)
    event_df = _load_event_features(s3, scenario)
    event_df = _merge_shared_layers(s3, event_df)
    folds_df = _load_folds(s3, scenario)

    if not folds_df.empty and "fold" not in event_df.columns:
        folds_df["zcta_id"] = folds_df["zcta_id"].astype(str)
        fold_map = folds_df[["zcta_id", "fold"]].drop_duplicates(subset="zcta_id")
        event_df = event_df.merge(fold_map, on="zcta_id", how="left")
    if "fold" not in event_df.columns:
        event_df["fold"] = event_df["zcta_id"].apply(lambda z: hash(z) % 5)

    feature_cols = _select_features(event_df)
    log.info("Selected %d features", len(feature_cols))

    features = (
        event_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    fold_ids = event_df["fold"].to_numpy()
    zcta_ids = event_df["zcta_id"].astype(str).to_numpy()

    # Fit-predict each construct and collect per-row predictions
    construct_predictions = {}
    construct_targets = {}

    for construct, target_col in CONSTRUCT_TARGET_COLUMNS.items():
        log.info("Fitting construct: %s (target: %s)", construct, target_col)

        if construct == "fast":
            target = _load_fast_target(s3, event_df, scenario).to_numpy(dtype=float)
        elif target_col in event_df.columns:
            target = event_df[target_col].to_numpy(dtype=float)
        else:
            log.warning("Target column %s not found, skipping %s", target_col, construct)
            continue

        n_finite = int(np.isfinite(target).sum())
        if n_finite < 30:
            log.warning("Construct %s: insufficient finite targets (%d), skipping",
                        construct, n_finite)
            continue

        pred = _fit_predict_construct(features, target, fold_ids, seed=seed)
        valid = np.isfinite(pred) & np.isfinite(target)
        if valid.sum() < 10:
            log.warning("Construct %s: insufficient valid predictions (%d), skipping",
                        construct, valid.sum())
            continue

        r2 = float(r2_score(target[valid], pred[valid]))
        log.info("  %s: R2=%.3f  n_valid=%d/%d", construct, r2, valid.sum(), len(target))

        construct_predictions[construct] = pred
        construct_targets[construct] = target

    if len(construct_predictions) < 2:
        log.error("Need at least 2 constructs for disagreement, got %d",
                  len(construct_predictions))
        return {"scenario": scenario, "status": "SKIP", "reason": "insufficient constructs"}

    # Compute per-ZCTA disagreement
    disagree_df = compute_per_zcta_disagreement(
        zcta_ids, construct_predictions, construct_targets,
    )
    log.info("Per-ZCTA disagreement: %d ZCTAs, median max_div=%.3f",
             len(disagree_df),
             disagree_df["max_pairwise_divergence"].median())

    # Join SVI + HIFLD (already in event_features, aggregate to ZCTA level)
    svi_hifld_cols = [c for c in SVI_COLUMNS + HIFLD_COLUMNS if c in event_df.columns]
    if svi_hifld_cols:
        zcta_svi = (
            event_df[["zcta_id"] + svi_hifld_cols]
            .groupby("zcta_id")[svi_hifld_cols]
            .mean()
        )
        disagree_df = disagree_df.join(zcta_svi, how="left")
        log.info("Joined %d SVI/HIFLD columns", len(svi_hifld_cols))
    else:
        log.warning("No SVI/HIFLD columns found in event_features")

    # Compute correlations
    svi_present = [c for c in SVI_COLUMNS if c in disagree_df.columns]
    hifld_present = [c for c in HIFLD_COLUMNS if c in disagree_df.columns]
    correlations = compute_correlations(disagree_df, svi_present, hifld_present)

    # Build summary
    summary = {
        "scenario": scenario,
        "status": "OK",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_zctas": len(disagree_df),
        "n_constructs": len(construct_predictions),
        "constructs_used": sorted(construct_predictions.keys()),
        "disagreement_stats": {
            "max_pairwise_divergence": {
                "mean": round(float(disagree_df["max_pairwise_divergence"].mean()), 4),
                "median": round(float(disagree_df["max_pairwise_divergence"].median()), 4),
                "std": round(float(disagree_df["max_pairwise_divergence"].std()), 4),
                "min": round(float(disagree_df["max_pairwise_divergence"].min()), 4),
                "max": round(float(disagree_df["max_pairwise_divergence"].max()), 4),
            },
        },
        "correlations": correlations,
        "seed": seed,
        "model": "HistGradientBoostingRegressor",
        "model_params": HGBDT_PARAMS,
    }

    # Print results
    print()
    print("=" * 72)
    print("C6 SVI-Disagreement Overlay: %s" % scenario)
    print("=" * 72)
    print("  Constructs: %s" % sorted(construct_predictions.keys()))
    print("  ZCTAs: %d" % len(disagree_df))
    print("  Median max pairwise divergence: %.3f" %
          disagree_df["max_pairwise_divergence"].median())
    print()
    print("  SVI correlations:")
    for col, vals in correlations.get("svi", {}).items():
        sig = "*" if vals["significant_005"] else ""
        print("    %-30s  rho=%.3f  p=%.4f  n=%d %s" % (
            col, vals["spearman_rho"], vals["p_value"], vals["n"], sig))
    print()
    print("  HIFLD correlations:")
    for col, vals in correlations.get("hifld", {}).items():
        sig = "*" if vals["significant_005"] else ""
        print("    %-30s  rho=%.3f  p=%.4f  n=%d %s" % (
            col, vals["spearman_rho"], vals["p_value"], vals["n"], sig))
    if "cert_vs_raw_divergence" in correlations.get("c8_baseline", {}):
        c8 = correlations["c8_baseline"]["cert_vs_raw_divergence"]
        print()
        print("  C8 baseline (cert vs raw divergence): rho=%.3f  p=%.4f" % (
            c8["spearman_rho"], c8["p_value"]))
        print("  ZCTAs with high cert / low raw divergence: %d (%.1f%%)" % (
            correlations["c8_baseline"]["n_high_cert_low_raw"],
            correlations["c8_baseline"]["pct_high_cert_low_raw"]))
    print("=" * 72)

    # Save local
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results" / "doe_c1"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_json = out_dir / f"c6_svi_overlay_{scenario}.json"
    with open(local_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Written local: %s", local_json)

    local_parquet = out_dir / f"c6_svi_overlay_{scenario}.parquet"
    disagree_df.reset_index().to_parquet(local_parquet, index=False)
    log.info("Written local: %s", local_parquet)

    # Upload to S3
    if upload:
        json_key = f"{RESULTS_PREFIX}/c6_svi_overlay_{scenario}.json"
        upload_json_result(s3, BUCKET, json_key, summary)
        log.info("Uploaded s3://%s/%s", BUCKET, json_key)

        pq_buf = io.BytesIO()
        disagree_df.reset_index().to_parquet(pq_buf, index=False)
        pq_buf.seek(0)
        pq_key = f"{RESULTS_PREFIX}/c6_svi_overlay_{scenario}.parquet"
        s3.put_object(Bucket=BUCKET, Key=pq_key, Body=pq_buf.getvalue())
        log.info("Uploaded s3://%s/%s", BUCKET, pq_key)

    return summary


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C6: SVI-disagreement overlay")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=SCENARIOS)
    group.add_argument("--all", action="store_true")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run:
        log.info("DRY RUN: C6 SVI overlay")
        log.info("Scenarios: %s", SCENARIOS if args.all else [args.scenario])
        log.info("Constructs: %s", CONSTRUCT_ORDER)
        log.info("SVI columns: %s", SVI_COLUMNS)
        log.info("HIFLD columns: %s", HIFLD_COLUMNS)
        return 0

    s3 = get_s3_client()
    scenarios = SCENARIOS if args.all else [args.scenario]
    all_summaries = []

    for scenario in scenarios:
        summary = run_scenario(s3, scenario, args.seed, args.upload)
        all_summaries.append(summary)

    # Cross-scenario summary
    if len(all_summaries) > 1:
        cross = {
            "artifact": "c6_svi_overlay",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_scenarios": len(all_summaries),
            "scenarios": [],
        }
        for s in all_summaries:
            entry = {
                "scenario": s["scenario"],
                "status": s.get("status", "SKIP"),
                "n_zctas": s.get("n_zctas", 0),
                "n_constructs": s.get("n_constructs", 0),
            }
            if "correlations" in s and "svi" in s["correlations"]:
                svi_overall = s["correlations"]["svi"].get("svi_overall", {})
                entry["svi_overall_rho"] = svi_overall.get("spearman_rho")
                entry["svi_overall_pval"] = svi_overall.get("p_value")
            cross["scenarios"].append(entry)

        out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results" / "doe_c1"
        local_cross = out_dir / "c6_svi_overlay_all.json"
        with open(local_cross, "w") as f:
            json.dump(cross, f, indent=2)
        log.info("Written local: %s", local_cross)

        if args.upload:
            cross_key = f"{RESULTS_PREFIX}/c6_svi_overlay_all.json"
            upload_json_result(s3, BUCKET, cross_key, cross)
            log.info("Uploaded s3://%s/%s", BUCKET, cross_key)

        print()
        print("=" * 72)
        print("C6 Cross-Scenario Summary")
        print("=" * 72)
        for entry in cross["scenarios"]:
            rho = entry.get("svi_overall_rho", "N/A")
            rho_str = "%.3f" % rho if isinstance(rho, float) else str(rho)
            print("  %-25s  n_zcta=%3d  n_const=%d  svi_rho=%s" % (
                entry["scenario"], entry["n_zctas"],
                entry["n_constructs"], rho_str))
        print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
