"""
compute_deployment_alignment.py -- Deployment-aligned validation (TWCV).

Phase 4e sidecar: reweights per-fold validation losses to match the
deployment task distribution, following Brenning & Suesse (2026).

Does NOT alter locked folds, solvers, features, or primary inference.
Writes only to results/s035/sidecar/deployment_alignment/.

Usage:
    python compute_deployment_alignment.py --scenario houston --upload
    python compute_deployment_alignment.py --all-scenarios --upload
    python compute_deployment_alignment.py --all-scenarios --all-levels --upload
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    BUCKET,
    get_s3_client,
    load_crosswalk,
    load_processed_parquet,
    level_prefix,
)

# Import TWCV library from georsct
try:
    from rsct.validation import (
        AlignmentGates,
        RegimeDomain,
        TaskDescriptorConfig,
        add_nearest_training_distance,
        alignment_summary,
        apply_bins,
        build_deployment_target_descriptors,
        compute_iwcv_weights,
        fit_quantile_edges,
        marginal_ratio_weights,
        nearest_distance_to_reference,
        rake_weights,
    )

    RSCT_VALIDATION_AVAILABLE = True
except ImportError:
    RSCT_VALIDATION_AVAILABLE = False
    log.warning("rsct.validation not available -- install georsct package")

# -----------------------------------------------------------------------
# Configuration (pre-registered in DOE_deployment_alignment.md)
# -----------------------------------------------------------------------

MODELABLE = [
    "houston",
    "southwest_florida",
    "nyc",
    "riverside_coachella",
    "new_orleans",
]

LEVELS = ["r0", "r1", "r2"]

# Task descriptors: spatial difficulty + 4 environmental axes
DESCRIPTOR_COLS = [
    "flood_pct_zone_a",
    "twi_twi",
    "impervious_pct",
    "population",
]

# Columns needed from processed parquet
FEATURE_COLS = DESCRIPTOR_COLS + ["latitude", "longitude"]

# TWCV configuration (frozen per DOE)
N_BINS = 5
SHRINKAGE = 0.20
CLIP = 10.0
IWCV_C = 1.0

# Pluvial regime: all ZCTAs eligible (rainfall risk is universal)
PLUVIAL_DOMAIN = RegimeDomain(
    regime_id="pluvial_full",
    name="Pluvial (general flood)",
    hazard_mechanism="rainfall",
    label_source="NFIP claims / 311 / HWM",
    full_universe=True,
)

SIDECAR_PREFIX = "results/s035/sidecar/deployment_alignment"

# Task descriptor config adapted for floodrsct column names
CFG = TaskDescriptorConfig(
    id_cols=("zcta_id", "event"),
    fold_col="fold_id",
    lat_col="latitude",
    lon_col="longitude",
    n_bins=N_BINS,
    distance_col="nearest_train_km",
)

GATES = AlignmentGates()


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_folds(s3, scenario: str) -> pd.DataFrame:
    """Load fold assignments for a scenario."""
    key = f"folds/{scenario}_folds.parquet"
    log.info("Loading s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    return pd.read_parquet(buf)


def load_predictions(s3, scenario: str, level: str) -> pd.DataFrame:
    """Load per-row predictions for a (scenario, level)."""
    prefix = level_prefix(level)
    key = f"results/s035/{prefix}_{scenario}_predictions.parquet"
    log.info("Loading s3://%s/%s", BUCKET, key)
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        buf = io.BytesIO(resp["Body"].read())
        return pd.read_parquet(buf)
    except Exception as e:
        log.warning("Could not load predictions for %s/%s: %s", scenario, level, e)
        return pd.DataFrame()


def load_results_json(s3, scenario: str, level: str) -> dict:
    """Load the results JSON for a (scenario, level) to get aggregate metrics."""
    prefix = level_prefix(level)
    key = f"results/s035/{prefix}_{scenario}.json"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception as e:
        log.warning("Could not load results for %s/%s: %s", scenario, level, e)
        return {}


# -----------------------------------------------------------------------
# Deployment universe
# -----------------------------------------------------------------------

def build_deployment_universe(
    features_df: pd.DataFrame,
    crosswalk: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """Build the deployment universe: all ZCTAs in the metro's county set.

    The deployment domain is defined by administrative boundaries (counties),
    NOT by where we have observations. This is the set of ZCTAs for which
    predictions will be issued.
    """
    # Get the county set for this scenario from observed ZCTAs
    metro_zctas = features_df["zcta_id"].unique()
    metro_counties = crosswalk[
        crosswalk["zcta_id"].isin(metro_zctas)
    ]["county_fips"].unique()

    # All ZCTAs in those counties = deployment universe
    universe_zctas = crosswalk[
        crosswalk["county_fips"].isin(metro_counties)
    ]["zcta_id"].unique()

    log.info(
        "%s: %d observed ZCTAs, %d counties, %d deployment ZCTAs",
        scenario, len(metro_zctas), len(metro_counties), len(universe_zctas),
    )

    # Build universe DataFrame with available features
    # For ZCTAs we have features for, use them; for others, use crosswalk only
    universe = pd.DataFrame({"zcta_id": universe_zctas})

    # Merge features where available (observed ZCTAs)
    feat_cols = ["zcta_id"] + [c for c in FEATURE_COLS if c in features_df.columns]
    observed = features_df[feat_cols].drop_duplicates(subset=["zcta_id"])
    universe = universe.merge(observed, on="zcta_id", how="left")

    # For deployment ZCTAs without features, mark as NaN (will get SENTINEL_BIN)
    n_missing = universe[DESCRIPTOR_COLS[0]].isna().sum() if DESCRIPTOR_COLS[0] in universe.columns else 0
    if n_missing > 0:
        log.info(
            "%s: %d deployment ZCTAs lack feature data (will get SENTINEL_BIN)",
            scenario, n_missing,
        )

    return universe


# -----------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------

def compute_alignment_for_cell(
    features_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    scenario: str,
    level: str,
    target: str,
) -> dict:
    """Compute deployment alignment for one (scenario, level, target) cell.

    Returns a dict with alignment summary, weights, and reweighted metrics.
    """
    # 1. Build validation task descriptors (features + folds + nearest distance)
    avail_descriptors = [c for c in DESCRIPTOR_COLS if c in features_df.columns]
    if not avail_descriptors:
        log.warning("%s/%s: no descriptor columns available, skipping", scenario, level)
        return {"scenario": scenario, "level": level, "target": target, "status": "SKIP_NO_DESCRIPTORS"}

    # Merge features with folds
    merge_cols = list(CFG.id_cols)
    task_df = features_df.merge(
        folds_df[merge_cols + [CFG.fold_col]].drop_duplicates(),
        on=merge_cols,
        how="inner",
    )

    if task_df.empty:
        log.warning("%s/%s: empty after fold merge, skipping", scenario, level)
        return {"scenario": scenario, "level": level, "target": target, "status": "SKIP_EMPTY"}

    # 2. Add nearest-training distance per fold
    task_df = add_nearest_training_distance(task_df, cfg=CFG)

    # 3. Build deployment target descriptors
    reference_df = features_df[
        ["zcta_id", "latitude", "longitude"]
    ].drop_duplicates(subset=["zcta_id"])

    deploy_df = universe_df.copy()
    deploy_df[CFG.distance_col] = nearest_distance_to_reference(
        deploy_df, reference_df, cfg=CFG,
    )

    # 4. Fit bin edges from deployment distribution
    bin_descriptors = avail_descriptors + [CFG.distance_col]
    edges = fit_quantile_edges(deploy_df, bin_descriptors, n_bins=N_BINS)

    # 5. Apply bins to both validation and deployment
    bin_cols = [f"{c}_bin" for c in bin_descriptors]
    task_binned = apply_bins(task_df, edges)
    deploy_binned = apply_bins(deploy_df, edges)

    # 6. Compute weights (all 3 strategies)
    w_twcv, converged, iterations = rake_weights(
        task_binned, deploy_binned, bin_cols,
        clip=CLIP, shrinkage=SHRINKAGE,
    )
    w_lite = marginal_ratio_weights(
        task_binned, deploy_binned, bin_cols,
        clip=CLIP, shrinkage=SHRINKAGE,
    )

    iwcv_vars = [c for c in avail_descriptors + [CFG.distance_col]
                 if c in task_df.columns and c in deploy_df.columns]
    try:
        w_iwcv = compute_iwcv_weights(
            task_df, deploy_df, iwcv_vars,
            shrinkage=0.0, C=IWCV_C,
        )
    except (ValueError, Exception) as e:
        log.warning("%s/%s: IWCV failed: %s", scenario, level, e)
        w_iwcv = np.ones(len(task_df)) / max(len(task_df), 1)

    # 7. Certificate summary
    support_col = f"{CFG.distance_col}_bin"
    summary = alignment_summary(
        task_binned, deploy_binned, bin_cols,
        weights=w_twcv,
        gates=GATES,
        support_col=support_col,
    )

    # 8. Reweight metrics if predictions are available
    metric_results = _compute_reweighted_metrics(
        task_df, predictions_df, w_twcv, w_lite, w_iwcv, target,
    )

    return {
        "scenario": scenario,
        "level": level,
        "target": target,
        "status": "COMPUTED",
        "raking_converged": converged,
        "raking_iterations": iterations,
        **summary,
        **metric_results,
        "n_descriptors": len(avail_descriptors),
        "descriptor_cols": avail_descriptors,
        "n_deployment_universe": len(universe_df),
    }


def _compute_reweighted_metrics(
    task_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    w_twcv: np.ndarray,
    w_lite: np.ndarray,
    w_iwcv: np.ndarray,
    target: str,
) -> dict:
    """Compute unweighted and TWCV-reweighted RMSE/MAE from predictions."""
    if predictions_df.empty:
        return {
            "metric_unweighted": float("nan"),
            "metric_twcv": float("nan"),
            "metric_iwcv": float("nan"),
            "delta_twcv": float("nan"),
            "delta_pct": float("nan"),
        }

    # Merge predictions onto task_df
    merge_on = [c for c in ["zcta_id", "event"] if c in predictions_df.columns and c in task_df.columns]
    if not merge_on:
        return {
            "metric_unweighted": float("nan"),
            "metric_twcv": float("nan"),
            "metric_iwcv": float("nan"),
            "delta_twcv": float("nan"),
            "delta_pct": float("nan"),
        }

    # Find y_true and y_pred columns
    y_true_col = None
    y_pred_col = None
    for col in predictions_df.columns:
        if "true" in col.lower() or col == "y_true" or col == target:
            y_true_col = col
        if "pred" in col.lower() or col == "y_pred":
            y_pred_col = col

    if y_true_col is None or y_pred_col is None:
        log.warning("Cannot find y_true/y_pred columns in predictions")
        return {
            "metric_unweighted": float("nan"),
            "metric_twcv": float("nan"),
            "metric_iwcv": float("nan"),
            "delta_twcv": float("nan"),
            "delta_pct": float("nan"),
        }

    merged = task_df.merge(
        predictions_df[merge_on + [y_true_col, y_pred_col]].drop_duplicates(),
        on=merge_on,
        how="inner",
    )

    if merged.empty or len(merged) != len(w_twcv):
        log.warning(
            "Prediction merge mismatch: %d task rows, %d merged, %d weights",
            len(task_df), len(merged), len(w_twcv),
        )
        return {
            "metric_unweighted": float("nan"),
            "metric_twcv": float("nan"),
            "metric_iwcv": float("nan"),
            "delta_twcv": float("nan"),
            "delta_pct": float("nan"),
        }

    residuals = merged[y_true_col].to_numpy(float) - merged[y_pred_col].to_numpy(float)
    sq_errors = residuals ** 2

    # Unweighted RMSE
    rmse_unw = float(np.sqrt(np.mean(sq_errors)))

    # TWCV-weighted RMSE
    w = w_twcv / w_twcv.sum() if w_twcv.sum() > 0 else np.ones(len(sq_errors)) / len(sq_errors)
    rmse_twcv = float(np.sqrt(np.sum(w * sq_errors)))

    # IWCV-weighted RMSE
    wi = w_iwcv / w_iwcv.sum() if w_iwcv.sum() > 0 else np.ones(len(sq_errors)) / len(sq_errors)
    rmse_iwcv = float(np.sqrt(np.sum(wi * sq_errors)))

    delta = rmse_twcv - rmse_unw
    delta_pct = (delta / rmse_unw * 100) if rmse_unw > 0 else float("nan")

    return {
        "metric_unweighted": rmse_unw,
        "metric_twcv": rmse_twcv,
        "metric_twcv_lite": float(np.sqrt(np.sum(
            (w_lite / w_lite.sum() if w_lite.sum() > 0 else np.ones(len(sq_errors)) / len(sq_errors))
            * sq_errors
        ))),
        "metric_iwcv": rmse_iwcv,
        "delta_twcv": delta,
        "delta_pct": delta_pct,
    }


# -----------------------------------------------------------------------
# Upload
# -----------------------------------------------------------------------

def upload_json(s3, key: str, data: dict) -> None:
    """Upload JSON to S3."""
    body = json.dumps(data, indent=2, default=str).encode()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/json")
    log.info("Uploaded s3://%s/%s", BUCKET, key)


def upload_parquet(s3, key: str, df: pd.DataFrame) -> None:
    """Upload parquet to S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    log.info("Uploaded s3://%s/%s (%d rows)", BUCKET, key, len(df))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run_scenario(s3, scenario: str, levels: list[str], upload: bool) -> list[dict]:
    """Run deployment alignment for one scenario across levels."""
    log.info("=== %s ===", scenario.upper())

    # Load data
    features_df = load_processed_parquet(s3, scenario)
    crosswalk = load_crosswalk(s3)

    try:
        folds_df = load_folds(s3, scenario)
    except Exception as e:
        log.error("Cannot load folds for %s: %s", scenario, e)
        return [{"scenario": scenario, "status": "SKIP_NO_FOLDS", "error": str(e)}]

    # Ensure column names match CFG
    if "zcta" in features_df.columns and "zcta_id" not in features_df.columns:
        features_df = features_df.rename(columns={"zcta": "zcta_id"})
    if "zcta" in folds_df.columns and "zcta_id" not in folds_df.columns:
        folds_df = folds_df.rename(columns={"zcta": "zcta_id"})

    # Build deployment universe (all ZCTAs in metro counties)
    universe_df = build_deployment_universe(features_df, crosswalk, scenario)

    # Detect targets from results
    targets = ["obs_nfip_event_claims"]  # primary target

    results = []
    for level in levels:
        predictions_df = load_predictions(s3, scenario, level)

        for target in targets:
            log.info("--- %s / %s / %s ---", scenario, level, target)
            cell_result = compute_alignment_for_cell(
                features_df=features_df,
                folds_df=folds_df,
                predictions_df=predictions_df,
                universe_df=universe_df,
                scenario=scenario,
                level=level,
                target=target,
            )
            results.append(cell_result)

            decision = cell_result.get("alignment_decision", "N/A")
            ess_frac = cell_result.get("ess_fraction", float("nan"))
            delta = cell_result.get("delta_pct", float("nan"))
            log.info(
                "  decision=%s  ESS_frac=%.3f  delta_pct=%.1f%%",
                decision, ess_frac, delta,
            )

    if upload:
        # Upload per-scenario summary
        key = f"{SIDECAR_PREFIX}/{scenario}_alignment.json"
        upload_json(s3, key, {
            "scenario": scenario,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gates_tag": GATES.registration_tag,
            "descriptor_cols": DESCRIPTOR_COLS,
            "n_bins": N_BINS,
            "shrinkage": SHRINKAGE,
            "clip": CLIP,
            "cells": results,
        })

    return results


def run_uplift_comparison(all_results: list[dict], s3=None, upload: bool = False) -> dict:
    """Compare unweighted vs TWCV uplift direction across levels."""
    comparisons = []

    # Group by (scenario, target)
    by_cell = {}
    for r in all_results:
        if r.get("status") != "COMPUTED":
            continue
        key = (r["scenario"], r["target"])
        by_cell.setdefault(key, {})[r["level"]] = r

    for (scenario, target), levels_dict in by_cell.items():
        r0 = levels_dict.get("r0", {})
        r1 = levels_dict.get("r1", {})

        if not r0 or not r1:
            continue

        uplift_unw = (r0.get("metric_unweighted", float("nan"))
                      - r1.get("metric_unweighted", float("nan")))
        uplift_twcv = (r0.get("metric_twcv", float("nan"))
                       - r1.get("metric_twcv", float("nan")))

        # Positive uplift = R1 improved (lower RMSE)
        direction_unw = "improved" if uplift_unw > 0 else "degraded"
        direction_twcv = "improved" if uplift_twcv > 0 else "degraded"

        comparisons.append({
            "scenario": scenario,
            "target": target,
            "uplift_R0_R1_unweighted": float(uplift_unw),
            "uplift_R0_R1_twcv": float(uplift_twcv),
            "direction_unweighted": direction_unw,
            "direction_twcv": direction_twcv,
            "direction_agrees": direction_unw == direction_twcv,
            "delta_magnitude": abs(float(uplift_twcv) - float(uplift_unw)),
        })

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comparisons": comparisons,
        "all_directions_agree": all(c["direction_agrees"] for c in comparisons) if comparisons else False,
        "n_cells": len(comparisons),
    }

    if upload and s3:
        key = f"{SIDECAR_PREFIX}/uplift_comparison.json"
        upload_json(s3, key, result)

    return result


def main():
    parser = argparse.ArgumentParser(description="Deployment-aligned validation (TWCV)")
    parser.add_argument("--scenario", choices=MODELABLE, help="Single scenario")
    parser.add_argument("--all-scenarios", action="store_true", help="Run all 5 metros")
    parser.add_argument("--level", choices=LEVELS, default=None, help="Single level")
    parser.add_argument("--all-levels", action="store_true", help="Run all levels (r0, r1, r2)")
    parser.add_argument("--upload", action="store_true", help="Upload results to S3")
    args = parser.parse_args()

    if not RSCT_VALIDATION_AVAILABLE:
        log.error("rsct.validation not available. Install georsct: pip install -e ~/github/georsct")
        sys.exit(1)

    if not args.scenario and not args.all_scenarios:
        parser.error("Specify --scenario or --all-scenarios")

    scenarios = MODELABLE if args.all_scenarios else [args.scenario]
    levels = LEVELS if args.all_levels else ([args.level] if args.level else ["r0", "r1"])

    s3 = get_s3_client()
    all_results = []

    for scenario in scenarios:
        results = run_scenario(s3, scenario, levels, args.upload)
        all_results.extend(results)

    # Uplift comparison (only meaningful with r0 + r1)
    if len(levels) >= 2:
        comparison = run_uplift_comparison(all_results, s3=s3, upload=args.upload)
        log.info("=== UPLIFT COMPARISON ===")
        for c in comparison.get("comparisons", []):
            log.info(
                "  %s: unweighted=%s, twcv=%s, agrees=%s",
                c["scenario"], c["direction_unweighted"],
                c["direction_twcv"], c["direction_agrees"],
            )
        if comparison.get("all_directions_agree"):
            log.info("ALL directions agree -- uplift is deployment-representative")
        else:
            log.info("DIRECTION DISAGREEMENT -- uplift may be sample-biased")

    # Summary
    computed = [r for r in all_results if r.get("status") == "COMPUTED"]
    n_pass = sum(1 for r in computed if r.get("alignment_decision") == "PASS")
    n_warn = sum(1 for r in computed if r.get("alignment_decision") == "WARN")
    n_fail = sum(1 for r in computed if r.get("alignment_decision") == "FAIL")
    log.info(
        "=== SUMMARY: %d cells computed, %d PASS, %d WARN, %d FAIL ===",
        len(computed), n_pass, n_warn, n_fail,
    )


if __name__ == "__main__":
    main()
