#!/usr/bin/env python3
"""
run_variance_stack.py -- 5-step variance-control decomposition (SageMaker).

Runs the full ladder from georsct.analysis.variance_stack:
  0. Unweighted (baseline)
  1. Coverage gate (positivity check)
  2. Marginal product (TWCV-lite)
  3. Marginal raking (IPF)
  4. Raking + shrinkage (regularized)

Fits inline HistGBDT per fold (same as primary training).
Produces per-cell ladder JSON + multi-panel figure.

Usage:
    python run_variance_stack.py --upload
    python run_variance_stack.py
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
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client, load_processed_parquet

from georsct.validation.task_descriptors import fit_quantile_edges, apply_bins
from georsct.analysis.variance_stack import decompose_stack

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELABLE = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]

DESCRIPTORS = [
    "flood_pct_zone_a",
    "twi_twi",
    "impervious_pct",
    "population",
    "elevation_m_msl",
    "slope_mean_pct",
]

R0_FEATURES = [
    "elevation_m_msl", "slope_mean_pct", "aspect_mean_deg",
    "flood_pct_zone_a", "flood_pct_zone_v", "flood_pct_zone_x",
    "population", "housing_units", "median_income", "median_year_built",
    "pct_pre_firm", "nfip_has_claims", "nfip_claim_count",
    "nfip_total_loss", "nfip_mean_loss_per_claim",
    "rainfall_total_mm", "peak_stage_ft", "peak_flow_cfs",
    "obs_gauge_count", "obs_gauge_distance_km",
]

TARGETS = [
    {"column": "obs_nfip_event_claims", "label": "NFIP claims", "task": "regression"},
    {"column": "obs_has_311", "label": "311 reports", "task": "classification"},
    {"column": "obs_has_hwm", "label": "HWM presence", "task": "classification"},
]

MIN_PREVALENCE = 0.03
MIN_FOLD_SIZE = 10
SIDECAR_PREFIX = "results/s035/sidecar/variance_stack"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(s3, scenario: str) -> pd.DataFrame:
    return load_processed_parquet(s3, scenario)


def load_folds(s3, scenario: str) -> pd.DataFrame:
    key = f"folds/{scenario}_folds.parquet"
    log.info("Loading s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(resp["Body"].read()))


# ---------------------------------------------------------------------------
# Per-fold model fitting
# ---------------------------------------------------------------------------

def fit_predict_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task: str,
) -> np.ndarray:
    """Fit HistGBDT on train, predict on val."""
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)

    if task == "classification":
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
    else:
        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

    return y_pred


# ---------------------------------------------------------------------------
# Per-cell processing
# ---------------------------------------------------------------------------

def process_cell(
    merged: pd.DataFrame,
    scenario: str,
    target: dict,
    descriptors: list[str],
    feature_cols: list[str],
) -> dict | None:
    """Run full 5-step ladder for one scenario-target cell."""
    target_col = target["column"]
    task = target["task"]
    label = f"{scenario} / {target['label']}"

    if target_col not in merged.columns:
        log.info("  %s: target column missing, skipping", label)
        return None

    valid = merged[target_col].notna()
    if valid.sum() < 20:
        log.info("  %s: too few non-null target values (%d), skipping", label, valid.sum())
        return None

    if task == "classification":
        prev = merged.loc[valid, target_col].mean()
        if prev < MIN_PREVALENCE or prev > (1 - MIN_PREVALENCE):
            log.info("  %s: prevalence=%.3f below threshold, skipping", label, prev)
            return None

    # Deployment domain = all descriptor-complete rows
    target_df = merged[descriptors].copy().dropna(subset=descriptors)
    if len(target_df) < 20:
        log.info("  %s: too few descriptor-complete rows (%d), skipping", label, len(target_df))
        return None

    edges = fit_quantile_edges(target_df, descriptors, n_bins=5)
    target_binned = apply_bins(target_df, edges)
    bin_cols = [f"{c}_bin" for c in descriptors if f"{c}_bin" in target_binned.columns]

    all_ladders = []
    folds = sorted(merged["fold_spatial_blocked"].unique())

    for fold_id in folds:
        val_mask = merged["fold_spatial_blocked"] == fold_id
        train_mask = ~val_mask

        train_df = merged[train_mask & merged[target_col].notna()].copy()
        val_df = merged[val_mask & merged[target_col].notna()].copy()
        val_df = val_df.dropna(subset=descriptors)

        if len(val_df) < MIN_FOLD_SIZE or len(train_df) < MIN_FOLD_SIZE:
            continue

        y_pred = fit_predict_fold(train_df, val_df, feature_cols, target_col, task)
        y_true = val_df[target_col].to_numpy(dtype=float)
        val_binned = apply_bins(val_df, edges)

        ladder = decompose_stack(
            val_binned, target_binned, bin_cols,
            y_true=y_true, y_pred=y_pred,
            identity={
                "scenario": scenario,
                "target": target_col,
                "fold": str(fold_id),
            },
        )
        all_ladders.append(ladder)

        log.info("  Fold %s (n=%d):", fold_id, len(val_df))
        for _, row in ladder.iterrows():
            log.info(
                "    %-25s  RMSE=%.3f  gap=%.3f  ESS=%.1f  frac=%.3f  mwxu=%.1f  %s",
                row["config"],
                row.get("rmse", float("nan")),
                row["residual_margin_gap"],
                row["ess"],
                row["ess_fraction"],
                row["max_weight_x_uniform"],
                row["stack_status"],
            )

    if not all_ladders:
        log.info("  %s: no valid folds, skipping", label)
        return None

    # Average across folds
    combined = pd.concat(all_ladders, ignore_index=True)
    numeric_cols = [
        "rmse", "mae", "residual_margin_gap", "ess", "ess_fraction",
        "max_weight_x_uniform", "max_uncovered_target_mass",
        "shrinkage_delta_ess", "shrinkage_delta_margin_gap",
    ]
    existing = [c for c in numeric_cols if c in combined.columns]
    avg = combined.groupby("config")[existing].mean().reset_index()

    status_mode = (
        combined.groupby("config")["stack_status"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    avg = avg.merge(status_mode, on="config")

    config_order = [
        "0_unweighted", "1_coverage_gate", "2_marginal_product",
        "3_marginal_raking", "4_raking_shrunk",
    ]
    avg["_order"] = avg["config"].map({c: i for i, c in enumerate(config_order)})
    avg = avg.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    avg["delta_rmse"] = avg["rmse"].diff()
    avg["delta_margin_gap"] = avg["residual_margin_gap"].diff()
    avg["delta_ess"] = avg["ess"].diff()

    log.info("\n  === %s AVERAGE ACROSS %d FOLDS ===", label, len(all_ladders))
    for _, row in avg.iterrows():
        log.info(
            "  %-25s  RMSE=%.3f  gap=%.3f  ESS=%.1f  dRMSE=%+.4f  dGap=%+.4f  %s",
            row["config"],
            row["rmse"],
            row["residual_margin_gap"],
            row["ess"],
            row.get("delta_rmse", float("nan")),
            row.get("delta_margin_gap", float("nan")),
            row["stack_status"],
        )

    return {
        "scenario": scenario,
        "target": target_col,
        "target_label": target["label"],
        "task": task,
        "n_folds": len(all_ladders),
        "n_folds_available": len(folds),
        "ladder": avg.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="5-step variance stack decomposition"
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()
    all_results = []

    for scenario in MODELABLE:
        log.info("\n=== %s ===", scenario.upper())

        try:
            features = load_features(s3, scenario)
            folds = load_folds(s3, scenario)
        except Exception as e:
            log.warning("Cannot load data for %s: %s", scenario, e)
            continue

        log.info("Loaded %d rows x %d columns", len(features), len(features.columns))

        merged = features.merge(
            folds[["zcta_id", "event", "fold_spatial_blocked"]],
            on=["zcta_id", "event"],
            how="inner",
        )
        log.info("Merged: %d rows", len(merged))

        # Available descriptors (>50% non-null)
        descriptors = [d for d in DESCRIPTORS
                       if d in merged.columns and merged[d].notna().mean() > 0.5]
        if len(descriptors) < 3:
            log.warning("%s: only %d descriptors, skipping", scenario, len(descriptors))
            continue
        log.info("Descriptors: %s", descriptors)

        feature_cols = [f for f in R0_FEATURES if f in merged.columns]
        log.info("Features: %d columns", len(feature_cols))

        for target in TARGETS:
            log.info("\nProcessing %s / %s...", scenario, target["label"])
            result = process_cell(merged, scenario, target, descriptors, feature_cols)
            if result:
                all_results.append(result)

    # Build output
    payload = {
        "phase": "variance_stack_decomposition",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "5-step variance-control ladder (Brenning & Suesse 2026)",
        "steps": [
            "0_unweighted: baseline (no weighting)",
            "1_coverage_gate: positivity gate (no weight change, status only)",
            "2_marginal_product: product of marginal ratios (TWCV-lite)",
            "3_marginal_raking: IPF raking (full TWCV)",
            "4_raking_shrunk: raking + shrinkage (lambda=0.2)",
        ],
        "n_cells": len(all_results),
        "cells": all_results,
    }

    output_json = json.dumps(payload, indent=2, default=str)

    if args.upload:
        json_key = f"{SIDECAR_PREFIX}/variance_stack_results.json"
        s3.put_object(
            Bucket=BUCKET, Key=json_key,
            Body=output_json.encode(),
            ContentType="application/json",
        )
        log.info("Uploaded s3://%s/%s", BUCKET, json_key)
    else:
        local = Path("/tmp/variance_stack_results.json")
        local.write_text(output_json)
        log.info("Wrote %s", local)

    # Summary
    print(f"\n{'='*60}")
    print(f"  VARIANCE STACK DECOMPOSITION -- {len(all_results)} cells")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n  {r['scenario']} / {r['target_label']} ({r['n_folds']} folds):")
        for step in r["ladder"]:
            print(
                f"    {step['config']:25s}  "
                f"RMSE={step.get('rmse', float('nan')):6.3f}  "
                f"gap={step['residual_margin_gap']:5.3f}  "
                f"ESS%={step['ess_fraction']:5.3f}  "
                f"{step['stack_status']}"
            )


if __name__ == "__main__":
    main()
