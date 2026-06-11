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
Produces per-cell ladder JSON with full outcome records.

Every attempted cell leaves a CellOutcome record -- no silent drops.
The readiness denominator is reconstructable from the JSON alone.

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
from cell_outcome import (
    CellOutcome, CellStatus, MAX_BINS, split_floor,
    audit_fold_sizes, readiness_summary,
)

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

# Gate thresholds -- derived from cell_outcome, not magic literals.
MIN_PREVALENCE = 0.03                # classification prevalence floor
FOLD_FLOOR = split_floor()           # 2 * min_samples_leaf = 40

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
    """Fit HistGBDT on train, predict on val.

    max_bins is fixed at MAX_BINS (32) for ALL cells. Never min(255, n_train)
    -- that makes binning resolution a function of cell size.
    """
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)

    if task == "classification":
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1,
            max_bins=MAX_BINS, random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
    else:
        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.1,
            max_bins=MAX_BINS, random_state=42,
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
) -> CellOutcome:
    """Run full 5-step ladder for one scenario-target cell.

    Always returns a CellOutcome. Never returns None.
    If a cell passes all gates and the trainer still crashes, that is a
    bug -- not a finding. The gate must catch all data-related failures.
    """
    target_col = target["column"]
    task = target["task"]
    label = f"{scenario} / {target['label']}"
    n_total = len(merged)

    # --- Gate 1: target column exists ---
    if target_col not in merged.columns:
        log.info("  %s: target column missing", label)
        return CellOutcome(scenario, target_col, CellStatus.ABSTAIN_ZERO_TARGETS,
                          reason=f"{target_col} not in columns",
                          n_total=n_total, n_nonnull_target=0)

    # --- Gate 2: sufficient non-null targets ---
    valid = merged[target_col].notna()
    n_nonnull = int(valid.sum())
    if n_nonnull == 0:
        log.info("  %s: zero non-null targets", label)
        return CellOutcome(scenario, target_col, CellStatus.ABSTAIN_ZERO_TARGETS,
                          n_total=n_total, n_nonnull_target=0)

    # --- Gate 3: prevalence (classification only) ---
    prevalence = None
    if task == "classification":
        prevalence = float(merged.loc[valid, target_col].mean())
        if prevalence < MIN_PREVALENCE or prevalence > (1.0 - MIN_PREVALENCE):
            log.info("  %s: prevalence=%.3f below floor", label, prevalence)
            return CellOutcome(scenario, target_col, CellStatus.ABSTAIN_LOW_PREVALENCE,
                              reason=f"prevalence={prevalence:.3f}",
                              n_total=n_total, n_nonnull_target=n_nonnull,
                              prevalence=prevalence)

    # --- Gate 4: constant target (regression) ---
    target_std = None
    if task == "regression":
        target_std = float(merged.loc[valid, target_col].std())
        if target_std == 0.0:
            log.info("  %s: constant target", label)
            return CellOutcome(scenario, target_col, CellStatus.ABSTAIN_CONSTANT_TARGET,
                              n_total=n_total, n_nonnull_target=n_nonnull,
                              target_std=0.0)

    # --- Gate 5: descriptor completeness ---
    target_df = merged[descriptors].copy().dropna(subset=descriptors)
    if len(target_df) < 20:
        log.info("  %s: too few descriptor-complete rows (%d)", label, len(target_df))
        return CellOutcome(scenario, target_col, CellStatus.ABSTAIN_INSUFFICIENT_N,
                          reason=f"descriptor-complete rows {len(target_df)} < 20",
                          n_total=n_total, n_nonnull_target=n_nonnull)

    # --- Gate 6: min fold train size (the real crash cause) ---
    # Binding constraint is the SMALLEST training fold after full NaN
    # filtering on target + features. This is where max_bins=min(255,n)
    # produced max_bins=1 and crashed.
    usable = merged[target_col].notna()
    for col in feature_cols:
        if col in merged.columns:
            usable = usable & merged[col].notna()

    folds = sorted(merged["fold_spatial_blocked"].unique())
    n_groups = len(folds)

    if n_groups < 2:
        log.info("  %s: only %d fold groups", label, n_groups)
        return CellOutcome(scenario, target_col, CellStatus.ABSTAIN_TOO_FEW_GROUPS,
                          n_total=n_total, n_nonnull_target=n_nonnull,
                          n_groups=n_groups)

    fold_train_sizes = []
    for fold_id in folds:
        train_n = int((usable & (merged["fold_spatial_blocked"] != fold_id)).sum())
        fold_train_sizes.append(train_n)

    min_fold_train = min(fold_train_sizes)
    log.info("  %s: %d folds, min train = %d (floor = %d)",
             label, n_groups, min_fold_train, FOLD_FLOOR)

    if min_fold_train < FOLD_FLOOR:
        log.info("  %s: min fold train %d < floor %d", label, min_fold_train, FOLD_FLOOR)
        return CellOutcome(scenario, target_col, CellStatus.ABSTAIN_INSUFFICIENT_N,
                          reason=f"min fold train {min_fold_train} < floor {FOLD_FLOOR}",
                          n_total=n_total, n_nonnull_target=n_nonnull,
                          n_groups=n_groups, min_fold_train_n=min_fold_train)

    # --- Passed all gates. Fit the ladder. ---
    edges = fit_quantile_edges(target_df, descriptors, n_bins=5)
    target_binned = apply_bins(target_df, edges)
    bin_cols = [f"{c}_bin" for c in descriptors if f"{c}_bin" in target_binned.columns]

    all_ladders = []

    for fold_id in folds:
        val_mask = merged["fold_spatial_blocked"] == fold_id
        train_mask = ~val_mask

        train_df = merged[train_mask & merged[target_col].notna()].copy()
        train_df = train_df.dropna(subset=feature_cols)
        val_df = merged[val_mask & merged[target_col].notna()].copy()
        val_df = val_df.dropna(subset=descriptors)

        if len(val_df) == 0 or len(train_df) == 0:
            log.warning("  Fold %s: empty after NaN filter (train=%d, val=%d)",
                       fold_id, len(train_df), len(val_df))
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
        log.info("  %s: all folds empty after NaN filter", label)
        return CellOutcome(scenario, target_col, CellStatus.ABSTAIN_INSUFFICIENT_N,
                          reason=f"0/{n_groups} folds produced results",
                          n_total=n_total, n_nonnull_target=n_nonnull,
                          n_groups=n_groups, min_fold_train_n=min_fold_train)

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

    return CellOutcome(
        scenario, target_col, CellStatus.RAN,
        n_total=n_total, n_nonnull_target=n_nonnull,
        n_groups=n_groups, min_fold_train_n=min_fold_train,
        prevalence=prevalence, target_std=target_std,
        metrics={
            "target_label": target["label"],
            "task": task,
            "n_folds": len(all_ladders),
            "n_folds_available": len(folds),
            "ladder": avg.to_dict(orient="records"),
        },
    )


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
    all_outcomes: list[CellOutcome] = []

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
            # No try/except: if the gate passes and the trainer crashes,
            # that is a bug to investigate, not a finding to report.
            outcome = process_cell(merged, scenario, target, descriptors, feature_cols)
            all_outcomes.append(outcome)

    # Post-hoc audit: flag RAN cells whose smallest fold < split_floor
    flagged = audit_fold_sizes(all_outcomes)
    if flagged:
        log.warning("STUMP RISK: %d cells with min_fold_train_n < %d",
                    len(flagged), FOLD_FLOOR)
        for f in flagged:
            log.warning("  %s/%s: min_fold=%d", f.scenario, f.target, f.min_fold_train_n)

    summary = readiness_summary(all_outcomes)
    ran_valid = [o for o in all_outcomes
                 if o.status is CellStatus.RAN and not o.stump_risk]

    # Build output -- backward-compatible cells + full audit trail
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
        "gate": {
            "max_bins": MAX_BINS,
            "fold_floor": FOLD_FLOOR,
            "min_prevalence": MIN_PREVALENCE,
            "min_samples_leaf": 20,
        },
        "readiness": summary,
        "n_cells": len(ran_valid),
        "cells": [
            {"scenario": o.scenario, "target": o.target, **o.metrics}
            for o in ran_valid
        ],
        "all_outcomes": [o.to_record() for o in all_outcomes],
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
    print(f"  VARIANCE STACK -- {summary['attempted']} attempted, "
          f"{summary['ran_valid']} valid, {summary['abstained']} abstained")
    print(f"{'='*60}")

    for o in all_outcomes:
        if o.abstained:
            print(f"\n  {o.scenario} / {o.target}: {o.status.value} -- {o.reason}")
        elif o.stump_risk:
            print(f"\n  {o.scenario} / {o.target}: STUMP_RISK "
                  f"(min_fold={o.min_fold_train_n} < {FOLD_FLOOR})")
        else:
            m = o.metrics
            print(f"\n  {o.scenario} / {m['target_label']} ({m['n_folds']} folds):")
            for step in m["ladder"]:
                print(
                    f"    {step['config']:25s}  "
                    f"RMSE={step.get('rmse', float('nan')):6.3f}  "
                    f"gap={step['residual_margin_gap']:5.3f}  "
                    f"ESS%={step['ess_fraction']:5.3f}  "
                    f"{step['stack_status']}"
                )

    log.info("\nReadiness: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
