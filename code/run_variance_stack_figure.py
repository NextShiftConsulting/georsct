#!/usr/bin/env python3
"""
Run the three-layer variance-control decomposition across all available
scenario-target cells and produce the paper figure.

Self-contained: fits inline HistGBDT per fold using features + folds
parquets only. No dependency on SageMaker R0 prediction outputs.

The key: deployment domain != validation evidence domain.
  - Deployment: ALL hazard-eligible ZCTAs (where decisions are needed).
  - Evidence: per spatial-block fold, only the held-out ZCTAs.

Spatial blocking creates systematic descriptor gaps. The ladder shows
how much each variance-control layer corrects for that mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georsct.validation.task_descriptors import fit_quantile_edges, apply_bins
from georsct.analysis.variance_stack import decompose_stack
from georsct.analysis.render_ladder import render_ladder_panel

DATA = Path("/tmp/vstack_data")
OUT = Path("/tmp/vstack_data")

DESCRIPTORS = [
    "elevation_m_msl",
    "slope_mean_pct",
    "flood_pct_zone_a",
    "population",
    "rainfall_total_mm",
]

# R0 tabular features (same as train_r0_baseline.py)
R0_FEATURES = [
    "elevation_m_msl",
    "slope_mean_pct",
    "aspect_mean_deg",
    "flood_pct_zone_a",
    "flood_pct_zone_v",
    "flood_pct_zone_x",
    "population",
    "housing_units",
    "median_income",
    "median_year_built",
    "pct_pre_firm",
    "nfip_has_claims",
    "nfip_claim_count",
    "nfip_total_loss",
    "nfip_mean_loss_per_claim",
    "rainfall_total_mm",
    "peak_stage_ft",
    "peak_flow_cfs",
    "obs_gauge_count",
    "obs_gauge_distance_km",
]

MIN_PREVALENCE = 0.03  # skip targets with <3% positive class
MIN_FOLD_SIZE = 10

# Scenario data registry: features, folds, available descriptors
SCENARIOS = [
    {
        "name": "Houston",
        "features": DATA / "houston_features.parquet",
        "folds": DATA / "houston_folds.parquet",
    },
    {
        "name": "New Orleans",
        "features": DATA / "nola_features.parquet",
        "folds": DATA / "nola_folds.parquet",
    },
    {
        "name": "NYC",
        "features": DATA / "nyc_features.parquet",
        "folds": DATA / "nyc_folds.parquet",
    },
    {
        "name": "Riverside",
        "features": DATA / "rc_features.parquet",
        "folds": DATA / "rc_folds.parquet",
    },
    {
        "name": "SW Florida",
        "features": DATA / "swfl_features.parquet",
        "folds": DATA / "swfl_folds.parquet",
    },
]

TARGETS = [
    {"column": "obs_nfip_event_claims", "label": "NFIP claims", "task": "regression"},
    {"column": "obs_has_311", "label": "311 reports", "task": "classification"},
    {"column": "obs_has_hwm", "label": "HWM presence", "task": "classification"},
]


def _available_descriptors(df: pd.DataFrame) -> list[str]:
    """Return descriptors that have <50% null in the features."""
    avail = []
    for d in DESCRIPTORS:
        if d in df.columns and df[d].notna().mean() > 0.5:
            avail.append(d)
    return avail


def _available_features(df: pd.DataFrame) -> list[str]:
    """Return R0 features present in the dataframe."""
    return [f for f in R0_FEATURES if f in df.columns]


def _fit_predict_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    task: str,
) -> np.ndarray:
    """Fit HistGBDT on train, predict on val. Returns y_pred array."""
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)

    n_bins = min(255, len(X_train))
    if task == "classification":
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1,
            max_bins=n_bins, random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
    else:
        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, learning_rate=0.1,
            max_bins=n_bins, random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

    return y_pred


def process_cell(
    scenario: dict,
    target: dict,
    features: pd.DataFrame,
    merged: pd.DataFrame,
    descriptors: list[str],
) -> pd.DataFrame | None:
    """Run variance stack for one scenario-target cell.

    Fits inline predictions per fold. Returns averaged ladder or None.
    """
    target_col = target["column"]
    task = target["task"]
    label = f"{scenario['name']} / {target['label']}"

    # Check target availability and prevalence
    if target_col not in merged.columns:
        print(f"  {label}: target column missing, skipping")
        return None

    valid = merged[target_col].notna()
    if valid.sum() < 20:
        print(f"  {label}: too few non-null target values ({valid.sum()}), skipping")
        return None

    if task == "classification":
        prev = merged.loc[valid, target_col].mean()
        if prev < MIN_PREVALENCE or prev > (1 - MIN_PREVALENCE):
            print(f"  {label}: prevalence={prev:.3f} below threshold, skipping")
            return None

    # Deployment domain descriptors
    target_df = merged[descriptors].copy().dropna(subset=descriptors)
    if len(target_df) < 20:
        print(f"  {label}: too few descriptor-complete rows ({len(target_df)}), skipping")
        return None
    edges = fit_quantile_edges(target_df, descriptors, n_bins=5)
    target_binned = apply_bins(target_df, edges)
    bin_cols = [f"{c}_bin" for c in descriptors if f"{c}_bin" in target_binned.columns]

    feature_cols = _available_features(merged)
    all_ladders = []

    for fold_id in sorted(merged["fold_spatial_blocked"].unique()):
        val_mask = merged["fold_spatial_blocked"] == fold_id
        train_mask = ~val_mask

        train_df = merged[train_mask & merged[target_col].notna()].copy()
        train_df = train_df.dropna(subset=feature_cols)
        val_df = merged[val_mask & merged[target_col].notna()].copy()
        val_df = val_df.dropna(subset=descriptors)

        if len(val_df) < MIN_FOLD_SIZE or len(train_df) < MIN_FOLD_SIZE:
            continue

        # Fit inline prediction
        y_pred = _fit_predict_fold(train_df, val_df, feature_cols, target_col, task)
        y_true = val_df[target_col].to_numpy(dtype=float)

        val_binned = apply_bins(val_df, edges)

        ladder = decompose_stack(
            val_binned, target_binned, bin_cols,
            y_true=y_true, y_pred=y_pred,
            identity={
                "scenario": label,
                "target": target_col,
                "fold": str(fold_id),
            },
        )
        all_ladders.append(ladder)

        print(f"  Fold {fold_id} (n={len(val_df)}):")
        for _, row in ladder.iterrows():
            print(
                f"    {row['config']:25s}  "
                f"RMSE={row.get('rmse', float('nan')):6.3f}  "
                f"gap={row['residual_margin_gap']:5.3f}  "
                f"ESS={row['ess']:6.1f}  "
                f"frac={row['ess_fraction']:5.3f}  "
                f"mwxu={row['max_weight_x_uniform']:5.1f}  "
                f"{row['stack_status']}"
            )

    if not all_ladders:
        print(f"  {label}: no valid folds, skipping")
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

    return avg


def run_all() -> dict[str, pd.DataFrame]:
    """Run decompose_stack for every viable scenario-target cell."""
    results = {}

    for scenario in SCENARIOS:
        if not scenario["features"].exists():
            print(f"\n{scenario['name']}: features file missing, skipping")
            continue
        if not scenario["folds"].exists():
            print(f"\n{scenario['name']}: folds file missing, skipping")
            continue

        features = pd.read_parquet(scenario["features"])
        folds = pd.read_parquet(scenario["folds"])

        merged = features.merge(
            folds[["zcta_id", "event", "fold_spatial_blocked"]],
            on=["zcta_id", "event"],
            how="inner",
        )

        descriptors = _available_descriptors(merged)
        if len(descriptors) < 3:
            print(f"\n{scenario['name']}: only {len(descriptors)} descriptors, skipping")
            continue

        dropped = [d for d in DESCRIPTORS if d not in descriptors]
        if dropped:
            print(f"\n{scenario['name']}: dropping unavailable descriptors: {dropped}")

        for target in TARGETS:
            cell_label = f"{scenario['name']}\n{target['label']}"
            print(f"\nProcessing {scenario['name']} / {target['label']}...")

            avg = process_cell(scenario, target, features, merged, descriptors)
            if avg is None:
                continue

            results[cell_label] = avg
            label = f"{scenario['name']} / {target['label']}"
            print(f"\n  === {label} AVERAGE ACROSS FOLDS ===")
            for _, row in avg.iterrows():
                drmse = row.get("delta_rmse", float("nan"))
                dgap = row.get("delta_margin_gap", float("nan"))
                print(
                    f"  {row['config']:25s}  "
                    f"RMSE={row['rmse']:6.3f}  "
                    f"gap={row['residual_margin_gap']:5.3f}  "
                    f"ESS={row['ess']:6.1f}  "
                    f"frac={row['ess_fraction']:5.3f}  "
                    f"dRMSE={drmse:+7.4f}  "
                    f"dGap={dgap:+7.4f}  "
                    f"{row['stack_status']}"
                )

    return results


def render_figure(results: dict[str, pd.DataFrame]) -> Path:
    """Render the multi-panel ladder figure."""
    if not results:
        print("No results to render")
        return OUT / "variance_stack_figure.pdf"

    pdf_path = OUT / "variance_stack_figure.pdf"
    render_ladder_panel(
        results, pdf_path,
        suptitle="Variance-Control Stack: Deployment-Aligned Decomposition (S035, spatial-blocked CV)",
        also_png=True,
    )
    print(f"\nSaved figure: {pdf_path}")
    print(f"Saved PNG:    {pdf_path.with_suffix('.png')}")

    csv_path = OUT / "variance_stack_table.csv"
    all_rows = []
    for label, df in results.items():
        df = df.copy()
        if "cell" not in df.columns:
            df.insert(0, "cell", label.replace("\n", " / "))
        all_rows.append(df)
    pd.concat(all_rows).to_csv(csv_path, index=False)
    print(f"Saved table:  {csv_path}")

    return pdf_path


def main():
    results = run_all()
    fig_path = render_figure(results)
    print(f"\nDone. {len(results)} cells rendered. Figure: {fig_path}")


if __name__ == "__main__":
    main()
