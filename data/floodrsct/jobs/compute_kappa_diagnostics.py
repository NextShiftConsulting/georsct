#!/usr/bin/env python3
"""
compute_kappa_diagnostics.py -- Phase 4: progressive kappa diagnostics.

Runs at each representation level (R0, R1, R2) to compute 4 kappa proxies
that diagnose where the next representation fix should help.

Kappa proxies:
  1. kappa_leakage:          random vs spatial metric gap (A.1 autocorrelation)
  2. kappa_transfer:         leave-event-out vs spatial metric ratio (D.3 transfer)
  3. kappa_solver:           HistGBDT vs Ridge agreement
  4. kappa_residual_spatial: Moran's I on spatial_blocked residuals (A.2/B.1)

Uses yrsn.core.kappa.spatial.compute.compute_kappa_spatial for Moran's I.
Uses _coverage_common.load_adjacency for ZCTA adjacency graph.

All kappa formulas use "higher is better" primary metric:
  - Regression: R2 score
  - Classification: ROC-AUC

Pre-registration: kappa_diagnostics_{level}.json is uploaded to S3 BEFORE
the next level trains, establishing temporal ordering proof.

Usage:
    python compute_kappa_diagnostics.py --level r0 --upload
    python compute_kappa_diagnostics.py --level r1 --upload
    python compute_kappa_diagnostics.py --level r2 --upload
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
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, load_adjacency

# Import Moran's I from yrsn (not reinvented)
try:
    from yrsn.core.kappa.spatial.compute import compute_kappa_spatial
except ImportError:
    # Fallback: yrsn not installed on SageMaker -- install via requirements
    compute_kappa_spatial = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"

# Primary metric per task type (higher = better)
PRIMARY_METRIC = {
    "regression": "r2",
    "classification": "roc_auc",
}

# Modelable scenarios (NOLA excluded, n=20)
MODELABLE = {"houston", "southwest_florida", "nyc", "riverside_coachella"}


def _load_results(s3, level: str, scenario: str) -> dict | None:
    """Load results JSON for a (level, scenario) pair."""
    key = f"{RESULTS_PREFIX}/{level}_{scenario}.json"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception as exc:
        log.warning("No results for %s/%s: %s", level, scenario, exc)
        return None


def _load_predictions(s3, level: str, scenario: str) -> pd.DataFrame:
    """Load predictions parquet for Moran's I computation."""
    key = f"{RESULTS_PREFIX}/{level}_{scenario}_predictions.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception as exc:
        log.warning("No predictions for %s/%s: %s", level, scenario, exc)
        return pd.DataFrame()


def _build_adjacency_dict(adj_df: pd.DataFrame, zcta_ids: list[str]) -> dict[int, list[int]]:
    """Convert adjacency edge-list DataFrame to index-based dict for compute_kappa_spatial.

    adj_df has columns like (zcta_id_1, zcta_id_2) or (source, target).
    Returns Dict[int, List[int]] where keys/values are positional indices
    into zcta_ids.
    """
    # Identify column names
    cols = adj_df.columns.tolist()
    if "zcta_id_1" in cols and "zcta_id_2" in cols:
        c1, c2 = "zcta_id_1", "zcta_id_2"
    elif "source" in cols and "target" in cols:
        c1, c2 = "source", "target"
    else:
        c1, c2 = cols[0], cols[1]

    zcta_to_idx = {z: i for i, z in enumerate(zcta_ids)}

    adj_dict: dict[int, list[int]] = {i: [] for i in range(len(zcta_ids))}
    for _, row in adj_df.iterrows():
        a, b = str(row[c1]), str(row[c2])
        if a in zcta_to_idx and b in zcta_to_idx:
            i, j = zcta_to_idx[a], zcta_to_idx[b]
            adj_dict[i].append(j)

    n_edges = sum(len(v) for v in adj_dict.values())
    log.info("Adjacency: %d ZCTAs, %d directed edges, mean degree %.1f",
             len(zcta_ids), n_edges, n_edges / max(len(zcta_ids), 1))
    return adj_dict


def _mean_metric(runs: list[dict], split: str, solver: str | None,
                 target: str, task_type: str) -> float | None:
    """Compute mean primary metric across folds for a (split, solver, target) combo."""
    metric_name = PRIMARY_METRIC[task_type]
    vals = []
    for r in runs:
        if r["split"] != split:
            continue
        if solver and r["solver"] != solver:
            continue
        if r["target"] != target:
            continue
        v = r["metrics"].get(metric_name)
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def compute_kappa_leakage(runs: list[dict], target: str, task_type: str) -> float | None:
    """kappa_leakage = 1 - (metric_random - metric_spatial) / max(metric_random, 0.01)

    Uses HistGBDT as reference solver (nonlinear, more sensitive to leakage).
    Low = random split inflated by spatial autocorrelation -> R1 should help.
    """
    m_random = _mean_metric(runs, "random", "histgbdt", target, task_type)
    m_spatial = _mean_metric(runs, "spatial_blocked", "histgbdt", target, task_type)
    if m_random is None or m_spatial is None:
        return None
    denom = max(abs(m_random), 0.01)
    return 1.0 - (m_random - m_spatial) / denom


def compute_kappa_transfer(runs: list[dict], target: str, task_type: str) -> float | None:
    """kappa_transfer = max(0, metric_leave_event / max(metric_spatial, 0.01))

    Uses HistGBDT as reference solver.
    Low = can't generalize across events -> R2 should help.
    """
    m_leo = _mean_metric(runs, "leave_event_out", "histgbdt", target, task_type)
    m_spatial = _mean_metric(runs, "spatial_blocked", "histgbdt", target, task_type)
    if m_leo is None or m_spatial is None:
        return None
    denom = max(abs(m_spatial), 0.01)
    return max(0.0, m_leo / denom)


def compute_kappa_solver(runs: list[dict], target: str, task_type: str) -> float | None:
    """kappa_solver = 1 - |metric_hgbdt - metric_ridge| / max(|both|, 0.01)

    Uses spatial_blocked as reference split.
    Low = solvers disagree -> complex structure R0 misses.
    """
    m_hgbdt = _mean_metric(runs, "spatial_blocked", "histgbdt", target, task_type)
    m_ridge = _mean_metric(runs, "spatial_blocked", "ridge", target, task_type)
    if m_hgbdt is None or m_ridge is None:
        return None
    denom = max(abs(m_hgbdt), abs(m_ridge), 0.01)
    return 1.0 - abs(m_hgbdt - m_ridge) / denom


def compute_kappa_residual_spatial(
    predictions_df: pd.DataFrame,
    adj_dict: dict[int, list[int]],
    zcta_ids: list[str],
    target: str,
) -> float | None:
    """kappa_residual_spatial via Moran's I on HistGBDT residuals.

    Uses yrsn.core.kappa.spatial.compute.compute_kappa_spatial.
    Convention: HIGH kappa_spatial from yrsn = clustered errors.
    We INVERT: kappa_residual_spatial = 1 - kappa_spatial (high = good = no clustering).
    """
    if compute_kappa_spatial is None:
        log.warning("yrsn not available -- skipping kappa_residual_spatial")
        return None

    if predictions_df.empty:
        return None

    # Filter to target + histgbdt (reference solver)
    mask = (predictions_df["target"] == target) & (predictions_df["solver"] == "histgbdt")
    sub = predictions_df[mask].copy()
    if len(sub) < 10:
        log.warning("Too few predictions for Moran's I (%d)", len(sub))
        return None

    # Aggregate residuals per ZCTA (average across events/folds)
    sub["residual"] = np.abs(sub["y_true"] - sub["y_pred"])
    zcta_resid = sub.groupby("zcta_id")["residual"].mean()

    # Align with zcta_ids ordering
    residuals = np.array([
        float(zcta_resid.get(z, np.nan)) for z in zcta_ids
    ])

    # Drop NaN ZCTAs from adjacency
    valid_mask = ~np.isnan(residuals)
    if valid_mask.sum() < 10:
        log.warning("Too few valid ZCTAs for Moran's I (%d)", valid_mask.sum())
        return None

    valid_idx = np.where(valid_mask)[0]
    old_to_new = {old: new for new, old in enumerate(valid_idx)}
    filtered_residuals = residuals[valid_mask]
    filtered_adj = {}
    for old_i in valid_idx:
        new_i = old_to_new[old_i]
        neighbors = [old_to_new[j] for j in adj_dict.get(old_i, [])
                      if j in old_to_new]
        filtered_adj[new_i] = neighbors

    result = compute_kappa_spatial(filtered_residuals, filtered_adj)
    log.info("Moran's I = %.4f, kappa_spatial = %.4f (target=%s, n=%d)",
             result.morans_i, result.kappa, target, result.n_samples)

    # Invert: our DOE convention is high = good (no spatial clustering)
    return 1.0 - result.kappa


def _determine_task_type(runs: list[dict], target: str) -> str | None:
    """Get task type (regression/classification) for a target from results."""
    for r in runs:
        if r["target"] == target:
            return r["task"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: progressive kappa diagnostics"
    )
    parser.add_argument("--level", required=True, choices=["r0", "r1", "r2"])
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    level = args.level
    s3 = get_s3_client()

    print(f"\n{'='*60}")
    print(f"  S035 PHASE 4: KAPPA DIAGNOSTICS -- {level.upper()}")
    print(f"{'='*60}\n")

    # Load adjacency once (shared across scenarios)
    try:
        adj_df = load_adjacency(s3)
        log.info("Adjacency edge list: %d rows", len(adj_df))
    except FileNotFoundError:
        log.warning("No adjacency data -- kappa_residual_spatial will be null")
        adj_df = pd.DataFrame()

    all_cells = []

    for scenario in sorted(MODELABLE):
        results = _load_results(s3, level, scenario)
        if results is None:
            log.warning("No %s results for %s -- skipping", level, scenario)
            continue

        runs = results.get("runs", [])
        if not runs:
            continue

        predictions_df = _load_predictions(s3, level, scenario)

        # Build adjacency dict for this scenario's ZCTAs
        zcta_ids = sorted(set(r.get("zcta_id", "") for r in
                              predictions_df.to_dict("records"))) if not predictions_df.empty else []
        if not zcta_ids:
            # Fall back to getting ZCTAs from prediction rows
            zcta_ids = sorted(predictions_df["zcta_id"].unique().tolist()) if not predictions_df.empty else []

        adj_dict = _build_adjacency_dict(adj_df, zcta_ids) if not adj_df.empty and zcta_ids else {}

        # Get unique targets from results
        targets = sorted(set(r["target"] for r in runs))

        for target in targets:
            task_type = _determine_task_type(runs, target)
            if not task_type:
                continue

            cell = {
                "scenario": scenario,
                "target": target,
                "task_type": task_type,
                "level": level,
                "kappa_leakage": compute_kappa_leakage(runs, target, task_type),
                "kappa_transfer": compute_kappa_transfer(runs, target, task_type),
                "kappa_solver": compute_kappa_solver(runs, target, task_type),
                "kappa_residual_spatial": compute_kappa_residual_spatial(
                    predictions_df, adj_dict, zcta_ids, target,
                ),
            }

            # Primary metric (spatial_blocked, histgbdt) for reference
            metric_name = PRIMARY_METRIC[task_type]
            cell["primary_metric_name"] = metric_name
            cell["primary_metric_value"] = _mean_metric(
                runs, "spatial_blocked", "histgbdt", target, task_type,
            )

            all_cells.append(cell)
            log.info("  %s / %s: leak=%.3f xfer=%.3f solver=%.3f resid=%s",
                     scenario, target,
                     cell["kappa_leakage"] or float("nan"),
                     cell["kappa_transfer"] or float("nan"),
                     cell["kappa_solver"] or float("nan"),
                     f"{cell['kappa_residual_spatial']:.3f}"
                     if cell["kappa_residual_spatial"] is not None else "N/A")

    # --- Pre-registration predictions ---
    # Median split: cells below median kappa are "flagged" (predicted to benefit)
    predictions = {}
    if level in ("r0", "r1"):
        next_level = "r1" if level == "r0" else "r2"
        kappa_key = "kappa_leakage" if next_level == "r1" else "kappa_transfer"

        valid_vals = [c[kappa_key] for c in all_cells if c[kappa_key] is not None]
        if valid_vals:
            median_val = float(np.median(valid_vals))
            flagged = [c["scenario"] for c in all_cells
                       if c[kappa_key] is not None and c[kappa_key] < median_val]
            unflagged = [c["scenario"] for c in all_cells
                         if c[kappa_key] is not None and c[kappa_key] >= median_val]
            predictions = {
                f"{next_level}_should_help_most": sorted(set(flagged)),
                f"{next_level}_should_help_least": sorted(set(unflagged)),
                "ordering_criterion": f"{kappa_key} ascending (lowest = most predicted uplift)",
                "flag_threshold": "median_split",
                "median_value": median_val,
            }
            log.info("Pre-registration: %s should help %s (median %s = %.3f)",
                     next_level, flagged, kappa_key, median_val)

    # --- Output ---
    payload = {
        "experiment": "s035-model-ladder",
        "phase": f"kappa_diagnostics_{level}",
        "level": level,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(all_cells),
        "cells": all_cells,
        "predictions": predictions,
        "methodology": {
            "kappa_leakage": "1 - (metric_random - metric_spatial) / max(|metric_random|, 0.01)",
            "kappa_transfer": "max(0, metric_leave_event / max(|metric_spatial|, 0.01))",
            "kappa_solver": "1 - |metric_hgbdt - metric_ridge| / max(|both|, 0.01)",
            "kappa_residual_spatial": "1 - kappa_spatial(yrsn) where kappa_spatial uses Moran's I",
            "primary_metric": "R2 for regression, ROC-AUC for classification",
            "reference_solver": "histgbdt (except kappa_solver which uses both)",
            "flag_threshold": "median split (pre-committed, no tuning)",
        },
    }

    output_json = json.dumps(payload, indent=2, default=str)

    if args.upload:
        key = f"{RESULTS_PREFIX}/kappa_diagnostics_{level}.json"
        s3.put_object(
            Bucket=BUCKET, Key=key,
            Body=output_json.encode(),
            ContentType="application/json",
        )
        log.info("Uploaded s3://%s/%s", BUCKET, key)
    else:
        local = f"/tmp/kappa_diagnostics_{level}.json"
        Path(local).write_text(output_json)
        log.info("Wrote %s", local)

    print(output_json)


if __name__ == "__main__":
    main()
