"""
compute_residual_lisa.py -- Local Moran's I cluster maps on model residuals.

Extends diag_residual_spatial (Global Moran's I) with Local Indicators of
Spatial Association (LISA). Identifies WHERE the model fails spatially, not
just whether residuals are spatially autocorrelated.

Outputs per-ZCTA cluster classification: HH (hot spot), HL (outlier),
LH (outlier), LL (cold spot), NS (not significant).

Global Moran's I is delegated to yrsn.core.kappa.spatial.compute_kappa_spatial
for consistency with compute_diagnostics.py. Local LISA uses esda (yrsn
doesn't provide Local Moran's I).

Design inspired by Hotspot Analysis v3 QGIS plugin (clean-room, no GPL code).

Usage:
    python compute_residual_lisa.py --level r0 --scenario houston --upload
    python compute_residual_lisa.py --level r0 --all-scenarios --upload
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
from _coverage_common import BUCKET, get_s3_client, load_adjacency, level_prefix

# yrsn for global Moran's I (same pattern as compute_diagnostics.py)
try:
    from yrsn.core.kappa.spatial.compute import compute_kappa_spatial
except ImportError:
    compute_kappa_spatial = None

MODELABLE = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


# ---------------------------------------------------------------------------
# Index-based adjacency for yrsn kappa_spatial
# ---------------------------------------------------------------------------

def _build_adjacency_dict(
    adj_df: pd.DataFrame, zcta_ids: list[str],
) -> dict[int, list[int]]:
    """Convert adjacency edge-list to index-based dict for compute_kappa_spatial.

    Same pattern as compute_diagnostics.py -- yrsn expects Dict[int, List[int]]
    keyed by positional index.
    """
    cols = adj_df.columns.tolist()
    if "zcta_from" in cols and "zcta_to" in cols:
        c1, c2 = "zcta_from", "zcta_to"
    elif "zcta_id_1" in cols and "zcta_id_2" in cols:
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

    return adj_dict


# ---------------------------------------------------------------------------
# Spatial weights from adjacency edge list (for libpysal LISA)
# ---------------------------------------------------------------------------

def build_weights_from_adjacency(
    adj_df: pd.DataFrame, zcta_ids: list[str],
) -> "libpysal.weights.W":
    """Build a libpysal spatial weights matrix from an adjacency edge list.

    Args:
        adj_df: DataFrame with columns [zcta_from, zcta_to].
        zcta_ids: List of ZCTA IDs to include (filters adjacency).

    Returns:
        libpysal W object.
    """
    from libpysal.weights import W

    id_set = set(zcta_ids)
    # Filter to edges where both endpoints are in our dataset
    # Adjacency columns may be named zcta_from/zcta_to or zcta_id_1/zcta_id_2
    cols = adj_df.columns.tolist()
    if "zcta_from" in cols:
        c1, c2 = "zcta_from", "zcta_to"
    elif "zcta_id_1" in cols:
        c1, c2 = "zcta_id_1", "zcta_id_2"
    else:
        c1, c2 = cols[0], cols[1]

    mask = adj_df[c1].astype(str).isin(id_set) & adj_df[c2].astype(str).isin(id_set)
    edges = adj_df[mask]

    neighbors = {z: [] for z in zcta_ids}
    for _, row in edges.iterrows():
        a, b = str(row[c1]), str(row[c2])
        if a in neighbors and b in neighbors:
            neighbors[a].append(b)
            neighbors[b].append(a)

    # Deduplicate
    neighbors = {k: list(set(v)) for k, v in neighbors.items()}

    # Islands (no neighbors) get empty list -- libpysal handles this
    w = W(neighbors, silence_warnings=True)
    log.info(
        "Spatial weights: %d zones, %d edges, %d islands",
        w.n, w.s0 // 2, sum(1 for v in w.cardinalities.values() if v == 0),
    )
    return w


# ---------------------------------------------------------------------------
# Local Moran's I (LISA)
# ---------------------------------------------------------------------------

def compute_lisa(
    residuals: pd.Series,
    w: "libpysal.weights.W",
    permutations: int = 999,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute Local Moran's I and classify into LISA clusters.

    Args:
        residuals: Series indexed by ZCTA ID with model residuals.
        w: Spatial weights matrix (same IDs as residuals).
        permutations: Number of permutations for significance testing.
        alpha: Significance threshold for cluster classification.

    Returns:
        DataFrame indexed by ZCTA with columns:
            lisa_i: Local Moran's I statistic
            lisa_p: Pseudo p-value
            lisa_z: Z-score
            lisa_cluster: HH, HL, LH, LL, or NS (not significant)
    """
    from esda.moran import Moran_Local

    # Align residuals to weight matrix order
    aligned = residuals.reindex(w.id_order)
    y = aligned.values.astype(np.float64)

    # Replace NaN with 0 (conservative -- treats missing as neutral)
    nan_mask = np.isnan(y)
    if nan_mask.any():
        log.warning("%d NaN residuals set to 0 for LISA", nan_mask.sum())
        y[nan_mask] = 0.0

    lisa = Moran_Local(y, w, permutations=permutations)

    # Classify into quadrants
    # esda uses q attribute: 1=HH, 2=LH, 3=LL, 4=HL
    quad_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    clusters = []
    for i in range(len(y)):
        if lisa.p_sim[i] > alpha:
            clusters.append("NS")
        else:
            clusters.append(quad_labels.get(lisa.q[i], "NS"))

    result = pd.DataFrame({
        "lisa_i": lisa.Is,
        "lisa_p": lisa.p_sim,
        "lisa_z": lisa.z_sim,
        "lisa_cluster": clusters,
    }, index=w.id_order)
    result.index.name = "zcta_id"

    counts = pd.Series(clusters).value_counts()
    log.info("LISA clusters: %s", dict(counts))
    log.info(
        "Global Moran's I = %.4f (p=%.4f)",
        lisa.z_sim.mean(), np.mean(lisa.p_sim),
    )
    return result


# ---------------------------------------------------------------------------
# Load predictions and compute residuals
# ---------------------------------------------------------------------------

def load_predictions(s3, level: str, scenario: str) -> pd.DataFrame:
    """Load per-row predictions from S3."""
    key = f"results/s035/{level_prefix(level)}_{scenario}_predictions.parquet"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        log.info("Loaded %d predictions from %s", len(df), key)
        return df
    except s3.exceptions.ClientError:
        log.warning("Predictions not found: %s", key)
        return pd.DataFrame()


def compute_residuals(preds_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute residuals per (target, solver) from predictions DataFrame.

    Returns dict keyed by "{target}__{solver}" with Series indexed by zcta_id.
    """
    if preds_df.empty:
        return {}

    required = {"zcta_id", "target", "solver", "y_true", "y_pred"}
    if not required.issubset(preds_df.columns):
        log.warning("Predictions missing columns: %s", required - set(preds_df.columns))
        return {}

    residuals = {}
    for (target, solver), group in preds_df.groupby(["target", "solver"]):
        resid = group.set_index("zcta_id")["y_true"] - group.set_index("zcta_id")["y_pred"]
        key = f"{target}__{solver}"
        residuals[key] = resid
        log.info(
            "Residuals %s: n=%d, mean=%.4f, std=%.4f",
            key, len(resid), resid.mean(), resid.std(),
        )
    return residuals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_lisa_analysis(
    s3, level: str, scenario: str, adj_df: pd.DataFrame, upload: bool = False,
) -> dict:
    """Run LISA analysis for one scenario at one representation level."""
    preds = load_predictions(s3, level, scenario)
    if preds.empty:
        return {"scenario": scenario, "level": level, "status": "NO_PREDICTIONS"}

    residuals_dict = compute_residuals(preds)
    if not residuals_dict:
        return {"scenario": scenario, "level": level, "status": "NO_RESIDUALS"}

    zcta_ids = sorted(preds["zcta_id"].unique().astype(str).tolist())
    w = build_weights_from_adjacency(adj_df, zcta_ids)

    results = {}
    all_lisa_dfs = []
    for cell_key, resid_series in residuals_dict.items():
        resid_series.index = resid_series.index.astype(str)
        lisa_df = compute_lisa(resid_series, w)
        lisa_df["cell"] = cell_key
        all_lisa_dfs.append(lisa_df)

        counts = lisa_df["lisa_cluster"].value_counts().to_dict()
        results[cell_key] = {
            "n_zctas": len(lisa_df),
            "n_significant": int((lisa_df["lisa_cluster"] != "NS").sum()),
            "clusters": counts,
            "global_mean_lisa_i": float(lisa_df["lisa_i"].mean()),
        }

        # Global Moran's I via yrsn (consistent with compute_diagnostics.py)
        if compute_kappa_spatial is not None and not adj_df.empty:
            try:
                idx_adj = _build_adjacency_dict(adj_df, zcta_ids)
                resid_aligned = np.array([
                    float(resid_series.get(z, 0.0)) for z in zcta_ids
                ])
                kappa_result = compute_kappa_spatial(resid_aligned, idx_adj)
                results[cell_key]["yrsn_morans_i"] = float(kappa_result.morans_i)
                results[cell_key]["yrsn_kappa_spatial"] = float(kappa_result.kappa)
                results[cell_key]["yrsn_expected_i"] = float(kappa_result.expected_i)
                log.info(
                    "  yrsn kappa_spatial: I=%.4f, kappa=%.4f (cell=%s)",
                    kappa_result.morans_i, kappa_result.kappa, cell_key,
                )
            except Exception as e:
                log.warning("yrsn kappa_spatial failed for %s: %s", cell_key, e)

    summary = {
        "scenario": scenario,
        "level": level,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cells": results,
        "status": "COMPLETE",
    }

    if upload and all_lisa_dfs:
        # Upload LISA cluster parquet
        combined = pd.concat(all_lisa_dfs, ignore_index=False)
        combined = combined.reset_index()
        buf = io.BytesIO()
        combined.to_parquet(buf, compression="zstd")
        key = f"results/s035/{level}_{scenario}_lisa.parquet"
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        log.info("Uploaded %s", key)
        summary["lisa_parquet"] = f"s3://{BUCKET}/{key}"

        # Upload summary JSON
        json_key = f"results/s035/{level}_{scenario}_lisa.json"
        s3.put_object(
            Bucket=BUCKET, Key=json_key,
            Body=json.dumps(summary, indent=2).encode(),
            ContentType="application/json",
        )
        log.info("Uploaded %s", json_key)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LISA residual cluster analysis")
    parser.add_argument("--level", required=True, choices=["r0", "r1", "r2"])
    parser.add_argument("--scenario", choices=MODELABLE)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    if not args.scenario and not args.all_scenarios:
        parser.error("specify --scenario or --all-scenarios")

    s3 = get_s3_client()

    try:
        adj_df = load_adjacency(s3)
    except FileNotFoundError:
        log.error("ZCTA adjacency not found on S3. Required for LISA.")
        sys.exit(1)

    scenarios = MODELABLE if args.all_scenarios else [args.scenario]
    for scenario in scenarios:
        log.info("=== LISA %s / %s ===", args.level, scenario)
        result = run_lisa_analysis(s3, args.level, scenario, adj_df, args.upload)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
