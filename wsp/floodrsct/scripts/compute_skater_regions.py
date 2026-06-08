#!/usr/bin/env python3
"""compute_skater_regions.py -- Skater regionalization vs county boundaries.

Supplementary spatial analysis for s035-model-ladder. Compares spatially-
constrained Skater regions (from prediction residuals) against county
boundaries using Adjusted Rand Index. Computes per-region Moran's I to
assess whether some regions resolve spatial autocorrelation while others
do not.

Governance notes:
    - Queen contiguity W-matrix is the canonical specification (locked by DOE).
    - Skater is supplementary analysis, not a replacement for LISA.
    - Framing: "the benchmark accommodates multi-scale evaluation."

Usage:
    python compute_skater_regions.py --scenario houston --k-regions 5
    python compute_skater_regions.py --scenario houston --k-regions 5 --upload
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Queen
from esda.moran import Moran
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import adjusted_rand_score
from spopt.region import Skater

from swarm_auth import get_aws_credentials

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET = "swarm-floodrsct-data"
SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]
DEFAULT_K = 5
OUTPUT_KEY = "results/s035/skater_regionalization.json"


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _s3_client() -> Any:
    """Create S3 client using swarm_auth credentials."""
    import boto3

    aws = get_aws_credentials()
    return boto3.client("s3", **aws)


def _read_parquet_from_s3(key: str) -> pd.DataFrame:
    """Read a parquet file from the floodrsct bucket."""
    s3 = _s3_client()
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    return pd.read_parquet(buf)


def _read_geoparquet_from_s3(key: str) -> gpd.GeoDataFrame:
    """Read a geoparquet file from the floodrsct bucket."""
    s3 = _s3_client()
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    return gpd.read_parquet(buf)


def _upload_json(data: dict[str, Any], key: str) -> None:
    """Upload JSON result to S3."""
    s3 = _s3_client()
    body = json.dumps(data, indent=2, default=str)
    s3.put_object(Bucket=BUCKET, Key=key, Body=body.encode("utf-8"))
    print(f"Uploaded: s3://{BUCKET}/{key}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_predictions(scenario: str) -> pd.DataFrame:
    """Load R0 spatial-blocked predictions for a scenario.

    Returns DataFrame with columns: zcta_id, target, solver, fold, y_true, y_pred.
    Uses the best-performing solver (HistGBDT) and primary target.
    """
    key = f"results/s035/r0_{scenario}_predictions.parquet"
    df = _read_parquet_from_s3(key)
    print(f"Loaded predictions: {len(df)} rows from {key}")
    return df


def load_adjacency(zcta_ids: list[str]) -> pd.DataFrame:
    """Load pre-computed ZCTA adjacency edgelist, filtered to scenario ZCTAs.

    Args:
        zcta_ids: List of ZCTA IDs in the scenario.

    Returns:
        DataFrame with columns: zcta_id_1, zcta_id_2 (filtered to scenario).
    """
    key = "raw/geocertdb2026/zcta_adjacency.parquet"
    df = _read_parquet_from_s3(key)
    df["zcta_id_1"] = df["zcta_id_1"].astype(str)
    df["zcta_id_2"] = df["zcta_id_2"].astype(str)
    zcta_set = set(zcta_ids)
    mask = df["zcta_id_1"].isin(zcta_set) & df["zcta_id_2"].isin(zcta_set)
    filtered = df[mask].reset_index(drop=True)
    print(f"Loaded adjacency: {len(filtered)} edges among {len(zcta_set)} ZCTAs")
    return filtered


def load_county_crosswalk() -> pd.DataFrame:
    """Load ZCTA-to-county crosswalk."""
    key = "raw/geocertdb2026/zcta_county_crosswalk.parquet"
    df = _read_parquet_from_s3(key)
    print(f"Loaded county crosswalk: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def compute_residuals(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute mean residual per ZCTA from spatial-blocked predictions.

    Filters to primary target (obs_nfip_event_claims) and HistGBDT solver,
    then averages residuals across folds and events.

    Args:
        predictions: Raw predictions DataFrame.

    Returns:
        DataFrame with columns: zcta_id, mean_residual, abs_mean_residual.
    """
    # Filter to primary target and best solver
    mask = predictions["target"] == "obs_nfip_event_claims"
    if "solver" in predictions.columns:
        # Prefer HistGBDT if available
        solvers = predictions["solver"].unique()
        if "HistGBDT" in solvers:
            mask = mask & (predictions["solver"] == "HistGBDT")
        elif "histgbdt" in solvers:
            mask = mask & (predictions["solver"] == "histgbdt")

    df = predictions[mask].copy()
    if df.empty:
        raise ValueError("No predictions found for primary target + solver")

    df["residual"] = df["y_true"] - df["y_pred"]

    # Average residuals per ZCTA (across folds and events)
    residuals = (
        df.groupby("zcta_id")["residual"]
        .agg(mean_residual="mean", abs_mean_residual=lambda x: np.abs(x).mean())
        .reset_index()
    )
    print(f"Computed residuals for {len(residuals)} ZCTAs")
    return residuals


def build_queen_weights(gdf: gpd.GeoDataFrame) -> Queen:
    """Build Queen contiguity W-matrix from ZCTA geometries.

    Args:
        gdf: GeoDataFrame with ZCTA polygons.

    Returns:
        Queen contiguity weights object.
    """
    w = Queen.from_dataframe(gdf, use_index=False)
    print(f"Built Queen W-matrix: {w.n} observations, {w.mean_neighbors:.1f} mean neighbors")
    return w


def run_skater(
    gdf: gpd.GeoDataFrame,
    attrs: list[str],
    w: Queen,
    k: int,
) -> np.ndarray:
    """Run Skater regionalization.

    Args:
        gdf: GeoDataFrame with attributes for clustering.
        attrs: Column names to use as clustering attributes.
        w: Spatial weights (Queen contiguity).
        k: Number of regions to produce.

    Returns:
        Array of region labels (length = n_zctas).
    """
    model = Skater(
        gdf,
        w,
        attrs_name=attrs,
        n_clusters=k,
        floor=3,  # minimum ZCTAs per region
    )
    model.solve()
    labels = np.array(model.labels_)
    print(f"Skater regionalization: {k} regions, sizes = {np.bincount(labels).tolist()}")
    return labels


def compute_morans_i_per_group(
    residuals: np.ndarray,
    w: Queen,
    group_labels: np.ndarray,
    group_names: dict[int, str] | None = None,
) -> dict[str, float | None]:
    """Compute Moran's I on residuals within each group.

    Args:
        residuals: Array of residual values (aligned with w).
        w: Full Queen weights matrix.
        group_labels: Group assignment for each observation.
        group_names: Optional mapping from label to name.

    Returns:
        Dict mapping group name to Moran's I (or None if too few observations).
    """
    results: dict[str, float | None] = {}
    unique_labels = np.unique(group_labels)

    for label in unique_labels:
        mask = group_labels == label
        n_in_group = mask.sum()

        name = group_names[label] if group_names else f"region_{label}"

        if n_in_group < 5:
            results[name] = None
            continue

        # Subset weights to this group
        indices = np.where(mask)[0]
        try:
            w_sub = w.full()[np.ix_(indices, indices)]
            # Row-standardize
            row_sums = w_sub.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            w_sub = w_sub / row_sums

            # Compute Moran's I manually for the subgroup
            y = residuals[indices]
            y_demean = y - y.mean()
            n = len(y)
            numerator = float(y_demean @ w_sub @ y_demean)
            denominator = float(y_demean @ y_demean)

            if denominator == 0:
                results[name] = None
            else:
                morans_i = (n / w_sub.sum()) * (numerator / denominator)
                results[name] = round(float(morans_i), 4)
        except Exception:
            results[name] = None

    return results


def compute_global_morans_i(residuals: np.ndarray, w: Queen) -> float:
    """Compute global Moran's I for the full set of residuals.

    Args:
        residuals: Array of residual values.
        w: Queen contiguity weights.

    Returns:
        Global Moran's I statistic.
    """
    mi = Moran(residuals, w)
    return round(float(mi.I), 4)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_analysis(scenario: str, k_regions: int) -> dict[str, Any]:
    """Run the full Skater regionalization analysis.

    Args:
        scenario: Scenario name (e.g. "houston").
        k_regions: Number of Skater regions.

    Returns:
        Results dictionary matching the expected output schema.
    """
    # 1. Load data
    predictions = load_predictions(scenario)
    geometry = load_zcta_geometry(scenario)
    crosswalk = load_county_crosswalk()

    # 2. Compute residuals
    residuals_df = compute_residuals(predictions)

    # 3. Merge geometry + residuals
    # Ensure zcta_id types match
    geometry["zcta_id"] = geometry["zcta_id"].astype(str)
    residuals_df["zcta_id"] = residuals_df["zcta_id"].astype(str)
    crosswalk["zcta_id"] = crosswalk["zcta_id"].astype(str)

    gdf = geometry.merge(residuals_df, on="zcta_id", how="inner")
    gdf = gdf.merge(
        crosswalk[["zcta_id", "county_name"]].drop_duplicates(),
        on="zcta_id",
        how="left",
    )
    gdf = gdf.dropna(subset=["mean_residual"]).reset_index(drop=True)
    print(f"Merged dataset: {len(gdf)} ZCTAs with residuals and geometry")

    if len(gdf) < k_regions * 3:
        raise ValueError(
            f"Too few ZCTAs ({len(gdf)}) for {k_regions} regions "
            f"(need at least {k_regions * 3})"
        )

    # 4. Build Queen W-matrix
    w = build_queen_weights(gdf)

    # 5. Run Skater regionalization on residuals
    attrs = ["mean_residual", "abs_mean_residual"]
    skater_labels = run_skater(gdf, attrs, w, k_regions)
    gdf["skater_region"] = skater_labels

    # 6. Compute Adjusted Rand Index (Skater vs county)
    county_labels = gdf["county_name"].fillna("Unknown")
    # Encode counties as integers for ARI
    county_unique = county_labels.unique()
    county_int = county_labels.map(
        {c: i for i, c in enumerate(county_unique)}
    ).values
    ari = adjusted_rand_score(county_int, skater_labels)
    print(f"Adjusted Rand Index (Skater vs County): {ari:.4f}")

    # 7. Compute global Moran's I
    residual_arr = gdf["mean_residual"].values
    global_mi = compute_global_morans_i(residual_arr, w)
    print(f"Global Moran's I: {global_mi}")

    # 8. Per-region Moran's I (Skater regions)
    per_region_mi = compute_morans_i_per_group(
        residual_arr, w, skater_labels
    )

    # 9. Per-county Moran's I
    county_names_map = {i: c for i, c in enumerate(county_unique)}
    per_county_mi = compute_morans_i_per_group(
        residual_arr, w, county_int, group_names=county_names_map
    )

    # 10. Build output
    skater_assignment = {
        str(row["zcta_id"]): int(row["skater_region"])
        for _, row in gdf.iterrows()
    }
    county_assignment = {
        str(row["zcta_id"]): str(row["county_name"])
        for _, row in gdf.iterrows()
    }

    # Interpretation
    if ari > 0.5:
        interpretation = (
            f"Skater regions strongly align with counties (ARI={ari:.3f}), "
            "suggesting county boundaries are a reasonable proxy for natural "
            "spatial regimes in prediction performance."
        )
    elif ari > 0.2:
        interpretation = (
            f"Skater regions partially align with counties (ARI={ari:.3f}). "
            "County boundaries capture some spatial structure but miss "
            "sub-county variation in model performance."
        )
    else:
        interpretation = (
            f"Skater regions diverge from counties (ARI={ari:.3f}), "
            "indicating that prediction residuals cluster along different "
            "spatial axes than administrative boundaries. Multi-scale "
            "evaluation reveals structure invisible to county blocking."
        )

    result: dict[str, Any] = {
        "scenario": scenario,
        "n_zctas": len(gdf),
        "k_regions": k_regions,
        "global_morans_i": global_mi,
        "adjusted_rand_index": round(float(ari), 4),
        "skater_regions": skater_assignment,
        "county_assignment": county_assignment,
        "per_region_morans_i": per_region_mi,
        "per_county_morans_i": per_county_mi,
        "region_sizes": np.bincount(skater_labels).tolist(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interpretation": interpretation,
        "governance": {
            "w_matrix": "Queen contiguity (DOE-locked)",
            "role": "supplementary (not replacement for LISA)",
            "framing": "benchmark accommodates multi-scale evaluation",
        },
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Skater regionalization analysis."""
    parser = argparse.ArgumentParser(
        description="Skater regionalization vs county boundaries"
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="houston",
        help="Scenario to analyze (default: houston)",
    )
    parser.add_argument(
        "--k-regions",
        type=int,
        default=DEFAULT_K,
        help=f"Number of Skater regions (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to S3",
    )
    parser.add_argument(
        "--output-local",
        type=str,
        default=None,
        help="Write results to local file (in addition to S3 if --upload)",
    )
    args = parser.parse_args()

    print(f"=== Skater Regionalization: {args.scenario}, k={args.k_regions} ===")
    result = run_analysis(args.scenario, args.k_regions)

    # Save locally
    if args.output_local:
        out_path = Path(args.output_local)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"Saved locally: {out_path}")

    # Upload to S3
    if args.upload:
        _upload_json(result, OUTPUT_KEY)

    # Summary
    print("\n=== Summary ===")
    print(f"  ZCTAs:              {result['n_zctas']}")
    print(f"  Skater regions:     {result['k_regions']}")
    print(f"  Region sizes:       {result['region_sizes']}")
    print(f"  Global Moran's I:   {result['global_morans_i']}")
    print(f"  ARI (Skater/County):{result['adjusted_rand_index']}")
    print(f"  Interpretation:     {result['interpretation']}")


if __name__ == "__main__":
    main()
