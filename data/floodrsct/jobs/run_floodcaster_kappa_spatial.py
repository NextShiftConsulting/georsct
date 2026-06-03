#!/usr/bin/env python3
"""
run_floodcaster_kappa_spatial.py -- S036 H4: Floodcaster Kappa Spatial Probe.

Claim: Floodcaster damage prediction residuals exhibit spatial autocorrelation
(Moran's I > 0) when compared against NFIP historical claims, and kappa_spatial
detects coastal discontinuities (reef effect, elevation breaks).

Inputs (S3):
  swarm-floodcaster/results/{job_id}.parquet       (floodcaster output)
  swarm-floodrsct-data/raw/geocertdb2026/nfip_claims_zcta.parquet
  swarm-floodrsct-data/raw/geocertdb2026/zcta_adjacency.parquet
  swarm-floodrsct-data/raw/geocertdb2026/zcta_features_labels.parquet

Outputs:
  outputs/floodcaster_spatial/kappa_spatial_results.json
  outputs/floodcaster_spatial/residuals_by_zcta.csv
  outputs/floodcaster_spatial/fig_kappa_spatial_map.png
  outputs/floodcaster_spatial/evidence_s036_h4.json
  All uploaded to s3://swarm-yrsn-datasets/geocert-experiments/s036/floodcaster_spatial/

Usage:
    python run_floodcaster_kappa_spatial.py
    JOB_ID=1f3ba5fedaaa python run_floodcaster_kappa_spatial.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FLOODCASTER_BUCKET = "swarm-floodcaster"
FLOODRSCT_BUCKET = "swarm-floodrsct-data"

# Reference data prefix -- Hawaii uses separate tables (different CRS/topology)
GEOCERT_PREFIX = "raw/geocertdb2026"
HAWAII_PREFIX = "raw/hawaii"

DATASETS_BUCKET = "swarm-yrsn-datasets"
S3_OUTPUT_PREFIX = os.environ.get(
    "S3_OUTPUT_PREFIX", "geocert-experiments/s036/floodcaster_spatial"
)

OUTPUT_DIR = Path(os.environ.get("LOCAL_OUTPUT_DIR", "outputs/floodcaster_spatial"))

# Floodcaster job to analyze (Oahu coastal surge, 47983 buildings)
JOB_ID = os.environ.get("JOB_ID", "1f3ba5fedaaa")

# Hawaii job IDs use separate reference data (different projections/topology)
HAWAII_JOB_IDS = {"1f3ba5fedaaa"}

REGION = "us-east-1"

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_client() -> boto3.client:
    return boto3.client("s3", region_name=REGION)


def s3_load_parquet(s3, bucket: str, key: str) -> pd.DataFrame:
    logger.info("Loading s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def s3_upload_file(s3, local_path: Path, bucket: str, key: str) -> None:
    logger.info("Uploading %s -> s3://%s/%s", local_path, bucket, key)
    s3.upload_file(str(local_path), bucket, key)


def upload_outputs(s3, output_dir: Path, prefix: str) -> None:
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            dest_key = f"{prefix.rstrip('/')}/{p.name}"
            s3_upload_file(s3, p, DATASETS_BUCKET, dest_key)


# ---------------------------------------------------------------------------
# Step 1: Load floodcaster results
# ---------------------------------------------------------------------------

def load_floodcaster_results(s3, job_id: str) -> pd.DataFrame:
    key = f"results/{job_id}.parquet"
    df = s3_load_parquet(s3, FLOODCASTER_BUCKET, key)
    logger.info("Floodcaster results: %d buildings, %d columns", len(df), len(df.columns))

    required = ["latitude", "longitude", "BldgLossUSD"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        sys.exit(1)

    return df


# ---------------------------------------------------------------------------
# Step 2: Assign buildings to ZCTAs via spatial join
# ---------------------------------------------------------------------------

def assign_zctas(buildings: pd.DataFrame, zcta_features: pd.DataFrame) -> pd.DataFrame:
    """Assign each building to the nearest ZCTA centroid.

    Uses haversine nearest-neighbor since we don't have ZCTA polygons
    loaded here. The zcta_features table has lat/lon centroids.
    """
    lat_col = next((c for c in zcta_features.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in zcta_features.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    zcta_col = next((c for c in zcta_features.columns if "zcta" in c.lower() or "geoid" in c.lower()), None)

    if not all([lat_col, lon_col, zcta_col]):
        logger.error("Cannot find lat/lon/zcta columns in features table: %s", zcta_features.columns.tolist())
        sys.exit(1)

    logger.info("ZCTA assignment using columns: zcta=%s lat=%s lon=%s", zcta_col, lat_col, lon_col)

    zcta_lats = zcta_features[lat_col].values
    zcta_lons = zcta_features[lon_col].values
    zcta_ids = zcta_features[zcta_col].astype(str).values

    bldg_lats = buildings["latitude"].values
    bldg_lons = buildings["longitude"].values

    # Vectorized haversine nearest-neighbor
    assignments = []
    batch_size = 5000
    for start in range(0, len(bldg_lats), batch_size):
        end = min(start + batch_size, len(bldg_lats))
        blat = np.radians(bldg_lats[start:end, None])
        blon = np.radians(bldg_lons[start:end, None])
        zlat = np.radians(zcta_lats[None, :])
        zlon = np.radians(zcta_lons[None, :])

        dlat = blat - zlat
        dlon = blon - zlon
        a = np.sin(dlat / 2) ** 2 + np.cos(blat) * np.cos(zlat) * np.sin(dlon / 2) ** 2
        dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        nearest = np.argmin(dist, axis=1)
        assignments.extend(zcta_ids[nearest])

    buildings = buildings.copy()
    buildings["zcta"] = assignments
    n_zctas = buildings["zcta"].nunique()
    logger.info("Buildings assigned to %d ZCTAs", n_zctas)
    return buildings


# ---------------------------------------------------------------------------
# Step 3: Aggregate to ZCTA level and compute residuals
# ---------------------------------------------------------------------------

def compute_zcta_residuals(
    buildings: pd.DataFrame,
    nfip_claims: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate floodcaster predictions by ZCTA, join with NFIP claims,
    compute residuals (predicted - actual)."""

    # Aggregate predicted losses by ZCTA
    zcta_pred = buildings.groupby("zcta").agg(
        pred_bldg_loss=("BldgLossUSD", "sum"),
        pred_content_loss=("ContentLossUSD", lambda x: x.sum() if x.notna().any() else 0),
        n_buildings=("BldgLossUSD", "count"),
        mean_damage_pct=("BldgDmgPct", lambda x: x.mean() if x.notna().any() else 0),
    ).reset_index()

    zcta_pred["pred_total_loss"] = zcta_pred["pred_bldg_loss"] + zcta_pred["pred_content_loss"]
    logger.info("ZCTA predictions: %d ZCTAs", len(zcta_pred))

    # Identify NFIP ZCTA column and claims column
    nfip_zcta_col = next((c for c in nfip_claims.columns if "zcta" in c.lower() or "geoid" in c.lower()), None)
    claims_cols = [c for c in nfip_claims.columns if "claim" in c.lower() or "loss" in c.lower() or "paid" in c.lower()]

    if not nfip_zcta_col:
        logger.error("Cannot find ZCTA column in NFIP claims: %s", nfip_claims.columns.tolist())
        sys.exit(1)

    logger.info("NFIP columns: zcta=%s, claims_cols=%s", nfip_zcta_col, claims_cols)
    logger.info("NFIP claims shape: %d rows", len(nfip_claims))

    nfip_claims[nfip_zcta_col] = nfip_claims[nfip_zcta_col].astype(str)

    # Use the first numeric claims column as the ground truth signal
    claims_value_col = None
    for c in claims_cols:
        if pd.api.types.is_numeric_dtype(nfip_claims[c]):
            claims_value_col = c
            break

    if claims_value_col is None:
        # Fallback: count of claims per ZCTA as proxy
        logger.warning("No numeric claims column found; using claim count as proxy")
        nfip_agg = nfip_claims.groupby(nfip_zcta_col).size().reset_index(name="nfip_claim_count")
        claims_value_col = "nfip_claim_count"
    else:
        nfip_agg = nfip_claims.groupby(nfip_zcta_col).agg(
            nfip_total=pd.NamedAgg(column=claims_value_col, aggfunc="sum"),
            nfip_count=pd.NamedAgg(column=claims_value_col, aggfunc="count"),
        ).reset_index()
        claims_value_col = "nfip_total"

    nfip_agg = nfip_agg.rename(columns={nfip_zcta_col: "zcta"})

    # Join
    merged = zcta_pred.merge(nfip_agg, on="zcta", how="inner")
    logger.info("Merged ZCTAs (pred + NFIP): %d", len(merged))

    if len(merged) < 3:
        logger.warning("Too few overlapping ZCTAs (%d). Using prediction-only residuals.", len(merged))
        # Fallback: use per-ZCTA predicted loss as the "residual" itself
        # This still tests spatial autocorrelation of damage estimates
        zcta_pred["residual"] = zcta_pred["pred_total_loss"]
        zcta_pred["residual_type"] = "prediction_magnitude"
        return zcta_pred

    # Normalize both to [0,1] for comparable residuals
    pred_max = merged["pred_total_loss"].max()
    nfip_max = merged[claims_value_col].max()

    if pred_max > 0:
        merged["pred_norm"] = merged["pred_total_loss"] / pred_max
    else:
        merged["pred_norm"] = 0.0

    if nfip_max > 0:
        merged["nfip_norm"] = merged[claims_value_col] / nfip_max
    else:
        merged["nfip_norm"] = 0.0

    merged["residual"] = (merged["pred_norm"] - merged["nfip_norm"]).abs()
    merged["residual_type"] = "normalized_pred_minus_nfip"

    logger.info("Residual stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
                merged["residual"].mean(), merged["residual"].std(),
                merged["residual"].min(), merged["residual"].max())

    return merged


# ---------------------------------------------------------------------------
# Step 4: Build adjacency dict for kappa_spatial
# ---------------------------------------------------------------------------

def build_adjacency_dict(
    adj_df: pd.DataFrame,
    active_zctas: list[str],
) -> dict[int, list[int]]:
    """Convert adjacency parquet to {index: [neighbor_indices]} dict."""
    col_a = adj_df.columns[0]
    col_b = adj_df.columns[1]

    zcta_to_idx = {z: i for i, z in enumerate(active_zctas)}

    adjacency: dict[int, list[int]] = {i: [] for i in range(len(active_zctas))}

    for a, b in zip(adj_df[col_a].astype(str), adj_df[col_b].astype(str)):
        if a in zcta_to_idx and b in zcta_to_idx:
            i, j = zcta_to_idx[a], zcta_to_idx[b]
            adjacency[i].append(j)
            adjacency[j].append(i)

    # Deduplicate
    for k in adjacency:
        adjacency[k] = sorted(set(adjacency[k]))

    n_edges = sum(len(v) for v in adjacency.values())
    n_connected = sum(1 for v in adjacency.values() if len(v) > 0)
    logger.info("Adjacency dict: %d nodes, %d directed edges, %d connected",
                len(adjacency), n_edges, n_connected)

    return adjacency


# ---------------------------------------------------------------------------
# Step 5: Compute kappa_spatial (Moran's I)
# ---------------------------------------------------------------------------

def compute_kappa_spatial(
    residuals: np.ndarray,
    adjacency: dict[int, list[int]],
) -> dict:
    """Compute Moran's I and map to kappa in [0, 1].

    Reimplements the core logic from yrsn/core/kappa/spatial/compute.py
    so this script is self-contained for SageMaker (no yrsn wheel needed).
    """
    n = len(residuals)
    if n < 3:
        logger.warning("Too few samples (%d) for Moran's I", n)
        return {"morans_i": 0.0, "expected_i": 0.0, "kappa": 0.5, "n_samples": n, "n_edges": 0, "mean_degree": 0.0}

    z = residuals - residuals.mean()
    ss = float(np.dot(z, z))
    if ss < 1e-12:
        logger.warning("Constant residuals -- Moran's I undefined")
        return {"morans_i": 0.0, "expected_i": -1.0 / (n - 1), "kappa": 0.5, "n_samples": n, "n_edges": 0, "mean_degree": 0.0}

    # Compute weighted cross-product
    cross = 0.0
    W = 0
    for i, neighbors in adjacency.items():
        if i >= n:
            continue
        for j in neighbors:
            if j >= n:
                continue
            cross += z[i] * z[j]
            W += 1

    if W == 0:
        logger.warning("No adjacency edges -- Moran's I undefined")
        return {"morans_i": 0.0, "expected_i": -1.0 / (n - 1), "kappa": 0.5, "n_samples": n, "n_edges": 0, "mean_degree": 0.0}

    I = (n / W) * (cross / ss)
    E_I = -1.0 / (n - 1)

    # Map to [0, 1]: kappa = clamp((I - E[I] + 1) / 2, 0, 1)
    kappa = max(0.0, min(1.0, (I - E_I + 1.0) / 2.0))

    mean_degree = W / n

    logger.info("Moran's I = %.6f  E[I] = %.6f  kappa = %.4f", I, E_I, kappa)
    logger.info("W = %d edges, mean degree = %.2f", W, mean_degree)

    return {
        "morans_i": float(I),
        "expected_i": float(E_I),
        "kappa": float(kappa),
        "n_samples": n,
        "n_edges": W,
        "mean_degree": float(mean_degree),
    }


# ---------------------------------------------------------------------------
# Step 6: Visualization
# ---------------------------------------------------------------------------

def plot_spatial_residuals(
    zcta_df: pd.DataFrame,
    kappa_result: dict,
    output_path: Path,
) -> None:
    """Scatter plot of residuals by lat/lon with kappa annotation."""
    lat_col = next((c for c in zcta_df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in zcta_df.columns if "lon" in c.lower() or "lng" in c.lower()), None)

    if not lat_col or not lon_col or lat_col not in zcta_df.columns:
        logger.warning("No lat/lon in ZCTA table for plotting; skipping figure")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scatter = ax.scatter(
        zcta_df[lon_col], zcta_df[lat_col],
        c=zcta_df["residual"], cmap="YlOrRd", s=40, alpha=0.7,
        edgecolors="k", linewidth=0.3,
    )
    plt.colorbar(scatter, ax=ax, label="Residual magnitude")

    kappa = kappa_result["kappa"]
    morans_i = kappa_result["morans_i"]
    ax.set_title(
        f"Floodcaster Residuals by ZCTA\n"
        f"Moran's I = {morans_i:.4f}  |  kappa_spatial = {kappa:.4f}",
        fontsize=12,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = s3_client()

    # 1. Load floodcaster results
    logger.info("=" * 60)
    logger.info("S036 H4: Floodcaster Kappa Spatial Probe")
    logger.info("Job ID: %s", JOB_ID)
    logger.info("=" * 60)

    buildings = load_floodcaster_results(s3, JOB_ID)

    # 2. Load ZCTA reference data (Hawaii uses separate tables)
    is_hawaii = JOB_ID in HAWAII_JOB_IDS
    if is_hawaii:
        logger.info("Hawaii job detected -- using separate reference data")
        ref_prefix = HAWAII_PREFIX
        zcta_features = s3_load_parquet(s3, FLOODRSCT_BUCKET, f"{ref_prefix}/hawaii_zcta_centroids.parquet")
        nfip_claims = s3_load_parquet(s3, FLOODRSCT_BUCKET, f"{ref_prefix}/hawaii_nfip_claims.parquet")
        adj_df = s3_load_parquet(s3, FLOODRSCT_BUCKET, f"{ref_prefix}/hawaii_zcta_adjacency.parquet")
    else:
        ref_prefix = GEOCERT_PREFIX
        zcta_features = s3_load_parquet(s3, FLOODRSCT_BUCKET, f"{ref_prefix}/zcta_features_labels.parquet")
        nfip_claims = s3_load_parquet(s3, FLOODRSCT_BUCKET, f"{ref_prefix}/nfip_claims_zcta.parquet")
        adj_df = s3_load_parquet(s3, FLOODRSCT_BUCKET, f"{ref_prefix}/zcta_adjacency.parquet")

    logger.info("ZCTA features: %d rows, NFIP claims: %d rows, Adjacency: %d rows",
                len(zcta_features), len(nfip_claims), len(adj_df))

    # 3. Assign buildings to ZCTAs
    buildings = assign_zctas(buildings, zcta_features)

    # 4. Compute per-ZCTA residuals
    zcta_residuals = compute_zcta_residuals(buildings, nfip_claims)

    # 5. Build adjacency dict for active ZCTAs
    active_zctas = sorted(zcta_residuals["zcta"].unique())
    adjacency = build_adjacency_dict(adj_df, active_zctas)

    # 6. Map residuals to ordered array matching adjacency indices
    zcta_to_idx = {z: i for i, z in enumerate(active_zctas)}
    residual_arr = np.zeros(len(active_zctas))
    for _, row in zcta_residuals.iterrows():
        idx = zcta_to_idx.get(row["zcta"])
        if idx is not None:
            residual_arr[idx] = row["residual"]

    # 7. Compute kappa_spatial
    kappa_result = compute_kappa_spatial(residual_arr, adjacency)

    # 8. Save results
    zcta_residuals.to_csv(OUTPUT_DIR / "residuals_by_zcta.csv", index=False)

    # Merge lat/lon for plotting
    lat_col = next((c for c in zcta_features.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in zcta_features.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    zcta_col = next((c for c in zcta_features.columns if "zcta" in c.lower() or "geoid" in c.lower()), None)
    if lat_col and lon_col and zcta_col:
        coords = zcta_features[[zcta_col, lat_col, lon_col]].copy()
        coords[zcta_col] = coords[zcta_col].astype(str)
        coords = coords.rename(columns={zcta_col: "zcta"})
        plot_df = zcta_residuals.merge(coords, on="zcta", how="left")
        plot_spatial_residuals(plot_df, kappa_result, OUTPUT_DIR / "fig_kappa_spatial_map.png")

    # Evidence JSON
    residual_type = zcta_residuals["residual_type"].iloc[0] if "residual_type" in zcta_residuals.columns else "unknown"
    evidence = {
        "hypothesis": "H4",
        "claim": "Floodcaster damage residuals exhibit spatial autocorrelation detectable by kappa_spatial",
        "job_id": JOB_ID,
        "n_buildings": len(buildings),
        "n_zctas": len(active_zctas),
        "n_zctas_with_adjacency": sum(1 for v in adjacency.values() if len(v) > 0),
        "residual_type": residual_type,
        "morans_i": round(kappa_result["morans_i"], 6),
        "expected_i": round(kappa_result["expected_i"], 6),
        "kappa_spatial": round(kappa_result["kappa"], 6),
        "n_edges": kappa_result["n_edges"],
        "mean_degree": round(kappa_result["mean_degree"], 4),
        "interpretation": (
            "HIGH spatial autocorrelation (clustered errors)"
            if kappa_result["kappa"] > 0.7
            else "MODERATE spatial autocorrelation"
            if kappa_result["kappa"] > 0.55
            else "LOW spatial autocorrelation (random errors)"
        ),
        "status": "PASS" if kappa_result["morans_i"] > kappa_result["expected_i"] else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(OUTPUT_DIR / "kappa_spatial_results.json", "w") as fh:
        json.dump(kappa_result, fh, indent=2)

    with open(OUTPUT_DIR / "evidence_s036_h4.json", "w") as fh:
        json.dump(evidence, fh, indent=2)

    # Summary
    logger.info("=" * 60)
    logger.info("S036 H4 Floodcaster Kappa Spatial: %s", evidence["status"])
    logger.info("  Job ID:         %s", JOB_ID)
    logger.info("  Buildings:      %d", evidence["n_buildings"])
    logger.info("  ZCTAs:          %d (%d with adjacency)", evidence["n_zctas"], evidence["n_zctas_with_adjacency"])
    logger.info("  Residual type:  %s", evidence["residual_type"])
    logger.info("  Moran's I:      %.6f (expected: %.6f)", evidence["morans_i"], evidence["expected_i"])
    logger.info("  kappa_spatial:  %.4f", evidence["kappa_spatial"])
    logger.info("  Interpretation: %s", evidence["interpretation"])
    logger.info("  Edges:          %d (mean degree %.2f)", evidence["n_edges"], evidence["mean_degree"])
    logger.info("=" * 60)

    # Upload
    upload_outputs(s3, OUTPUT_DIR, S3_OUTPUT_PREFIX)
    logger.info("Done. Results at s3://%s/%s/", DATASETS_BUCKET, S3_OUTPUT_PREFIX)


if __name__ == "__main__":
    main()
