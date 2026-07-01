#!/usr/bin/env python3
"""run_fetch_hydrology_3dep.py -- Compute HAND/TWI/SPI/GFI from local 3DEP tiles.

Replaces the STAC-based hydrology extraction (run_fetch_hydrology.py) with a
local-DEM approach that eliminates two problems:

  1. GDAL/curl ABI mismatch segfault (rasterio VSICURL on PyTorch base image)
  2. Sensory bias from 1 km centroid buffers (too small for watershed-scale
     HAND; systematically fails on coastal/flat ZCTAs)

This job downloads USGS 3DEP 1/3 arc-second (~10m) GeoTIFFs that are already
staged on S3, computes all four hydrology metrics per tile using floodcaster's
numpy-based hydrology module, and samples at ZCTA centroids.

Each 3DEP tile covers 1x1 degree (~111 km x 85 km at 30N), providing proper
watershed-scale drainage routing for HAND computation.

DEM source:
  s3://swarm-floodrsct-data/raw/dem/3dep/v1/{region}/*.tif

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_hydrology_{scenario}.parquet
  s3://swarm-floodrsct-data/results/s035/hydrology_3dep_{scenario}.json

Usage:
    python run_fetch_hydrology_3dep.py --scenario houston --upload
    python run_fetch_hydrology_3dep.py --scenario all --upload
    python run_fetch_hydrology_3dep.py --scenario houston --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client
from _s3_result import upload_json_result

# ---------------------------------------------------------------------------
# Scenario -> 3DEP region mapping
# ---------------------------------------------------------------------------
# 3DEP tiles are stored under raw/dem/3dep/v1/{region}/ on S3.
# Region names match the directory structure, not scenario names.
SCENARIO_DEM_REGION = {
    "houston": "houston",
    "new_orleans": "new_orleans",
    "nyc": "nyc",
    "riverside_coachella": "socal",
    "southwest_florida": "southwest_florida",
}

# Centroids source
STATIC_KEY = "raw/geocertdb2026/zcta_features_labels.parquet"
CROSSWALK_KEY = "raw/geocertdb2026/zcta_county_crosswalk.parquet"
HYDROLOGY_KEY_TEMPLATE = "processed/shared/zcta_hydrology_{scenario}.parquet"
OUTPUT_COLUMNS = ["zcta_id", "hand_mean_m", "twi_mean", "gfi_mean", "spi_mean"]

# County FIPS codes per scenario (from build_event_dataset.py)
SCENARIO_COUNTY_FIPS = {
    "houston": ["48201", "48157", "48339", "48167", "48039", "48071"],
    "new_orleans": ["22051", "22071", "22075", "22087", "22103"],
    "nyc": ["36061", "36047", "36081", "36005", "36085"],
    "riverside_coachella": ["06065", "06025"],
    "southwest_florida": ["12021", "12071", "12015", "12115", "12081", "12057", "12103"],
}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_read_parquet(s3, key: str):
    """Read parquet from S3; return None if missing."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        log.warning("Could not read %s: %s", key, e)
        return None


def s3_write_parquet(s3, df: pd.DataFrame, key: str) -> None:
    """Write DataFrame as parquet to S3."""
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    log.info("Uploaded %d rows x %d cols to s3://%s/%s",
             len(df), len(df.columns), BUCKET, key)


# ---------------------------------------------------------------------------
# Tile processing
# ---------------------------------------------------------------------------

def _tile_bounds(tif_path: str) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) from a GeoTIFF."""
    import rasterio
    with rasterio.open(tif_path) as src:
        b = src.bounds
        return (b.left, b.bottom, b.right, b.top)


def _process_tile(tif_path: str, centroids: pd.DataFrame,
                  stream_threshold: int = 1000) -> pd.DataFrame:
    """Compute HAND/TWI/SPI/GFI for a single 3DEP tile and sample at centroids.

    Args:
        tif_path: Path to local GeoTIFF.
        centroids: DataFrame with zcta_id, lat, lon.
        stream_threshold: Flow accumulation threshold for stream definition.

    Returns:
        DataFrame with zcta_id, hand_mean_m, twi_mean, gfi_mean, spi_mean
        for centroids within this tile.
    """
    import rasterio
    from floodcaster.hydrology import (
        compute_flow_accumulation,
        compute_hand,
        compute_twi,
        compute_spi,
        compute_gfi,
    )

    tile_name = Path(tif_path).stem

    # Get tile bounds and filter centroids
    west, south, east, north = _tile_bounds(tif_path)
    # Shrink bounds by 0.01 deg to avoid edge effects
    margin = 0.01
    in_tile = centroids[
        (centroids["lat"] >= south + margin)
        & (centroids["lat"] <= north - margin)
        & (centroids["lon"] >= west + margin)
        & (centroids["lon"] <= east - margin)
    ]

    if in_tile.empty:
        log.info("  Tile %s: 0 centroids in bounds, skipping", tile_name)
        return pd.DataFrame(columns=["zcta_id"] + OUTPUT_COLUMNS[1:])

    log.info("  Tile %s: %d centroids, computing flow accumulation...",
             tile_name, len(in_tile))

    # Compute shared flow accumulation (reused by all 4 metrics)
    t0 = time.time()
    flow_acc = compute_flow_accumulation(tif_path)
    log.info("  Tile %s: flow_acc done (%.1fs)", tile_name, time.time() - t0)

    # Compute each metric
    metrics = {}
    for name, fn, kwargs in [
        ("hand", compute_hand, {"flow_acc": flow_acc, "stream_threshold": stream_threshold}),
        ("twi", compute_twi, {"flow_acc": flow_acc}),
        ("spi", compute_spi, {"flow_acc": flow_acc}),
        ("gfi", compute_gfi, {"flow_acc": flow_acc, "stream_threshold": stream_threshold}),
    ]:
        try:
            t1 = time.time()
            arr = fn(tif_path, **kwargs)
            metrics[name] = arr
            log.info("  Tile %s: %s done (%.1fs), shape=%s",
                     tile_name, name, time.time() - t1, arr.shape)
        except Exception as e:
            log.warning("  Tile %s: %s failed: %s", tile_name, name, e)
            metrics[name] = None

    # Sample at centroids using rasterio
    with rasterio.open(tif_path) as src:
        transform = src.transform

    rows_list = []
    for _, row in in_tile.iterrows():
        zcta_id = row["zcta_id"]
        lat, lon = row["lat"], row["lon"]

        # Convert lat/lon to pixel coords
        col_px = int((lon - transform.c) / transform.a)
        row_px = int((lat - transform.f) / transform.e)

        # Sample in a 5x5 window (~50m at 10m resolution) for mean
        half = 2
        values = {}
        for metric_name, arr in metrics.items():
            if arr is None:
                values[metric_name] = np.nan
                continue
            r_lo = max(0, row_px - half)
            r_hi = min(arr.shape[0], row_px + half + 1)
            c_lo = max(0, col_px - half)
            c_hi = min(arr.shape[1], col_px + half + 1)
            window = arr[r_lo:r_hi, c_lo:c_hi]
            valid = window[np.isfinite(window)]
            values[metric_name] = float(np.nanmean(valid)) if len(valid) > 0 else np.nan

        rows_list.append({
            "zcta_id": zcta_id,
            "hand_mean_m": values.get("hand", np.nan),
            "twi_mean": values.get("twi", np.nan),
            "gfi_mean": values.get("gfi", np.nan),
            "spi_mean": values.get("spi", np.nan),
        })

    result = pd.DataFrame(rows_list)
    n_valid = result["hand_mean_m"].notna().sum()
    log.info("  Tile %s: %d/%d centroids with valid HAND",
             tile_name, n_valid, len(result))
    return result


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_scenario(s3, scenario: str, upload: bool, dry_run: bool) -> dict:
    """Run hydrology extraction for one scenario."""
    region = SCENARIO_DEM_REGION[scenario]
    t0 = time.time()

    # 1. List 3DEP tiles
    prefix = f"raw/dem/3dep/v1/{region}/"
    paginator = s3.get_paginator("list_objects_v2")
    tif_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tif"):
                tif_keys.append(obj["Key"])

    if not tif_keys:
        log.error("No 3DEP tiles found at s3://%s/%s", BUCKET, prefix)
        return {"scenario": scenario, "status": "ERROR", "error": "no_tiles"}

    log.info("%s: %d 3DEP tiles found", scenario, len(tif_keys))

    if dry_run:
        log.info("[DRY RUN] Would process %d tiles for %s", len(tif_keys), scenario)
        return {"scenario": scenario, "status": "DRY_RUN", "n_tiles": len(tif_keys)}

    # 2. Load centroids via county crosswalk (same logic as build_event_dataset.py)
    xwalk = s3_read_parquet(s3, CROSSWALK_KEY)
    if xwalk is None:
        return {"scenario": scenario, "status": "ERROR", "error": "no_crosswalk"}

    fips = SCENARIO_COUNTY_FIPS[scenario]
    scenario_zctas = xwalk[xwalk["county_fips"].isin(fips)]["zcta_id"].astype(str).tolist()
    log.info("%s: %d ZCTAs from %d counties", scenario, len(scenario_zctas), len(fips))

    static = s3_read_parquet(s3, STATIC_KEY)
    if static is None:
        return {"scenario": scenario, "status": "ERROR", "error": "no_static"}

    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    static = static[[zcta_col, lat_col, lon_col]].rename(
        columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"},
    )
    static["zcta_id"] = static["zcta_id"].astype(str)
    centroids = static[static["zcta_id"].isin(scenario_zctas)].dropna(subset=["lat", "lon"])

    log.info("%s: %d centroids loaded", scenario, len(centroids))

    # 3. Download all tiles, then process in parallel
    # Memory budget: 128 GB (ml.m5.8xlarge). Per tile peak: ~8 GB (DEM
    # 900 MB + flow_dir 900 MB + flow_acc 900 MB + 4 metric arrays 3.6 GB
    # + Python/numpy overhead). 128/8 = 16 safe workers, use 8 for headroom.
    n_workers = min(8, len(tif_keys), os.cpu_count() or 1)
    log.info("%s: processing %d tiles with %d parallel workers",
             scenario, len(tif_keys), n_workers)

    all_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download all tiles first (I/O bound, fast)
        local_paths = []
        for i, key in enumerate(tif_keys):
            fname = Path(key).name
            local_path = os.path.join(tmpdir, fname)
            log.info("Downloading tile %d/%d: %s", i + 1, len(tif_keys), fname)
            s3.download_file(BUCKET, key, local_path)
            local_paths.append(local_path)

        # Process tiles in parallel (CPU bound)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_process_tile, path, centroids): Path(path).stem
                for path in local_paths
            }
            for future in futures:
                tile_name = futures[future]
                try:
                    tile_result = future.result()
                    if not tile_result.empty:
                        all_results.append(tile_result)
                except Exception as e:
                    log.warning("Tile %s failed: %s", tile_name, e)

    if not all_results:
        log.error("%s: no results from any tile", scenario)
        return {"scenario": scenario, "status": "ERROR", "error": "no_results"}

    # 4. Merge results (a centroid might appear in overlapping tiles; take first)
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.drop_duplicates(subset=["zcta_id"], keep="first")
    combined = combined[OUTPUT_COLUMNS]

    n_total = len(centroids)
    n_with_hand = combined["hand_mean_m"].notna().sum()
    coverage = n_with_hand / n_total if n_total > 0 else 0

    log.info("%s: %d/%d ZCTAs with HAND data (%.1f%% coverage)",
             scenario, n_with_hand, n_total, coverage * 100)

    # 5. Upload
    if upload:
        out_key = HYDROLOGY_KEY_TEMPLATE.format(scenario=scenario)
        s3_write_parquet(s3, combined, out_key)

    elapsed = time.time() - t0
    result = {
        "scenario": scenario,
        "status": "OK",
        "n_tiles": len(tif_keys),
        "n_zctas": n_total,
        "n_with_hand": int(n_with_hand),
        "coverage_pct": round(coverage * 100, 1),
        "elapsed_s": round(elapsed, 1),
        "dem_source": "USGS_3DEP_1_3_arcsecond",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if upload:
        upload_json_result(
            s3, result,
            f"results/s035/hydrology_3dep_{scenario}.json",
            BUCKET,
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True,
                        choices=SCENARIOS + ["all"])
    parser.add_argument("--upload", action="store_true",
                        help="Upload results to S3")
    parser.add_argument("--dry-run", action="store_true",
                        help="List tiles only, no computation")
    args = parser.parse_args()

    s3 = get_s3_client()

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    results = []

    for scenario in scenarios:
        log.info("=" * 60)
        log.info("Processing scenario: %s", scenario)
        log.info("=" * 60)
        r = extract_scenario(s3, scenario, args.upload, args.dry_run)
        results.append(r)
        log.info("Result: %s", json.dumps(r, indent=2))

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    for r in results:
        log.info("  %s: %s (coverage=%.1f%%)",
                 r["scenario"], r["status"],
                 r.get("coverage_pct", 0))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
