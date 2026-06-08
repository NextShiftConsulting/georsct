#!/usr/bin/env python3
"""
run_fast_zcta.py -- Phase 7a: FEMA FAST depth-damage features per ZCTA.

Loads pre-fetched NSI 2.0 structures from S3, runs Hazus depth-damage via
sphere (through floodcaster.engine), aggregates to ZCTA level via
floodcaster.aggregation.

Data dependencies (all on S3):
  - raw/nsi/v2/{scenario}_structures.parquet   (from launch_fetch_nsi_structures.py)
  - raw/floodsimbench/6hr_max/{TILE}_{mm}mm_MaxDepth.tif  (Houston, NYC)
  - raw/noaa_slosh/mom_national/us_Category{N}_MOM_Inundation_HIGH.tif  (SW FL)
  - raw/geocertdb2026/zcta5_boundaries.parquet  (from stage_zcta_geometry.py)

Outputs:
  processed/{scenario}/{scenario}_fast_zcta_{return_period}.parquet

Usage:
    python run_fast_zcta.py --scenario houston --return-period 100yr --upload
    python run_fast_zcta.py --scenario houston --all-return-periods --upload
"""

import argparse
import io
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client

# Sphere types via georsct.flood.engine (not reimplementing -- per DOE constraint)
from georsct.flood.engine import (
    FastBuildings,
    HazusFloodAnalysis,
    DefaultFloodVulnerability,
    SingleValueRaster,
)
from georsct.flood.aggregation import aggregate_by_zcta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_PREFIX = "processed"

# FloodSimBench return period -> rainfall mm (scenario-specific)
RETURN_PERIOD_MM = {
    "houston": {
        "1yr": 48, "2yr": 57, "5yr": 70, "10yr": 82, "25yr": 98,
        "50yr": 110, "100yr": 123, "200yr": 138, "500yr": 162, "1000yr": 181,
    },
    "nyc": {
        "1yr": 48, "2yr": 58, "5yr": 65, "10yr": 73,
    },
}

# Scenario -> FloodSimBench tile prefix
SCENARIO_TILE_PREFIX = {
    "houston": "HOU",
    "nyc": "NYC",
}

# SW Florida uses SLOSH MOM instead of FloodSimBench
SLOSH_CATEGORIES = {
    "cat1": 1, "cat2": 2, "cat3": 3, "cat4": 4, "cat5": 5,
}

# Scenarios with FAST support (Riverside excluded -- no depth data)
FAST_SCENARIOS = ["houston", "nyc", "southwest_florida"]

# NSI -> sphere field mapping (mirrors floodcaster.nsi_sources._map_nsi_to_sphere)
SPHERE_OVERRIDES = {
    "id": "id",
    "occupancy_type": "occupancy_type",
    "first_floor_height": "first_floor_height",
    "foundation_type": "foundation_type",
    "number_stories": "number_stories",
    "area": "area",
    "building_cost": "building_cost",
    "content_cost": "content_cost",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_nsi(s3, scenario: str) -> gpd.GeoDataFrame:
    """Load pre-fetched NSI structures from S3."""
    key = f"raw/nsi/v2/{scenario}_structures.parquet"
    log.info("Loading NSI: s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    gdf = gpd.read_parquet(io.BytesIO(resp["Body"].read()))
    log.info("NSI: %d structures, CRS=%s", len(gdf), gdf.crs)
    return gdf


def map_nsi_to_sphere(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map NSI column names to sphere FastBuildings field aliases."""
    out = gdf.copy()
    out["id"] = out["fd_id"]
    out["occupancy_type"] = out["occtype"]
    out["first_floor_height"] = pd.to_numeric(out["found_ht"], errors="coerce").fillna(0.0)
    out["foundation_type"] = pd.to_numeric(out["found_type"], errors="coerce").fillna(7).astype(int)
    out["number_stories"] = pd.to_numeric(out["num_story"], errors="coerce").fillna(1).astype(int).clip(lower=1)
    out["area"] = pd.to_numeric(out["sqft"], errors="coerce").clip(lower=100)
    out["building_cost"] = pd.to_numeric(out["val_struct"], errors="coerce").fillna(0.0)
    out["content_cost"] = pd.to_numeric(out["val_cont"], errors="coerce").fillna(0.0)
    out["longitude"] = out.geometry.x
    out["latitude"] = out.geometry.y
    return out


def load_zcta_boundaries(s3) -> gpd.GeoDataFrame:
    """Load ZCTA5 polygon boundaries from S3."""
    key = "raw/geocertdb2026/zcta5_boundaries.parquet"
    log.info("Loading ZCTA boundaries: s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    gdf = gpd.read_parquet(io.BytesIO(resp["Body"].read()))
    # Rename to match floodcaster aggregation convention
    if "zcta_id" not in gdf.columns:
        zcta_col = next((c for c in gdf.columns if "zcta" in c.lower()), None)
        if zcta_col:
            gdf = gdf.rename(columns={zcta_col: "zcta_id"})
    # Ensure column name matches aggregation's zcta_id_col default
    if "zcta" not in gdf.columns and "zcta_id" in gdf.columns:
        gdf["zcta"] = gdf["zcta_id"]
    log.info("ZCTA boundaries: %d polygons", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Raster discovery and download
# ---------------------------------------------------------------------------

def find_floodsimbench_rasters(s3, scenario: str, return_period: str) -> list[str]:
    """Find all FloodSimBench tiles for a scenario + return period."""
    tile_prefix = SCENARIO_TILE_PREFIX.get(scenario)
    if not tile_prefix:
        return []

    rain_mm = RETURN_PERIOD_MM.get(scenario, {}).get(return_period)
    if rain_mm is None:
        log.error("Unknown return period: %s", return_period)
        return []

    s3_prefix = "raw/floodsimbench/6hr_max/"
    tiles = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key.split("/")[-1]
            if name.startswith(tile_prefix) and f"{rain_mm}mm" in name:
                tiles.append(key)

    log.info("FloodSimBench %s/%s: %d tiles found", scenario, return_period, len(tiles))
    return sorted(tiles)


def find_slosh_rasters(s3, category: str) -> list[str]:
    """Find SLOSH MOM raster for a hurricane category."""
    cat_num = SLOSH_CATEGORIES.get(category)
    if cat_num is None:
        return []

    key = f"raw/noaa_slosh/mom_national/us_Category{cat_num}_MOM_Inundation_HIGH.tif"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return [key]
    except Exception:
        log.warning("SLOSH raster not found: %s", key)
        return []



def download_and_merge_rasters(s3, keys: list[str], tmp_dir: str) -> str:
    """Download raster tiles from S3 and merge into single GeoTIFF."""
    if not keys:
        raise ValueError("No raster keys to download")

    local_paths = []
    for key in keys:
        local = Path(tmp_dir) / Path(key).name
        s3.download_file(BUCKET, key, str(local))
        local_paths.append(str(local))
        log.info("Downloaded: %s", key)

    if len(local_paths) == 1:
        return local_paths[0]

    # Manual numpy merge to avoid rasterio.merge's negative-pixel-height
    # rejection (rasterio >= 1.4).  Reads each tile, normalizes to north-up,
    # and paints into a union-bounds canvas.
    log.info("Merging %d tiles (numpy)...", len(local_paths))

    datasets = [rasterio.open(p) for p in local_paths]
    try:
        # Compute union bounds (bounds are always ordered regardless of pixel sign)
        all_bounds = [ds.bounds for ds in datasets]
        min_left = min(b.left for b in all_bounds)
        min_bottom = min(b.bottom for b in all_bounds)
        max_right = max(b.right for b in all_bounds)
        max_top = max(b.top for b in all_bounds)

        res_x = abs(datasets[0].transform.a)
        res_y = abs(datasets[0].transform.e)
        out_width = int(round((max_right - min_left) / res_x))
        out_height = int(round((max_top - min_bottom) / res_y))

        # North-up transform (standard: negative pixel height)
        out_transform = rasterio.Affine(res_x, 0, min_left, 0, -res_y, max_top)

        nodata = datasets[0].nodata if datasets[0].nodata is not None else 0
        mosaic = np.full(
            (datasets[0].count, out_height, out_width),
            nodata, dtype=datasets[0].dtypes[0],
        )

        for ds in datasets:
            data = ds.read()
            # Normalize to north-up if needed
            if ds.transform.e > 0:
                data = data[:, ::-1, :]

            # Pixel offsets into the output canvas
            col_off = int(round((ds.bounds.left - min_left) / res_x))
            row_off = int(round((max_top - ds.bounds.top) / res_y))

            h = min(data.shape[1], out_height - row_off)
            w = min(data.shape[2], out_width - col_off)
            if h <= 0 or w <= 0:
                continue

            tile = data[:, :h, :w]
            if nodata == 0:
                mask = tile != 0
            else:
                mask = ~np.isnan(tile) if np.issubdtype(tile.dtype, np.floating) else (tile != nodata)
            mosaic[:, row_off:row_off+h, col_off:col_off+w] = np.where(
                mask, tile, mosaic[:, row_off:row_off+h, col_off:col_off+w],
            )

        out_meta = datasets[0].meta.copy()
        out_meta.update({
            "height": out_height,
            "width": out_width,
            "transform": out_transform,
            "driver": "GTiff",
        })
    finally:
        for ds in datasets:
            ds.close()

    merged_path = str(Path(tmp_dir) / "merged_depth.tif")
    with rasterio.open(merged_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    log.info("Merged raster: %s (%d x %d)", merged_path, mosaic.shape[2], mosaic.shape[1])
    return merged_path


# ---------------------------------------------------------------------------
# FAST analysis
# ---------------------------------------------------------------------------

def run_fast_for_raster(
    nsi_gdf: gpd.GeoDataFrame,
    raster_path: str,
    zcta_gdf: gpd.GeoDataFrame,
    flood_type: str = "R",
) -> pd.DataFrame:
    """Run Hazus depth-damage on NSI structures, aggregate to ZCTA.

    Uses sphere types via floodcaster.engine (not reimplementing).
    Uses floodcaster.aggregation for ZCTA rollup (not reimplementing).
    """
    # Write NSI to temp CSV for sphere's FastBuildings loader
    csv_path = Path(raster_path).parent / "_fast_nsi_buildings.csv"
    nsi_gdf.to_csv(csv_path, index=False)

    try:
        buildings = FastBuildings(str(csv_path), overrides=SPHERE_OVERRIDES)

        raster = SingleValueRaster(raster_path)
        vulnerability = DefaultFloodVulnerability(buildings, flood_type)
        analysis = HazusFloodAnalysis(buildings, vulnerability, raster)
        analysis.calculate_losses()

        result_gdf = buildings.gdf
        loss_col = buildings.fields.get_field_name("building_loss")
        total_loss = result_gdf[loss_col].sum(skipna=True) if loss_col in result_gdf.columns else 0
        n_damaged = (result_gdf[loss_col] > 0).sum() if loss_col in result_gdf.columns else 0
        log.info(
            "Hazus complete: %d structures, %d damaged, total loss $%.0f",
            len(result_gdf), n_damaged, total_loss,
        )

        # Kill rule: all zeros
        if total_loss == 0:
            log.warning("KILL RULE: total loss is $0 -- depth raster may not overlap NSI structures")

        # Aggregate to ZCTA using floodcaster.aggregation
        fast_df = aggregate_by_zcta(result_gdf, zcta_gdf)
        log.info("ZCTA aggregation: %d ZCTAs with data", len(fast_df))
        return fast_df

    finally:
        if csv_path.exists():
            csv_path.unlink()


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_result(df: pd.DataFrame, s3, scenario: str, period_label: str) -> None:
    """Upload FAST ZCTA parquet to S3."""
    key = f"{RESULTS_PREFIX}/{scenario}/{scenario}_fast_zcta_{period_label}.parquet"
    local = f"/tmp/{scenario}_fast_zcta_{period_label}.parquet"
    df.to_parquet(local, index=True)  # zcta is the index
    s3.upload_file(local, BUCKET, key)
    log.info("Uploaded s3://%s/%s (%d ZCTAs)", BUCKET, key, len(df))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_return_periods(scenario: str) -> list[str]:
    """Return available return periods for a scenario."""
    if scenario == "southwest_florida":
        return list(SLOSH_CATEGORIES.keys())
    return list(RETURN_PERIOD_MM.get(scenario, {}).keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7a: compute FEMA FAST depth-damage features per ZCTA"
    )
    parser.add_argument("--scenario", required=True, choices=FAST_SCENARIOS)
    parser.add_argument("--return-period",
                        help="Single return period (e.g. 100yr, cat3)")
    parser.add_argument("--all-return-periods", action="store_true",
                        help="Run all available return periods")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    if not args.return_period and not args.all_return_periods:
        parser.error("Specify --return-period or --all-return-periods")

    scenario = args.scenario
    s3 = get_s3_client()

    print(f"\n{'='*60}")
    print(f"  S035 PHASE 7a: FAST ZCTA -- {scenario.upper()}")
    print(f"{'='*60}\n")

    # Load NSI structures
    nsi_gdf = load_nsi(s3, scenario)
    nsi_gdf = map_nsi_to_sphere(nsi_gdf)

    # Load ZCTA boundaries
    zcta_gdf = load_zcta_boundaries(s3)

    # Determine flood type
    flood_type = "C" if scenario == "southwest_florida" else "R"

    # Determine return periods to process
    if args.all_return_periods:
        periods = get_return_periods(scenario)
    else:
        periods = [args.return_period]

    results_summary = []

    for period in periods:
        log.info("--- Processing %s / %s ---", scenario, period)

        # Find rasters
        if scenario == "southwest_florida":
            raster_keys = find_slosh_rasters(s3, period)
        else:
            raster_keys = find_floodsimbench_rasters(s3, scenario, period)

        if not raster_keys:
            log.warning("No rasters found for %s/%s -- skipping", scenario, period)
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path = download_and_merge_rasters(s3, raster_keys, tmp_dir)
            fast_df = run_fast_for_raster(nsi_gdf, raster_path, zcta_gdf, flood_type)

        if fast_df.empty:
            log.warning("Empty FAST result for %s/%s", scenario, period)
            continue

        results_summary.append({
            "scenario": scenario,
            "return_period": period,
            "n_zctas": len(fast_df),
            "total_loss": float(fast_df["fast_total_loss_usd"].sum()),
            "pct_damaged": float(fast_df["fast_pct_damaged"].mean()),
            "n_structures_total": int(fast_df["fast_n_structures"].sum()),
        })

        if args.upload:
            upload_result(fast_df, s3, scenario, period)
        else:
            local = f"/tmp/{scenario}_fast_zcta_{period}.parquet"
            fast_df.to_parquet(local, index=True)
            log.info("Saved locally: %s", local)

    # Summary
    print(f"\n--- FAST RESULTS SUMMARY ---")
    for r in results_summary:
        print(f"  {r['scenario']:20s} {r['return_period']:8s} "
              f"ZCTAs={r['n_zctas']:5d}  "
              f"structures={r['n_structures_total']:8d}  "
              f"total_loss=${r['total_loss']:,.0f}  "
              f"pct_damaged={r['pct_damaged']:.1%}")

    # Upload summary metadata
    if args.upload and results_summary:
        meta = {
            "experiment": "s035-model-ladder",
            "phase": "fast_zcta",
            "scenario": scenario,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results_summary,
        }
        meta_key = f"{RESULTS_PREFIX}/{scenario}/{scenario}_fast_zcta_meta.json"
        s3.put_object(
            Bucket=BUCKET, Key=meta_key,
            Body=json.dumps(meta, indent=2).encode(),
            ContentType="application/json",
        )
        log.info("Uploaded s3://%s/%s", BUCKET, meta_key)

    log.info("run_fast_zcta complete")


if __name__ == "__main__":
    main()
