#!/usr/bin/env python3
"""run_fetch_hydrology_owp_hand.py -- Fetch pre-computed HAND from NOAA OWP.

Replaces compute-from-scratch HAND with NOAA Office of Water Prediction's
hydrologically conditioned HAND rasters (Height Above Nearest Drainage).

OWP HAND is production-grade: hydrologically conditioned DEMs, proper
stream network delineation, and drainage enforcement. Computing HAND from
raw DEMs is reinventing the wheel and produces inferior results on flat
coastal terrain (Houston: 9.4% coverage from raw 3DEP).

Data source:
  s3://noaa-nws-owp-fim/hand_fim/hand_4_9_9_0/{HUC8}/branches/0/rem_zeroed_masked_0.tif
  - 10m resolution, EPSG:5070 (NAD83 / CONUS Albers)
  - int16 with nodata = -32768
  - Requester-pays bucket (nsc-swarm profile covers cost)

Workflow:
  1. Download WBD HUC8 polygons from USGS (one-time, cached on S3)
  2. Load ZCTA centroids for scenario counties
  3. Spatial join: ZCTA centroids -> HUC8 polygons
  4. Download OWP HAND rasters for matched HUC8s
  5. Mosaic into single raster covering scenario extent
  6. Run zonal_hydro_stats() from floodcaster.hydrology
  7. Merge with existing TWI/SPI/GFI from 3DEP pipeline
  8. Contract validation -> upload

TWI/SPI/GFI remain computed from 3DEP DEMs (DEM-only metrics, no drainage
dependency). Only HAND switches to OWP.

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_hydrology_{scenario}.parquet
  s3://swarm-floodrsct-data/results/s035/hydrology_owp_hand_{scenario}.json

Usage:
    python run_fetch_hydrology_owp_hand.py --scenario houston --upload
    python run_fetch_hydrology_owp_hand.py --scenario all --upload
    python run_fetch_hydrology_owp_hand.py --scenario houston --dry-run
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
# Constants
# ---------------------------------------------------------------------------

# OWP HAND S3 bucket (requester-pays)
OWP_BUCKET = "noaa-nws-owp-fim"
OWP_HAND_PREFIX = "hand_fim/hand_4_9_9_0"
OWP_HAND_FILE = "branches/0/rem_zeroed_masked_0.tif"

# WBD HUC8 shapefile cached on our S3 to avoid re-downloading from USGS
WBD_CACHE_KEY = "raw/reference/wbd_hu8_conus.parquet"

# WBD download URL (USGS National Map)
WBD_DOWNLOAD_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip"

# Centroids + crosswalk source (same as 3DEP pipeline)
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

def s3_read_parquet(s3, key: str, bucket: str = BUCKET):
    """Read parquet from S3; return None if missing."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        log.warning("Could not read s3://%s/%s: %s", bucket, key, e)
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
# WBD HUC8 loading
# ---------------------------------------------------------------------------

def _load_huc8_polygons(s3, tmpdir: str) -> "gpd.GeoDataFrame":
    """Load HUC8 polygons from cached parquet on S3, or download from USGS.

    Returns GeoDataFrame with columns: huc8 (str), geometry (Polygon).
    CRS: EPSG:4326 (for centroid join compatibility).
    """
    import geopandas as gpd

    # Try cached GeoParquet first (must use gpd.read_parquet, not pd)
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=WBD_CACHE_KEY)
        gdf = gpd.read_parquet(io.BytesIO(obj["Body"].read()))
        log.info("Loaded HUC8 polygons from cache: %d rows", len(gdf))
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return gdf
    except Exception as e:
        log.info("HUC8 cache miss (%s), will download from USGS", e)

    # Download WBD from USGS (large file ~1.5 GB zipped)
    log.info("HUC8 cache not found. Downloading WBD from USGS...")
    import urllib.request
    import zipfile

    zip_path = os.path.join(tmpdir, "wbd.zip")
    urllib.request.urlretrieve(WBD_DOWNLOAD_URL, zip_path)
    log.info("Downloaded WBD: %.1f MB", os.path.getsize(zip_path) / 1e6)

    # Extract and read HUC8 layer from GDB
    extract_dir = os.path.join(tmpdir, "wbd")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    # Find the .gdb directory
    gdb_path = None
    for root, dirs, files in os.walk(extract_dir):
        for d in dirs:
            if d.endswith(".gdb"):
                gdb_path = os.path.join(root, d)
                break
        if gdb_path:
            break

    if gdb_path is None:
        raise FileNotFoundError("WBD .gdb not found in downloaded archive")

    log.info("Reading WBDHU8 layer from %s", gdb_path)
    gdf = gpd.read_file(gdb_path, layer="WBDHU8")
    gdf = gdf[["huc8", "geometry"]].copy()
    gdf = gdf.to_crs("EPSG:4326")

    # Cache to S3 as parquet for future runs
    log.info("Caching HUC8 polygons to s3://%s/%s", BUCKET, WBD_CACHE_KEY)
    buf = BytesIO()
    gdf.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=WBD_CACHE_KEY, Body=buf.getvalue())

    return gdf


# ---------------------------------------------------------------------------
# OWP HAND download + mosaic
# ---------------------------------------------------------------------------

def _list_available_huc8s(s3, huc8_list: list[str]) -> list[str]:
    """Check which HUC8s have OWP HAND data available."""
    available = []
    for huc8 in huc8_list:
        key = f"{OWP_HAND_PREFIX}/{huc8}/{OWP_HAND_FILE}"
        try:
            s3.head_object(
                Bucket=OWP_BUCKET, Key=key,
                RequestPayer="requester",
            )
            available.append(huc8)
        except Exception:
            log.warning("OWP HAND not available for HUC8 %s", huc8)
    return available


def _download_owp_hand(s3, huc8: str, tmpdir: str) -> str:
    """Download a single OWP HAND raster. Returns local path."""
    key = f"{OWP_HAND_PREFIX}/{huc8}/{OWP_HAND_FILE}"
    local_path = os.path.join(tmpdir, f"hand_{huc8}.tif")
    log.info("Downloading OWP HAND for HUC8 %s ...", huc8)
    s3.download_file(
        OWP_BUCKET, key, local_path,
        ExtraArgs={"RequestPayer": "requester"},
    )
    size_mb = os.path.getsize(local_path) / 1e6
    log.info("  HUC8 %s: %.1f MB", huc8, size_mb)
    return local_path


def _mosaic_hand_rasters(tif_paths: list[str], output_path: str) -> str:
    """Mosaic multiple HAND rasters into a single GeoTIFF.

    OWP HAND rasters are EPSG:5070 (NAD83/CONUS Albers), int16.
    Mosaic output is float32 with nodata=NaN for compatibility with
    zonal_hydro_stats().

    Returns path to mosaic raster.
    """
    import rasterio
    from rasterio.merge import merge

    datasets = []
    for p in tif_paths:
        ds = rasterio.open(p)
        datasets.append(ds)

    log.info("Mosaicking %d HAND rasters...", len(datasets))
    mosaic_arr, mosaic_transform = merge(datasets)

    # Close input datasets
    for ds in datasets:
        ds.close()

    # OWP HAND is int16. nodata = 32767 (verified from tile headers).
    # Values are in millimeters. 0 = drainage/water cells.
    # Convert to float32 meters, set nodata + drainage to NaN.
    mosaic = mosaic_arr[0].astype(np.float32)
    mosaic[mosaic >= 32767] = np.nan  # nodata sentinel
    mosaic[mosaic == 0] = np.nan      # drainage/water cells
    mosaic = mosaic / 1000.0          # mm -> meters

    # Write mosaic
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": mosaic.shape[1],
        "height": mosaic.shape[0],
        "count": 1,
        "crs": datasets[0].crs if datasets else "EPSG:5070",
        "transform": mosaic_transform,
        "nodata": np.nan,
        "compress": "lzw",
    }
    # Re-read CRS from first input since datasets are closed
    with rasterio.open(tif_paths[0]) as src:
        profile["crs"] = src.crs

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic, 1)

    log.info("Mosaic written: %s (%d x %d, %.1f MB)",
             output_path, mosaic.shape[0], mosaic.shape[1],
             os.path.getsize(output_path) / 1e6)

    return output_path


# ---------------------------------------------------------------------------
# Zonal stats via floodcaster
# ---------------------------------------------------------------------------

def _owp_hand_zonal_stats(
    mosaic_path: str,
    zcta_gdf: "gpd.GeoDataFrame",
    zone_id_col: str = "zcta_id",
) -> pd.DataFrame:
    """Compute HAND zonal statistics per ZCTA using floodcaster.

    Returns DataFrame with columns: zcta_id, hand_mean, hand_median,
    hand_p10, hand_p90, hand_min, hand_max.
    """
    from floodcaster.hydrology import zonal_hydro_stats

    log.info("Computing HAND zonal stats for %d ZCTAs...", len(zcta_gdf))
    stats = zonal_hydro_stats(
        raster_path=mosaic_path,
        zones_gdf=zcta_gdf,
        zone_id_col=zone_id_col,
        prefix="hand_",
    )

    # zonal_hydro_stats returns DataFrame indexed by zone_id
    stats = stats.reset_index()
    log.info("HAND zonal stats: %d ZCTAs with data", stats["hand_mean"].notna().sum())
    return stats


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_scenario(s3, scenario: str, upload: bool, dry_run: bool) -> dict:
    """Run OWP HAND extraction for one scenario."""
    t0 = time.time()

    # 1. Load ZCTA centroids for this scenario
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

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2. Load HUC8 polygons
        import geopandas as gpd
        from shapely.geometry import Point

        huc8_gdf = _load_huc8_polygons(s3, tmpdir)
        log.info("HUC8 polygons loaded: %d watersheds", len(huc8_gdf))

        # 3. Spatial join: ZCTA centroids -> HUC8
        centroid_gdf = gpd.GeoDataFrame(
            centroids,
            geometry=[Point(lon, lat) for lon, lat in zip(centroids["lon"], centroids["lat"])],
            crs="EPSG:4326",
        )

        joined = gpd.sjoin(centroid_gdf, huc8_gdf, how="left", predicate="within")
        n_matched = joined["huc8"].notna().sum()
        n_unmatched = joined["huc8"].isna().sum()
        log.info("%s: %d/%d ZCTAs matched to HUC8 (%d unmatched)",
                 scenario, n_matched, len(joined), n_unmatched)

        if n_unmatched > 0:
            unmatched_ids = joined[joined["huc8"].isna()]["zcta_id"].tolist()
            log.warning("Unmatched ZCTAs (offshore or non-CONUS): %s",
                        unmatched_ids[:10])

        # Drop unmatched ZCTAs
        joined = joined.dropna(subset=["huc8"])
        unique_huc8s = sorted(joined["huc8"].unique().tolist())
        log.info("%s: %d unique HUC8 watersheds needed", scenario, len(unique_huc8s))

        if dry_run:
            log.info("[DRY RUN] Would download %d HUC8 HAND rasters for %s",
                     len(unique_huc8s), scenario)
            log.info("[DRY RUN] HUC8s: %s", unique_huc8s)
            return {
                "scenario": scenario,
                "status": "DRY_RUN",
                "n_zctas": len(centroids),
                "n_matched": int(n_matched),
                "n_huc8s": len(unique_huc8s),
                "huc8s": unique_huc8s,
            }

        # 4. Check OWP HAND availability
        available_huc8s = _list_available_huc8s(s3, unique_huc8s)
        log.info("%s: %d/%d HUC8s have OWP HAND data",
                 scenario, len(available_huc8s), len(unique_huc8s))

        if not available_huc8s:
            return {"scenario": scenario, "status": "ERROR",
                    "error": "no_owp_hand_available",
                    "requested_huc8s": unique_huc8s}

        missing_huc8s = set(unique_huc8s) - set(available_huc8s)
        if missing_huc8s:
            log.warning("Missing OWP HAND for HUC8s: %s", sorted(missing_huc8s))
            # Continue with available data -- missing HUC8 ZCTAs get NaN

        # 5. Download OWP HAND rasters (parallel -- I/O-bound)
        from concurrent.futures import ThreadPoolExecutor

        hand_paths = []

        def _download_one(huc8):
            return _download_owp_hand(s3, huc8, tmpdir)

        with ThreadPoolExecutor(max_workers=min(4, len(available_huc8s))) as pool:
            futures = {pool.submit(_download_one, h): h for h in available_huc8s}
            for future in futures:
                huc8 = futures[future]
                try:
                    path = future.result()
                    hand_paths.append(path)
                except Exception as e:
                    log.warning("Failed to download HAND for HUC8 %s: %s", huc8, e)

        if not hand_paths:
            return {"scenario": scenario, "status": "ERROR",
                    "error": "all_downloads_failed"}

        # 6. Mosaic HAND rasters
        mosaic_path = os.path.join(tmpdir, f"hand_mosaic_{scenario}.tif")
        _mosaic_hand_rasters(hand_paths, mosaic_path)

        # 7. Build ZCTA polygon GeoDataFrame for zonal stats
        # We need actual ZCTA polygons, not just centroids.
        # Load from Census TIGER/Line via S3 cache or download.
        zcta_polygons = _load_zcta_polygons(s3, scenario_zctas, tmpdir)

        if zcta_polygons is not None and not zcta_polygons.empty:
            # Use zonal_hydro_stats with polygons (preferred -- captures
            # full ZCTA extent, not just centroid neighborhood)
            hand_stats = _owp_hand_zonal_stats(
                mosaic_path, zcta_polygons, zone_id_col="zcta_id",
            )
        else:
            # Fallback: sample at centroids with buffer (like 3DEP pipeline)
            log.warning("ZCTA polygons not available, falling back to centroid sampling")
            hand_stats = _sample_hand_at_centroids(mosaic_path, centroids)

        # Rename hand_mean -> hand_mean_m for output compatibility
        if "hand_mean" in hand_stats.columns:
            hand_stats = hand_stats.rename(columns={"hand_mean": "hand_mean_m"})

    # 8. Load existing TWI/SPI/GFI from 3DEP pipeline (if available)
    existing_key = HYDROLOGY_KEY_TEMPLATE.format(scenario=scenario)
    existing = s3_read_parquet(s3, existing_key)

    if existing is not None and "twi_mean" in existing.columns:
        # Merge: take HAND from OWP, TWI/SPI/GFI from existing 3DEP
        log.info("Merging OWP HAND with existing TWI/SPI/GFI from 3DEP")
        hand_stats["zcta_id"] = hand_stats["zcta_id"].astype(str)
        existing["zcta_id"] = existing["zcta_id"].astype(str)

        combined = hand_stats[["zcta_id", "hand_mean_m"]].merge(
            existing[["zcta_id", "twi_mean", "gfi_mean", "spi_mean"]],
            on="zcta_id", how="outer",
        )
    else:
        # No existing data -- HAND only (TWI/SPI/GFI will be NaN)
        log.warning("No existing TWI/SPI/GFI found for %s", scenario)
        combined = hand_stats[["zcta_id", "hand_mean_m"]].copy()
        combined["twi_mean"] = np.nan
        combined["gfi_mean"] = np.nan
        combined["spi_mean"] = np.nan

    combined = combined[OUTPUT_COLUMNS]

    n_total = len(centroids)
    n_with_hand = combined["hand_mean_m"].notna().sum()
    coverage = n_with_hand / n_total if n_total > 0 else 0

    log.info("%s: %d/%d ZCTAs with HAND data (%.1f%% coverage)",
             scenario, n_with_hand, n_total, coverage * 100)

    # 9. Contract validation BEFORE upload
    from _validate_contract import COVERAGE_THRESHOLDS, Status

    HYDRO_BOUNDS = {
        "hand_mean_m": (0, 200),
        "twi_mean": (0, 30),
        "gfi_mean": (-5, 10),
        "spi_mean": (-15, 20),
    }

    contract_fails = []
    for col in ["hand_mean_m", "twi_mean", "gfi_mean", "spi_mean"]:
        if col not in combined.columns:
            continue
        non_null = combined[col].notna().mean()
        threshold = COVERAGE_THRESHOLDS.get(col, 0.50)
        if non_null < threshold:
            msg = "%s: %.1f%% non-null < %.0f%% threshold" % (
                col, non_null * 100, threshold * 100)
            log.error("CONTRACT FAIL: %s", msg)
            contract_fails.append(msg)
        else:
            log.info("CONTRACT PASS: %s %.1f%% non-null >= %.0f%%",
                     col, non_null * 100, threshold * 100)

        lo, hi = HYDRO_BOUNDS.get(col, (None, None))
        if lo is not None:
            vals = combined[col].dropna()
            if len(vals) > 0:
                below = (vals < lo).sum()
                above = (vals > hi).sum()
                if below > 0 or above > 0:
                    msg = "%s: %d below %.0f, %d above %.0f (range: %.2f to %.2f)" % (
                        col, below, lo, above, hi, vals.min(), vals.max())
                    log.warning("CONTRACT WARN: %s", msg)

    if contract_fails:
        log.error("%s: CONTRACT BLOCKED -- %d coverage failures. "
                  "Data NOT uploaded.", scenario, len(contract_fails))
        elapsed = time.time() - t0
        return {
            "scenario": scenario,
            "status": "CONTRACT_FAIL",
            "contract_fails": contract_fails,
            "n_zctas": n_total,
            "n_with_hand": int(n_with_hand),
            "coverage_pct": round(coverage * 100, 1),
            "elapsed_s": round(elapsed, 1),
        }

    # 10. Upload (only after contract passes)
    if upload:
        out_key = HYDROLOGY_KEY_TEMPLATE.format(scenario=scenario)
        s3_write_parquet(s3, combined, out_key)

    elapsed = time.time() - t0
    result = {
        "scenario": scenario,
        "status": "OK",
        "hand_source": "NOAA_OWP_HAND_4.9.9.0",
        "twi_spi_gfi_source": "USGS_3DEP_1_3_arcsecond",
        "n_huc8s": len(available_huc8s),
        "n_zctas": n_total,
        "n_with_hand": int(n_with_hand),
        "coverage_pct": round(coverage * 100, 1),
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if upload:
        upload_json_result(
            s3, BUCKET,
            f"results/s035/hydrology_owp_hand_{scenario}.json",
            result,
        )

    return result


# ---------------------------------------------------------------------------
# ZCTA polygon loading
# ---------------------------------------------------------------------------

def _load_zcta_polygons(
    s3, zcta_ids: list[str], tmpdir: str
) -> "gpd.GeoDataFrame | None":
    """Load ZCTA polygon boundaries from S3 cache or Census TIGER/Line.

    Returns GeoDataFrame with zcta_id and geometry columns, or None.
    """
    import geopandas as gpd

    # Try cached ZCTA boundaries on S3
    cache_key = "raw/reference/zcta_boundaries_2020.parquet"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=cache_key)
        gdf = gpd.read_parquet(io.BytesIO(obj["Body"].read()))
        log.info("Loaded ZCTA polygons from cache: %d rows", len(gdf))

        # Normalize column names
        zcta_col = next((c for c in gdf.columns if "zcta" in c.lower() or "geoid" in c.lower()), None)
        if zcta_col and zcta_col != "zcta_id":
            gdf = gdf.rename(columns={zcta_col: "zcta_id"})
        gdf["zcta_id"] = gdf["zcta_id"].astype(str)
        gdf = gdf[gdf["zcta_id"].isin(zcta_ids)]
        return gdf[["zcta_id", "geometry"]]
    except Exception as e:
        log.warning("ZCTA polygon cache not found: %s", e)

    # Download from Census TIGER/Line (2020 ZCTAs)
    try:
        url = "https://www2.census.gov/geo/tiger/TIGER2020/ZCTA520/tl_2020_us_zcta520.zip"
        zip_path = os.path.join(tmpdir, "zcta.zip")
        log.info("Downloading ZCTA boundaries from Census...")
        import urllib.request
        urllib.request.urlretrieve(url, zip_path)
        gdf = gpd.read_file(f"zip://{zip_path}")
        log.info("Downloaded ZCTA boundaries: %d polygons", len(gdf))

        # Normalize
        zcta_col = next((c for c in gdf.columns if "ZCTA" in c or "GEOID" in c), None)
        if zcta_col:
            gdf = gdf.rename(columns={zcta_col: "zcta_id"})
        gdf["zcta_id"] = gdf["zcta_id"].astype(str)

        # Cache the full set to S3 for future runs
        log.info("Caching ZCTA polygons to S3...")
        buf = BytesIO()
        gdf[["zcta_id", "geometry"]].to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=cache_key, Body=buf.getvalue())

        gdf = gdf[gdf["zcta_id"].isin(zcta_ids)]
        return gdf[["zcta_id", "geometry"]]
    except Exception as e:
        log.warning("Failed to download ZCTA boundaries: %s", e)
        return None


def _sample_hand_at_centroids(
    mosaic_path: str, centroids: pd.DataFrame
) -> pd.DataFrame:
    """Fallback: sample HAND raster at ZCTA centroids with buffer.

    Used when ZCTA polygon boundaries are not available.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform

    rows_list = []
    with rasterio.open(mosaic_path) as src:
        raster_crs = src.crs
        transform = src.transform
        data = src.read(1)

        for _, row in centroids.iterrows():
            # Transform centroid from EPSG:4326 to raster CRS (EPSG:5070)
            xs, ys = warp_transform("EPSG:4326", raster_crs, [row["lon"]], [row["lat"]])
            col_px = int((xs[0] - transform.c) / transform.a)
            row_px = int((ys[0] - transform.f) / transform.e)

            # Sample with 500px buffer (~5km at 10m resolution)
            half = 500
            r_lo = max(0, row_px - half)
            r_hi = min(data.shape[0], row_px + half + 1)
            c_lo = max(0, col_px - half)
            c_hi = min(data.shape[1], col_px + half + 1)

            if r_lo >= r_hi or c_lo >= c_hi:
                rows_list.append({"zcta_id": row["zcta_id"], "hand_mean_m": np.nan})
                continue

            window = data[r_lo:r_hi, c_lo:c_hi].astype(np.float32)
            valid = window[np.isfinite(window) & (window > 0)]
            hand_val = float(np.nanmean(valid)) if len(valid) > 0 else np.nan
            rows_list.append({"zcta_id": row["zcta_id"], "hand_mean_m": hand_val})

    return pd.DataFrame(rows_list)


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
                        help="Show HUC8 mapping only, no downloads")
    args = parser.parse_args()

    s3 = get_s3_client()

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    results = []

    for scenario in scenarios:
        log.info("=" * 60)
        log.info("Processing scenario: %s (OWP HAND)", scenario)
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
