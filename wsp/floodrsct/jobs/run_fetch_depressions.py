#!/usr/bin/env python3
"""run_fetch_depressions.py -- Depression delineation from Copernicus DEM.

Runs depression delineation on Copernicus 30m DEM tiles for one scenario's
ZCTA universe. Uses whitebox-tools directly (fill_depressions + depth_in_sink)
with rasterio for I/O and scipy for connected-component labeling.

Outputs per-ZCTA statistics:
  - depression_count: number of depressions intersecting the ZCTA
  - depression_volume_m3: total depression volume (flood storage capacity)
  - max_depression_depth_m: deepest depression in the ZCTA
  - depression_area_m2: total depressed area

These are scenario-independent (topographic, not event-based), so results
live in processed/shared/ and are built once. Cache-first: ZCTAs already in
the shared parquet are not re-processed.

Source:
  Copernicus GLO-30 DEM via Planetary Computer STAC

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_depressions.parquet
  s3://swarm-floodrsct-data/results/s035/depressions_extraction_{scenario}.json

Usage:
    python run_fetch_depressions.py --scenario houston --upload
    python run_fetch_depressions.py --scenario houston --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
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

# S3 infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client
from _s3_result import upload_json_result

# Output keys
DEPRESSIONS_CACHE_KEY = "processed/shared/zcta_depressions.parquet"
OUTPUT_COLUMNS = [
    "zcta_id", "depression_count", "depression_volume_m3",
    "max_depression_depth_m", "depression_area_m2",
]

# ZCTA boundaries (GeoParquet with polygon geometry)
ZCTA_BOUNDARIES_KEYS = [
    "raw/geocertdb2026/zcta_boundaries_5070.parquet",
    "raw/geocertdb2026/zcta_boundaries.parquet",
    "raw/geocertdb2026/zcta5_boundaries.parquet",
]

# Scenario event_features keys
SCENARIO_EVENT_KEYS = {
    "houston": "processed/houston/houston_event_features.parquet",
    "new_orleans": "processed/new_orleans/no_event_features.parquet",
    "nyc": "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
}

# Depression delineation parameters
MIN_SINK_SIZE = 100       # minimum sink area in pixels (~100 * 30m^2 = 90000 m2)
MIN_DEPRESSION_DEPTH = 0.5  # minimum depth in meters


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
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    log.info("Uploaded %d rows x %d cols to s3://%s/%s",
             len(df), len(df.columns), BUCKET, key)


# ---------------------------------------------------------------------------
# Load ZCTA universe + boundaries
# ---------------------------------------------------------------------------

def load_scenario_zcta_ids(s3, scenario: str) -> list[str]:
    """Load unique ZCTA IDs for one scenario from event_features."""
    key = SCENARIO_EVENT_KEYS[scenario]
    df = s3_read_parquet(s3, key)
    if df is None:
        raise RuntimeError(f"event_features not found: s3://{BUCKET}/{key}")
    zcta_col = next((c for c in df.columns if "zcta" in c.lower()), None)
    if zcta_col is None:
        raise RuntimeError(f"No zcta column in {key}")
    ids = sorted(df[zcta_col].astype(str).unique().tolist())
    log.info("%s: %d unique ZCTAs", scenario, len(ids))
    return ids


def load_zcta_boundaries(s3, zcta_ids: list[str]):
    """Load ZCTA polygon boundaries as GeoDataFrame, projected to EPSG:5070."""
    import geopandas as gpd

    for key in ZCTA_BOUNDARIES_KEYS:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            gdf = gpd.read_parquet(io.BytesIO(obj["Body"].read()))
            if "zcta_id" in gdf.columns:
                gdf["zcta_id"] = gdf["zcta_id"].astype(str)
            elif "ZCTA5CE20" in gdf.columns:
                gdf = gdf.rename(columns={"ZCTA5CE20": "zcta_id"})
                gdf["zcta_id"] = gdf["zcta_id"].astype(str)
            else:
                zcta_col = next((c for c in gdf.columns if "zcta" in c.lower()), None)
                if zcta_col:
                    gdf = gdf.rename(columns={zcta_col: "zcta_id"})
                    gdf["zcta_id"] = gdf["zcta_id"].astype(str)
                else:
                    continue

            gdf = gdf[gdf["zcta_id"].isin(zcta_ids)]
            if gdf.crs is None or gdf.crs.to_epsg() != 5070:
                gdf = gdf.to_crs("EPSG:5070")
            log.info("Loaded %d ZCTA boundaries from %s (EPSG:5070)", len(gdf), key)
            return gdf[["zcta_id", "geometry"]]
        except Exception as e:
            log.warning("Could not load %s: %s", key, e)
            continue

    raise RuntimeError("No ZCTA boundary file found on S3")


# ---------------------------------------------------------------------------
# Cache-first pattern
# ---------------------------------------------------------------------------

def load_cache(s3) -> pd.DataFrame | None:
    """Load existing depressions cache from S3."""
    return s3_read_parquet(s3, DEPRESSIONS_CACHE_KEY)


def merge_and_write_cache(s3, existing: pd.DataFrame | None,
                          new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append new rows to cache, dedupe on zcta_id, write back."""
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()
    combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")
    for col in OUTPUT_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[OUTPUT_COLUMNS]
    s3_write_parquet(s3, combined, DEPRESSIONS_CACHE_KEY)
    return combined


# ---------------------------------------------------------------------------
# DEM fetching — one tile at a time, merge to disk
# ---------------------------------------------------------------------------

def fetch_dem_for_bbox(bbox: tuple[float, float, float, float],
                       dst_path: str) -> str | None:
    """Fetch Copernicus GLO-30 DEM, writing merged result to disk.

    Streams tiles one at a time through rasterio.merge to avoid
    loading all tiles into memory simultaneously.
    """
    try:
        import planetary_computer
        import pystac_client
        import rasterio
        from rasterio.merge import merge

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox,
        )
        items = list(search.items())
        log.info("Found %d COP-DEM tiles for bbox", len(items))

        if not items:
            return None

        # Open all as rasterio datasets (lazy — data read on demand by merge)
        datasets = []
        for item in items:
            href = item.assets["data"].href
            datasets.append(rasterio.open(href))

        # Merge writes the result; peak memory = one output array
        merged, merged_transform = merge(datasets)

        for ds in datasets:
            ds.close()

        # Write to disk immediately, free the array
        profile = datasets[0].profile.copy()
        profile.update(
            height=merged.shape[1],
            width=merged.shape[2],
            transform=merged_transform,
            count=1,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(merged)

        pixel_count = merged.shape[1] * merged.shape[2]
        mem_mb = pixel_count * 4 / 1024 / 1024  # float32
        log.info("Wrote merged DEM: %s (%d x %d px, ~%.0f MB)",
                 dst_path, merged.shape[2], merged.shape[1], mem_mb)
        del merged  # free immediately
        return dst_path

    except Exception as e:
        log.error("DEM fetch failed for bbox %s: %s", bbox, e)
        return None


# ---------------------------------------------------------------------------
# Depression delineation
# ---------------------------------------------------------------------------

def delineate_depressions(dem_path: str, work_dir: str) -> pd.DataFrame | None:
    """Run whitebox depression delineation on a DEM tile.

    Uses whitebox-tools directly instead of the lidar package to avoid
    the osgeo/GDAL dependency (lidar uses osgeo internally, which requires
    system-level GDAL matching the container Python version).

    Pipeline: depth_in_sink -> connected-component labeling -> per-sink stats.
    """
    try:
        import rasterio
        import whitebox
        from scipy.ndimage import label

        wbt = whitebox.WhiteboxTools()
        wbt.set_verbose_mode(True)

        dep_dir = Path(work_dir) / "depressions"
        dep_dir.mkdir(exist_ok=True)

        depth_path = str(dep_dir / "depth_in_sink.tif")

        # Compute per-pixel depression depth (filled DEM minus original)
        log.info("Computing depth-in-sink...")
        ret = wbt.depth_in_sink(dem_path, depth_path, zero_background=True)
        if ret != 0:
            log.error("whitebox depth_in_sink returned %s", ret)
            return None

        # Read the depth raster
        with rasterio.open(depth_path) as src:
            depth = src.read(1)
            transform = src.transform
            nodata = src.nodata
            crs = src.crs

        # Mask nodata
        if nodata is not None:
            depth = np.where(depth == nodata, 0.0, depth)

        # Approximate pixel area in m^2 for geographic CRS (Copernicus is EPSG:4326)
        if crs and crs.is_geographic:
            center_lat = transform.f + depth.shape[0] * transform.e / 2
            cos_lat = np.cos(np.radians(abs(center_lat)))
            pixel_w_m = abs(transform.a) * 111320.0 * cos_lat
            pixel_h_m = abs(transform.e) * 110540.0
        else:
            pixel_w_m = abs(transform.a)
            pixel_h_m = abs(transform.e)
        pixel_area_m2 = pixel_w_m * pixel_h_m
        log.info("Pixel area: %.1f m2 (%.1f x %.1f m)", pixel_area_m2, pixel_w_m, pixel_h_m)

        # Label connected components of depressed pixels
        depression_mask = depth > 0
        labeled, n_features = label(depression_mask)
        log.info("Found %d raw sink regions", n_features)

        records = []
        for sid in range(1, n_features + 1):
            mask = labeled == sid
            n_pixels = int(mask.sum())
            if n_pixels < MIN_SINK_SIZE:
                continue

            dep_depths = depth[mask]
            max_depth = float(dep_depths.max())
            if max_depth < MIN_DEPRESSION_DEPTH:
                continue

            volume = float(dep_depths.sum() * pixel_area_m2)
            area = float(n_pixels * pixel_area_m2)

            # Centroid in pixel coords
            rows, cols = np.where(mask)
            records.append({
                "row": int(rows.mean()),
                "col": int(cols.mean()),
                "volume": volume,
                "max_depth": max_depth,
                "area": area,
            })

        if not records:
            log.warning("No depressions passed filters (min_size=%d, min_depth=%.1f)",
                        MIN_SINK_SIZE, MIN_DEPRESSION_DEPTH)
            return None

        deps = pd.DataFrame(records)
        log.info("Found %d depressions after filtering", len(deps))
        return deps

    except Exception as e:
        log.error("Depression delineation failed: %s", e)
        return None


def assign_depressions_to_zctas(deps: pd.DataFrame, dem_path: str,
                                 zcta_gdf) -> pd.DataFrame:
    """Assign depression centroids to ZCTAs and aggregate statistics.

    deps has columns: row, col, volume, max_depth, area (from delineate_depressions).
    """
    import geopandas as gpd
    import rasterio
    from shapely.geometry import Point

    if deps.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Convert pixel coords to geographic
    if "row" in deps.columns and "col" in deps.columns:
        with rasterio.open(dem_path) as src:
            transform = src.transform
            crs = src.crs
        xs, ys = rasterio.transform.xy(transform, deps["row"].values, deps["col"].values)
        dep_points = gpd.GeoDataFrame(
            deps, geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=crs,
        )
    elif "x" in deps.columns and "y" in deps.columns:
        with rasterio.open(dem_path) as src:
            crs = src.crs
        dep_points = gpd.GeoDataFrame(
            deps, geometry=[Point(x, y) for x, y in zip(deps["x"], deps["y"])],
            crs=crs,
        )
    else:
        log.warning("Depression CSV has no spatial columns; cannot assign to ZCTAs")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    dep_points = dep_points.to_crs("EPSG:5070")

    joined = gpd.sjoin(dep_points, zcta_gdf, how="inner", predicate="within")

    agg = joined.groupby("zcta_id").agg(
        depression_count=("volume", "count"),
        depression_volume_m3=("volume", "sum"),
        max_depression_depth_m=("max_depth", "max"),
        depression_area_m2=("area", "sum"),
    ).reset_index()

    log.info("Assigned depressions to %d ZCTAs", len(agg))
    return agg


# ---------------------------------------------------------------------------
# Coverage logging
# ---------------------------------------------------------------------------

def log_coverage(df: pd.DataFrame, zcta_ids: list[str]) -> dict:
    """Log per-column coverage and return coverage dict."""
    coverage = {}
    for col in OUTPUT_COLUMNS:
        if col == "zcta_id":
            continue
        if col in df.columns:
            pct = df[col].notna().mean() * 100
        else:
            pct = 0.0
        coverage[col] = round(pct, 1)
        log.info("  %s: %.1f%% non-null", col, pct)

    hit_rate = len(df[df["zcta_id"].isin(zcta_ids)]) / len(zcta_ids) * 100
    coverage["zcta_hit_rate"] = round(hit_rate, 1)
    log.info("  ZCTA hit rate: %.1f%% (%d / %d)",
             hit_rate, len(df[df["zcta_id"].isin(zcta_ids)]), len(zcta_ids))
    return coverage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scenario: str, upload: bool, dry_run: bool) -> None:
    s3 = get_s3_client()
    t0 = time.time()

    # 1. Load target ZCTAs
    zcta_ids = load_scenario_zcta_ids(s3, scenario)

    # 2. Cache-first: check which ZCTAs are already done
    cache = load_cache(s3)
    if cache is not None:
        cached_ids = set(cache["zcta_id"].astype(str).tolist())
        needed_ids = [z for z in zcta_ids if z not in cached_ids]
        log.info("Cache has %d ZCTAs; %d of %d needed are already cached",
                 len(cached_ids), len(zcta_ids) - len(needed_ids), len(zcta_ids))
    else:
        needed_ids = zcta_ids
        log.info("No existing cache; all %d ZCTAs needed", len(needed_ids))

    if not needed_ids:
        log.info("All ZCTAs already in cache. Nothing to do.")
        log_coverage(cache, zcta_ids)
        return

    if dry_run:
        log.info("[DRY RUN] Would delineate depressions for %d ZCTAs (%d cached, %d new)",
                 len(zcta_ids), len(zcta_ids) - len(needed_ids), len(needed_ids))
        return

    # 3. Load ZCTA boundaries for needed ZCTAs
    zcta_gdf = load_zcta_boundaries(s3, needed_ids)

    # 4. Compute bounding box
    import geopandas as gpd
    zcta_4326 = zcta_gdf.to_crs("EPSG:4326")
    bounds = zcta_4326.total_bounds
    bbox = (bounds[0] - 0.01, bounds[1] - 0.01,
            bounds[2] + 0.01, bounds[3] + 0.01)
    log.info("BBox for %s: %.4f, %.4f, %.4f, %.4f", scenario, *bbox)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 5. Fetch + merge DEM
        dem_path = str(Path(tmpdir) / f"dem_{scenario}.tif")
        dem_result = fetch_dem_for_bbox(bbox, dem_path)

        if dem_result is None:
            raise RuntimeError(f"DEM fetch failed for {scenario}")

        # 6. Delineate depressions
        deps = delineate_depressions(dem_path, tmpdir)

        if deps is None or deps.empty:
            log.warning("No depressions found for %s", scenario)
            new_rows = pd.DataFrame(columns=OUTPUT_COLUMNS)
        else:
            # 7. Assign to ZCTAs
            new_rows = assign_depressions_to_zctas(deps, dem_path, zcta_gdf)

    elapsed = time.time() - t0

    # 8. Merge into cache, dedupe, write back
    if upload and not new_rows.empty:
        combined = merge_and_write_cache(s3, cache, new_rows)
    elif cache is not None and not cache.empty:
        combined = pd.concat([cache, new_rows], ignore_index=True) if not new_rows.empty else cache
        combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")
    else:
        combined = new_rows

    # 9. Coverage logging
    log.info("=== Coverage for %s ===", scenario)
    coverage = log_coverage(combined, zcta_ids) if not combined.empty else {}

    # 10. Metadata JSON
    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "parameters": {
            "min_sink_size_px": MIN_SINK_SIZE,
            "min_depression_depth_m": MIN_DEPRESSION_DEPTH,
            "depression_interval_m": DEPRESSION_INTERVAL,
        },
        "n_zctas_requested": len(zcta_ids),
        "n_zctas_cached": len(zcta_ids) - len(needed_ids),
        "n_zctas_fetched": len(new_rows),
        "n_depressions": len(deps) if deps is not None else 0,
        "n_zctas_in_cache_total": len(combined) if not combined.empty else 0,
        "total_volume_m3": float(new_rows["depression_volume_m3"].sum()) if len(new_rows) > 0 else 0.0,
        "max_depth_m": float(new_rows["max_depression_depth_m"].max()) if len(new_rows) > 0 else 0.0,
        "elapsed_sec": round(elapsed, 1),
        "coverage": coverage,
    }

    if upload:
        evidence_key = f"results/s035/depressions_extraction_{scenario}.json"
        upload_json_result(s3, BUCKET, evidence_key, evidence)
    else:
        print(json.dumps(evidence, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Depression delineation from Copernicus DEM")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS,
                        help="Scenario to process (one per job)")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run(args.scenario, args.upload, args.dry_run)


if __name__ == "__main__":
    main()
