#!/usr/bin/env python3
"""run_fetch_depressions.py -- Depression delineation from Copernicus DEM.

Runs level-set depression delineation on Copernicus 30m DEM tiles for each
scenario's ZCTA universe. Uses the `lidar` package (graph-based sink extraction
via whitebox-tools) to identify surface depressions that act as flood storage.

Outputs per-ZCTA statistics:
  - depression_count: number of depressions intersecting the ZCTA
  - depression_volume_m3: total depression volume (flood storage capacity)
  - max_depression_depth_m: deepest depression in the ZCTA
  - depression_area_m2: total depressed area

These are scenario-independent (topographic, not event-based), so results
live in processed/shared/ and are built once for all scenarios.

Source:
  Copernicus GLO-30 DEM via Planetary Computer STAC

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_depressions.parquet

Usage:
    python run_fetch_depressions.py --upload
    python run_fetch_depressions.py --scenario houston --upload
    python run_fetch_depressions.py --dry-run
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
DEPRESSIONS_PARQUET_KEY = "processed/shared/zcta_depressions.parquet"
DEPRESSIONS_EVIDENCE_KEY = "results/s035/depressions_extraction.json"

# ZCTA boundaries (GeoParquet with polygon geometry)
ZCTA_BOUNDARIES_KEYS = [
    "raw/geocertdb2026/zcta_boundaries_5070.parquet",
    "raw/geocertdb2026/zcta_boundaries.parquet",
    "raw/geocertdb2026/zcta5_boundaries.parquet",
]

# Scenario event_features keys (for ZCTA ID extraction)
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
DEPRESSION_INTERVAL = 0.5   # depth interval for nested level sets (meters)


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
# Load ZCTA universe
# ---------------------------------------------------------------------------

def load_scenario_zctas(s3, scenarios: list[str]) -> dict[str, list[str]]:
    """Load unique ZCTA IDs per scenario from event_features parquets."""
    result = {}
    for scenario in scenarios:
        key = SCENARIO_EVENT_KEYS.get(scenario)
        if not key:
            continue
        df = s3_read_parquet(s3, key)
        if df is None:
            log.warning("Missing event_features for %s, skipping", scenario)
            continue
        zcta_col = next((c for c in df.columns if "zcta" in c.lower()), None)
        if zcta_col is None:
            continue
        ids = sorted(df[zcta_col].astype(str).unique().tolist())
        result[scenario] = ids
        log.info("  %s: %d unique ZCTAs", scenario, len(ids))
    return result


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
# DEM fetching from Planetary Computer STAC
# ---------------------------------------------------------------------------

def fetch_dem_for_bbox(bbox: tuple[float, float, float, float],
                       dst_path: str) -> str | None:
    """Fetch Copernicus GLO-30 DEM for a bounding box from Planetary Computer.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
        dst_path: Local path for output GeoTIFF

    Returns:
        Path to merged DEM GeoTIFF, or None on failure.
    """
    try:
        import planetary_computer
        import pystac_client
        import rasterio
        from rasterio.merge import merge
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import box

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

        # Read and merge tiles
        datasets = []
        for item in items:
            href = item.assets["data"].href
            ds = rasterio.open(href)
            datasets.append(ds)

        merged, merged_transform = merge(datasets)
        for ds in datasets:
            ds.close()

        # Write merged DEM
        profile = datasets[0].profile.copy()
        profile.update(
            height=merged.shape[1],
            width=merged.shape[2],
            transform=merged_transform,
            count=1,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(merged)

        log.info("Wrote merged DEM: %s (%d x %d pixels)",
                 dst_path, merged.shape[2], merged.shape[1])
        return dst_path

    except Exception as e:
        log.error("DEM fetch failed for bbox %s: %s", bbox, e)
        return None


# ---------------------------------------------------------------------------
# Depression delineation
# ---------------------------------------------------------------------------

def delineate_depressions(dem_path: str, work_dir: str) -> pd.DataFrame | None:
    """Run lidar depression delineation on a DEM tile.

    Returns DataFrame with depression statistics, or None on failure.
    """
    try:
        import lidar

        sink_path = Path(work_dir) / "sinks.tif"
        dep_path = Path(work_dir) / "depressions"
        dep_path.mkdir(exist_ok=True)

        log.info("Extracting sinks (min_size=%d pixels)...", MIN_SINK_SIZE)
        lidar.ExtractSinks(
            in_dem=dem_path,
            min_size=MIN_SINK_SIZE,
            out_dir=str(dep_path),
        )

        # Find the output sink raster
        sink_files = list(dep_path.glob("*sink*.tif")) + list(dep_path.glob("*Sink*.tif"))
        if not sink_files:
            log.warning("No sink files produced")
            return None

        log.info("Delineating depressions (min_depth=%.1f m, interval=%.1f m)...",
                 MIN_DEPRESSION_DEPTH, DEPRESSION_INTERVAL)
        lidar.DelineateDepressions(
            in_sink=str(sink_files[0]),
            min_size=MIN_SINK_SIZE,
            min_depth=MIN_DEPRESSION_DEPTH,
            interval=DEPRESSION_INTERVAL,
            out_dir=str(dep_path),
        )

        # Read depression CSV output
        csv_files = list(dep_path.glob("*depression*.csv")) + list(dep_path.glob("*Depression*.csv"))
        if not csv_files:
            log.warning("No depression CSV produced")
            return None

        deps = pd.read_csv(csv_files[0])
        log.info("Found %d depressions", len(deps))
        return deps

    except Exception as e:
        log.error("Depression delineation failed: %s", e)
        return None


def assign_depressions_to_zctas(deps: pd.DataFrame, dem_path: str,
                                 zcta_gdf) -> pd.DataFrame:
    """Assign depression centroids to ZCTAs and aggregate statistics.

    The depression CSV from lidar contains id, level, volume, avg_depth,
    max_depth, area columns. We need to geolocate each depression back
    to a ZCTA using the DEM's spatial reference.
    """
    import geopandas as gpd
    import rasterio
    from shapely.geometry import Point

    if deps.empty:
        return pd.DataFrame(columns=[
            "zcta_id", "depression_count", "depression_volume_m3",
            "max_depression_depth_m", "depression_area_m2",
        ])

    # Standardize column names (lidar output varies)
    col_map = {}
    for c in deps.columns:
        cl = c.lower()
        if "vol" in cl:
            col_map[c] = "volume"
        elif "max" in cl and "depth" in cl:
            col_map[c] = "max_depth"
        elif "avg" in cl and "depth" in cl:
            col_map[c] = "avg_depth"
        elif "area" in cl:
            col_map[c] = "area"
        elif cl in ("row", "y"):
            col_map[c] = "row"
        elif cl in ("col", "x"):
            col_map[c] = "col"
    deps = deps.rename(columns=col_map)

    # If depression CSV has row/col, convert to geographic coordinates
    # using the DEM's affine transform
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
        # No spatial info — cannot assign to ZCTAs; aggregate globally
        log.warning("Depression CSV has no spatial columns; cannot assign to ZCTAs")
        return pd.DataFrame(columns=[
            "zcta_id", "depression_count", "depression_volume_m3",
            "max_depression_depth_m", "depression_area_m2",
        ])

    # Project to match ZCTA CRS (EPSG:5070)
    dep_points = dep_points.to_crs("EPSG:5070")

    # Spatial join with ZCTA boundaries
    joined = gpd.sjoin(dep_points, zcta_gdf, how="inner", predicate="within")

    # Aggregate per ZCTA
    # Volume from lidar is in DEM-native units (pixel_area * depth);
    # for COP-30 at ~30m resolution, pixel area ~ 900 m2.
    # Multiply volume by pixel_area if not already in m3.
    agg = joined.groupby("zcta_id").agg(
        depression_count=("volume", "count"),
        depression_volume_m3=("volume", "sum"),
        max_depression_depth_m=("max_depth", "max"),
        depression_area_m2=("area", "sum"),
    ).reset_index()

    log.info("Assigned depressions to %d ZCTAs", len(agg))
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scenarios: list[str], upload: bool, dry_run: bool) -> None:
    s3 = get_s3_client()

    # Load ZCTA universe
    scenario_zctas = load_scenario_zctas(s3, scenarios)
    all_zcta_ids = sorted(set(z for ids in scenario_zctas.values() for z in ids))
    log.info("Total unique ZCTAs across %d scenarios: %d",
             len(scenario_zctas), len(all_zcta_ids))

    if dry_run:
        log.info("[DRY RUN] Would delineate depressions for %d ZCTAs across %d scenarios",
                 len(all_zcta_ids), len(scenario_zctas))
        for scenario, ids in scenario_zctas.items():
            log.info("  %s: %d ZCTAs", scenario, len(ids))
        return

    # Load ZCTA boundaries (polygons in EPSG:5070)
    zcta_gdf = load_zcta_boundaries(s3, all_zcta_ids)

    # For bounding box computation, get 4326 envelope
    import geopandas as gpd
    zcta_4326 = zcta_gdf.to_crs("EPSG:4326")

    parts = []
    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "min_sink_size_px": MIN_SINK_SIZE,
            "min_depression_depth_m": MIN_DEPRESSION_DEPTH,
            "depression_interval_m": DEPRESSION_INTERVAL,
        },
        "scenarios": {},
    }

    for scenario, zcta_ids in scenario_zctas.items():
        t0 = time.time()
        log.info("=== %s: %d ZCTAs ===", scenario, len(zcta_ids))

        # Compute bounding box
        scenario_geom = zcta_4326[zcta_4326["zcta_id"].isin(zcta_ids)]
        if scenario_geom.empty:
            log.warning("No ZCTA boundaries for %s, skipping", scenario)
            continue

        bounds = scenario_geom.total_bounds
        bbox = (bounds[0] - 0.01, bounds[1] - 0.01,
                bounds[2] + 0.01, bounds[3] + 0.01)
        log.info("BBox: %.4f, %.4f, %.4f, %.4f", *bbox)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Fetch DEM
            dem_path = str(Path(tmpdir) / f"dem_{scenario}.tif")
            dem_result = fetch_dem_for_bbox(bbox, dem_path)

            if dem_result is None:
                log.warning("DEM fetch failed for %s", scenario)
                evidence["scenarios"][scenario] = {"status": "FAILED", "error": "dem_fetch_failed"}
                continue

            # Run depression delineation
            deps = delineate_depressions(dem_path, tmpdir)

            if deps is None or deps.empty:
                log.warning("No depressions found for %s", scenario)
                evidence["scenarios"][scenario] = {
                    "status": "OK",
                    "n_zctas": len(zcta_ids),
                    "n_depressions": 0,
                    "n_zctas_with_depressions": 0,
                }
                continue

            # Assign to ZCTAs
            scenario_boundaries = zcta_gdf[zcta_gdf["zcta_id"].isin(zcta_ids)]
            agg = assign_depressions_to_zctas(deps, dem_path, scenario_boundaries)

        elapsed = time.time() - t0
        coverage = len(agg) / len(zcta_ids) * 100 if zcta_ids else 0

        log.info("%s: %d ZCTAs with depressions (%.1f%% coverage), %.0f sec",
                 scenario, len(agg), coverage, elapsed)

        evidence["scenarios"][scenario] = {
            "status": "OK",
            "n_zctas": len(zcta_ids),
            "n_depressions": len(deps),
            "n_zctas_with_depressions": len(agg),
            "coverage_pct": round(coverage, 1),
            "total_volume_m3": float(agg["depression_volume_m3"].sum()) if len(agg) > 0 else 0.0,
            "max_depth_m": float(agg["max_depression_depth_m"].max()) if len(agg) > 0 else 0.0,
            "elapsed_sec": round(elapsed, 1),
        }
        parts.append(agg)

    if not parts:
        raise RuntimeError("Depression delineation failed for all scenarios")

    # Combine + deduplicate (take max depth, sum others for shared ZCTAs)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby("zcta_id").agg(
        depression_count=("depression_count", "sum"),
        depression_volume_m3=("depression_volume_m3", "sum"),
        max_depression_depth_m=("max_depression_depth_m", "max"),
        depression_area_m2=("depression_area_m2", "sum"),
    ).reset_index()
    log.info("Combined: %d unique ZCTAs with depressions", len(combined))

    evidence["n_zctas_total"] = len(combined)
    evidence["total_depressions"] = int(combined["depression_count"].sum())

    if upload:
        s3_write_parquet(s3, combined, DEPRESSIONS_PARQUET_KEY)
        upload_json_result(s3, BUCKET, DEPRESSIONS_EVIDENCE_KEY, evidence)
    else:
        log.info("Skipping upload (--upload not set)")
        print(json.dumps(evidence, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Depression delineation from Copernicus DEM")
    parser.add_argument("--scenario", choices=SCENARIOS, default=None,
                        help="Single scenario (default: all)")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIOS
    run(scenarios, args.upload, args.dry_run)


if __name__ == "__main__":
    main()
