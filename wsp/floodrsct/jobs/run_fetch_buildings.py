#!/usr/bin/env python3
"""run_fetch_buildings.py -- Extract Overture building footprints per ZCTA.

Downloads building polygons from Overture Maps via open-buildings (DuckDB+S3),
spatially joins with ZCTA boundaries, and aggregates per-ZCTA statistics:

  - building_count: number of building footprints in the ZCTA
  - total_footprint_area_m2: total building footprint area (EPSG:5070 equal-area)

Output is scenario-independent (keyed by ZCTA, not by event), so results live
in the processed/shared/ prefix and are built once for all scenarios.

Source:
  Overture Maps Foundation (via open-buildings DuckDB adapter)

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_buildings.parquet

Usage:
    python run_fetch_buildings.py --upload
    python run_fetch_buildings.py --scenario houston --upload
    python run_fetch_buildings.py --dry-run
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
BUILDINGS_PARQUET_KEY = "processed/shared/zcta_buildings.parquet"
BUILDINGS_EVIDENCE_KEY = "results/s035/buildings_extraction.json"

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
                    log.warning("No zcta_id column in %s, trying next", key)
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
# Building extraction per scenario
# ---------------------------------------------------------------------------

def extract_buildings_for_bbox(bbox: tuple[float, float, float, float],
                               dst_path: str) -> str | None:
    """Download Overture buildings for a bounding box to GeoParquet.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
        dst_path: Local file path for output GeoParquet

    Returns:
        Path to output file, or None on failure.
    """
    try:
        from open_buildings.download_buildings import download
        download(
            geojson_data=json.dumps({
                "type": "Polygon",
                "coordinates": [[
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                    [bbox[0], bbox[1]],
                ]],
            }),
            dst=dst_path,
            source="overture",
            country_iso="US",
            verbose=True,
        )
        return dst_path
    except Exception as e:
        log.error("Building download failed for bbox %s: %s", bbox, e)
        return None


def aggregate_buildings_per_zcta(buildings_path: str,
                                  zcta_gdf) -> pd.DataFrame:
    """Spatial join buildings with ZCTA polygons and aggregate.

    Returns DataFrame with columns: zcta_id, building_count, total_footprint_area_m2
    """
    import geopandas as gpd

    log.info("Reading buildings from %s...", buildings_path)
    buildings = gpd.read_parquet(buildings_path)
    log.info("Loaded %d building footprints", len(buildings))

    if buildings.empty:
        return pd.DataFrame(columns=["zcta_id", "building_count", "total_footprint_area_m2"])

    # Project buildings to EPSG:5070 for area calculation
    if buildings.crs is None:
        buildings = buildings.set_crs("EPSG:4326")
    buildings = buildings.to_crs("EPSG:5070")

    # Compute footprint area in m2
    buildings["area_m2"] = buildings.geometry.area

    # Spatial join: which ZCTA does each building fall in?
    joined = gpd.sjoin(buildings, zcta_gdf, how="inner", predicate="intersects")

    # Aggregate per ZCTA
    agg = joined.groupby("zcta_id").agg(
        building_count=("area_m2", "count"),
        total_footprint_area_m2=("area_m2", "sum"),
    ).reset_index()

    log.info("Aggregated buildings for %d ZCTAs (%.0f total buildings matched)",
             len(agg), agg["building_count"].sum())
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
        log.info("[DRY RUN] Would extract buildings for %d ZCTAs across %d scenarios",
                 len(all_zcta_ids), len(scenario_zctas))
        for scenario, ids in scenario_zctas.items():
            log.info("  %s: %d ZCTAs", scenario, len(ids))
        return

    # Load ZCTA boundaries (polygons in EPSG:5070)
    zcta_gdf = load_zcta_boundaries(s3, all_zcta_ids)

    # For bounding box computation, get 4326 envelope per scenario
    import geopandas as gpd
    zcta_4326 = zcta_gdf.to_crs("EPSG:4326")

    parts = []
    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": {},
    }

    for scenario, zcta_ids in scenario_zctas.items():
        t0 = time.time()
        log.info("=== %s: %d ZCTAs ===", scenario, len(zcta_ids))

        # Compute bounding box for this scenario
        scenario_geom = zcta_4326[zcta_4326["zcta_id"].isin(zcta_ids)]
        if scenario_geom.empty:
            log.warning("No ZCTA boundaries for %s, skipping", scenario)
            continue

        bounds = scenario_geom.total_bounds  # (minx, miny, maxx, maxy)
        # Add small buffer (0.05 deg ~ 5km) to avoid edge clipping
        bbox = (bounds[0] - 0.05, bounds[1] - 0.05,
                bounds[2] + 0.05, bounds[3] + 0.05)
        log.info("BBox: %.4f, %.4f, %.4f, %.4f", *bbox)

        # Download buildings for this scenario bbox
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = str(Path(tmpdir) / f"buildings_{scenario}.parquet")
            result_path = extract_buildings_for_bbox(bbox, dst)

            if result_path is None:
                log.warning("Building extraction failed for %s", scenario)
                evidence["scenarios"][scenario] = {"status": "FAILED", "error": "download_failed"}
                continue

            # Spatial join + aggregate
            scenario_boundaries = zcta_gdf[zcta_gdf["zcta_id"].isin(zcta_ids)]
            agg = aggregate_buildings_per_zcta(result_path, scenario_boundaries)

        elapsed = time.time() - t0
        coverage = len(agg) / len(zcta_ids) * 100 if zcta_ids else 0

        log.info("%s: %d ZCTAs with buildings (%.1f%% coverage), %.0f sec",
                 scenario, len(agg), coverage, elapsed)

        evidence["scenarios"][scenario] = {
            "status": "OK",
            "n_zctas": len(zcta_ids),
            "n_zctas_with_buildings": len(agg),
            "coverage_pct": round(coverage, 1),
            "total_buildings": int(agg["building_count"].sum()) if len(agg) > 0 else 0,
            "total_area_m2": float(agg["total_footprint_area_m2"].sum()) if len(agg) > 0 else 0.0,
            "elapsed_sec": round(elapsed, 1),
        }
        parts.append(agg)

    if not parts:
        raise RuntimeError("Building extraction failed for all scenarios")

    # Combine + deduplicate (ZCTAs can overlap between scenarios)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.groupby("zcta_id").agg(
        building_count=("building_count", "sum"),
        total_footprint_area_m2=("total_footprint_area_m2", "sum"),
    ).reset_index()
    log.info("Combined: %d unique ZCTAs, %d total buildings",
             len(combined), combined["building_count"].sum())

    evidence["n_zctas_total"] = len(combined)
    evidence["total_buildings"] = int(combined["building_count"].sum())

    if upload:
        s3_write_parquet(s3, combined, BUILDINGS_PARQUET_KEY)
        upload_json_result(s3, BUCKET, BUILDINGS_EVIDENCE_KEY, evidence)
    else:
        log.info("Skipping upload (--upload not set)")
        print(json.dumps(evidence, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Overture building footprints per ZCTA")
    parser.add_argument("--scenario", choices=SCENARIOS, default=None,
                        help="Single scenario (default: all)")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIOS
    run(scenarios, args.upload, args.dry_run)


if __name__ == "__main__":
    main()
