#!/usr/bin/env python3
"""run_fetch_buildings.py -- Extract Overture building footprints per ZCTA.

Downloads building polygons from Overture Maps via open-buildings (DuckDB+S3),
spatially joins with ZCTA boundaries, and aggregates per-ZCTA statistics:

  - building_count: number of building footprints in the ZCTA
  - total_footprint_area_m2: total building footprint area (EPSG:5070 equal-area)

Output is scenario-independent (keyed by ZCTA, not by event), so results live
in processed/shared/ and are built once for all scenarios. Cache-first: if the
shared parquet already contains a ZCTA, it is not re-fetched.

Source:
  Overture Maps Foundation (via open-buildings DuckDB adapter)

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_buildings.parquet
  s3://swarm-floodrsct-data/results/s035/buildings_extraction_{scenario}.json

Usage:
    python run_fetch_buildings.py --scenario houston --upload
    python run_fetch_buildings.py --scenario houston --dry-run
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
BUILDINGS_CACHE_KEY = "processed/shared/zcta_buildings.parquet"
OUTPUT_COLUMNS = ["zcta_id", "building_count", "total_footprint_area_m2"]

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
            # Normalize zcta_id column
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
    """Load existing buildings cache from S3."""
    return s3_read_parquet(s3, BUILDINGS_CACHE_KEY)


def merge_and_write_cache(s3, existing: pd.DataFrame | None,
                          new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append new rows to cache, dedupe on zcta_id, write back."""
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()
    combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")
    # Enforce output schema
    for col in OUTPUT_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[OUTPUT_COLUMNS]
    s3_write_parquet(s3, combined, BUILDINGS_CACHE_KEY)
    return combined


# ---------------------------------------------------------------------------
# Building extraction
# ---------------------------------------------------------------------------

def extract_buildings_for_bbox(bbox: tuple[float, float, float, float],
                               dst_path: str) -> str | None:
    """Download Overture buildings for a bounding box to GeoParquet."""
    try:
        from open_buildings.download_buildings import download
        geojson_str = json.dumps({
            "type": "Polygon",
            "coordinates": [[
                [float(bbox[0]), float(bbox[1])],
                [float(bbox[2]), float(bbox[1])],
                [float(bbox[2]), float(bbox[3])],
                [float(bbox[0]), float(bbox[3])],
                [float(bbox[0]), float(bbox[1])],
            ]],
        })
        # open-buildings 0.10.0: geojson_input is a file handle (json.load),
        # all params are positional from the Click CLI wrapper.
        download(
            geojson_input=io.StringIO(geojson_str),
            format="parquet",
            generate_sql=False,
            dst=dst_path,
            silent=False,
            overwrite=True,
            verbose=True,
            data_path=None,
            hive_partitioning=False,
            country_iso="US",
        )
        return dst_path
    except Exception as e:
        log.error("Building download failed for bbox %s: %s", bbox, e)
        return None


def aggregate_buildings_per_zcta(buildings_path: str,
                                  zcta_gdf) -> pd.DataFrame:
    """Spatial join buildings with ZCTA polygons and aggregate."""
    import geopandas as gpd

    log.info("Reading buildings from %s...", buildings_path)
    buildings = gpd.read_parquet(buildings_path)
    log.info("Loaded %d building footprints", len(buildings))

    if buildings.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

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

    # 1. Load target ZCTAs for this scenario
    zcta_ids = load_scenario_zcta_ids(s3, scenario)

    # 2. Load existing cache — check which ZCTAs are already done
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
        log.info("[DRY RUN] Would extract buildings for %d ZCTAs (%d cached, %d new)",
                 len(zcta_ids), len(zcta_ids) - len(needed_ids), len(needed_ids))
        return

    # 3. Load ZCTA boundaries for needed ZCTAs
    zcta_gdf = load_zcta_boundaries(s3, needed_ids)

    # 4. Compute bounding box for this scenario's needed ZCTAs
    import geopandas as gpd
    zcta_4326 = zcta_gdf.to_crs("EPSG:4326")
    bounds = zcta_4326.total_bounds  # (minx, miny, maxx, maxy)
    bbox = (bounds[0] - 0.05, bounds[1] - 0.05,
            bounds[2] + 0.05, bounds[3] + 0.05)
    log.info("BBox for %s: %.4f, %.4f, %.4f, %.4f", scenario, *bbox)

    # 5. Download buildings for bbox
    with tempfile.TemporaryDirectory() as tmpdir:
        dst = str(Path(tmpdir) / f"buildings_{scenario}.parquet")
        result_path = extract_buildings_for_bbox(bbox, dst)

        if result_path is None:
            raise RuntimeError(f"Building extraction failed for {scenario}")

        # 6. Spatial join + aggregate
        new_rows = aggregate_buildings_per_zcta(result_path, zcta_gdf)

    elapsed = time.time() - t0

    # 7. Merge into cache, dedupe, write back
    if upload:
        combined = merge_and_write_cache(s3, cache, new_rows)
    else:
        if cache is not None and not cache.empty:
            combined = pd.concat([cache, new_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")
        else:
            combined = new_rows

    # 8. Coverage logging
    log.info("=== Coverage for %s ===", scenario)
    coverage = log_coverage(combined, zcta_ids)

    # 9. Metadata JSON
    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "n_zctas_requested": len(zcta_ids),
        "n_zctas_cached": len(zcta_ids) - len(needed_ids),
        "n_zctas_fetched": len(new_rows),
        "n_zctas_in_cache_total": len(combined),
        "total_buildings": int(new_rows["building_count"].sum()) if len(new_rows) > 0 else 0,
        "total_area_m2": float(new_rows["total_footprint_area_m2"].sum()) if len(new_rows) > 0 else 0.0,
        "elapsed_sec": round(elapsed, 1),
        "coverage": coverage,
    }

    if upload:
        evidence_key = f"results/s035/buildings_extraction_{scenario}.json"
        upload_json_result(s3, BUCKET, evidence_key, evidence)
    else:
        print(json.dumps(evidence, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Overture building footprints per ZCTA")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS,
                        help="Scenario to process (one per job)")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run(args.scenario, args.upload, args.dry_run)


if __name__ == "__main__":
    main()
