#!/usr/bin/env python3
"""run_fetch_buildings_v2.py -- Push-down buildings extraction via DuckDB.

V2 rewrites the extraction pipeline to push spatial join + aggregation
into DuckDB SQL, eliminating the geopandas materialization bottleneck.

Old flow (v1):
  DuckDB scan -> materialize ALL buildings -> WKB->GeoParquet ->
  reproject 5070 -> compute area -> geopandas sjoin -> aggregate

New flow (v2):
  Load ZCTA polygons into DuckDB -> single SQL query:
  scan Overture + spatial join + ST_Area_Spheroid + GROUP BY ->
  return ~200 aggregate rows directly

Benefits:
  - No materialization of 1M+ building geometries
  - No geopandas sjoin (DuckDB does it)
  - No WKB conversion or reprojection step
  - Memory: ~200 MB vs ~2 GB for dense metros (NYC)
  - ST_Area_Spheroid: geodesic area from WGS84, no PROJ needed

Output schema is identical to v1:
  zcta_id, building_count, total_footprint_area_m2

Usage:
    python run_fetch_buildings_v2.py --scenario houston --upload
    python run_fetch_buildings_v2.py --scenario nyc --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
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

BUILDINGS_CACHE_KEY = "processed/shared/zcta_buildings.parquet"
BUILDINGS_SCENARIO_PREFIX = "processed/shared/staging/zcta_buildings_"
OUTPUT_COLUMNS = ["zcta_id", "building_count", "total_footprint_area_m2"]

ZCTA_BOUNDARIES_KEYS = [
    "raw/geocertdb2026/zcta_boundaries_5070.parquet",
    "raw/geocertdb2026/zcta_boundaries.parquet",
    "raw/geocertdb2026/zcta5_boundaries.parquet",
]

SCENARIO_EVENT_KEYS = {
    "houston": "processed/houston/houston_event_features.parquet",
    "new_orleans": "processed/new_orleans/no_event_features.parquet",
    "nyc": "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
}

OVERTURE_DATA_PATH = (
    "s3://us-west-2.opendata.source.coop/cholmes/overture/"
    "geoparquet-country-quad-hive/*/*.parquet"
)

MIN_QK_LEN = 5


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_read_parquet(s3, key: str):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        log.warning("Could not read %s: %s", key, e)
        return None


def s3_write_parquet(s3, df: pd.DataFrame, key: str) -> None:
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    log.info("Uploaded %d rows to s3://%s/%s", len(df), BUCKET, key)


def get_git_hash() -> str:
    gh = os.environ.get("S035_GIT_HASH")
    if gh:
        return gh
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scenario_zcta_ids(s3, scenario: str) -> list[str]:
    key = SCENARIO_EVENT_KEYS[scenario]
    df = s3_read_parquet(s3, key)
    if df is None:
        raise RuntimeError(f"event_features not found: s3://{BUCKET}/{key}")
    zcta_col = next((c for c in df.columns if "zcta" in c.lower()), None)
    if zcta_col is None:
        raise RuntimeError(f"No zcta column in {key}")
    return sorted(df[zcta_col].astype(str).unique().tolist())


def download_zcta_boundaries(s3, zcta_ids: list[str], tmpdir: str) -> str:
    """Download ZCTA boundaries and save as WGS84 GeoParquet for DuckDB.

    Returns local path to the WGS84 parquet file.
    """
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

            gdf = gdf[gdf["zcta_id"].isin(zcta_ids)][["zcta_id", "geometry"]]

            # Ensure WGS84 for DuckDB spatial join with Overture (also WGS84)
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")

            local_path = os.path.join(tmpdir, "zcta_boundaries_4326.parquet")
            gdf.to_parquet(local_path)
            log.info("Saved %d ZCTA boundaries to %s (WGS84)", len(gdf), local_path)
            return local_path
        except Exception as e:
            log.warning("Could not load %s: %s", key, e)
            continue

    raise RuntimeError("No ZCTA boundary file found on S3")


def _quadkey_to_bbox(qk: str) -> tuple[float, float, float, float]:
    minx, maxx = -180.0, 180.0
    miny, maxy = -85.05112878, 85.05112878
    for ch in qk:
        midx = (minx + maxx) / 2
        midy = (miny + maxy) / 2
        if ch == "0":
            maxx, miny = midx, midy
        elif ch == "1":
            minx, miny = midx, midy
        elif ch == "2":
            maxx, maxy = midx, midy
        elif ch == "3":
            minx, maxy = midx, midy
    return (minx, miny, maxx, maxy)


# ---------------------------------------------------------------------------
# Push-down extraction: spatial join + aggregate in DuckDB
# ---------------------------------------------------------------------------

def extract_and_aggregate_pushdown(
    zcta_parquet_path: str,
    bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Run the full pipeline in DuckDB: scan Overture, spatial join with
    ZCTA polygons, compute geodesic area, aggregate per ZCTA.

    Returns DataFrame with columns: zcta_id, building_count, total_footprint_area_m2.
    """
    import duckdb
    from open_buildings.download_buildings import geojson_to_quadkey

    geojson_data = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [float(bbox[0]), float(bbox[1])],
                [float(bbox[2]), float(bbox[1])],
                [float(bbox[2]), float(bbox[3])],
                [float(bbox[0]), float(bbox[3])],
                [float(bbox[0]), float(bbox[1])],
            ]],
        },
        "properties": {},
    }
    quadkey = geojson_to_quadkey(geojson_data)

    # Expand short quadkeys and pre-filter by bbox intersection
    if len(quadkey) < MIN_QK_LEN:
        from itertools import product as iterproduct
        pad = MIN_QK_LEN - len(quadkey)
        all_sub = [quadkey + "".join(d)
                   for d in iterproduct("0123", repeat=pad)]
        sub_quadkeys = []
        for qk in all_sub:
            tb = _quadkey_to_bbox(qk)
            if (tb[0] <= bbox[2] and tb[2] >= bbox[0]
                    and tb[1] <= bbox[3] and tb[3] >= bbox[1]):
                sub_quadkeys.append(qk)
        log.info("Short quadkey '%s' -> %d sub-queries (filtered from %d)",
                 quadkey, len(sub_quadkeys), len(all_sub))
    else:
        sub_quadkeys = [quadkey]

    conn = duckdb.connect(database=":memory:")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("SET s3_region = 'us-west-2';")
    conn.execute("SET s3_url_style = 'path';")
    conn.execute("INSTALL spatial; LOAD spatial;")

    # Let DuckDB use all available threads for the scan
    # Only constrain memory (not threads) for short quadkeys
    if len(quadkey) < MIN_QK_LEN:
        conn.execute("SET memory_limit = '6GB';")
        log.info("Short quadkey: DuckDB memory capped at 6GB (spill-to-disk)")

    # Load ZCTA polygons into DuckDB
    conn.execute(f"""
        CREATE TABLE zctas AS
        SELECT zcta_id, geometry
        FROM ST_Read('{zcta_parquet_path}')
    """)
    n_zctas = conn.execute("SELECT COUNT(*) FROM zctas").fetchone()[0]
    log.info("Loaded %d ZCTA polygons into DuckDB", n_zctas)

    # Build UNION ALL query across all sub-quadkeys
    # Each sub-query: scan Overture partition, join with ZCTAs, aggregate
    # This avoids materializing building geometries -- streams and aggregates
    union_parts = []
    for qk in sub_quadkeys:
        part = f"""
            SELECT
                z.zcta_id,
                COUNT(*) AS building_count,
                COALESCE(SUM(ST_Area_Spheroid(b.geometry)), 0) AS total_footprint_area_m2
            FROM read_parquet('{OVERTURE_DATA_PATH}', hive_partitioning=1) b
            JOIN zctas z ON ST_Within(b.geometry, z.geometry)
            WHERE b.country_iso = 'US' AND b.quadkey LIKE '{qk}%'
            GROUP BY z.zcta_id
        """
        union_parts.append(part)

    if len(union_parts) == 1:
        full_sql = union_parts[0]
    else:
        # Wrap in outer GROUP BY to merge across sub-queries
        inner = "\nUNION ALL\n".join(union_parts)
        full_sql = f"""
            SELECT
                zcta_id,
                SUM(building_count) AS building_count,
                SUM(total_footprint_area_m2) AS total_footprint_area_m2
            FROM ({inner}) sub
            GROUP BY zcta_id
        """

    log.info("Executing push-down query (%d sub-queries)...", len(sub_quadkeys))
    t0 = time.time()

    try:
        result = conn.execute(full_sql).fetchdf()
    except Exception as e:
        # Fallback: run sub-queries one at a time and concat
        log.warning("Batch query failed (%s), falling back to sequential", e)
        parts = []
        for i, qk in enumerate(sub_quadkeys):
            log.info("  Sub-query %d/%d (qk=%s)...", i + 1, len(sub_quadkeys), qk)
            try:
                part_sql = f"""
                    SELECT
                        z.zcta_id,
                        COUNT(*) AS building_count,
                        COALESCE(SUM(ST_Area_Spheroid(b.geometry)), 0) AS total_footprint_area_m2
                    FROM read_parquet('{OVERTURE_DATA_PATH}', hive_partitioning=1) b
                    JOIN zctas z ON ST_Within(b.geometry, z.geometry)
                    WHERE b.country_iso = 'US' AND b.quadkey LIKE '{qk}%'
                    GROUP BY z.zcta_id
                """
                part_df = conn.execute(part_sql).fetchdf()
                if len(part_df) > 0:
                    parts.append(part_df)
                    log.info("    -> %d ZCTAs, %d buildings",
                             len(part_df), int(part_df["building_count"].sum()))
            except Exception as sub_e:
                log.warning("    Sub-query %s failed: %s", qk, sub_e)
                continue

        if parts:
            result = pd.concat(parts, ignore_index=True)
            result = result.groupby("zcta_id", as_index=False).agg({
                "building_count": "sum",
                "total_footprint_area_m2": "sum",
            })
        else:
            result = pd.DataFrame(columns=OUTPUT_COLUMNS)

    elapsed = time.time() - t0
    conn.close()

    log.info("Push-down query complete: %d ZCTAs, %d buildings in %.1fs",
             len(result),
             int(result["building_count"].sum()) if len(result) > 0 else 0,
             elapsed)

    return result


# ---------------------------------------------------------------------------
# Cache (same as v1)
# ---------------------------------------------------------------------------

def load_cache(s3) -> pd.DataFrame | None:
    return s3_read_parquet(s3, BUILDINGS_CACHE_KEY)


def write_scenario_staging(s3, scenario: str, new_rows: pd.DataFrame) -> None:
    key = f"{BUILDINGS_SCENARIO_PREFIX}{scenario}.parquet"
    s3_write_parquet(s3, new_rows, key)


def consolidate_cache(s3) -> pd.DataFrame:
    cache = load_cache(s3)
    parts = [cache] if cache is not None and not cache.empty else []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=BUILDINGS_SCENARIO_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                df = s3_read_parquet(s3, obj["Key"])
                if df is not None and not df.empty:
                    parts.append(df)
                    log.info("Loaded staging: %s (%d rows)", obj["Key"], len(df))

    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")
    for col in OUTPUT_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[OUTPUT_COLUMNS]
    s3_write_parquet(s3, combined, BUILDINGS_CACHE_KEY)
    return combined


# ---------------------------------------------------------------------------
# QA (same as v1)
# ---------------------------------------------------------------------------

def qa_buildings(df: pd.DataFrame, zcta_ids: list[str]) -> dict:
    findings = []
    for col in ["building_count", "total_footprint_area_m2"]:
        if df[col].isna().sum() > 0:
            findings.append({"severity": "FAIL", "check": f"{col}_nulls",
                             "detail": f"{df[col].isna().sum()} null values"})
        if (df[col] < 0).sum() > 0:
            findings.append({"severity": "FAIL", "check": f"{col}_negative",
                             "detail": f"{(df[col] < 0).sum()} negative values"})

    if df["zcta_id"].duplicated().sum() > 0:
        findings.append({"severity": "FAIL", "check": "duplicate_zcta_ids",
                         "detail": f"{df['zcta_id'].duplicated().sum()} duplicates"})

    if len(df) > 0 and (df["building_count"] > 0).all():
        avg = df["total_footprint_area_m2"] / df["building_count"]
        n_low = (avg < 50).sum()
        n_high = (avg > 5000).sum()
        if n_low > 0:
            findings.append({"severity": "WARN", "check": "avg_footprint_low",
                             "detail": f"{n_low} ZCTAs with avg < 50 m2"})
        if n_high > 0:
            findings.append({"severity": "WARN", "check": "avg_footprint_high",
                             "detail": f"{n_high} ZCTAs with avg > 5000 m2"})

    hit_rate = len(df[df["zcta_id"].isin(zcta_ids)]) / max(len(zcta_ids), 1) * 100
    if hit_rate < 95:
        findings.append({"severity": "WARN", "check": "low_hit_rate",
                         "detail": f"hit rate {hit_rate:.1f}%"})

    passed = all(f["severity"] != "FAIL" for f in findings)
    log.info("=== QA: %s (%d findings) ===", "PASS" if passed else "FAIL", len(findings))
    for f in findings:
        log.info("  [%s] %s: %s", f["severity"], f["check"], f["detail"])
    return {"passed": passed, "findings": findings}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scenario: str, upload: bool, dry_run: bool) -> None:
    s3 = get_s3_client()
    t0 = time.time()
    git_hash = get_git_hash()

    print("=" * 60)
    print(f"  BUILDINGS V2 (push-down) -- {scenario}")
    print("=" * 60)
    sys.stdout.flush()

    zcta_ids = load_scenario_zcta_ids(s3, scenario)

    # Cache check
    cache = load_cache(s3)
    if cache is not None:
        cached_ids = set(cache["zcta_id"].astype(str).tolist())
        needed_ids = [z for z in zcta_ids if z not in cached_ids]
        log.info("Cache: %d ZCTAs cached, %d new for %s",
                 len(cached_ids), len(needed_ids), scenario)
    else:
        needed_ids = zcta_ids
        log.info("No cache; all %d ZCTAs needed", len(needed_ids))

    if not needed_ids:
        log.info("All ZCTAs cached. Nothing to do.")
        return

    if dry_run:
        log.info("[DRY RUN] Would extract %d ZCTAs for %s", len(needed_ids), scenario)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download ZCTA boundaries as WGS84 parquet for DuckDB
        zcta_path = download_zcta_boundaries(s3, needed_ids, tmpdir)

        # Compute bbox from ZCTA boundaries
        import geopandas as gpd
        zcta_gdf = gpd.read_parquet(zcta_path)
        bounds = zcta_gdf.total_bounds
        bbox = (bounds[0] - 0.05, bounds[1] - 0.05,
                bounds[2] + 0.05, bounds[3] + 0.05)
        log.info("BBox for %s: %.4f, %.4f, %.4f, %.4f", scenario, *bbox)

        # Push-down: scan + join + aggregate in DuckDB
        new_rows = extract_and_aggregate_pushdown(zcta_path, bbox)

    elapsed = time.time() - t0

    if len(new_rows) == 0:
        log.warning("No buildings extracted for %s", scenario)

    # Stage + consolidate
    if upload and len(new_rows) > 0:
        write_scenario_staging(s3, scenario, new_rows)
        combined = consolidate_cache(s3)
    else:
        combined = new_rows

    # QA
    qa_result = qa_buildings(new_rows, zcta_ids) if len(new_rows) > 0 else {"passed": True, "findings": []}

    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "version": "v2-pushdown",
        "n_zctas_requested": len(zcta_ids),
        "n_zctas_cached": len(zcta_ids) - len(needed_ids),
        "n_zctas_fetched": len(new_rows),
        "n_zctas_in_cache_total": len(combined) if combined is not None else 0,
        "total_buildings": int(new_rows["building_count"].sum()) if len(new_rows) > 0 else 0,
        "total_area_m2": float(new_rows["total_footprint_area_m2"].sum()) if len(new_rows) > 0 else 0.0,
        "elapsed_sec": round(elapsed, 1),
        "qa": qa_result,
        "git_hash": git_hash,
    }

    print("\n" + "=" * 60)
    print(f"  RESULT: {scenario}")
    for k, v in evidence.items():
        if k not in ("qa",):
            print(f"    {k}: {v}")
    print("=" * 60)
    sys.stdout.flush()

    if upload:
        key = f"results/s035/buildings_extraction_{scenario}_v2.json"
        upload_json_result(s3, BUCKET, key, evidence, git_hash=git_hash)

    print(json.dumps(evidence, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Overture buildings per ZCTA (v2 push-down)"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.scenario, args.upload, args.dry_run)


if __name__ == "__main__":
    main()
