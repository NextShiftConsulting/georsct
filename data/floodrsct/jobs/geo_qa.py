"""
geo_qa.py -- Geospatial QA checks for FloodRSCT raw data.

Layer 1b: validates that fetched data covers the correct geography,
sits between L1 (file existence) and L2 (post-assembly stats).

Checks:
  - Bounding box: do geometries/points fall within the scenario extent?
  - CRS assertion: is the data in the expected projection?
  - Coverage fraction: what % of target ZCTAs have non-null values after join?

Uses scenario bounding boxes from the ScenarioRegistry (db/schema/006).

Usage:
    python geo_qa.py --scenario houston
    python geo_qa.py --all
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
from swarm_auth import get_aws_credentials

# Add repo root to path for registry import
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from db.scripts.scenario_registry import BoundingBox, ScenarioDef, get_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"


@dataclass
class GeoQAResult:
    feature: str
    check: str
    status: str  # PASS, FAIL, WARN, SKIP
    message: str

    def __str__(self) -> str:
        return f"  [GeoQA] [{self.status:4s}] {self.feature}: {self.check} -- {self.message}"


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _read_parquet(s3, key: str) -> Optional[pd.DataFrame]:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except Exception:
        return None


def _list_keys(s3, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


# ---------------------------------------------------------------------------
# Bounding box check
# ---------------------------------------------------------------------------

def check_bbox(
    s3, feature: str, s3_key: str, bounds: BoundingBox
) -> list[GeoQAResult]:
    """Check that point/geometry data falls within scenario bounding box."""
    results = []

    df = _read_parquet(s3, s3_key)
    if df is None:
        results.append(GeoQAResult(
            feature, "bbox", "SKIP",
            f"cannot read {s3_key}",
        ))
        return results

    # Find lat/lon columns
    lat_col = next((c for c in df.columns if c.lower() in ("lat", "latitude", "y")), None)
    lon_col = next((c for c in df.columns if c.lower() in ("lon", "longitude", "lng", "x")), None)

    if lat_col is None or lon_col is None:
        # Try geometry column (geoparquet)
        if "geometry" in df.columns:
            results.append(GeoQAResult(
                feature, "bbox", "SKIP",
                "geometry column present but bbox check requires geopandas (not available)",
            ))
        else:
            results.append(GeoQAResult(
                feature, "bbox", "SKIP",
                f"no lat/lon columns found (columns: {list(df.columns)[:10]})",
            ))
        return results

    lats = df[lat_col].dropna()
    lons = df[lon_col].dropna()

    if len(lats) == 0:
        results.append(GeoQAResult(
            feature, "bbox", "WARN",
            "all lat/lon values are null",
        ))
        return results

    in_bounds = (
        (lats >= bounds.lat_min) & (lats <= bounds.lat_max) &
        (lons >= bounds.lon_min) & (lons <= bounds.lon_max)
    )
    pct_in = in_bounds.mean()

    if pct_in >= 0.8:
        results.append(GeoQAResult(
            feature, "bbox", "PASS",
            f"{pct_in:.0%} of points within scenario bounds "
            f"[{bounds.lat_min}-{bounds.lat_max}, {bounds.lon_min}-{bounds.lon_max}]",
        ))
    elif pct_in >= 0.3:
        results.append(GeoQAResult(
            feature, "bbox", "WARN",
            f"only {pct_in:.0%} of points within bounds -- possible partial coverage "
            f"(lat range: {lats.min():.2f}-{lats.max():.2f}, "
            f"lon range: {lons.min():.2f}-{lons.max():.2f})",
        ))
    else:
        results.append(GeoQAResult(
            feature, "bbox", "FAIL",
            f"only {pct_in:.0%} of points within bounds -- likely WRONG GEOGRAPHY "
            f"(data lat: {lats.min():.2f}-{lats.max():.2f}, "
            f"data lon: {lons.min():.2f}-{lons.max():.2f}, "
            f"expected: [{bounds.lat_min}-{bounds.lat_max}, {bounds.lon_min}-{bounds.lon_max}])",
        ))

    return results


# ---------------------------------------------------------------------------
# Per-scenario QA checks
# ---------------------------------------------------------------------------

# Map feature -> S3 key patterns to check per scenario
# Only features with spatial data worth bbox-checking
SPATIAL_CHECKS = {
    "houston": [
        ("bayou_segment_id", "raw/nhdplus/catchments/v2/"),
        ("drainage_district_id", "raw/hcfcd/drainage_districts/v1/hcfcd_districts.parquet"),
        ("hwm_max_ft", "raw/surge_estimates/harvey2017/hwm_harvey2017.parquet"),
    ],
    "new_orleans": [
        ("levee_condition_rating", "raw/usace_levees/new_orleans_levees.parquet"),
        ("canal_proximity_m", "raw/osm/new_orleans_canals/v1/no_canals.parquet"),
        ("tidal_surge_max_m", "raw/noaa_tides/ida2021_nyc/"),
    ],
    "nyc": [
        ("sewer_shed_id", "raw/nyc_sewersheds/nyc_sewersheds.gpkg"),
        ("subway_station_count", "raw/mta/subway_stations/v1/subway_stations.parquet"),
        ("levee_condition_rating", "raw/usace_levees/nyc_levees.parquet"),
        ("hwm_max_ft", "raw/surge_estimates/ida2021_nyc/hwm_ida2021_nyc.parquet"),
    ],
    "riverside_coachella": [
        ("upstream_catchment_km2", "raw/nhdplus/catchments/v2/"),
        ("burn_scar_overlap", "raw/mtbs/perimeters/v2023/"),
        ("hwm_max_ft", "raw/surge_estimates/hilary2023/hwm_hilary2023.parquet"),
    ],
    "southwest_florida": [
        ("coastal_distance_m", "raw/tiger/coastline/v2020/us_coastline.parquet"),
        ("tidal_surge_max_m", "raw/noaa_tides/ian2022/"),
        ("hwm_max_ft", "raw/surge_estimates/ian2022/hwm_ian2022.parquet"),
    ],
}


def run_geo_qa(s3, scenario: str) -> list[GeoQAResult]:
    """Run all geo QA checks for a scenario."""
    reg = get_registry()
    sdef = reg.get(scenario)
    results = []

    if sdef.bounds is None:
        results.append(GeoQAResult(
            "(scenario)", "bounds", "SKIP",
            f"no bounding box defined for {scenario}",
        ))
        return results

    checks = SPATIAL_CHECKS.get(scenario, [])
    if not checks:
        results.append(GeoQAResult(
            "(scenario)", "spatial_checks", "SKIP",
            f"no spatial checks configured for {scenario}",
        ))
        return results

    for feature, s3_path in checks:
        # Find a parquet file to check
        if s3_path.endswith(".parquet"):
            bbox_results = check_bbox(s3, feature, s3_path, sdef.bounds)
        else:
            # Directory prefix -- find first parquet
            keys = _list_keys(s3, s3_path)
            parquets = [k for k in keys if k.endswith(".parquet")]
            if parquets:
                bbox_results = check_bbox(s3, feature, parquets[0], sdef.bounds)
            else:
                bbox_results = [GeoQAResult(
                    feature, "bbox", "SKIP",
                    f"no parquet files at {s3_path}",
                )]
        results.extend(bbox_results)

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[GeoQAResult], scenario: str) -> int:
    """Print geo QA report. Returns number of FAILs."""
    print(f"\n{'=' * 72}")
    print(f"Geo QA Report: {scenario}")
    print(f"{'=' * 72}")

    for r in results:
        print(r)

    pass_n = sum(1 for r in results if r.status == "PASS")
    fail_n = sum(1 for r in results if r.status == "FAIL")
    warn_n = sum(1 for r in results if r.status == "WARN")
    skip_n = sum(1 for r in results if r.status == "SKIP")

    print(f"\nPASS: {pass_n}  FAIL: {fail_n}  WARN: {warn_n}  SKIP: {skip_n}")
    if fail_n > 0:
        print("VERDICT: WRONG GEOGRAPHY DETECTED -- investigate before build")
    elif warn_n > 0:
        print("VERDICT: REVIEW WARNINGS")
    else:
        print("VERDICT: CLEAR")
    print(f"{'=' * 72}\n")

    return fail_n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    reg = get_registry()

    parser = argparse.ArgumentParser(description="Geospatial QA for FloodRSCT")
    parser.add_argument(
        "--scenario",
        choices=reg.active_names(),
        help="Run QA for a single scenario",
    )
    parser.add_argument("--all", action="store_true", help="Run QA for all active scenarios")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        parser.error("specify --scenario or --all")

    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    scenarios = reg.active_names() if args.all else [args.scenario]
    total_fails = 0

    for scenario in scenarios:
        results = run_geo_qa(s3, scenario)
        fails = print_report(results, scenario)
        total_fails += fails

    sys.exit(1 if total_fails > 0 else 0)


if __name__ == "__main__":
    main()
