#!/usr/bin/env python3
"""
fetch_nsi_structures.py -- SageMaker job: download NSI 2.0 building inventory.

Queries USACE National Structure Inventory (NSI) 2.0 API for all structures
in each scenario's counties. Required for FEMA FAST depth-damage analysis.

NSI 2.0 API: https://nsi.sec.usace.army.mil/nsiapi/structures
Public, no authentication required. Returns GeoJSON per FIPS code.

Outputs GeoParquet (one per scenario) to S3:
  s3://swarm-floodrsct-data/raw/nsi/v2/houston_structures.parquet
  s3://swarm-floodrsct-data/raw/nsi/v2/nyc_structures.parquet
  s3://swarm-floodrsct-data/raw/nsi/v2/southwest_florida_structures.parquet

Usage:
    python fetch_nsi_structures.py --scenario houston
    python fetch_nsi_structures.py --scenario nyc
    python fetch_nsi_structures.py --scenario southwest_florida
    python fetch_nsi_structures.py --all
"""

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import boto3
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from swarm_auth import get_aws_credentials

sys.path.insert(0, str(Path(__file__).parent))
from _manifest_writer import write_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
NSI_API_BASE = "https://nsi.sec.usace.army.mil/nsiapi/structures"
REQUEST_SLEEP = 2.0  # seconds between county requests (be polite)

# Required fields from NSI 2.0 (per DOE_FAST_validation.md)
KEEP_COLS = [
    "fd_id", "x", "y", "occtype", "sqft", "found_type",
    "num_story", "found_ht", "val_struct", "val_cont", "med_yr_blt",
    "st_damcat", "cbfips",
]

# Scenario → county FIPS codes
SCENARIO_COUNTIES = {
    "houston": ["48201"],  # Harris County
    "nyc": ["36061", "36047", "36081", "36005", "36085"],  # 5 boroughs
    "southwest_florida": [
        "12021", "12071", "12115", "12081", "12057", "12103",
    ],  # Charlotte, Lee, Sarasota, Manatee, Hillsborough, Pinellas
}


def fetch_county_structures(fips: str) -> gpd.GeoDataFrame:
    """Fetch all NSI 2.0 structures for a single county FIPS."""
    url = f"{NSI_API_BASE}"
    params = {"fips": fips}
    log.info("Fetching NSI structures for FIPS %s", fips)

    resp = requests.get(url, params=params, timeout=300)
    if resp.status_code == 404:
        log.warning("No structures found for FIPS %s (404)", fips)
        return gpd.GeoDataFrame()
    resp.raise_for_status()

    data = resp.json()
    features = data.get("features", [])
    if not features:
        log.warning("Empty feature collection for FIPS %s", fips)
        return gpd.GeoDataFrame()

    log.info("FIPS %s: %d structures returned", fips, len(features))

    # Parse GeoJSON features
    rows = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        row = {k: props.get(k) for k in KEEP_COLS}
        # Fallback: use x/y from properties if geometry missing
        if coords[0] is None:
            row["_lon"] = props.get("x")
            row["_lat"] = props.get("y")
        else:
            row["_lon"] = coords[0]
            row["_lat"] = coords[1]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Build geometry
    geometry = [
        Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
        for lon, lat in zip(df["_lon"], df["_lat"])
    ]
    gdf = gpd.GeoDataFrame(df.drop(columns=["_lon", "_lat"]), geometry=geometry, crs="EPSG:4326")
    return gdf


def fetch_scenario(scenario: str) -> gpd.GeoDataFrame:
    """Fetch NSI structures for all counties in a scenario."""
    counties = SCENARIO_COUNTIES[scenario]
    frames = []
    for fips in counties:
        gdf = fetch_county_structures(fips)
        if not gdf.empty:
            frames.append(gdf)
        time.sleep(REQUEST_SLEEP)

    if not frames:
        log.error("No structures fetched for scenario %s", scenario)
        return gpd.GeoDataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

    # Deduplicate by fd_id (structures near county boundaries)
    if "fd_id" in combined.columns:
        n_before = len(combined)
        combined = combined.drop_duplicates(subset=["fd_id"], keep="first")
        n_dupes = n_before - len(combined)
        if n_dupes:
            log.info("Dropped %d duplicate structures (cross-county)", n_dupes)

    log.info("Scenario %s: %d structures total across %d counties",
             scenario, len(combined), len(counties))
    return combined


def upload(gdf: gpd.GeoDataFrame, s3_key: str, s3) -> str:
    """Write GeoParquet to S3, return SHA-256."""
    local = f"/tmp/{Path(s3_key).name}"
    gdf.to_parquet(local, index=False)
    file_size = Path(local).stat().st_size

    with open(local, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    s3.upload_file(local, BUCKET, s3_key)
    log.info("Uploaded s3://%s/%s (%.1f MB, %d structures)",
             BUCKET, s3_key, file_size / 1e6, len(gdf))
    return sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIO_COUNTIES.keys()))
    parser.add_argument("--all", action="store_true",
                        help="Fetch all scenarios")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        parser.error("Specify --scenario or --all")

    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    scenarios = list(SCENARIO_COUNTIES.keys()) if args.all else [args.scenario]

    for scenario in scenarios:
        s3_key = f"raw/nsi/v2/{scenario}_structures.parquet"

        # Check if already uploaded
        try:
            s3.head_object(Bucket=BUCKET, Key=s3_key)
            log.info("Already exists: s3://%s/%s -- skipping", BUCKET, s3_key)
            continue
        except s3.exceptions.ClientError:
            pass

        gdf = fetch_scenario(scenario)
        if gdf.empty:
            log.error("No data for %s -- skipping upload", scenario)
            continue

        sha = upload(gdf, s3_key, s3)

        write_manifest(
            s3=s3,
            dataset="nsi_structures",
            version=f"v2/{scenario}",
            source_url=NSI_API_BASE,
            s3_key=s3_key,
            crs="EPSG:4326",
            record_count=len(gdf),
            sha256=sha,
            license_="Public Domain -- USACE National Structure Inventory 2.0",
            notes=f"NSI 2.0 structures for {scenario}. "
                  f"Counties: {', '.join(SCENARIO_COUNTIES[scenario])}. "
                  f"Required for FEMA FAST depth-damage analysis (DOE Change 11).",
        )

    log.info("fetch_nsi_structures complete")


if __name__ == "__main__":
    main()
