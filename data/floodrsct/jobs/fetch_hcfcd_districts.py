#!/usr/bin/env python3
"""
fetch_hcfcd_districts.py -- Fetch Harris County Flood Control District
(HCFCD) watershed/drainage district boundaries.

Source: Harris County Open Data (data.hcfcd.org)
        ArcGIS REST endpoint for HCFCD watershed boundaries.

Output: s3://swarm-floodrsct-data/raw/hcfcd/drainage_districts/v1/hcfcd_districts.parquet
"""

import hashlib
import logging
import sys
from pathlib import Path

import boto3
import geopandas as gpd
import pandas as pd
import requests
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
S3_KEY = "raw/hcfcd/drainage_districts/v1/hcfcd_districts.parquet"

# HCFCD ArcGIS REST endpoint for watershed boundaries
# This is the public MapServer layer for HCFCD drainage areas
ARCGIS_URL = (
    "https://www.gis.hctx.net/arcgishcpid/rest/services/"
    "HCFCD/Watershed/MapServer/1/query"
)

# Fallback: ArcGIS Online hosted feature layer
GEOJSON_URL = (
    "https://services.arcgis.com/0ZrJhVGlMgJkF4iG/arcgis/rest/services/"
    "HCFCD_Watersheds/FeatureServer/0/query"
)


def fetch_from_arcgis() -> gpd.GeoDataFrame:
    """Fetch HCFCD district boundaries via ArcGIS REST API."""
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "returnGeometry": "true",
    }
    log.info("Querying HCFCD ArcGIS endpoint")
    resp = requests.get(ARCGIS_URL, params=params, timeout=120)
    resp.raise_for_status()

    gdf = gpd.read_file(resp.text, driver="GeoJSON")
    log.info("ArcGIS returned %d features", len(gdf))
    return gdf


def fetch_from_geojson() -> gpd.GeoDataFrame:
    """Fallback: fetch from ArcGIS Online hosted feature layer."""
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "returnGeometry": "true",
    }
    log.info("Trying ArcGIS Online feature layer endpoint")
    resp = requests.get(GEOJSON_URL, params=params, timeout=120)
    resp.raise_for_status()

    gdf = gpd.read_file(resp.text, driver="GeoJSON")
    log.info("ArcGIS Online returned %d features", len(gdf))
    return gdf


def fetch_districts() -> gpd.GeoDataFrame:
    """Try ArcGIS first, fall back to GeoJSON export."""
    try:
        gdf = fetch_from_arcgis()
        if not gdf.empty:
            return gdf
    except Exception as e:
        log.warning("ArcGIS fetch failed: %s", e)

    try:
        gdf = fetch_from_geojson()
        if not gdf.empty:
            return gdf
    except Exception as e:
        log.warning("GeoJSON fetch failed: %s", e)

    log.error(
        "Both endpoints failed. HCFCD may require manual download from "
        "https://data.hcfcd.org/ -- search for 'Watersheds' dataset."
    )
    return gpd.GeoDataFrame()


def main() -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    # Check if already uploaded
    try:
        s3.head_object(Bucket=BUCKET, Key=S3_KEY)
        log.info("Already exists: s3://%s/%s -- skipping", BUCKET, S3_KEY)
        return
    except s3.exceptions.ClientError:
        pass

    gdf = fetch_districts()
    if gdf.empty:
        log.error("No district data fetched; aborting upload")
        sys.exit(1)

    # Ensure EPSG:4326
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
        log.info("Reprojected to EPSG:4326")

    local_path = "/tmp/hcfcd_districts.parquet"
    gdf.to_parquet(local_path, index=False)
    file_size = Path(local_path).stat().st_size

    with open(local_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    s3.upload_file(local_path, BUCKET, S3_KEY)
    log.info("Uploaded s3://%s/%s (%.1f KB)", BUCKET, S3_KEY, file_size / 1e3)

    write_manifest(
        s3=s3,
        dataset="hcfcd_drainage_districts",
        version="v1",
        source_url="https://data.hcfcd.org/",
        s3_key=S3_KEY,
        crs="EPSG:4326",
        record_count=len(gdf),
        sha256=sha,
        license_="Public Domain -- Harris County Flood Control District",
        notes="HCFCD watershed/drainage district boundaries for Harris County. "
              "Used for drainage_district_id feature in houston scenario.",
    )

    log.info("fetch_hcfcd_districts complete: %d districts", len(gdf))


if __name__ == "__main__":
    main()
