#!/usr/bin/env python3
"""
fetch_tiger_coastline.py -- Fetch TIGER/Line coastline shapefile from Census Bureau.

Downloads the national coastline shapefile (tl_2020_us_coastline.zip), converts
to GeoParquet, and uploads to S3.

Source: https://www2.census.gov/geo/tiger/TIGER2020/COASTLINE/
Output: s3://swarm-floodrsct-data/raw/tiger/coastline/v2020/us_coastline.parquet
"""

import hashlib
import io
import logging
import sys
import tempfile
import zipfile
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
S3_KEY = "raw/tiger/coastline/v2020/us_coastline.parquet"
SOURCE_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2020/COASTLINE/"
    "tl_2020_us_coastline.zip"
)


def fetch_coastline() -> gpd.GeoDataFrame:
    """Download TIGER coastline shapefile and return as GeoDataFrame."""
    log.info("Downloading TIGER coastline from %s", SOURCE_URL)
    resp = requests.get(SOURCE_URL, timeout=300)
    resp.raise_for_status()
    log.info("Downloaded %.1f MB", len(resp.content) / 1e6)

    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        zf.extractall(tmpdir)
        shp_files = list(Path(tmpdir).glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError("No .shp file in TIGER coastline zip")
        gdf = gpd.read_file(shp_files[0])

    log.info("Loaded %d coastline segments, CRS=%s", len(gdf), gdf.crs)

    # Ensure EPSG:4326
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
        log.info("Reprojected to EPSG:4326")

    return gdf


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

    gdf = fetch_coastline()

    # Write to parquet
    local_path = "/tmp/us_coastline.parquet"
    gdf.to_parquet(local_path, index=False)
    file_size = Path(local_path).stat().st_size

    with open(local_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    s3.upload_file(local_path, BUCKET, S3_KEY)
    log.info("Uploaded s3://%s/%s (%.1f MB)", BUCKET, S3_KEY, file_size / 1e6)

    write_manifest(
        s3=s3,
        dataset="tiger_coastline",
        version="v2020",
        source_url=SOURCE_URL,
        s3_key=S3_KEY,
        crs="EPSG:4326",
        record_count=len(gdf),
        sha256=sha,
        license_="Public Domain -- U.S. Census Bureau TIGER/Line",
        notes="National coastline shapefile. Used for coastal_distance_m "
              "feature in southwest_florida scenario.",
    )

    log.info("fetch_tiger_coastline complete: %d segments", len(gdf))


if __name__ == "__main__":
    main()
