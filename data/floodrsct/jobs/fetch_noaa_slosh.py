#!/usr/bin/env python3
"""
fetch_noaa_slosh.py -- Download NHC SLOSH MOM national inundation grids.

MOM (Maximum of MEOWs) grids are basin-specific, not storm-specific.
They represent worst-case surge envelopes by Saffir-Simpson category
for the entire US coastline. One download covers all events.

Source: https://www.nhc.noaa.gov/gis/hazardmaps/US_SLOSH_MOM_Inundation_v4.zip

Output:
  s3://swarm-floodrsct-data/raw/noaa_slosh/mom_national/
    US_SLOSH_MOM_Inundation_v4.*  (GeoTIFF + world file + metadata)

Feature contract fields derived downstream:
  - slosh_max_surge_m (max MOM inundation within ZCTA, by storm category)
  - slosh_category (Saffir-Simpson category from MOM)

Usage:
    python fetch_noaa_slosh.py
"""

import logging
import os
import sys
import zipfile
from pathlib import Path

import boto3
import requests
from swarm_auth import get_aws_credentials

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
S3_PREFIX = "raw/noaa_slosh/mom_national"

MOM_URL = "https://www.nhc.noaa.gov/gis/hazardmaps/US_SLOSH_MOM_Inundation_v4.zip"
MOM_ZIP_NAME = "US_SLOSH_MOM_Inundation_v4.zip"

# Extensions worth keeping from the zip
KEEP_EXTENSIONS = {".tif", ".tfw", ".xml", ".prj", ".dbf", ".shp", ".shx", ".cpg"}


def download_mom_zip(dest_dir: str) -> str:
    """Download the national MOM zip. Returns path to local zip file."""
    local_path = os.path.join(dest_dir, MOM_ZIP_NAME)
    log.info("Downloading %s", MOM_URL)
    resp = requests.get(MOM_URL, timeout=600, stream=True)
    resp.raise_for_status()

    total = 0
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            total += len(chunk)
    log.info("Downloaded %s (%.1f MB)", MOM_ZIP_NAME, total / 1e6)
    return local_path


def extract_and_upload(zip_path: str, dest_dir: str) -> int:
    """Extract zip, upload relevant files to S3. Returns file count."""
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    uploaded = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        log.info("Zip contains %d entries", len(members))

        for member in members:
            # Skip directories
            if member.endswith("/"):
                continue

            ext = os.path.splitext(member)[1].lower()
            if ext not in KEEP_EXTENSIONS:
                log.debug("Skipping %s (extension %s)", member, ext)
                continue

            # Extract to temp
            basename = os.path.basename(member)
            local_file = os.path.join(dest_dir, basename)
            with zf.open(member) as src, open(local_file, "wb") as dst:
                dst.write(src.read())

            # Upload to S3
            s3_key = f"{S3_PREFIX}/{basename}"
            file_size = os.path.getsize(local_file)
            log.info("Uploading %s (%.1f MB) -> s3://%s/%s",
                     basename, file_size / 1e6, BUCKET, s3_key)
            s3.upload_file(local_file, BUCKET, s3_key)
            uploaded += 1

            # Clean up local to save disk
            os.unlink(local_file)

    return uploaded


def main() -> None:
    work_dir = "/tmp/slosh_mom"
    os.makedirs(work_dir, exist_ok=True)

    # Check if already uploaded
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=S3_PREFIX, MaxKeys=1)
    existing = resp.get("KeyCount", 0)
    if existing > 0:
        log.info("MOM data already exists at s3://%s/%s/ (%d objects). "
                 "Delete prefix to re-download.", BUCKET, S3_PREFIX, existing)
        return

    zip_path = download_mom_zip(work_dir)
    count = extract_and_upload(zip_path, work_dir)
    log.info("Done. Uploaded %d files to s3://%s/%s/", count, BUCKET, S3_PREFIX)

    # Clean up zip
    os.unlink(zip_path)


if __name__ == "__main__":
    main()
