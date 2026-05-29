#!/usr/bin/env python3
"""
fetch_usgs_subsidence_no.py -- Fetch USGS InSAR-derived subsidence velocity
raster for the New Orleans / Mississippi River Delta region.

Source: USGS data release doi:10.5066/P9VFMF9K (Qu et al.)
        Vertical displacement rate (mm/yr) from InSAR time series analysis.

The USGS ScienceBase API provides download links for the GeoTIFF rasters.
This script downloads the velocity raster and uploads to S3.

Output: s3://swarm-floodrsct-data/raw/usgs_subsidence/no_subsidence_v1/velocity_mmyr.tif
"""

import hashlib
import logging
import sys
from pathlib import Path

import boto3
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
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
S3_PREFIX = "raw/usgs_subsidence/no_subsidence_v1"

# USGS ScienceBase item for the InSAR subsidence dataset
SCIENCEBASE_ITEM_URL = (
    "https://www.sciencebase.gov/catalog/item/5f6a285d82ce38aaa24217e0"
)
SCIENCEBASE_FILES_URL = (
    "https://www.sciencebase.gov/catalog/item/5f6a285d82ce38aaa24217e0"
    "?format=json&fields=files"
)

# Direct download URL for the velocity raster (backup if ScienceBase API changes)
DIRECT_URL = (
    "https://www.sciencebase.gov/catalog/file/get/5f6a285d82ce38aaa24217e0"
)


def find_velocity_tif_url() -> str:
    """Query ScienceBase API to find the velocity GeoTIFF download URL."""
    log.info("Querying ScienceBase for subsidence dataset files")
    try:
        resp = requests.get(SCIENCEBASE_FILES_URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        files = data.get("files", [])
        for f in files:
            name = f.get("name", "").lower()
            if "velocity" in name and name.endswith(".tif"):
                url = f.get("downloadUri") or f.get("url")
                if url:
                    log.info("Found velocity raster: %s (%s)", f["name"], url)
                    return url
        # Fallback: look for any GeoTIFF
        for f in files:
            if f.get("name", "").lower().endswith(".tif"):
                url = f.get("downloadUri") or f.get("url")
                if url:
                    log.info("Using GeoTIFF: %s (%s)", f["name"], url)
                    return url
    except requests.RequestException as e:
        log.warning("ScienceBase API query failed: %s", e)

    log.info("Falling back to direct download URL")
    return DIRECT_URL


def download_raster(url: str, local_path: str) -> None:
    """Download a raster file with progress logging."""
    log.info("Downloading from %s", url)
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    total = 0
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)

    log.info("Downloaded %.1f MB to %s", total / 1e6, local_path)


def main() -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    s3_key = f"{S3_PREFIX}/velocity_mmyr.tif"

    # Check if already uploaded
    try:
        s3.head_object(Bucket=BUCKET, Key=s3_key)
        log.info("Already exists: s3://%s/%s -- skipping", BUCKET, s3_key)
        return
    except s3.exceptions.ClientError:
        pass

    url = find_velocity_tif_url()

    local_path = "/tmp/velocity_mmyr.tif"
    download_raster(url, local_path)

    file_size = Path(local_path).stat().st_size
    with open(local_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    s3.upload_file(local_path, BUCKET, s3_key)
    log.info("Uploaded s3://%s/%s (%.1f MB)", BUCKET, s3_key, file_size / 1e6)

    write_manifest(
        s3=s3,
        dataset="usgs_subsidence_no",
        version="v1",
        source_url=SCIENCEBASE_ITEM_URL,
        s3_key=s3_key,
        crs="EPSG:4326",
        record_count=1,
        sha256=sha,
        license_="Public Domain -- USGS",
        notes="InSAR-derived vertical displacement velocity (mm/yr) for "
              "Mississippi River Delta / NOLA metro. Qu et al. "
              "doi:10.5066/P9VFMF9K. Used for subsidence_rate_mm_yr feature.",
    )

    log.info("fetch_usgs_subsidence_no complete")


if __name__ == "__main__":
    main()
