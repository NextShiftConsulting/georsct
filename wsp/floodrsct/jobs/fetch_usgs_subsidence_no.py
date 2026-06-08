#!/usr/bin/env python3
"""
fetch_usgs_subsidence_no.py -- Fetch Delta-X total subsidence rate raster for
the Mississippi River Delta region.

Source: Delta-X Total Subsidence Rate, MRD, Louisiana, USA, 2016-2023
        ORNL DAAC, doi:10.3334/ORNLDAAC/2349
        Cloud Optimized GeoTIFF at 30 m resolution.
        Landing page: https://daac.ornl.gov/DELTAX/guides/DeltaX_TotalSubsidenceRate_MRD.html

NOTE: ORNL DAAC requires NASA Earthdata authentication.  If the download
fails with HTTP 401/403, set the EARTHDATA_TOKEN environment variable to a
valid NASA Earthdata bearer token, or download the file manually and upload
it to S3 at the key shown below.

Output: s3://swarm-floodrsct-data/raw/usgs_subsidence/no_subsidence_v1/velocity_mmyr.tif
"""

import hashlib
import logging
import os
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
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
S3_PREFIX = "raw/usgs_subsidence/no_subsidence_v1"

# ORNL DAAC landing page for the Delta-X total subsidence rate dataset
ORNL_LANDING_URL = (
    "https://daac.ornl.gov/DELTAX/guides/DeltaX_TotalSubsidenceRate_MRD.html"
)

# ORNL DAAC data directory for the dataset (doi:10.3334/ORNLDAAC/2349)
ORNL_DATA_DIR_URL = (
    "https://daac.ornl.gov/daacdata/deltax/DeltaX_TotalSubsidenceRate_MRD/data/"
)

# Known direct download URL for the COG GeoTIFF.
# File name is based on the ORNL DAAC dataset naming convention; update if
# ORNL renames the file.
DIRECT_URL = (
    "https://daac.ornl.gov/daacdata/deltax/DeltaX_TotalSubsidenceRate_MRD/"
    "data/DeltaX_TotalSubsidenceRate_MRD.tif"
)


def _earthdata_headers() -> dict:
    """Return Authorization header if EARTHDATA_TOKEN is set, else empty dict."""
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def find_velocity_tif_url() -> str:
    """Find the GeoTIFF download URL from the ORNL DAAC data directory.

    Tries to scrape the data directory listing for a .tif file.  Falls back
    to the known direct URL if the directory listing is unavailable.
    """
    log.info("Querying ORNL DAAC data directory: %s", ORNL_DATA_DIR_URL)
    try:
        resp = requests.get(
            ORNL_DATA_DIR_URL,
            headers=_earthdata_headers(),
            timeout=60,
            allow_redirects=True,
        )
        if resp.status_code in (401, 403):
            log.warning(
                "ORNL DAAC returned HTTP %s -- directory listing requires "
                "NASA Earthdata login.  Falling back to known direct URL.",
                resp.status_code,
            )
        else:
            resp.raise_for_status()
            # Parse a simple Apache/nginx directory listing for .tif hrefs
            # Prefer the rate file over the uncertainty file
            tif_urls = []
            for line in resp.text.splitlines():
                lower = line.lower()
                if ".tif" in lower and 'href="' in lower:
                    start = line.index('href="') + len('href="')
                    end = line.index('"', start)
                    filename = line[start:end]
                    if filename.endswith(".tif"):
                        url = (
                            filename
                            if filename.startswith("http")
                            else ORNL_DATA_DIR_URL.rstrip("/") + "/" + filename
                        )
                        tif_urls.append(url)
            # Pick rate file (not uncertainty)
            for url in tif_urls:
                if "uncertainty" not in url.lower():
                    log.info("Found rate GeoTIFF: %s", url)
                    return url
            if tif_urls:
                log.info("Found GeoTIFF (fallback): %s", tif_urls[0])
                return tif_urls[0]
    except requests.RequestException as e:
        log.warning("ORNL DAAC directory query failed: %s", e)

    log.info("Using known direct URL: %s", DIRECT_URL)
    return DIRECT_URL


def download_raster(url: str, local_path: str) -> None:
    """Download a raster file with progress logging.

    Passes a NASA Earthdata bearer token if EARTHDATA_TOKEN is set.
    Raises a clear RuntimeError on HTTP 401/403 so the user knows what to do.
    """
    log.info("Downloading from %s", url)
    headers = _earthdata_headers()
    if not headers:
        log.info(
            "No EARTHDATA_TOKEN set -- if the download fails with HTTP 401/403, "
            "set EARTHDATA_TOKEN to a valid NASA Earthdata bearer token, or "
            "download the file manually from %s and upload it to S3.",
            ORNL_LANDING_URL,
        )

    resp = requests.get(url, headers=headers, stream=True, timeout=600, allow_redirects=True)

    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"ORNL DAAC returned HTTP {resp.status_code} -- NASA Earthdata "
            "authentication required.  Set the EARTHDATA_TOKEN environment "
            "variable to a valid bearer token (obtain one at "
            "https://urs.earthdata.nasa.gov/), or download the file manually "
            f"from {ORNL_LANDING_URL} and upload it to "
            f"s3://{BUCKET}/{S3_PREFIX}/velocity_mmyr.tif"
        )
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
        source_url=ORNL_LANDING_URL,
        s3_key=s3_key,
        crs="EPSG:4326",
        record_count=1,
        sha256=sha,
        license_="NASA Open Data -- see ORNL DAAC data use policy",
        notes="Delta-X Total Subsidence Rate, MRD, Louisiana, USA, 2016-2023. "
              "Cloud Optimized GeoTIFF at 30 m resolution. "
              "ORNL DAAC doi:10.3334/ORNLDAAC/2349. "
              "Used for subsidence_rate_mm_yr feature.",
    )

    log.info("fetch_usgs_subsidence_no complete")


if __name__ == "__main__":
    main()
