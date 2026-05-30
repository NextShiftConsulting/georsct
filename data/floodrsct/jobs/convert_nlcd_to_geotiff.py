#!/usr/bin/env python3
"""
convert_nlcd_to_geotiff.py -- One-time conversion of NLCD .img to GeoTIFF.

Downloads the 26 GB Erdas Imagine HFA file from S3, converts to LZW-compressed
GeoTIFF via gdal_translate (lossless), uploads the .tif back to S3.

This eliminates the 10-15 min runtime conversion on every build_event_dataset
run for houston/nyc scenarios.

Output:
  s3://swarm-floodrsct-data/raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.tif

Usage (via SageMaker):
    python convert_nlcd_to_geotiff.py
"""

import logging
import subprocess
import sys
from pathlib import Path

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
IMG_KEY = "raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.img"
TIF_KEY = "raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.tif"


def main() -> None:
    s3 = boto3.client("s3", region_name="us-east-1")

    # Check if .tif already exists
    try:
        s3.head_object(Bucket=BUCKET, Key=TIF_KEY)
        log.info("Output already exists: s3://%s/%s -- skipping", BUCKET, TIF_KEY)
        return
    except s3.exceptions.ClientError:
        pass

    # Download .img
    local_img = "/tmp/nlcd_2021_impervious_l48.img"
    log.info("Downloading s3://%s/%s (26 GB, ~3-5 min)", BUCKET, IMG_KEY)
    s3.download_file(BUCKET, IMG_KEY, local_img)
    log.info("Download complete: %.1f GB", Path(local_img).stat().st_size / 1e9)

    # Convert to GeoTIFF (lossless LZW compression)
    local_tif = "/tmp/nlcd_2021_impervious_l48.tif"
    log.info("Converting .img -> .tif via gdal_translate (LZW, lossless)")
    result = subprocess.run(
        ["gdal_translate", "-of", "GTiff", "-co", "COMPRESS=LZW",
         "-co", "TILED=YES", "-co", "BIGTIFF=YES",
         local_img, local_tif],
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        log.error("gdal_translate failed: %s", result.stderr[:1000])
        sys.exit(1)

    tif_size = Path(local_tif).stat().st_size
    log.info("Conversion complete: %.1f GB", tif_size / 1e9)

    # Free disk -- delete .img before uploading .tif
    Path(local_img).unlink(missing_ok=True)
    log.info("Deleted local .img to free disk space")

    # Upload .tif to S3
    log.info("Uploading s3://%s/%s (%.1f GB)", BUCKET, TIF_KEY, tif_size / 1e9)
    s3.upload_file(local_tif, BUCKET, TIF_KEY)
    log.info("Upload complete")

    # Verify
    resp = s3.head_object(Bucket=BUCKET, Key=TIF_KEY)
    log.info("Verified: s3://%s/%s (%d bytes)", BUCKET, TIF_KEY, resp["ContentLength"])

    # Cleanup
    Path(local_tif).unlink(missing_ok=True)
    log.info("Done -- NLCD GeoTIFF ready for build_event_dataset.py")


if __name__ == "__main__":
    main()
