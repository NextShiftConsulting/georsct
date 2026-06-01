#!/usr/bin/env python3
"""
stage_nlcd_geotiff.py -- Convert NLCD impervious .img to GeoTIFF on S3.

The NLCD 2021 impervious surface raster is on S3 as .img (Erdas Imagine HFA
format). pip-installed rasterio on the SageMaker PyTorch image cannot read
HFA without the GDAL HFA driver. This script converts to GeoTIFF using
gdal_translate (available on images with apt-get install gdal-bin).

Input:  s3://swarm-floodrsct-data/raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.img
Output: s3://swarm-floodrsct-data/raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.tif

One-time job. Unblocks impervious_pct in build_event_dataset.py.

Usage:
    python stage_nlcd_geotiff.py --upload
"""

import argparse
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

SRC_KEY = "raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.img"
DST_KEY = "raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.tif"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    # Check if .tif already exists
    try:
        s3.head_object(Bucket=BUCKET, Key=DST_KEY)
        log.info("Already exists: s3://%s/%s -- nothing to do", BUCKET, DST_KEY)
        return
    except Exception:
        pass

    # Download .img (26 GB)
    local_img = "/tmp/nlcd_impervious.img"
    local_tif = "/tmp/nlcd_impervious.tif"
    log.info("Downloading %s (26 GB, ~5 min)...", SRC_KEY)
    s3.download_file(BUCKET, SRC_KEY, local_img)
    img_size = Path(local_img).stat().st_size
    log.info("Downloaded: %.1f GB", img_size / 1e9)

    # Convert via gdal_translate
    log.info("Converting .img -> .tif via gdal_translate (LZW compression)...")
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
    log.info("Conversion complete: %.1f GB (%.0f%% of original)",
             tif_size / 1e9, tif_size / max(img_size, 1) * 100)

    # Remove .img to free disk space before upload
    Path(local_img).unlink(missing_ok=True)

    if args.upload:
        log.info("Uploading to s3://%s/%s...", BUCKET, DST_KEY)
        s3.upload_file(local_tif, BUCKET, DST_KEY)

        with open(local_tif, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        log.info("Upload complete. SHA-256: %s", sha[:16])
    else:
        log.info("Saved locally: %s (use --upload to stage on S3)", local_tif)

    log.info("stage_nlcd_geotiff complete")


if __name__ == "__main__":
    main()
