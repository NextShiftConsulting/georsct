#!/usr/bin/env python3
"""
stage_nlcd_geotiff.py -- Download NLCD zip, convert HFA to GeoTIFF, upload to S3.

The NLCD 2021 impervious surface raster is distributed as a zip from ScienceBase
containing Erdas Imagine HFA format (.img + .ige companion for large files).
The prior fetch script only extracted one file, producing an unreadable artifact.

This script downloads the zip fresh, extracts ALL companions into the same
directory so rasterio/GDAL can resolve them, converts to GeoTIFF via windowed
I/O, and uploads the result.

Input:  ScienceBase zip (~1.2 GB)
Output: s3://swarm-floodrsct-data/raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.tif

One-time job. Unblocks impervious_pct in build_event_dataset.py.

Usage:
    python stage_nlcd_geotiff.py --upload
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

SOURCE_URL = (
    "https://www.sciencebase.gov/catalog/file/get/"
    "649595c3d34ef77fcb01dc9e?name=nlcd_2021_impervious_l48_20230630.zip"
)
DST_KEY = "raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.tif"
EXTRACT_DIR = "/tmp/nlcd_extract"


def download_zip(url: str, dest: str, timeout: int = 1800) -> None:
    """Stream-download zip from URL."""
    log.info("Downloading NLCD zip (~1.2 GB) from ScienceBase...")
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    written = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
            f.write(chunk)
            written += len(chunk)
            if written % (100 * 1024 * 1024) < len(chunk):
                log.info("  downloaded %.0f MB so far", written / 1e6)
    size_mb = Path(dest).stat().st_size / 1e6
    log.info("Download complete: %.1f MB", size_mb)


def extract_all(zip_path: str, dest_dir: str) -> str:
    """Extract ALL files from zip, preserving basenames. Returns .img path."""
    os.makedirs(dest_dir, exist_ok=True)
    img_path = None

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        log.info("Zip contains %d files:", len(members))
        for name in members:
            log.info("  %s", name)

        for name in members:
            basename = Path(name).name
            if basename.startswith("__") or basename.startswith("."):
                continue
            out_path = os.path.join(dest_dir, basename)
            log.info("Extracting: %s", basename)
            with zf.open(name) as src, open(out_path, "wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            size_mb = Path(out_path).stat().st_size / 1e6
            log.info("  -> %.1f MB", size_mb)
            if out_path.lower().endswith(".img"):
                img_path = out_path

    if img_path is None:
        raise FileNotFoundError("No .img file found in zip")
    return img_path


def convert_to_geotiff(img_path: str, tif_path: str) -> None:
    """Convert HFA .img to GeoTIFF via rasterio windowed I/O."""
    import rasterio

    log.info("Opening %s with rasterio...", img_path)
    with rasterio.open(img_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            compress="lzw",
            tiled=True,
            bigtiff="YES",
        )
        log.info(
            "Source: %d x %d, %d band(s), dtype=%s, driver=%s",
            src.width, src.height, src.count, src.dtypes[0], src.driver,
        )
        log.info("Converting to GeoTIFF (LZW, tiled, windowed)...")
        with rasterio.open(tif_path, "w", **profile) as dst:
            windows = list(src.block_windows(1))
            log.info("Total blocks: %d", len(windows))
            for i, (_, window) in enumerate(windows):
                data = src.read(1, window=window)
                dst.write(data, 1, window=window)
                if (i + 1) % 500 == 0:
                    log.info("  block %d / %d", i + 1, len(windows))

    tif_size = Path(tif_path).stat().st_size
    log.info("Conversion complete: %.1f GB", tif_size / 1e9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    # Check if .tif already exists on S3
    try:
        s3.head_object(Bucket=BUCKET, Key=DST_KEY)
        log.info("Already exists: s3://%s/%s -- nothing to do", BUCKET, DST_KEY)
        return
    except Exception:
        pass

    zip_path = "/tmp/nlcd_impervious.zip"
    tif_path = "/tmp/nlcd_impervious.tif"

    # Step 1: Download zip from ScienceBase
    download_zip(SOURCE_URL, zip_path)

    # Step 2: Extract ALL files (preserving .img + .ige companions)
    img_path = extract_all(zip_path, EXTRACT_DIR)

    # Free zip disk space
    Path(zip_path).unlink(missing_ok=True)
    log.info("Deleted zip to free disk space")

    # Log all extracted files
    log.info("Extracted files in %s:", EXTRACT_DIR)
    for f in sorted(Path(EXTRACT_DIR).iterdir()):
        log.info("  %s: %.1f MB", f.name, f.stat().st_size / 1e6)

    # Step 3: Convert to GeoTIFF
    convert_to_geotiff(img_path, tif_path)

    # Free extracted files
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    log.info("Deleted extracted files to free disk space")

    # Step 4: Upload
    if args.upload:
        log.info("Uploading to s3://%s/%s ...", BUCKET, DST_KEY)
        s3.upload_file(tif_path, BUCKET, DST_KEY)
        tif_size = Path(tif_path).stat().st_size
        log.info("Upload complete. Size: %.1f GB", tif_size / 1e9)
    else:
        log.info("Saved locally: %s (use --upload to stage on S3)", tif_path)

    log.info("stage_nlcd_geotiff complete")


if __name__ == "__main__":
    main()
