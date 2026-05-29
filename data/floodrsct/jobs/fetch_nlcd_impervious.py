"""
fetch_nlcd_impervious.py -- Download NLCD 2021 Impervious Surface raster.

SageMaker container script (NOT a launcher).
Source: MRLC NLCD 2021 Impervious Surface, CONUS (via USGS ScienceBase)
URL: https://www.sciencebase.gov/catalog/file/get/649595c3d34ef77fcb01dc9e?name=nlcd_2021_impervious_l48_20230630.zip

Output:
  s3://swarm-floodrsct-data/raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.img
"""

import logging
import os
import sys
import uuid
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _manifest_writer import write_manifest
from _s3_stream import s3_key_exists, get_s3, stream_to_tmp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("fetch_nlcd_impervious")

BUCKET = "swarm-floodrsct-data"
SOURCE_URL = (
    "https://www.sciencebase.gov/catalog/file/get/"
    "649595c3d34ef77fcb01dc9e?name=nlcd_2021_impervious_l48_20230630.zip"
)
S3_KEY = "raw/nlcd/impervious_2021/nlcd_2021_impervious_l48.img"

RASTER_EXTENSIONS = (".tif", ".tiff", ".img", ".ige")
TMP_DIR = "/tmp"


def find_raster_in_zip(zip_path: str) -> str:
    """Return the name of the first raster file (.tif/.img) inside the zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            lower = name.lower()
            if any(lower.endswith(ext) for ext in RASTER_EXTENSIONS):
                return name
    raise FileNotFoundError(
        f"No raster file found in zip (looked for {RASTER_EXTENSIONS})"
    )


def main() -> None:
    s3 = get_s3()

    # ---- checkpoint: skip if already uploaded ----
    if s3_key_exists(s3, BUCKET, S3_KEY):
        log.info("Checkpoint hit -- already in S3: s3://%s/%s", BUCKET, S3_KEY)
        log.info("Done (skipped).")
        return

    # ---- download zip ----
    zip_name = f"{uuid.uuid4().hex}_nlcd_impervious.zip"
    zip_path = os.path.join(TMP_DIR, zip_name)

    log.info("Downloading NLCD impervious zip (~1.2 GB) from ScienceBase")
    ok = stream_to_tmp(SOURCE_URL, zip_path, retries=3, timeout=1800)
    if not ok:
        raise RuntimeError(f"Failed to download {SOURCE_URL}")

    zip_size_mb = Path(zip_path).stat().st_size / 1e6
    log.info("Downloaded zip: %.1f MB", zip_size_mb)

    # ---- extract raster ----
    raster_member = find_raster_in_zip(zip_path)
    log.info("Extracting raster member: %s", raster_member)

    extracted_name = f"{uuid.uuid4().hex}_{Path(raster_member).name}"
    extracted_path = os.path.join(TMP_DIR, extracted_name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(raster_member) as src, open(extracted_path, "wb") as dst:
            while True:
                chunk = src.read(4 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

    raster_size_mb = Path(extracted_path).stat().st_size / 1e6
    log.info("Extracted raster: %.1f MB", raster_size_mb)

    # ---- cleanup zip (free disk) ----
    os.unlink(zip_path)
    log.info("Deleted zip to free disk space.")

    # ---- upload raster to S3 ----
    log.info("Uploading to s3://%s/%s", BUCKET, S3_KEY)
    s3.upload_file(extracted_path, BUCKET, S3_KEY)
    log.info("Upload complete.")

    # ---- cleanup extracted raster ----
    os.unlink(extracted_path)

    # ---- manifest ----
    write_manifest(
        s3,
        dataset="nlcd_impervious",
        version="2021",
        source_url=SOURCE_URL,
        s3_key=S3_KEY,
        crs="EPSG:5070",  # NLCD native CRS is Albers Equal Area (CONUS)
        notes="NLCD 2021 Impervious Surface, CONUS, 30m resolution. "
              "Downloaded from USGS ScienceBase (MRLC legacy S3 decomm).",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
