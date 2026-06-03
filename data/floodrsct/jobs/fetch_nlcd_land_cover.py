"""
fetch_nlcd_land_cover.py -- Download NLCD 2021 Land Cover raster.

SageMaker container script (NOT a launcher).
Source: MRLC NLCD 2021 Land Cover, CONUS (via USGS ScienceBase)
Product page: https://www.sciencebase.gov/catalog/item/647626cbd34e4e58932d9d4e

Land cover is a CATEGORICAL raster (16 classes). Relevant classes for
cropland_pct:
  81 = Pasture/Hay
  82 = Cultivated Crops

Output:
  s3://swarm-floodrsct-data/raw/nlcd/land_cover_2021/nlcd_2021_land_cover_l48.img
  s3://swarm-floodrsct-data/raw/nlcd/land_cover_2021/nlcd_2021_land_cover_l48.tif
    (converted via gdal_translate for rasterio compatibility)
"""

import logging
import os
import subprocess
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
log = logging.getLogger("fetch_nlcd_land_cover")

BUCKET = "swarm-floodrsct-data"

# ScienceBase item ID for NLCD 2021 Land Cover: 649595d8d34ef77fcb01dca1
# (confirmed via ScienceBase catalog; sibling of impervious 649595c3d34ef77fcb01dc9e)
SCIENCEBASE_URLS = [
    # Primary: verified working ScienceBase item ID
    "https://www.sciencebase.gov/catalog/file/get/649595d8d34ef77fcb01dca1"
    "?name=nlcd_2021_land_cover_l48_20230630.zip",
    # Fallback: S3 mirror (may be decommissioned)
    "https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2021_land_cover_l48_20230630.zip",
    # Fallback: MRLC direct download
    "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/nlcd_2021_land_cover_l48_20230630.zip",
]

S3_KEY_IMG = "raw/nlcd/land_cover_2021/nlcd_2021_land_cover_l48.img"
S3_KEY_TIF = "raw/nlcd/land_cover_2021/nlcd_2021_land_cover_l48.tif"

RASTER_EXTENSIONS = (".tif", ".tiff", ".img", ".ige")
TMP_DIR = "/tmp"


def find_raster_in_zip(zip_path: str) -> str:
    """Return the name of the first land cover raster file inside the zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Prefer .img (the primary raster) over .tif
        for name in sorted(zf.namelist()):
            lower = name.lower()
            if "land_cover" in lower and lower.endswith((".img", ".tif", ".tiff")):
                return name
        # Fallback: any raster
        for name in zf.namelist():
            lower = name.lower()
            if any(lower.endswith(ext) for ext in RASTER_EXTENSIONS):
                return name
    raise FileNotFoundError(
        f"No raster file found in zip (looked for {RASTER_EXTENSIONS})"
    )


def convert_img_to_tif(img_path: str) -> str:
    """Convert .img (Erdas HFA) to compressed GeoTIFF via gdal_translate."""
    tif_path = img_path.replace(".img", ".tif")
    log.info("Converting .img -> .tif via gdal_translate (LZW compression)")
    result = subprocess.run(
        ["gdal_translate", "-of", "GTiff", "-co", "COMPRESS=LZW",
         "-co", "TILED=YES", "-co", "BIGTIFF=YES",
         img_path, tif_path],
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode == 0 and Path(tif_path).exists():
        tif_size = Path(tif_path).stat().st_size / 1e9
        log.info("Conversion OK: %.2f GB", tif_size)
        return tif_path
    else:
        log.error("gdal_translate failed: %s", result.stderr[:500])
        raise RuntimeError("gdal_translate failed")


def main() -> None:
    s3 = get_s3()

    # Checkpoint: skip if .tif already uploaded (preferred format)
    if s3_key_exists(s3, BUCKET, S3_KEY_TIF):
        log.info("Checkpoint hit -- .tif already in S3: s3://%s/%s", BUCKET, S3_KEY_TIF)
        log.info("Done (skipped).")
        return

    # Checkpoint: .img exists but .tif doesn't -- just need conversion
    if s3_key_exists(s3, BUCKET, S3_KEY_IMG):
        log.info(".img exists but .tif missing. Downloading .img for conversion.")
        img_local = os.path.join(TMP_DIR, "nlcd_land_cover.img")
        s3.download_file(BUCKET, S3_KEY_IMG, img_local)
        tif_local = convert_img_to_tif(img_local)
        log.info("Uploading .tif to s3://%s/%s", BUCKET, S3_KEY_TIF)
        s3.upload_file(tif_local, BUCKET, S3_KEY_TIF)
        os.unlink(img_local)
        os.unlink(tif_local)
        log.info("Done (.img -> .tif conversion only).")
        return

    # Download zip from one of the known URLs
    zip_name = f"{uuid.uuid4().hex}_nlcd_land_cover.zip"
    zip_path = os.path.join(TMP_DIR, zip_name)

    downloaded = False
    for url in SCIENCEBASE_URLS:
        log.info("Trying: %s", url[:120])
        ok = stream_to_tmp(url, zip_path, retries=2, timeout=1800)
        if ok and Path(zip_path).exists() and Path(zip_path).stat().st_size > 1e6:
            downloaded = True
            break
        log.warning("Failed or too small, trying next URL")

    if not downloaded:
        raise RuntimeError(
            "Failed to download NLCD 2021 Land Cover from any known URL. "
            "Check ScienceBase availability or download manually."
        )

    zip_size_mb = Path(zip_path).stat().st_size / 1e6
    log.info("Downloaded zip: %.1f MB", zip_size_mb)

    # Extract raster
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

    # Free disk: delete zip
    os.unlink(zip_path)
    log.info("Deleted zip to free disk space.")

    # Upload .img to S3
    log.info("Uploading .img to s3://%s/%s", BUCKET, S3_KEY_IMG)
    s3.upload_file(extracted_path, BUCKET, S3_KEY_IMG)
    log.info(".img upload complete.")

    # Convert to .tif and upload
    try:
        tif_path = convert_img_to_tif(extracted_path)
        log.info("Uploading .tif to s3://%s/%s", BUCKET, S3_KEY_TIF)
        s3.upload_file(tif_path, BUCKET, S3_KEY_TIF)
        log.info(".tif upload complete.")
        os.unlink(tif_path)
    except Exception as e:
        log.warning("TIF conversion failed: %s. .img is still usable.", e)

    # Cleanup
    os.unlink(extracted_path)

    # Manifest
    write_manifest(
        s3,
        dataset="nlcd_land_cover",
        version="2021",
        source_url=SCIENCEBASE_URLS[0],
        s3_key=S3_KEY_IMG,
        crs="EPSG:5070",
        notes="NLCD 2021 Land Cover, CONUS, 30m resolution, 16 classes. "
              "Classes 81 (Pasture/Hay) and 82 (Cultivated Crops) used for "
              "cropland_pct feature. Downloaded from USGS ScienceBase (MRLC).",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
