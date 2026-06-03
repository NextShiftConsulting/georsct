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
from _s3_stream import s3_key_exists, get_s3

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
    # Primary: MRLC direct download (most reliable from SageMaker)
    "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/nlcd_2021_land_cover_l48_20230630.zip",
    # Fallback: ScienceBase (can throttle/stall on repeat downloads)
    "https://www.sciencebase.gov/catalog/file/get/649595d8d34ef77fcb01dca1"
    "?name=nlcd_2021_land_cover_l48_20230630.zip",
    # Fallback: S3 mirror (may be decommissioned / 403)
    "https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2021_land_cover_l48_20230630.zip",
]

S3_KEY_IMG = "raw/nlcd/land_cover_2021/nlcd_2021_land_cover_l48.img"
S3_KEY_TIF = "raw/nlcd/land_cover_2021/nlcd_2021_land_cover_l48.tif"

RASTER_EXTENSIONS = (".tif", ".tiff", ".img", ".ige")
TMP_DIR = "/tmp"


def find_raster_files_in_zip(zip_path: str) -> tuple[str, list[str]]:
    """Return (primary_raster, all_sidecar_members) from the zip.

    Erdas HFA splits data across .img (header) + .ige (pixels).  Both must
    be extracted with their original basenames so gdal_translate can find
    the sidecar.  Returns the .img member name plus all members sharing
    the same stem (e.g., .img, .ige, .rrd, .xml).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Find the primary .img raster
        primary = None
        for name in sorted(names):
            lower = name.lower()
            if "land_cover" in lower and lower.endswith((".img", ".tif", ".tiff")):
                primary = name
                break
        if primary is None:
            for name in names:
                lower = name.lower()
                if any(lower.endswith(ext) for ext in RASTER_EXTENSIONS):
                    primary = name
                    break
        if primary is None:
            raise FileNotFoundError(
                f"No raster file found in zip (looked for {RASTER_EXTENSIONS})"
            )
        # Collect all sidecar files with the same stem
        stem = Path(primary).stem.lower()
        sidecars = [n for n in names if Path(n).stem.lower() == stem]
        return primary, sidecars


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

    # NOTE: .img without .ige sidecar is NOT convertible.  The previous run
    # uploaded a bare .img header; skip it and re-download from source.
    if s3_key_exists(s3, BUCKET, S3_KEY_IMG):
        log.info(".img exists on S3 but may lack .ige sidecar. Re-downloading from source.")

    # Download zip from one of the known URLs
    zip_name = f"{uuid.uuid4().hex}_nlcd_land_cover.zip"
    zip_path = os.path.join(TMP_DIR, zip_name)

    downloaded = False
    for url in SCIENCEBASE_URLS:
        log.info("Trying wget: %s", url[:120])
        # wget handles slow/unreliable connections far better than Python requests.
        # --tries=3 --read-timeout=120 fails fast on stalls, retries on transient errors.
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "--progress=dot:mega",
             "--tries=3", "--read-timeout=120", "--connect-timeout=30",
             "-O", zip_path, url],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0 and Path(zip_path).exists() and Path(zip_path).stat().st_size > 1e6:
            downloaded = True
            log.info("wget succeeded: %s", result.stderr.strip().split("\n")[-1][:200] if result.stderr else "OK")
            break
        log.warning("wget failed (rc=%d): %s", result.returncode,
                    result.stderr.strip()[-300:] if result.stderr else "no stderr")
        if Path(zip_path).exists():
            os.unlink(zip_path)

    if not downloaded:
        raise RuntimeError(
            "Failed to download NLCD 2021 Land Cover from any known URL. "
            "Check ScienceBase availability or download manually."
        )

    zip_size_mb = Path(zip_path).stat().st_size / 1e6
    log.info("Downloaded zip: %.1f MB", zip_size_mb)

    # Extract raster + sidecar files (Erdas HFA needs .img + .ige)
    primary_member, sidecar_members = find_raster_files_in_zip(zip_path)
    log.info("Primary raster: %s, sidecars: %s", primary_member,
             [Path(m).suffix for m in sidecar_members])

    # Extract with ORIGINAL filenames so GDAL can resolve the .ige reference
    extracted_paths = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in sidecar_members:
            dest = os.path.join(TMP_DIR, Path(member).name)
            log.info("Extracting: %s -> %s", member, dest)
            with zf.open(member) as src, open(dest, "wb") as dst:
                while True:
                    chunk = src.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            size_mb = Path(dest).stat().st_size / 1e6
            log.info("  %.1f MB", size_mb)
            extracted_paths.append(dest)

    img_path = os.path.join(TMP_DIR, Path(primary_member).name)

    # Free disk: delete zip
    os.unlink(zip_path)
    log.info("Deleted zip to free disk space.")

    # Convert to GeoTIFF (HARD requirement -- .img alone is not usable
    # without the .ige sidecar, and we standardize on GeoTIFF)
    tif_path = convert_img_to_tif(img_path)
    log.info("Uploading .tif to s3://%s/%s", BUCKET, S3_KEY_TIF)
    s3.upload_file(tif_path, BUCKET, S3_KEY_TIF)
    log.info(".tif upload complete.")

    # Also upload .img for archival
    log.info("Uploading .img to s3://%s/%s", BUCKET, S3_KEY_IMG)
    s3.upload_file(img_path, BUCKET, S3_KEY_IMG)
    log.info(".img upload complete.")

    # Cleanup all extracted files
    for p in extracted_paths:
        if os.path.exists(p):
            os.unlink(p)
    if os.path.exists(tif_path):
        os.unlink(tif_path)

    # Manifest
    write_manifest(
        s3,
        dataset="nlcd_land_cover",
        version="2021",
        source_url=SCIENCEBASE_URLS[0],
        s3_key=S3_KEY_TIF,
        crs="EPSG:5070",
        notes="NLCD 2021 Land Cover, CONUS, 30m resolution, 16 classes. "
              "Classes 81 (Pasture/Hay) and 82 (Cultivated Crops) used for "
              "cropland_pct feature. Downloaded from USGS ScienceBase (MRLC).",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
