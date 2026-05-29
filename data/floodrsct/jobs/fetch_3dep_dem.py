"""
fetch_3dep_dem.py -- Download USGS 3DEP 1/3 arc-second DEM tiles for a scenario.

Uses the USGS National Map (TNM) API to discover tiles within a bounding box,
downloads the latest version of each tile, and uploads to S3.

Source: https://tnmaccess.nationalmap.gov/api/v1/products
Product: National Elevation Dataset (NED) 1/3 arc-second, GeoTIFF

Output per scenario:
  s3://swarm-floodrsct-data/raw/dem/3dep/v1/{scenario}/USGS_13_{tile}_{date}.tif

Usage (SageMaker container):
    python3 -u fetch_3dep_dem.py --scenario houston
"""

import argparse
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import requests

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUCKET = "swarm-floodrsct-data"
DST_PREFIX = "raw/dem/3dep/v1"
TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
MAX_WORKERS = 4  # Large files, limit concurrent downloads
TMP_DIR = "/tmp"

SCENARIOS = {
    "houston": {
        "bbox": "-96.5,28.5,-93.5,31.0",
        "description": "Houston metro + SE Texas (Harvey/Imelda/Beryl flood zone)",
    },
    "nyc": {
        "bbox": "-74.5,40.3,-73.5,41.2",
        "description": "NYC metro + NJ (Ida 2021 flood zone)",
    },
    "socal": {
        "bbox": "-118.0,33.0,-115.5,34.5",
        "description": "Southern California (Hilary 2023 flood zone)",
    },
    "southwest_florida": {
        "bbox": "-82.5,25.5,-80.5,27.5",
        "description": "SW Florida (Ian 2022 flood zone)",
    },
    "new_orleans": {
        "bbox": "-91.5,29.0,-89.5,31.0",
        "description": "New Orleans metro (Ida 2021 LA flood zone)",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_tiles(bbox: str) -> list[dict]:
    """Query USGS TNM API for 3DEP 1/3 arc-second tiles within bbox.

    Returns list of dicts with 'title', 'url', 'size_bytes', 'tile_id', 'date'.
    Only returns the latest version of each tile.
    """
    params = {
        "datasets": "National Elevation Dataset (NED) 1/3 arc-second",
        "bbox": bbox,
        "prodFormats": "GeoTIFF",
        "max": 200,
    }
    log.info("Querying TNM API: bbox=%s", bbox)
    resp = requests.get(TNM_API, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    total = data.get("total", 0)
    items = data.get("items", [])
    log.info("TNM API returned %d items (total=%d)", len(items), total)

    # Group by tile, keep latest version
    tiles: dict[str, dict] = {}
    for item in items:
        url = item.get("downloadURL", "")
        title = item.get("title", "")
        size = item.get("sizeInBytes", 0)

        # Extract tile ID and date from filename
        # URL pattern: .../USGS_13_n30w095_20240229.tif
        fname = Path(url).name
        parts = fname.replace(".tif", "").split("_")
        if len(parts) < 4:
            continue

        tile_id = parts[2]  # e.g., n30w095
        date_str = parts[3]  # e.g., 20240229

        if tile_id not in tiles or date_str > tiles[tile_id]["date"]:
            tiles[tile_id] = {
                "tile_id": tile_id,
                "date": date_str,
                "url": url,
                "size_bytes": size,
                "title": title,
                "filename": fname,
            }

    result = sorted(tiles.values(), key=lambda x: x["tile_id"])
    log.info("Unique tiles (latest version each): %d", len(result))
    return result


def s3_key_exists(s3, key: str) -> bool:
    """Check if key exists in BUCKET."""
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def download_and_upload(s3, scenario: str, tile: dict) -> dict:
    """Download a single DEM tile and upload to S3.

    Returns dict with tile_id, status, size_mb.
    """
    s3_key = f"{DST_PREFIX}/{scenario}/{tile['filename']}"

    # Checkpoint: skip if already in S3
    if s3_key_exists(s3, s3_key):
        log.info("  SKIP %s (already in S3)", tile["tile_id"])
        return {"tile_id": tile["tile_id"], "status": "SKIPPED", "size_mb": 0}

    # Download to /tmp
    tmp_name = f"{uuid.uuid4().hex}_{tile['filename']}"
    tmp_path = os.path.join(TMP_DIR, tmp_name)

    try:
        log.info("  Downloading %s (%d MB) from TNM...",
                 tile["tile_id"], tile["size_bytes"] // (1024 * 1024))

        resp = requests.get(tile["url"], stream=True, timeout=600)
        resp.raise_for_status()

        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                fh.write(chunk)

        file_size = Path(tmp_path).stat().st_size
        if file_size < 1_000_000:  # < 1MB is suspicious for a DEM tile
            log.warning("  %s too small (%d bytes) -- skipping", tile["tile_id"], file_size)
            return {"tile_id": tile["tile_id"], "status": "TOO_SMALL", "size_mb": 0}

        # Upload to S3
        size_mb = file_size / 1e6
        log.info("  Uploading %s (%.0f MB) to s3://%s/%s",
                 tile["tile_id"], size_mb, BUCKET, s3_key)
        s3.upload_file(tmp_path, BUCKET, s3_key)

        return {"tile_id": tile["tile_id"], "status": "OK", "size_mb": size_mb}

    except Exception as exc:
        log.error("  FAILED %s: %s", tile["tile_id"], exc)
        return {"tile_id": tile["tile_id"], "status": f"FAILED: {exc}", "size_mb": 0}

    finally:
        if Path(tmp_path).exists():
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch USGS 3DEP DEM tiles for a flood scenario"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=list(SCENARIOS.keys()),
        help="Scenario name",
    )
    args = parser.parse_args()

    scenario = args.scenario
    cfg = SCENARIOS[scenario]

    log.info("=== 3DEP DEM Fetch ===")
    log.info("Scenario:    %s", scenario)
    log.info("Description: %s", cfg["description"])
    log.info("Bbox:        %s", cfg["bbox"])

    # Discover tiles
    tiles = discover_tiles(cfg["bbox"])
    if not tiles:
        log.error("No tiles found for bbox %s", cfg["bbox"])
        sys.exit(1)

    total_expected_mb = sum(t["size_bytes"] for t in tiles) / 1e6
    log.info("Tiles to fetch: %d (%.0f MB expected)", len(tiles), total_expected_mb)

    for t in tiles:
        log.info("  %s  %s  %d MB  %s",
                 t["tile_id"], t["date"], t["size_bytes"] // (1024 * 1024), t["filename"])

    # Parallel downloads with per-thread S3 clients
    # /tmp cleanup in download_and_upload() ensures disk stays bounded
    results = []
    log.info("Launching %d parallel workers (MAX_WORKERS=%d)", MAX_WORKERS, MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        _aws = get_aws_credentials()
        _aws.pop("region_name", None)
        s3 = boto3.client("s3", region_name="us-east-1", **_aws)
        futures = {
            pool.submit(download_and_upload, s3, scenario, tile): tile
            for tile in tiles
        }
        for i, future in enumerate(as_completed(futures), 1):
            tile = futures[future]
            try:
                r = future.result()
            except Exception as exc:
                log.error("Unhandled error for %s: %s", tile["tile_id"], exc)
                r = {"tile_id": tile["tile_id"], "status": f"FAILED: {exc}", "size_mb": 0}
            results.append(r)
            if i % 5 == 0 or i == len(tiles):
                log.info("Progress: %d/%d tiles", i, len(tiles))
                sys.stdout.flush()

    # Summary
    ok_count = sum(1 for r in results if r["status"] == "OK")
    skip_count = sum(1 for r in results if r["status"] == "SKIPPED")
    fail_count = sum(1 for r in results if r["status"].startswith("FAILED"))
    total_mb = sum(r["size_mb"] for r in results)

    log.info("")
    log.info("=== SUMMARY ===")
    log.info("Scenario:   %s", scenario)
    log.info("Tiles:      %d total, %d new, %d skipped, %d failed",
             len(results), ok_count, skip_count, fail_count)
    log.info("Uploaded:   %.0f MB", total_mb)

    # Write manifest
    write_manifest(
        s3=s3,
        dataset=f"3dep_dem_{scenario}",
        version=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        source_url="https://tnmaccess.nationalmap.gov/api/v1/products",
        s3_key=f"{DST_PREFIX}/{scenario}/",
        crs="EPSG:4269",  # 3DEP native is NAD83
        record_count=ok_count + skip_count,
        notes=(
            f"USGS 3DEP 1/3 arc-second DEM tiles for {scenario}. "
            f"bbox={cfg['bbox']}. "
            f"{ok_count} new + {skip_count} existing tiles."
        ),
    )

    if fail_count > 0:
        log.warning("%d tiles failed", fail_count)
        sys.exit(1)

    log.info("Done.")


if __name__ == "__main__":
    main()
