#!/usr/bin/env python3
"""
fetch_dem_elevation.py -- SageMaker job: download USGS 3DEP 1/3 arc-second
DEM GeoTIFFs for SW Florida and New Orleans and upload raw to S3.

THIS SCRIPT ONLY FETCHES AND NORMALIZES RAW DATA.
ZCTA elevation aggregation (build_elevation_features) happens in build_event_dataset.py.

Source: USGS 3DEP via The National Map (TNM) API
  API: https://tnmapi.cr.usgs.gov/api/products
  Product: 1/3 arc-second DEM (National Elevation Dataset)

Output:
  s3://swarm-floodrsct-data/raw/dem/3dep/v1/{region}/{tile}.tif
  s3://swarm-floodrsct-data/manifests/usgs_3dep_dem/v1/manifest.json

Regions:
  southwest_florida  — Lee, Collier, Sarasota, Manatee, Hillsborough, Pinellas
  new_orleans        — Orleans Parish + Jefferson Parish

Note: 3DEP tiles are ~1 degree x 1 degree. This script downloads all tiles
intersecting the bounding box of each target region.
"""

import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import requests

sys.path.insert(0, str(Path(__file__).parent))
from _manifest_writer import write_manifest
from _s3_stream import s3_key_exists, stream_download_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

DST_BUCKET = "swarm-floodrsct-data"
DST_PREFIX = "raw/dem/3dep/v1"
VERSION = "v1"
DATASET = "usgs_3dep_dem"
TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
SOURCE_URL = "https://apps.nationalmap.gov/downloader/"

# Bounding boxes (W, S, E, N) per region
REGIONS = {
    "southwest_florida": {"bbox": (-83.0, 25.5, -80.0, 29.0)},
    "new_orleans":       {"bbox": (-90.5, 29.0, -89.5, 30.3)},
}

# TNM product tag for 1/3 arc-second DEM
PRODUCT_TAG = "National Elevation Dataset (NED) 1/3 arc-second"


def query_tnm_tiles(bbox: tuple, product_tag: str) -> list[dict]:
    """Query TNM API for 1/3 arc-second DEM tiles intersecting a bounding box."""
    w, s, e, n = bbox
    params = {
        "datasets": product_tag,
        "bbox": f"{w},{s},{e},{n}",
        "max": 50,
        "offset": 0,
        "outputFormat": "JSON",
    }
    log.info("Querying TNM for bbox=%s", bbox)
    resp = requests.get(TNM_API, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("items", [])
    log.info("TNM returned %d tiles", len(items))
    return items


def main() -> None:
    s3 = boto3.client("s3", region_name="us-east-1")
    total_uploaded = 0

    for region, spec in REGIONS.items():
        log.info("=== Region: %s ===", region)
        tiles = query_tnm_tiles(spec["bbox"], PRODUCT_TAG)

        if not tiles:
            tiles = query_tnm_tiles(spec["bbox"], "Digital Elevation Model (DEM) 1/3 arc-second")

        if not tiles:
            log.warning("No DEM tiles found for %s; writing empty manifest entry", region)
            continue

        def _fetch_tile(tile: dict) -> int:
            """Stream one tile to disk → S3 multipart. Returns 1 on success.

            Uses s3=None — each thread in the pool gets its own boto3 client.
            """
            download_url = (
                tile.get("downloadURL")
                or (tile.get("urls") or {}).get("TIFF")
                or (tile.get("urls") or {}).get("ALL")
            )
            if not download_url:
                log.warning("No download URL for tile: %s", tile.get("title", "unknown"))
                return 0
            filename = Path(download_url).name
            dst_key = f"{DST_PREFIX}/{region}/{filename}"
            ok = stream_download_to_s3(
                None, download_url, DST_BUCKET, dst_key, timeout=300
            )
            return 1 if ok else 0

        with ThreadPoolExecutor(max_workers=8) as pool:
            for n in pool.map(_fetch_tile, tiles):
                total_uploaded += n
        sys.stdout.flush()

    write_manifest(
        s3=s3,
        dataset=DATASET,
        version=VERSION,
        source_url=SOURCE_URL,
        s3_key=DST_PREFIX + "/",
        crs="EPSG:4269",  # NAD83, standard for NED
        record_count=total_uploaded,
        license_="Public Domain — USGS National Map",
        notes=(
            "USGS 3DEP 1/3 arc-second DEM tiles for SW Florida and New Orleans. "
            "ZCTA mean elevation computed in build_event_dataset.py "
            "via build_elevation_features(). "
            "Also provides base raster for subsidence_rate_mm_yr derivation (NO)."
        ),
    )

    log.info("fetch_dem_elevation complete: %d tiles uploaded", total_uploaded)


if __name__ == "__main__":
    main()
