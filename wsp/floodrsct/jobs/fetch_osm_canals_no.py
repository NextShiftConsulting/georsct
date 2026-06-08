#!/usr/bin/env python3
"""
fetch_osm_canals_no.py -- Fetch canal/drainage channel geometries for New Orleans
from OpenStreetMap via the Overpass API.

Queries for waterway=canal and waterway=drain within Orleans Parish bounding box.

Output: s3://swarm-floodrsct-data/raw/osm/new_orleans_canals/v1/no_canals.parquet
"""

import hashlib
import logging
import sys
import time
from pathlib import Path

import boto3
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString
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
S3_KEY = "raw/osm/new_orleans_canals/v1/no_canals.parquet"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Orleans Parish + Jefferson Parish bounding box (generous)
NOLA_BBOX = (29.85, -90.25, 30.10, -89.85)  # (south, west, north, east)

OVERPASS_QUERY = (
    '[out:json][timeout:120];'
    '(way["waterway"="canal"]({south},{west},{north},{east});'
    'way["waterway"="drain"]({south},{west},{north},{east}););'
    'out body;>;out skel qt;'
)


def fetch_canals() -> gpd.GeoDataFrame:
    """Query Overpass API for canal/drain geometries in NOLA area."""
    south, west, north, east = NOLA_BBOX
    query = OVERPASS_QUERY.format(south=south, west=west, north=north, east=east)

    log.info("Querying Overpass API for canals in bbox %s", NOLA_BBOX)
    resp = requests.post(
        OVERPASS_URL,
        data={"data": query},
        headers={"User-Agent": "floodrsct/1.0 (geospatial-qa)"},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()

    elements = data.get("elements", [])
    log.info("Overpass returned %d elements", len(elements))

    # Build node lookup
    nodes = {}
    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    # Build way geometries
    rows = []
    for el in elements:
        if el["type"] != "way":
            continue
        coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        rows.append({
            "osm_id": el["id"],
            "waterway": tags.get("waterway", ""),
            "name": tags.get("name", ""),
            "geometry": LineString(coords),
        })

    if not rows:
        log.warning("No canal geometries found in NOLA bbox")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    log.info("Built %d canal/drain geometries", len(gdf))
    return gdf


def main() -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    # Check if already uploaded
    try:
        s3.head_object(Bucket=BUCKET, Key=S3_KEY)
        log.info("Already exists: s3://%s/%s -- skipping", BUCKET, S3_KEY)
        return
    except s3.exceptions.ClientError:
        pass

    gdf = fetch_canals()
    if gdf.empty:
        log.error("No canal data fetched; aborting upload")
        sys.exit(1)

    local_path = "/tmp/no_canals.parquet"
    gdf.to_parquet(local_path, index=False)
    file_size = Path(local_path).stat().st_size

    with open(local_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    s3.upload_file(local_path, BUCKET, S3_KEY)
    log.info("Uploaded s3://%s/%s (%.1f KB)", BUCKET, S3_KEY, file_size / 1e3)

    write_manifest(
        s3=s3,
        dataset="osm_canals_no",
        version="v1",
        source_url="https://overpass-api.de/api/interpreter",
        s3_key=S3_KEY,
        crs="EPSG:4326",
        record_count=len(gdf),
        sha256=sha,
        license_="ODbL -- OpenStreetMap contributors",
        notes="Canal and drain waterways in Orleans/Jefferson Parish. "
              "Used for canal_proximity_m feature in new_orleans scenario.",
    )

    log.info("fetch_osm_canals_no complete: %d features", len(gdf))


if __name__ == "__main__":
    main()
