#!/usr/bin/env python3
"""
fetch_mta_stations.py -- SageMaker job: download NYC MTA subway station
locations from NYC Open Data and upload raw to S3.

THIS SCRIPT ONLY FETCHES AND NORMALIZES RAW DATA.
ZCTA-level station counts and distances (build_subway_features) happen
in build_event_dataset.py.

Source: NY State Open Data — MTA Subway Stations
  Dataset: 39hk-dx4f  (data.ny.gov — verified 2026-05-28)
  API: https://data.ny.gov/resource/39hk-dx4f.json
  License: Public Domain / NY Open Data Terms

Output:
  s3://swarm-floodrsct-data/raw/mta/subway_stations/v1/subway_stations.parquet
  s3://swarm-floodrsct-data/manifests/mta_subway_stations/v1/manifest.json

Schema: station_id, name, line, borough, latitude, longitude
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from _manifest_writer import write_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

DST_BUCKET = "swarm-floodrsct-data"
DST_KEY = "raw/mta/subway_stations/v1/subway_stations.parquet"
VERSION = "v1"
DATASET = "mta_subway_stations"

# MTA Subway Stations — lives on data.ny.gov (not data.cityofnewyork.us)
SOCRATA_URL = "https://data.ny.gov/resource/39hk-dx4f.json"
PAGE_SIZE = 1000


def fetch_all_stations() -> pd.DataFrame:
    """Paginate through Socrata API and return all subway stations."""
    records = []
    offset = 0
    while True:
        params = {
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$order": "station_id ASC",
        }
        resp = requests.get(SOCRATA_URL, params=params, timeout=60)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        records.extend(page)
        offset += len(page)
        if len(page) < PAGE_SIZE:
            break
        log.info("Fetched %d stations so far...", len(records))

    log.info("Total stations fetched: %d", len(records))
    return pd.DataFrame(records)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and types."""
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("objectid", "station_id", "the_geom_id"):
            col_map[col] = "station_id"
        elif lc in ("name", "stop_name", "station_name"):
            col_map[col] = "name"
        elif lc in ("line", "lines", "daytime_routes"):
            col_map[col] = "line"
        elif lc in ("borough", "boro"):
            col_map[col] = "borough"
        elif "latitude" in lc or lc == "lat":
            col_map[col] = "latitude"
        elif "longitude" in lc or lc in ("long", "lon"):
            col_map[col] = "longitude"

    df = df.rename(columns=col_map)
    # Drop duplicate column names that result from multiple source cols mapping to same target
    df = df.loc[:, ~df.columns.duplicated()]

    # Handle nested the_geom (GeoJSON point)
    if "the_geom" in df.columns and "latitude" not in df.columns:
        def extract_coords(geom):
            if isinstance(geom, dict) and geom.get("type") == "Point":
                return geom["coordinates"][1], geom["coordinates"][0]  # lat, lon
            return None, None
        df[["latitude", "longitude"]] = df["the_geom"].apply(
            lambda g: pd.Series(extract_coords(g))
        )
        df = df.drop(columns=["the_geom"])

    # Coerce types
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [c for c in ["station_id", "name", "line", "borough", "latitude", "longitude"]
            if c in df.columns]
    df = df[keep].drop_duplicates().reset_index(drop=True)
    log.info("Normalized: %d stations, columns: %s", len(df), list(df.columns))
    return df


def main() -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    try:
        s3.head_object(Bucket=DST_BUCKET, Key=DST_KEY)
        log.info("Already exists: s3://%s/%s — skipping", DST_BUCKET, DST_KEY)
        return
    except s3.exceptions.ClientError:
        pass

    df = fetch_all_stations()
    df = normalize(df)

    local = "/tmp/subway_stations.parquet"
    df.to_parquet(local, index=False)

    checksum = hashlib.sha256(open(local, "rb").read()).hexdigest()
    s3.upload_file(local, DST_BUCKET, DST_KEY)
    log.info("Uploaded s3://%s/%s (%d stations)", DST_BUCKET, DST_KEY, len(df))

    write_manifest(
        s3=s3,
        dataset=DATASET,
        version=VERSION,
        source_url=SOCRATA_URL,
        s3_key=DST_KEY,
        crs="EPSG:4326",
        record_count=len(df),
        sha256=checksum,
        license_="Public Domain — NYC Open Data",
        notes=(
            "NYC MTA subway station locations (all lines). "
            "ZCTA station counts and nearest-station distances computed in "
            "build_event_dataset.py via build_subway_features()."
        ),
    )

    log.info("fetch_mta_stations complete: %d stations", len(df))


if __name__ == "__main__":
    main()
