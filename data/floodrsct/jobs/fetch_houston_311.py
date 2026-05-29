#!/usr/bin/env python3
"""
fetch_houston_311.py -- SageMaker job: pull Houston 311 flood service requests.

Queries the City of Houston open GIS server (mycity2.houstontx.gov) for
service requests categorized as Flooding or Drainage, within each event window.

Source: mycity2.houstontx.gov — Houston 311 Archives MapServer
  Archive endpoint covers Nov 2011 to present; paginated at 1,000 records/page.
  No auth required.

Field mapping (ArcGIS → canonical):
  Case_Number          → sr_number
  Incident_Case_Type   → sr_type
  Created_Date_Local   → create_date   (CST; coerced to UTC)
  Latitude, Longitude  → latitude, longitude
  Zip_Code             → zcta_id

Outputs:
  s3://swarm-floodrsct-data/raw/houston_311/harvey2017_311.parquet
  s3://swarm-floodrsct-data/raw/houston_311/imelda2019_311.parquet
  s3://swarm-floodrsct-data/raw/houston_311/beryl2024_311.parquet
"""

import logging
import sys
import time
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"

# Houston 311 Archive — covers Nov 2011 to present (Harvey, Imelda, Beryl all included)
ARCGIS_BASE = (
    "https://mycity2.houstontx.gov/gisweb01/rest/services"
    "/311/Houston311_Archives/MapServer/0/query"
)
PAGE_SIZE = 1_000  # server max per request

EVENTS = {
    "harvey2017": {
        "start": "2017-08-25 00:00:00",
        "end": "2017-09-10 23:59:59",
    },
    "imelda2019": {
        "start": "2019-09-17 00:00:00",
        "end": "2019-09-25 23:59:59",
    },
    "beryl2024": {
        "start": "2024-07-07 00:00:00",
        "end": "2024-07-15 23:59:59",
    },
}

# ArcGIS Incident_Case_Type values for flood/drainage requests
FLOOD_TYPES = ["Flooding", "Drainage"]


def build_where(start: str, end: str) -> str:
    type_list = ", ".join(f"'{t}'" for t in FLOOD_TYPES)
    return (
        f"Incident_Case_Type IN ({type_list}) "
        f"AND Created_Date_Local >= '{start}' "
        f"AND Created_Date_Local <= '{end}'"
    )


def fetch_311(event_name: str, start: str, end: str) -> pd.DataFrame:
    """Paginated ArcGIS REST pull for one event window."""
    where = build_where(start, end)
    all_records = []
    offset = 0

    while True:
        params = {
            "where": where,
            "outFields": (
                "Case_Number,Incident_Case_Type,Created_Date_Local,"
                "Latitude,Longitude,Zip_Code,Status,Department"
            ),
            "resultRecordCount": PAGE_SIZE,
            "resultOffset": offset,
            "f": "json",
        }
        log.info("Fetching 311 for %s, offset %d", event_name, offset)
        resp = requests.get(ARCGIS_BASE, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            log.error("ArcGIS error for %s: %s", event_name, data["error"])
            return pd.DataFrame()

        features = data.get("features", [])
        if not features:
            break

        records = [f["attributes"] for f in features]
        all_records.extend(records)
        log.info("  %d records so far", len(all_records))

        if not data.get("exceededTransferLimit", False):
            break
        offset += PAGE_SIZE
        time.sleep(0.3)

    if not all_records:
        log.warning("No 311 records for event %s", event_name)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["event"] = event_name

    # Rename to canonical schema
    df = df.rename(columns={
        "Case_Number": "sr_number",
        "Incident_Case_Type": "sr_type",
        "Created_Date_Local": "create_date",
        "Zip_Code": "zcta_id",
    })

    # ArcGIS returns epoch ms for date fields
    if "create_date" in df.columns:
        df["create_date"] = pd.to_datetime(
            df["create_date"], unit="ms", utc=True, errors="coerce"
        )

    # Lowercase coordinate columns
    for raw, canon in [("Latitude", "latitude"), ("Longitude", "longitude")]:
        if raw in df.columns:
            df = df.rename(columns={raw: canon})
        if canon in df.columns:
            df[canon] = pd.to_numeric(df[canon], errors="coerce")

    # Normalize ZIP
    if "zcta_id" in df.columns:
        df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)

    valid_coords = df[["latitude", "longitude"]].notna().all(axis=1).sum()
    log.info("Event %s: %d records, %d with valid coordinates",
             event_name, len(df), valid_coords)
    return df


def upload(df: pd.DataFrame, s3_key: str) -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    local = f"/tmp/{Path(s3_key).name}"
    df.to_parquet(local, index=False)
    s3.upload_file(local, BUCKET, s3_key)
    log.info("Uploaded %d rows to s3://%s/%s", len(df), BUCKET, s3_key)


def main() -> None:
    for event_name, window in EVENTS.items():
        df = fetch_311(event_name, window["start"], window["end"])
        if df.empty:
            log.error("No data for %s — check ArcGIS endpoint or date filter", event_name)
            continue
        s3_key = f"raw/houston_311/{event_name}_311.parquet"
        upload(df, s3_key)

    log.info("fetch_houston_311 complete")


if __name__ == "__main__":
    main()
