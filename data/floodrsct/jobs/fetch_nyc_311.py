#!/usr/bin/env python3
"""
fetch_nyc_311.py -- SageMaker job: pull NYC 311 flood complaints.

Queries NYC Open Data (Socrata) for flooding-related 311 service requests
during NYC flood event windows.

Source: NYC Open Data — "311 Service Requests from 2010 to Present"
Dataset: erm2-nwe9

Outputs:
  s3://swarm-floodrsct-data/raw/nyc_311/{event}_flooding_311.parquet
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
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
SOCRATA_DOMAIN = "data.cityofnewyork.us"
DATASET_ID = "erm2-nwe9"
SOCRATA_BASE = f"https://{SOCRATA_DOMAIN}/resource/{DATASET_ID}.json"
PAGE_SIZE = 50_000

EVENTS = {
    "sandy2012": {
        "start": "2012-10-28T00:00:00",
        "end": "2012-10-31T23:59:59",
    },
    "ida2021": {
        "start": "2021-09-01T00:00:00",
        "end": "2021-09-03T23:59:59",
    },
    "henri2021": {
        "start": "2021-08-21T00:00:00",
        "end": "2021-08-23T23:59:59",
    },
    "nyc_flood_2023": {
        "start": "2023-09-28T00:00:00",
        "end": "2023-09-30T23:59:59",
    },
}

FLOOD_COMPLAINT_TYPES = [
    "Sewer",
    "Street Flooding",
    "Catch Basin Clogged/Flooding",
    "Flooded Basement",
    "Water System",
]


def build_type_filter() -> str:
    quoted = [f"'{t}'" for t in FLOOD_COMPLAINT_TYPES]
    return f"complaint_type in({','.join(quoted)})"


def fetch_nyc_311(event_name: str, start: str, end: str) -> pd.DataFrame:
    """Paginated Socrata pull for one NYC event window."""
    type_filter = build_type_filter()
    all_records = []
    offset = 0

    while True:
        params = {
            "$where": (
                f"created_date >= '{start}' AND created_date <= '{end}' "
                f"AND {type_filter}"
            ),
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$select": (
                "unique_key,created_date,complaint_type,descriptor,"
                "incident_zip,latitude,longitude,bbl,"
                "borough,community_board,resolution_description"
            ),
            "$order": "created_date ASC",
        }
        log.info("Fetching NYC 311 for %s, offset %d", event_name, offset)
        resp = requests.get(SOCRATA_BASE, params=params, timeout=120)
        resp.raise_for_status()
        records = resp.json()
        if not records:
            break
        all_records.extend(records)
        log.info("  %d records so far", len(all_records))
        if len(records) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.5)

    if not all_records:
        log.warning("No NYC 311 records for event %s", event_name)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["event"] = event_name

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize ZIP
    if "incident_zip" in df.columns:
        df = df.rename(columns={"incident_zip": "zcta_id"})
        df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)

    df["created_date"] = pd.to_datetime(df["created_date"], utc=True, errors="coerce")
    geocoded = df[["latitude", "longitude"]].notna().all(axis=1).sum()
    log.info("Event %s: %d total records, %d geocoded", event_name, len(df), geocoded)
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
        df = fetch_nyc_311(event_name, window["start"], window["end"])
        if df.empty:
            log.error("No data for %s — check dataset ID or complaint type filter", event_name)
            continue
        s3_key = f"raw/nyc_311/{event_name}_flooding_311.parquet"
        upload(df, s3_key)

    log.info("fetch_nyc_311 complete")


if __name__ == "__main__":
    main()
