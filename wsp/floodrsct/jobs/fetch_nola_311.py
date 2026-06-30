#!/usr/bin/env python3
"""
fetch_nola_311.py -- SageMaker job: pull NOLA 311 flood complaints.

Queries Data.NOLA.gov (Socrata) for flooding-related 311 service requests
during New Orleans flood event windows.

Two datasets required (different schemas, different date ranges):
  - 3iz8-nghx: "311 Calls (Historic Data: 2012-2018)"
    Columns: ticket_id, issue_type, ticket_created_date_time, zip_code, ...
    Flood types: "Street Flooding/Drainage", "Catch Basin Maintenance"

  - 2jgv-pqrq: "311 OPCD Calls (2012-Present)"
    Columns: service_request, request_type, date_created, latitude, longitude, ...
    Flood types: "Drainage", "Roads/Drainage"

Event coverage:
  katrina2005 -- predates both datasets (no 311 data available)
  isaac2012   -- historic dataset (3iz8-nghx)
  barry2019   -- current dataset (2jgv-pqrq)
  ida2021     -- current dataset (2jgv-pqrq)

Outputs:
  s3://swarm-floodrsct-data/raw/nola_311/{event}_flooding_311.parquet
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
PAGE_SIZE = 50_000

# -- Historic dataset (2012-2018) ------------------------------------------
HISTORIC_DOMAIN = "data.nola.gov"
HISTORIC_DATASET_ID = "3iz8-nghx"
HISTORIC_BASE = f"https://{HISTORIC_DOMAIN}/resource/{HISTORIC_DATASET_ID}.json"
HISTORIC_FLOOD_TYPES = [
    "Street Flooding/Drainage",
    "Catch Basin Maintenance",
]

# -- Current dataset (2012-present) ----------------------------------------
CURRENT_DOMAIN = "data.nola.gov"
CURRENT_DATASET_ID = "2jgv-pqrq"
CURRENT_BASE = f"https://{CURRENT_DOMAIN}/resource/{CURRENT_DATASET_ID}.json"
CURRENT_FLOOD_TYPES = [
    "Drainage",
    "Roads/Drainage",
]

# -- Events ----------------------------------------------------------------
# katrina2005 predates the 311 portal; excluded.
EVENTS = {
    "isaac2012": {
        "start": "2012-08-28T00:00:00",
        "end": "2012-08-31T23:59:59",
        "source": "historic",
    },
    "barry2019": {
        "start": "2019-07-11T00:00:00",
        "end": "2019-07-15T23:59:59",
        "source": "current",
    },
    "ida2021": {
        "start": "2021-08-28T00:00:00",
        "end": "2021-09-02T23:59:59",
        "source": "current",
    },
}


def _build_type_filter(types: list[str], col: str) -> str:
    quoted = [f"'{t}'" for t in types]
    return f"{col} in({','.join(quoted)})"


def _paginated_fetch(
    base_url: str, where_clause: str, select: str, event_name: str,
) -> list[dict]:
    """Generic paginated Socrata pull."""
    all_records = []
    offset = 0
    while True:
        params = {
            "$where": where_clause,
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$select": select,
            "$order": select.split(",")[1].strip() + " ASC",
        }
        log.info("Fetching NOLA 311 for %s, offset %d", event_name, offset)
        resp = requests.get(base_url, params=params, timeout=120)
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
    return all_records


def fetch_historic(event_name: str, start: str, end: str) -> pd.DataFrame:
    """Fetch from 3iz8-nghx (2012-2018 historic dataset)."""
    type_filter = _build_type_filter(HISTORIC_FLOOD_TYPES, "issue_type")
    where = (
        f"ticket_created_date_time >= '{start}' "
        f"AND ticket_created_date_time <= '{end}' "
        f"AND {type_filter}"
    )
    select = (
        "ticket_id, ticket_created_date_time, issue_type, "
        "issue_description, zip_code, latitude, longitude, "
        "neighborhood_district, council_district, ticket_status"
    )
    records = _paginated_fetch(HISTORIC_BASE, where, select, event_name)
    if not records:
        log.warning("No historic 311 records for %s", event_name)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["event"] = event_name
    df["source_dataset"] = "3iz8-nghx"

    # Normalize to common schema
    df = df.rename(columns={
        "ticket_id": "request_id",
        "ticket_created_date_time": "created_date",
        "issue_type": "complaint_type",
        "issue_description": "description",
        "zip_code": "zcta_id",
    })
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "zcta_id" in df.columns:
        df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)
    df["created_date"] = pd.to_datetime(df["created_date"], utc=True, errors="coerce")

    geocoded = df[["latitude", "longitude"]].notna().all(axis=1).sum()
    log.info("Event %s (historic): %d records, %d geocoded",
             event_name, len(df), geocoded)
    return df


def fetch_current(event_name: str, start: str, end: str) -> pd.DataFrame:
    """Fetch from 2jgv-pqrq (2012-present dataset)."""
    type_filter = _build_type_filter(CURRENT_FLOOD_TYPES, "request_type")
    where = (
        f"date_created >= '{start}' "
        f"AND date_created <= '{end}' "
        f"AND {type_filter}"
    )
    select = (
        "service_request, date_created, request_type, "
        "request_reason, latitude, longitude, "
        "responsible_agency, request_status"
    )
    records = _paginated_fetch(CURRENT_BASE, where, select, event_name)
    if not records:
        log.warning("No current 311 records for %s", event_name)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["event"] = event_name
    df["source_dataset"] = "2jgv-pqrq"

    # Normalize to common schema
    df = df.rename(columns={
        "service_request": "request_id",
        "date_created": "created_date",
        "request_type": "complaint_type",
        "request_reason": "description",
    })
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["created_date"] = pd.to_datetime(df["created_date"], utc=True, errors="coerce")

    # Current dataset has no ZIP column -- assign via lat/lon in aggregate_311
    geocoded = df[["latitude", "longitude"]].notna().all(axis=1).sum()
    log.info("Event %s (current): %d records, %d geocoded",
             event_name, len(df), geocoded)
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
    for event_name, cfg in EVENTS.items():
        if cfg["source"] == "historic":
            df = fetch_historic(event_name, cfg["start"], cfg["end"])
        else:
            df = fetch_current(event_name, cfg["start"], cfg["end"])

        if df.empty:
            log.error("No data for %s -- check dataset or type filter", event_name)
            continue

        s3_key = f"raw/nola_311/{event_name}_flooding_311.parquet"
        upload(df, s3_key)

    log.info("fetch_nola_311 complete")


if __name__ == "__main__":
    main()
