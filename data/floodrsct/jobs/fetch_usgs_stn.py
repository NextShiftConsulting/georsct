#!/usr/bin/env python3
"""
fetch_usgs_stn.py -- SageMaker job: pull USGS Short-Term Network high-water marks.

Queries the USGS STN (Short-Term Network) web services for post-event
surveyed high-water marks. Harvey 2017 has ~2000 marks; Imelda 2019
has several hundred.

API: https://stn.wim.usgs.gov/STNServices/HWMs.json
Filter by event ID (not date range); event IDs determined from STN event list.

Outputs:
  s3://swarm-floodrsct-data/raw/usgs_stn/harvey2017_hwm.parquet
  s3://swarm-floodrsct-data/raw/usgs_stn/imelda2019_hwm.parquet
"""

import logging
import sys
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
STN_BASE = "https://stn.wim.usgs.gov/STNServices"

# STN event IDs — confirmed from stn.wim.usgs.gov/STNServices/Events.json
# Harvey 2017: eventID 23 ("Hurricane Harvey")
# Imelda 2019: eventID 56 ("Tropical Storm Imelda")
EVENTS = {
    "harvey2017": 23,
    "imelda2019": 56,
}


def fetch_event_list() -> pd.DataFrame:
    """Fetch all STN events for reference / event ID verification."""
    url = f"{STN_BASE}/Events.json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def fetch_hwm(event_id: int, event_name: str) -> pd.DataFrame:
    """Fetch all high-water marks for a given STN event ID via FilteredHWMs CSV.

    The STN REST API returns 415 for the .json suffix endpoint; the CSV
    endpoint is the correct programmatic access path (verified 2026-05-28).
    """
    url = f"{STN_BASE}/HWMs/FilteredHWMs.csv"
    params = {"Event": event_id}
    log.info("Fetching HWMs for event %s (ID %d) via CSV endpoint", event_name, event_id)
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()

    import io
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    if df.empty:
        log.warning("No HWMs returned for event %s", event_name)
        return df

    log.info("Retrieved %d HWMs for %s", len(df), event_name)

    # CSV column names differ slightly from JSON — map to canonical names.
    # Prefer latitude_dd/longitude_dd (decimal degrees) over latitude/longitude
    # to avoid duplicate column names after rename.
    if "latitude_dd" in df.columns:
        df = df.drop(columns=[c for c in ["latitude", "longitude"] if c in df.columns])
    rename = {
        "hwm_id": "hwm_id",
        "latitude": "latitude",
        "longitude": "longitude",
        "latitude_dd": "latitude",
        "longitude_dd": "longitude",
        "elev_ft": "elev_ft",
        "verticalDatumName": "datum",
        "hwm_uncertainty": "uncertainty_ft",
        "hwmQualityName": "hwm_quality",
        "hwm_type_id": "hwm_type_id",
        "countyName": "county",
        "stateName": "state",
        "event_id": "event_id",
    }
    cols = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=cols)
    keep = ["hwm_id", "latitude", "longitude", "elev_ft", "datum",
            "uncertainty_ft", "hwm_quality", "hwm_type_id", "county", "state", "event_id"]
    df = df[[c for c in keep if c in df.columns]]
    df["event_name"] = event_name

    # Drop marks without coordinates
    lat_col = "latitude" if "latitude" in df.columns else None
    lon_col = "longitude" if "longitude" in df.columns else None
    if lat_col and lon_col:
        df = df.dropna(subset=[lat_col, lon_col])
    log.info("%d HWMs with valid coordinates for %s", len(df), event_name)
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
    # Verify event IDs from live STN event list
    log.info("Fetching STN event list for verification...")
    try:
        events_df = fetch_event_list()
        for event_name, event_id in EVENTS.items():
            match = events_df[events_df.get("event_id", pd.Series()) == event_id]
            if match.empty:
                log.warning("Event ID %d not found in live STN list — verify manually", event_id)
            else:
                log.info("Confirmed event %d: %s", event_id, match.iloc[0].get("event_name", "?"))
    except Exception as e:
        log.warning("Could not fetch event list: %s — proceeding with hardcoded IDs", e)

    for event_name, event_id in EVENTS.items():
        df = fetch_hwm(event_id, event_name)
        if df.empty:
            log.error("No data for %s — check STN event ID %d", event_name, event_id)
            continue
        s3_key = f"raw/usgs_stn/{event_name}_hwm.parquet"
        upload(df, s3_key)

    log.info("fetch_usgs_stn complete")


if __name__ == "__main__":
    main()
