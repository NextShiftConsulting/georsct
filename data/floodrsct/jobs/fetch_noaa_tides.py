#!/usr/bin/env python3
"""
fetch_noaa_tides.py -- Fetch NOAA CO-OPS tidal predictions and verified water levels.

Downloads hourly water level data (observed + predicted) for tide stations
within a flood event window, then computes storm surge residual.

Products fetched per station:
  - hourly_height: verified (observed) water levels
  - predictions: astronomical tide predictions

Derived column:
  - surge_m = observed_m - predicted_m (storm surge signal)

Output per station:
  - raw/noaa_tides/{event}/{station_id}_hourly_height.json
  - raw/noaa_tides/{event}/{station_id}_predictions.json
  - raw/noaa_tides/{event}/tidal_surge_{station_id}.parquet

Usage (SageMaker container):
    python3 -u fetch_noaa_tides.py --event harvey2017
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from _manifest_writer import write_manifest

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUCKET = "swarm-floodrsct-data"
DST_PREFIX = "raw/noaa_tides"

API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

STATIONS = {
    "houston": [
        {"id": "8771450", "name": "Galveston Pier 21"},
        {"id": "8771013", "name": "Eagle Point"},
        {"id": "8770822", "name": "Texas Point"},
    ],
    "nyc_nj": [
        {"id": "8518750", "name": "The Battery"},
        {"id": "8531680", "name": "Sandy Hook"},
    ],
    "southwest_florida": [
        {"id": "8725520", "name": "Fort Myers"},
        {"id": "8726520", "name": "St. Petersburg"},
    ],
    "new_orleans": [
        {"id": "8761724", "name": "Grand Isle"},
        {"id": "8761927", "name": "New Canal Station"},
    ],
    "southern_california": [
        {"id": "9410230", "name": "La Jolla"},
        {"id": "9410660", "name": "Los Angeles"},
    ],
}

EVENT_WINDOWS = {
    "harvey2017": {"start": "20170817", "end": "20170904", "scenario": "houston"},
    "imelda2019": {"start": "20190917", "end": "20190922", "scenario": "houston"},
    "beryl2024": {"start": "20240707", "end": "20240713", "scenario": "houston"},
    "ida2021_nyc": {"start": "20210901", "end": "20210904", "scenario": "nyc_nj"},
    "ian2022": {"start": "20220923", "end": "20221001", "scenario": "southwest_florida"},
    "hilary2023": {"start": "20230819", "end": "20230823", "scenario": "southern_california"},
}

PRODUCTS = ["hourly_height", "predictions"]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_product(
    station_id: str,
    product: str,
    begin_date: str,
    end_date: str,
) -> dict:
    """Fetch a single product from NOAA CO-OPS API.

    Args:
        station_id: NOAA station ID.
        product: "hourly_height" or "predictions".
        begin_date: YYYYMMDD start.
        end_date: YYYYMMDD end.

    Returns:
        Raw JSON response as dict.
    """
    params = {
        "begin_date": begin_date,
        "end_date": end_date,
        "station": station_id,
        "product": product,
        "datum": "NAVD",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "application": "swarm_floodrsct",
    }
    log.info("  Fetching %s for station %s (%s - %s)",
             product, station_id, begin_date, end_date)

    resp = requests.get(API_URL, params=params, timeout=120)

    if resp.status_code == 400:
        log.warning("  HTTP 400 for station %s/%s -- station may not support "
                    "NAVD datum or date range. Skipping.", station_id, product)
        return {"error": {"message": f"HTTP 400 for station {station_id}"}}

    resp.raise_for_status()
    payload = resp.json()

    if "error" in payload:
        log.warning("  API error for %s/%s: %s",
                    station_id, product, payload["error"].get("message", ""))
        return payload

    n_records = len(payload.get("data", []))
    log.info("  Received %d records for %s", n_records, product)
    return payload


def parse_timeseries(payload: dict) -> pd.DataFrame:
    """Parse CO-OPS JSON data array into a DataFrame.

    Handles missing values where "v" is empty string or null.

    Returns:
        DataFrame with columns: timestamp (datetime64), value_m (float64).
    """
    records = payload.get("data", [])
    if not records:
        return pd.DataFrame(columns=["timestamp", "value_m"])

    rows = []
    for rec in records:
        ts = rec.get("t", "")
        raw_val = rec.get("v", "")
        if raw_val is None or str(raw_val).strip() == "":
            val = float("nan")
        else:
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                val = float("nan")
        rows.append({"timestamp": ts, "value_m": val})

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M")
    return df


def build_surge_df(
    observed_payload: dict,
    predicted_payload: dict,
) -> pd.DataFrame:
    """Combine observed and predicted into a surge residual DataFrame.

    Returns:
        DataFrame with columns: timestamp, observed_m, predicted_m, surge_m.
    """
    obs = parse_timeseries(observed_payload)
    pred = parse_timeseries(predicted_payload)

    if obs.empty and pred.empty:
        return pd.DataFrame(
            columns=["timestamp", "observed_m", "predicted_m", "surge_m"]
        )

    obs = obs.rename(columns={"value_m": "observed_m"})
    pred = pred.rename(columns={"value_m": "predicted_m"})

    merged = pd.merge(obs, pred, on="timestamp", how="outer")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["surge_m"] = merged["observed_m"] - merged["predicted_m"]

    return merged


# ---------------------------------------------------------------------------
# S3 upload helpers
# ---------------------------------------------------------------------------

def upload_json(s3, key: str, payload: dict) -> None:
    """Upload a JSON payload to S3."""
    body = json.dumps(payload, indent=2).encode()
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType="application/json")
    log.info("  Uploaded s3://%s/%s (%.1f KB)", BUCKET, key, len(body) / 1024)


def upload_parquet(s3, key: str, df: pd.DataFrame) -> None:
    """Upload a DataFrame as Parquet to S3."""
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue(),
                  ContentType="application/octet-stream")
    log.info("  Uploaded s3://%s/%s (%d rows, %.1f KB)",
             BUCKET, key, len(df), len(buf.getvalue()) / 1024)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch NOAA tidal data for a flood event window"
    )
    parser.add_argument("--event", required=True, choices=list(EVENT_WINDOWS.keys()),
                        help="Event key (e.g. harvey2017)")
    args = parser.parse_args()

    event = args.event
    window = EVENT_WINDOWS[event]
    scenario = window["scenario"]
    stations = STATIONS[scenario]

    log.info("=== NOAA Tides Fetch ===")
    log.info("Event:    %s", event)
    log.info("Scenario: %s", scenario)
    log.info("Window:   %s - %s", window["start"], window["end"])
    log.info("Stations: %d", len(stations))

    s3 = boto3.client("s3", region_name="us-east-1")

    total_records = 0
    uploaded_keys = []

    for station in stations:
        sid = station["id"]
        sname = station["name"]
        log.info("")
        log.info("Station %s (%s)", sid, sname)

        payloads = {}
        for product in PRODUCTS:
            payload = fetch_product(sid, product, window["start"], window["end"])
            payloads[product] = payload

            # Upload raw JSON
            json_key = f"{DST_PREFIX}/{event}/{sid}_{product}.json"
            upload_json(s3, json_key, payload)
            uploaded_keys.append(json_key)

        # Build surge residual
        surge_df = build_surge_df(
            payloads.get("hourly_height", {}),
            payloads.get("predictions", {}),
        )

        if surge_df.empty:
            log.warning("  No data for station %s -- skipping parquet", sid)
            continue

        n_valid_obs = surge_df["observed_m"].notna().sum()
        n_valid_pred = surge_df["predicted_m"].notna().sum()
        n_valid_surge = surge_df["surge_m"].notna().sum()
        log.info("  Surge DF: %d rows, %d obs valid, %d pred valid, %d surge valid",
                 len(surge_df), n_valid_obs, n_valid_pred, n_valid_surge)

        if n_valid_surge > 0:
            max_surge = surge_df["surge_m"].max()
            min_surge = surge_df["surge_m"].min()
            log.info("  Peak surge: %.3f m, min surge: %.3f m", max_surge, min_surge)

        pq_key = f"{DST_PREFIX}/{event}/tidal_surge_{sid}.parquet"
        upload_parquet(s3, pq_key, surge_df)
        uploaded_keys.append(pq_key)
        total_records += len(surge_df)

    # Write manifest
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("Event:         %s", event)
    log.info("Stations:      %d", len(stations))
    log.info("Total records: %d", total_records)
    log.info("Uploaded keys: %d", len(uploaded_keys))

    write_manifest(
        s3=s3,
        dataset=f"noaa_tides_{event}",
        version=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        source_url=API_URL,
        s3_key=f"{DST_PREFIX}/{event}/",
        crs="N/A",
        record_count=total_records,
        license_="Public Domain / U.S. Government",
        notes=(
            f"NOAA CO-OPS hourly_height + predictions for {event} "
            f"({window['start']}-{window['end']}), "
            f"datum=NAVD, units=metric, {len(stations)} stations"
        ),
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
