#!/usr/bin/env python3
"""
fetch_noaa_tides_swfl.py -- SageMaker job: pull NOAA Tides and Currents for
Southwest Florida during Ian 2022, Helene 2024, and Milton 2024.

Same CO-OPS API as fetch_noaa_tides.py (New Orleans), but different stations
and event windows.

Outputs:
  s3://swarm-floodrsct-data/raw/noaa_tides/swfl_{station_id}_{event}.parquet
"""

import logging
import sys
import time
from pathlib import Path

import boto3
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
COOPS_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

STATIONS = [
    {"id": "8725520", "name": "Fort_Myers"},
    {"id": "8726520", "name": "St_Petersburg"},
    {"id": "8726607", "name": "Old_Port_Tampa"},
    {"id": "8725110", "name": "Naples"},
    {"id": "8726724", "name": "Clearwater_Beach"},
]

EVENTS = {
    "ian2022":    {"begin": "20220923", "end": "20221001"},
    "helene2024": {"begin": "20240924", "end": "20241001"},
    "milton2024": {"begin": "20241007", "end": "20241012"},
}


def fetch_station_event(station_id: str, station_name: str, event: str,
                        begin_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "station": station_id,
        "begin_date": begin_date,
        "end_date": end_date,
        "product": "water_level",
        "datum": "NAVD",
        "time_zone": "GMT",
        "units": "metric",
        "format": "json",
        "application": "s035_floodrsct",
    }
    log.info("Station %s (%s) / %s", station_id, station_name, event)
    resp = requests.get(COOPS_BASE, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        log.warning("Station %s / %s error: %s", station_id, event,
                    data["error"].get("message", "unknown"))
        return pd.DataFrame()

    obs = data.get("data", [])
    if not obs:
        log.warning("No data for station %s / %s", station_id, event)
        return pd.DataFrame()

    df = pd.DataFrame(obs)
    df = df.rename(columns={"t": "datetime_utc", "v": "water_level_m", "q": "quality_flag"})
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df["water_level_m"] = pd.to_numeric(df["water_level_m"], errors="coerce")
    df["station_id"] = station_id
    df["station_name"] = station_name
    df["event"] = event

    # Fetch predictions for surge
    pred_params = {**params, "product": "predictions", "interval": "h"}
    try:
        time.sleep(0.3)
        pred_resp = requests.get(COOPS_BASE, params=pred_params, timeout=120)
        pred_resp.raise_for_status()
        preds = pred_resp.json().get("predictions", [])
        if preds:
            pred_df = pd.DataFrame(preds)
            pred_df = pred_df.rename(columns={"t": "datetime_utc", "v": "predicted_m"})
            pred_df["datetime_utc"] = pd.to_datetime(pred_df["datetime_utc"], utc=True, errors="coerce")
            pred_df["predicted_m"] = pd.to_numeric(pred_df["predicted_m"], errors="coerce")
            df = pd.merge(df, pred_df[["datetime_utc", "predicted_m"]], on="datetime_utc", how="left")
            df["surge_m"] = df["water_level_m"] - df["predicted_m"]
    except Exception as e:
        log.warning("Predictions unavailable for %s / %s: %s", station_id, event, e)

    return df


def upload(df: pd.DataFrame, s3_key: str) -> None:
    s3 = boto3.client("s3", region_name="us-east-1")
    local = f"/tmp/{Path(s3_key).name}"
    df.to_parquet(local, index=False)
    s3.upload_file(local, BUCKET, s3_key)
    log.info("Uploaded %d rows to s3://%s/%s", len(df), BUCKET, s3_key)


def main() -> None:
    for event, window in EVENTS.items():
        for station in STATIONS:
            df = fetch_station_event(
                station_id=station["id"],
                station_name=station["name"],
                event=event,
                begin_date=window["begin"],
                end_date=window["end"],
            )
            if not df.empty:
                s3_key = f"raw/noaa_tides/swfl_{station['id']}_{event}.parquet"
                upload(df, s3_key)
            time.sleep(1.0)

    log.info("fetch_noaa_tides_swfl complete")


if __name__ == "__main__":
    main()
