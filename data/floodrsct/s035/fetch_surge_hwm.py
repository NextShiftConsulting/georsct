"""
fetch_surge_hwm.py -- Download and normalize USGS STN high-water marks for
storm surge estimation.

Sources:
  - Existing parquets in S3: s3://swarm-floodrsct-data/raw/usgs_stn/{event}_hwm.parquet
  - USGS STN Flood Event API (fallback):
    https://stn.wim.usgs.gov/STNServices/HWMs.json?Event={event_id}&State={state}

Output per event:
  s3://swarm-floodrsct-data/raw/surge_estimates/{event}/hwm_{event}.parquet
  Columns: hwm_id, latitude, longitude, elev_ft,
           elev_m, event, source

Note: elev_ft is elevation above datum (typically NAVD88), not height above
ground level. This is the standard STN HWM measurement.

Downstream build_event_dataset.py handles the ZCTA spatial join.
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from _manifest_writer import write_manifest

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("fetch_surge_hwm")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUCKET = "swarm-floodrsct-data"
FT_TO_M = 0.3048

# Map event slug -> (STN event_id, state abbreviation)
EVENT_REGISTRY = {
    "harvey2017": (180, "TX"),
    "imelda2019": (None, "TX"),  # No STN deployment for Imelda
    "ian2022": (325, "FL"),
    "ida2021_nyc": (312, "NY"),
    "beryl2024": (342, "TX"),
    "hilary2023": (335, "CA"),
}

# Existing S3 parquets (pre-uploaded)
S3_HWM_KEYS = {
    "harvey2017": "raw/usgs_stn/harvey2017_hwm.parquet",
    "imelda2019": "raw/usgs_stn/imelda2019_hwm.parquet",
}

STN_API_BASE = "https://stn.wim.usgs.gov/STNServices/HWMs/FilteredHWMs.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _s3_key_exists(s3, key: str) -> bool:
    """Check if key exists in BUCKET."""
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def _download_parquet_from_s3(s3, key: str) -> pd.DataFrame:
    """Download a parquet from S3 and return as DataFrame."""
    log.info("Downloading s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    return pd.read_parquet(buf)


def _fetch_from_stn_api(event_id: int, state: str) -> pd.DataFrame:
    """Fetch HWMs from USGS STN FilteredHWMs API."""
    if event_id is None:
        log.warning("No STN event ID for this event -- no API data available")
        return pd.DataFrame()

    url = f"{STN_API_BASE}?Event={event_id}"
    log.info("Fetching STN API: %s", url)

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    records = resp.json()

    if not records:
        log.warning("STN API returned 0 records for event=%d", event_id)
        return pd.DataFrame()

    log.info("STN API returned %d total HWM records", len(records))

    rows = []
    for rec in records:
        lat = rec.get("latitude")
        lon = rec.get("longitude")
        elev = rec.get("elev_ft")
        hwm_id = rec.get("hwm_id")

        if lat is None or lon is None:
            continue
        if elev is None or elev <= 0:
            continue

        rows.append({
            "hwm_id": int(hwm_id) if hwm_id else None,
            "latitude": float(lat),
            "longitude": float(lon),
            "elev_ft": float(elev),
        })

    df = pd.DataFrame(rows)
    log.info("After filtering: %d HWMs with valid coordinates and elevation", len(df))
    return df


def _normalize_s3_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names from an existing S3 parquet.

    The pre-uploaded parquets have columns:
      hwm_id, latitude, longitude, elev_ft, datum, uncertainty_ft,
      hwm_quality, hwm_type_id, county, state, event_id, event_name

    Standardize to: hwm_id, latitude, longitude, elev_ft.
    """
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    # latitude
    for candidate in ["latitude", "lat", "site_latitude"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "latitude"
            break

    # longitude
    for candidate in ["longitude", "lon", "lng", "site_longitude"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "longitude"
            break

    # elevation above datum
    for candidate in ["elev_ft", "elevation_ft", "height_above_gnd", "height_above_gnd_ft"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "elev_ft"
            break

    # hwm id
    for candidate in ["hwm_id", "id"]:
        if candidate in lower_cols:
            col_map[lower_cols[candidate]] = "hwm_id"
            break

    df = df.rename(columns=col_map)

    # Keep only relevant columns that exist
    keep = [c for c in ["hwm_id", "latitude", "longitude", "elev_ft"] if c in df.columns]
    df = df[keep].copy()

    # Drop rows missing lat/lon or elevation
    required = [c for c in ["latitude", "longitude", "elev_ft"] if c in df.columns]
    if required:
        df = df.dropna(subset=required)

    # Filter out non-positive elevations
    if "elev_ft" in df.columns:
        df = df[df["elev_ft"] > 0].copy()

    return df


def _add_derived_columns(df: pd.DataFrame, event: str, source: str) -> pd.DataFrame:
    """Add meters conversion, event tag, and source label."""
    df = df.copy()
    df["elev_m"] = df["elev_ft"] * FT_TO_M
    df["event"] = event
    df["source"] = source
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and normalize USGS STN HWMs")
    parser.add_argument(
        "--event",
        required=True,
        help="Event slug, e.g. harvey2017, imelda2019, ian2022",
    )
    args = parser.parse_args()
    event = args.event.lower()

    if event not in EVENT_REGISTRY:
        log.error(
            "Unknown event '%s'. Known events: %s",
            event,
            ", ".join(sorted(EVENT_REGISTRY)),
        )
        sys.exit(1)

    event_entry = EVENT_REGISTRY[event]
    event_id, state = event_entry
    s3 = boto3.client("s3", region_name="us-east-1")

    # Output key
    out_key = f"raw/surge_estimates/{event}/hwm_{event}.parquet"

    # Check if output already exists
    if _s3_key_exists(s3, out_key):
        log.info("Output already exists: s3://%s/%s -- skipping", BUCKET, out_key)
        return

    # Try existing S3 parquet first, then fall back to STN API
    source_label = "unknown"
    s3_key_raw = S3_HWM_KEYS.get(event)

    if s3_key_raw and _s3_key_exists(s3, s3_key_raw):
        log.info("Using existing S3 parquet for %s", event)
        df = _download_parquet_from_s3(s3, s3_key_raw)
        df = _normalize_s3_parquet(df)
        source_label = f"s3://{BUCKET}/{s3_key_raw}"
    else:
        log.info("No S3 parquet for %s -- fetching from STN API", event)
        df = _fetch_from_stn_api(event_id, state)
        source_label = f"{STN_API_BASE}?Event={event_id}"

    if df.empty:
        log.warning("No valid HWM records for event %s -- writing empty manifest", event)
        write_manifest(
            s3=s3,
            dataset=f"surge_hwm_{event}",
            version=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
            source_url="N/A",
            s3_key="N/A",
            record_count=0,
            notes=f"No STN HWM data available for {event}.",
        )
        log.info("Done -- %s has no HWM data", event)
        return

    # Add derived columns
    df = _add_derived_columns(df, event, source_label)

    log.info(
        "Event %s: %d HWMs, elevation range %.2f - %.2f m",
        event,
        len(df),
        df["elev_m"].min(),
        df["elev_m"].max(),
    )

    # Write to parquet in memory and upload
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    payload = buf.getvalue()

    log.info("Uploading s3://%s/%s (%d bytes)", BUCKET, out_key, len(payload))
    s3.put_object(Bucket=BUCKET, Key=out_key, Body=payload)

    # Write manifest
    write_manifest(
        s3=s3,
        dataset=f"surge_hwm_{event}",
        version=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        source_url=source_label,
        s3_key=out_key,
        record_count=len(df),
        notes=f"USGS STN high-water marks for {event}. Heights in feet and meters above ground.",
    )

    log.info("Done -- %s complete", event)


if __name__ == "__main__":
    main()
