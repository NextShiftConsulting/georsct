"""
fetch_hurdat2.py -- Download and parse NHC HURDAT2 Best Track Data (Atlantic).

SageMaker container script (NOT a launcher).
Source: NHC HURDAT2, Atlantic basin, 1851-2023
URL: https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt

Output:
  s3://swarm-floodrsct-data/raw/hurdat2/hurdat2_atlantic.txt    (raw text)
  s3://swarm-floodrsct-data/raw/hurdat2/storm_tracks.parquet    (parsed, filtered)
"""

import io
import logging
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from _manifest_writer import write_manifest
from _s3_stream import s3_key_exists, get_s3, stream_to_tmp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("fetch_hurdat2")

BUCKET = "swarm-floodrsct-data"
SOURCE_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
RAW_KEY = "raw/hurdat2/hurdat2_atlantic.txt"
PARQUET_KEY = "raw/hurdat2/storm_tracks.parquet"

# Storms of interest for the flood-cert study
STORMS_OF_INTEREST = {
    "AL092017",  # Harvey
    "AL132019",  # Imelda
    "AL022024",  # Beryl
    "AL092021",  # Ida
    "AL092022",  # Ian
    "EP092023",  # Hilary
}

TMP_DIR = "/tmp"


def parse_lat(raw: str) -> float:
    """Parse HURDAT2 lat like '13.5N' -> 13.5 or '13.5S' -> -13.5."""
    raw = raw.strip()
    val = float(raw[:-1])
    if raw[-1] == "S":
        val = -val
    return val


def parse_lon(raw: str) -> float:
    """Parse HURDAT2 lon like '53.0W' -> -53.0 or '53.0E' -> 53.0."""
    raw = raw.strip()
    val = float(raw[:-1])
    if raw[-1] == "W":
        val = -val
    return val


def parse_int_or_none(raw: str) -> Optional[int]:
    """Parse integer field; return None for missing (-999 or blank)."""
    raw = raw.strip()
    if not raw or raw == "-999":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def parse_hurdat2(text: str) -> pd.DataFrame:
    """Parse full HURDAT2 text into a DataFrame, filtering to storms of interest.

    HURDAT2 format:
      Header: AL092017,                HARVEY,     42,
      Data:   20170817, 0000,  , TD, 13.5N,  53.0W,  30, 1007, ...

    Returns DataFrame with columns:
      storm_id, storm_name, timestamp, status, lat, lon, max_wind_kt, min_pressure_mb
    """
    rows = []
    current_id = None
    current_name = None

    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue

        # Header line: starts with basin code (AL/EP/CP), has exactly 3 comma-separated fields
        parts = line.split(",")
        first = parts[0].strip()

        if re.match(r"^[A-Z]{2}\d{6}$", first):
            # Header line
            current_id = first
            current_name = parts[1].strip()
            continue

        # Data line: starts with date (8 digits)
        date_str = parts[0].strip()
        if not re.match(r"^\d{8}$", date_str):
            continue

        # Only keep storms of interest
        if current_id not in STORMS_OF_INTEREST:
            continue

        time_str = parts[1].strip().zfill(4)  # e.g. "0000"
        # parts[2] = record identifier (L, W, etc. or blank)
        status = parts[3].strip()

        lat = parse_lat(parts[4])
        lon = parse_lon(parts[5])
        max_wind = parse_int_or_none(parts[6])
        min_pressure = parse_int_or_none(parts[7])

        timestamp = datetime.strptime(
            f"{date_str}{time_str}", "%Y%m%d%H%M"
        )

        rows.append({
            "storm_id": current_id,
            "storm_name": current_name,
            "timestamp": timestamp,
            "status": status,
            "lat": lat,
            "lon": lon,
            "max_wind_kt": max_wind,
            "min_pressure_mb": min_pressure,
        })

    df = pd.DataFrame(rows)
    log.info(
        "Parsed %d track records for %d storms of interest",
        len(df),
        df["storm_id"].nunique() if len(df) > 0 else 0,
    )
    return df


def main() -> None:
    s3 = get_s3()

    # ---- checkpoint: skip if both outputs already exist ----
    raw_exists = s3_key_exists(s3, BUCKET, RAW_KEY)
    parquet_exists = s3_key_exists(s3, BUCKET, PARQUET_KEY)
    if raw_exists and parquet_exists:
        log.info("Checkpoint hit -- both outputs already in S3.")
        log.info("Done (skipped).")
        return

    # ---- download raw text ----
    tmp_name = f"{uuid.uuid4().hex}_hurdat2.txt"
    tmp_path = os.path.join(TMP_DIR, tmp_name)

    log.info("Downloading HURDAT2 text (~3 MB) from %s", SOURCE_URL)
    ok = stream_to_tmp(SOURCE_URL, tmp_path, retries=3, timeout=120)
    if not ok:
        raise RuntimeError(f"Failed to download {SOURCE_URL}")

    raw_size = Path(tmp_path).stat().st_size
    log.info("Downloaded: %d bytes", raw_size)

    # ---- upload raw text to S3 ----
    if not raw_exists:
        log.info("Uploading raw text to s3://%s/%s", BUCKET, RAW_KEY)
        s3.upload_file(tmp_path, BUCKET, RAW_KEY)

    # ---- parse and filter ----
    with open(tmp_path, "r", encoding="utf-8") as fh:
        raw_text = fh.read()

    os.unlink(tmp_path)

    df = parse_hurdat2(raw_text)

    if len(df) == 0:
        log.warning("No matching storms found -- check STORMS_OF_INTEREST IDs.")

    # Log per-storm summary
    for sid, grp in df.groupby("storm_id"):
        log.info(
            "  %s (%s): %d records, wind range %s-%s kt",
            sid,
            grp["storm_name"].iloc[0],
            len(grp),
            grp["max_wind_kt"].min(),
            grp["max_wind_kt"].max(),
        )

    # ---- write parquet to S3 ----
    if not parquet_exists:
        parquet_tmp = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}_storm_tracks.parquet")
        df.to_parquet(parquet_tmp, index=False, engine="pyarrow")
        parquet_size_mb = Path(parquet_tmp).stat().st_size / 1e6
        log.info(
            "Uploading parquet to s3://%s/%s (%.2f MB)",
            BUCKET, PARQUET_KEY, parquet_size_mb,
        )
        s3.upload_file(parquet_tmp, BUCKET, PARQUET_KEY)
        os.unlink(parquet_tmp)

    # ---- manifest ----
    write_manifest(
        s3,
        dataset="hurdat2",
        version="2023",
        source_url=SOURCE_URL,
        s3_key=PARQUET_KEY,
        crs="EPSG:4326",
        record_count=len(df),
        notes="HURDAT2 Atlantic best-track data, filtered to storms of interest: "
              "Harvey, Imelda, Beryl, Ida, Ian, Hilary. "
              "Raw text also stored at raw/hurdat2/hurdat2_atlantic.txt.",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
