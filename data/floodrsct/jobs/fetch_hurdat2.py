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

# IBTrACS provides best-track data for recent storms not yet in HURDAT2,
# and covers ALL basins (Atlantic, East Pacific, etc.).
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.last3years.list.v04r01.csv"
)

# Storms of interest for the flood-cert study
STORMS_OF_INTEREST = {
    "AL092017",  # Harvey
    "AL112019",  # Imelda
    "AL082021",  # Henri (NYC)
    "AL092021",  # Ida (NOLA + NYC)
    "AL092022",  # Ian (SW Florida)
    "EP082023",  # Hilary (Riverside-Coachella) -- East Pacific basin
    "AL022024",  # Beryl (Houston)
    "AL092024",  # Helene (SW Florida)
    "AL142024",  # Milton (SW Florida)
}

# IBTrACS SID format differs from HURDAT2.  Mapping: HURDAT2 -> IBTrACS SID.
# IBTrACS uses YYYY{basin}{num} pattern, e.g. 2024219N13314 for Beryl.
# We match by name+year instead of SID for robustness.
IBTRACS_FALLBACK = {
    "AL022024": {"name": "BERYL",   "year": 2024, "basin": "NA"},
    "AL092024": {"name": "HELENE",  "year": 2024, "basin": "NA"},
    "AL142024": {"name": "MILTON",  "year": 2024, "basin": "NA"},
    "EP082023": {"name": "HILARY",  "year": 2023, "basin": "EP"},
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


def fetch_ibtracs_fallback(missing_ids: set[str]) -> pd.DataFrame:
    """Fetch tracks from IBTrACS for storms not in HURDAT2.

    IBTrACS is updated more frequently than HURDAT2 and covers all basins.
    We match storms by name + year since IBTrACS SIDs differ from HURDAT2.
    """
    needs = {sid: cfg for sid, cfg in IBTRACS_FALLBACK.items() if sid in missing_ids}
    if not needs:
        return pd.DataFrame()

    log.info("Fetching IBTrACS for %d missing storms: %s",
             len(needs), list(needs.keys()))

    import requests
    try:
        resp = requests.get(IBTRACS_URL, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        log.warning("IBTrACS download failed: %s", e)
        return pd.DataFrame()

    # IBTrACS CSV: SID, SEASON, NAME, ISO_TIME, LAT, LON, WMO_WIND, WMO_PRES, BASIN, ...
    # Skip first row (units row) after header
    lines = resp.text.splitlines()
    if len(lines) < 3:
        log.warning("IBTrACS CSV too short (%d lines)", len(lines))
        return pd.DataFrame()

    # Parse header
    header = [h.strip() for h in lines[0].split(",")]
    # Row 1 is units row in IBTrACS format -- skip it
    data_lines = lines[2:]

    rows = []
    for hurdat_id, cfg in needs.items():
        target_name = cfg["name"].upper()
        target_year = cfg["year"]
        target_basin = cfg["basin"]

        for line in data_lines:
            parts = line.split(",")
            if len(parts) < 12:
                continue
            try:
                season = int(parts[header.index("SEASON")].strip()) if "SEASON" in header else 0
                name = parts[header.index("NAME")].strip().upper() if "NAME" in header else ""
                basin = parts[header.index("BASIN")].strip() if "BASIN" in header else ""
            except (ValueError, IndexError):
                continue

            if name != target_name or season != target_year:
                continue
            if target_basin and basin != target_basin:
                continue

            try:
                iso_time = parts[header.index("ISO_TIME")].strip()
                lat_str = parts[header.index("LAT")].strip()
                lon_str = parts[header.index("LON")].strip()
                wind_str = parts[header.index("WMO_WIND")].strip() if "WMO_WIND" in header else ""
                pres_str = parts[header.index("WMO_PRES")].strip() if "WMO_PRES" in header else ""
                nature = parts[header.index("NATURE")].strip() if "NATURE" in header else ""

                if not lat_str or not lon_str or lat_str == " ":
                    continue

                ts = datetime.strptime(iso_time[:19], "%Y-%m-%d %H:%M:%S")
                lat = float(lat_str)
                lon = float(lon_str)
                wind = int(float(wind_str)) if wind_str and wind_str.strip() not in ("", " ") else None
                pres = int(float(pres_str)) if pres_str and pres_str.strip() not in ("", " ") else None

                # Map IBTrACS NATURE to HURDAT2-style status
                status_map = {"TS": "TS", "HU": "HU", "TD": "TD",
                              "ET": "EX", "SS": "SS", "DS": "LO"}
                status = status_map.get(nature, nature if nature else "TS")

                rows.append({
                    "storm_id": hurdat_id,
                    "storm_name": target_name,
                    "timestamp": ts,
                    "status": status,
                    "lat": lat,
                    "lon": lon,
                    "max_wind_kt": wind,
                    "min_pressure_mb": pres,
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        for sid in df["storm_id"].unique():
            n = len(df[df["storm_id"] == sid])
            log.info("IBTrACS: %s (%s) -> %d track records",
                     sid, df[df["storm_id"] == sid]["storm_name"].iloc[0], n)
    else:
        log.warning("IBTrACS: no matching records for %s", list(needs.keys()))
    return df


def main() -> None:
    s3 = get_s3()

    # Always rebuild parquet (storms of interest may have changed)
    raw_exists = s3_key_exists(s3, BUCKET, RAW_KEY)

    # ---- download raw HURDAT2 text ----
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

    # ---- parse HURDAT2 ----
    with open(tmp_path, "r", encoding="utf-8") as fh:
        raw_text = fh.read()

    os.unlink(tmp_path)

    df = parse_hurdat2(raw_text)
    found_ids = set(df["storm_id"].unique()) if not df.empty else set()
    missing_ids = STORMS_OF_INTEREST - found_ids

    if missing_ids:
        log.info("Missing from HURDAT2: %s -- trying IBTrACS fallback", missing_ids)
        ibtracs_df = fetch_ibtracs_fallback(missing_ids)
        if not ibtracs_df.empty:
            df = pd.concat([df, ibtracs_df], ignore_index=True)

    if df.empty:
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

    still_missing = STORMS_OF_INTEREST - set(df["storm_id"].unique()) if not df.empty else STORMS_OF_INTEREST
    if still_missing:
        log.warning("Still missing after IBTrACS fallback: %s", still_missing)

    # ---- write parquet to S3 (always overwrite -- storm list may have changed) ----
    parquet_tmp = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}_storm_tracks.parquet")
    df.to_parquet(parquet_tmp, index=False, engine="pyarrow")
    parquet_size_mb = Path(parquet_tmp).stat().st_size / 1e6
    log.info(
        "Uploading parquet to s3://%s/%s (%.2f MB, %d storms, %d records)",
        BUCKET, PARQUET_KEY, parquet_size_mb,
        df["storm_id"].nunique() if not df.empty else 0, len(df),
    )
    s3.upload_file(parquet_tmp, BUCKET, PARQUET_KEY)
    os.unlink(parquet_tmp)

    # ---- manifest ----
    write_manifest(
        s3,
        dataset="hurdat2",
        version="2024",
        source_url=SOURCE_URL,
        s3_key=PARQUET_KEY,
        crs="EPSG:4326",
        record_count=len(df),
        notes="HURDAT2 Atlantic best-track + IBTrACS fallback for 2024 and EP-basin storms. "
              "Storms: Harvey, Imelda, Henri, Ida, Ian, Hilary, Beryl, Helene, Milton. "
              "Raw HURDAT2 text at raw/hurdat2/hurdat2_atlantic.txt.",
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
