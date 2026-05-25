#!/usr/bin/env python3
"""
build_noaa_storm_events.py -- NOAA Storm Events flood history per ZCTA.

Downloads NOAA Storm Events Database (1996-2024) for flood-type events,
aggregates to county level, then distributes to ZCTAs via the county crosswalk.

Flood event types included:
  FLASH FLOOD, FLOOD, COASTAL FLOOD, LAKESHORE FLOOD

Output: noaa_storm_events_zcta.parquet
  - zcta_id              (str, 5-digit zero-padded)
  - flood_event_count    (int)   total flood events 1996-2024
  - flood_event_count_5y (int)   flood events 2019-2024
  - flood_deaths         (int)   direct + indirect deaths
  - flood_injuries       (int)   direct + indirect injuries
  - flood_property_damage_k (float) property damage in $1000s
  - flood_crop_damage_k     (float) crop damage in $1000s
  - flood_events_per_year   (float) annualized rate 1996-2024

Source: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

Join method: county_fips majority assignment from zcta_county_crosswalk.parquet.
ZCTAs inherit the flood history of their majority county.

Usage:
    python build_noaa_storm_events.py --dry-run
    python build_noaa_storm_events.py --upload
    python build_noaa_storm_events.py --years 2015 2024  # subset
"""

import argparse
import gzip
import io
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-yrsn-datasets"
PREFIX = "rsct_curriculum/series_018/processed"
NOAA_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

FLOOD_EVENT_TYPES = {
    "Flash Flood",
    "Flood",
    "Coastal Flood",
    "Lakeshore Flood",
}

FIRST_YEAR = 1996
RECENT_CUTOFF = 2019  # "recent 5-year" window
N_YEARS = 2024 - FIRST_YEAR + 1  # for annualized rate

DAMAGE_MULTIPLIERS = {"K": 1.0, "M": 1000.0, "B": 1_000_000.0}


def list_detail_files() -> list[tuple[int, str]]:
    """Fetch directory listing and return (year, url) pairs for detail CSVs."""
    log.info("Fetching NOAA directory listing from %s", NOAA_BASE)
    resp = requests.get(NOAA_BASE, timeout=60)
    resp.raise_for_status()

    # Files match: StormEvents_details-ftp_v1.0_dYYYY_c*.csv.gz
    pattern = re.compile(
        r'StormEvents_details-ftp_v1\.0_d(\d{4})_c\d{8}\.csv\.gz'
    )
    seen_years = {}
    for match in pattern.finditer(resp.text):
        fname = match.group(0)
        year = int(match.group(1))
        if FIRST_YEAR <= year <= 2024:
            # Keep only the most recent creation-date file per year
            seen_years[year] = fname

    result = [(yr, NOAA_BASE + fname) for yr, fname in sorted(seen_years.items())]
    log.info("Found %d annual detail files (%d-%d)",
             len(result), result[0][0] if result else 0,
             result[-1][0] if result else 0)
    return result


def parse_damage(value: str) -> float:
    """Convert NOAA damage string ('10K', '1.5M', '2B') to $1000s."""
    if not value or str(value).strip() in ("", "0", "nan"):
        return 0.0
    v = str(value).strip().upper()
    for suffix, mult in DAMAGE_MULTIPLIERS.items():
        if v.endswith(suffix):
            try:
                return float(v[:-1]) * mult
            except ValueError:
                return 0.0
    try:
        return float(v) / 1000.0  # raw dollars → $1000s
    except ValueError:
        return 0.0


def fetch_year(year: int, url: str) -> pd.DataFrame:
    """Download and parse one year's detail file. Returns flood events only."""
    log.info("  %d: fetching %s", year, url.split("/")[-1])
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            break
        except Exception as exc:
            if attempt == 2:
                log.error("  %d: failed after 3 attempts: %s", year, exc)
                return pd.DataFrame()
            time.sleep(5 * (attempt + 1))

    with gzip.open(io.BytesIO(resp.content), "rt", encoding="latin-1") as f:
        df = pd.read_csv(f, dtype=str, low_memory=False)

    # Filter to flood events only
    df["EVENT_TYPE"] = df.get("EVENT_TYPE", pd.Series(dtype=str)).str.strip().str.title()
    df = df[df["EVENT_TYPE"].isin(FLOOD_EVENT_TYPES)].copy()

    if df.empty:
        log.info("  %d: 0 flood events", year)
        return pd.DataFrame()

    # Normalize county FIPS: STATE_FIPS (2) + CZ_FIPS (3, zero-padded)
    # NOAA uses CZ_TYPE: C=county, Z=zone; we want county-type entries
    if "CZ_TYPE" in df.columns:
        df = df[df["CZ_TYPE"].str.strip().str.upper() == "C"].copy()

    df["state_fips"] = df.get("STATE_FIPS", pd.Series(dtype=str)).astype(str).str.zfill(2)
    df["cz_fips"] = df.get("CZ_FIPS", pd.Series(dtype=str)).astype(str).str.zfill(3)
    df["county_fips"] = df["state_fips"] + df["cz_fips"]

    # Parse damage and deaths/injuries
    df["prop_dmg_k"] = df.get("DAMAGE_PROPERTY", pd.Series(dtype=str)).apply(parse_damage)
    df["crop_dmg_k"] = df.get("DAMAGE_CROPS", pd.Series(dtype=str)).apply(parse_damage)

    for col in ("DEATHS_DIRECT", "DEATHS_INDIRECT", "INJURIES_DIRECT", "INJURIES_INDIRECT"):
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    df["deaths"] = df["DEATHS_DIRECT"] + df["DEATHS_INDIRECT"]
    df["injuries"] = df["INJURIES_DIRECT"] + df["INJURIES_INDIRECT"]
    df["year"] = year

    log.info("  %d: %d flood events across %d counties",
             year, len(df), df["county_fips"].nunique())

    return df[["county_fips", "year", "prop_dmg_k", "crop_dmg_k",
               "deaths", "injuries"]].copy()


def aggregate_to_county(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all flood events to county-level summary."""
    log.info("Aggregating to county level (%d total events)...", len(events))

    # Total counts
    by_county = events.groupby("county_fips").agg(
        flood_event_count=("year", "count"),
        flood_deaths=("deaths", "sum"),
        flood_injuries=("injuries", "sum"),
        flood_property_damage_k=("prop_dmg_k", "sum"),
        flood_crop_damage_k=("crop_dmg_k", "sum"),
    ).reset_index()

    # Recent 5-year count
    recent = (
        events[events["year"] >= RECENT_CUTOFF]
        .groupby("county_fips")
        .size()
        .reset_index(name="flood_event_count_5y")
    )
    by_county = by_county.merge(recent, on="county_fips", how="left")
    by_county["flood_event_count_5y"] = (
        by_county["flood_event_count_5y"].fillna(0).astype(int)
    )

    # Annualized rate
    by_county["flood_events_per_year"] = (
        by_county["flood_event_count"] / N_YEARS
    ).round(3)

    log.info("  %d counties with flood history", len(by_county))
    return by_county


def county_to_zcta(
    county_df: pd.DataFrame, crosswalk_path: Path
) -> pd.DataFrame:
    """Distribute county flood statistics to ZCTAs via majority-county crosswalk."""
    log.info("Joining to ZCTAs via county crosswalk...")
    xwalk = pd.read_parquet(crosswalk_path)
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    xwalk["county_fips"] = xwalk["county_fips"].astype(str).str.zfill(5)
    log.info("  Crosswalk: %d ZCTAs", len(xwalk))

    merged = xwalk[["zcta_id", "county_fips"]].merge(
        county_df, on="county_fips", how="left"
    )

    flood_cols = [
        "flood_event_count", "flood_event_count_5y", "flood_deaths",
        "flood_injuries", "flood_property_damage_k", "flood_crop_damage_k",
        "flood_events_per_year",
    ]
    for col in flood_cols:
        merged[col] = merged[col].fillna(0)

    merged["flood_event_count"] = merged["flood_event_count"].astype(int)
    merged["flood_event_count_5y"] = merged["flood_event_count_5y"].astype(int)
    merged["flood_deaths"] = merged["flood_deaths"].astype(int)
    merged["flood_injuries"] = merged["flood_injuries"].astype(int)

    result = merged[["zcta_id"] + flood_cols].copy()
    log.info("  %d ZCTAs enriched", len(result))
    log.info("  ZCTAs with any flood event: %d (%.1f%%)",
             (result["flood_event_count"] > 0).sum(),
             100 * (result["flood_event_count"] > 0).mean())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build NOAA storm events flood features per ZCTA"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Build locally, skip S3 upload")
    parser.add_argument("--upload", action="store_true",
                        help="Upload result to S3")
    parser.add_argument("--output", default="/tmp/noaa_storm_events_zcta.parquet")
    parser.add_argument("--crosswalk", default=None,
                        help="Path to zcta_county_crosswalk.parquet (local)")
    parser.add_argument("--years", nargs=2, type=int, metavar=("START", "END"),
                        help="Year range to download (default: 1996 2024)")
    args = parser.parse_args()

    year_start = args.years[0] if args.years else FIRST_YEAR
    year_end = args.years[1] if args.years else 2024
    timestamp = datetime.now(timezone.utc).isoformat()

    # Resolve crosswalk
    if args.crosswalk:
        crosswalk_path = Path(args.crosswalk)
    else:
        # Try local v24 directory first
        here = Path(__file__).parent
        crosswalk_path = here / "zcta_county_crosswalk.parquet"
        if not crosswalk_path.exists():
            log.error(
                "crosswalk not found at %s. "
                "Run build_zcta_county_crosswalk.py first, or pass --crosswalk.",
                crosswalk_path,
            )
            sys.exit(1)

    # Fetch file list
    all_files = list_detail_files()
    files = [(yr, url) for yr, url in all_files if year_start <= yr <= year_end]
    log.info("Processing %d years (%d-%d)", len(files), year_start, year_end)

    # Download and parse each year
    all_events = []
    for year, url in files:
        year_df = fetch_year(year, url)
        if not year_df.empty:
            all_events.append(year_df)

    if not all_events:
        log.error("No flood events found. Check year range and NOAA endpoint.")
        sys.exit(1)

    events = pd.concat(all_events, ignore_index=True)
    log.info("Total flood events: %d across %d years", len(events), len(all_events))

    # Aggregate to county then ZCTA
    county_df = aggregate_to_county(events)
    result = county_to_zcta(county_df, crosswalk_path)

    # Summary statistics
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:                     %d", len(result))
    log.info("ZCTAs with flood events:   %d (%.1f%%)",
             (result["flood_event_count"] > 0).sum(),
             100 * (result["flood_event_count"] > 0).mean())
    log.info("Total events catalogued:   %d", events.shape[0])
    log.info("Total deaths:              %d", int(result["flood_deaths"].sum()))
    log.info("Total property damage:     $%.1fB",
             result["flood_property_damage_k"].sum() / 1_000_000)
    log.info("Mean events/year (active): %.2f",
             result[result["flood_event_count"] > 0]["flood_events_per_year"].mean())

    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
from swarm_auth import get_aws_credentials
        key = f"{PREFIX}/noaa_storm_events_zcta.parquet"
        _aws = get_aws_credentials()
        s3 = boto3.client("s3", region_name=REGION, **_aws)
        s3.upload_file(args.output, BUCKET, key)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)

        provenance = {
            "operation": "build_noaa_storm_events",
            "timestamp": timestamp,
            "source": NOAA_BASE,
            "year_range": [year_start, year_end],
            "flood_event_types": sorted(FLOOD_EVENT_TYPES),
            "n_zctas": len(result),
            "n_zctas_with_events": int((result["flood_event_count"] > 0).sum()),
            "total_events": int(events.shape[0]),
        }
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{PREFIX}/noaa_storm_events_zcta_provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
