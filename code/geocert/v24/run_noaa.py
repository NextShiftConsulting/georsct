#!/usr/bin/env python3
"""
run_noaa.py -- SageMaker run script for NOAA Storm Events flood enrichment.

Designed for ml.m5.large (2 vCPU, 8 GB RAM). Expected runtime: 15-25 min.

NOAA dataset: ~29 annual gzip files (1996-2024), ~200-400 MB compressed total.
Strategy:
  1. Fetch NOAA NCEI directory listing to find current annual file URLs
  2. Download + parse each year's detail CSV (filter to flood event types)
  3. Aggregate to county FIPS level (sum events, deaths, injuries, damage)
  4. Join to ZCTAs via county crosswalk (majority-county assignment)
  5. Upload parquet + provenance directly to S3

S3 output:
  s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/noaa_storm_events_zcta.parquet
"""

import gzip
import io
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
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
for _h in logging.root.handlers:
    _h.flush = lambda _orig=_h.flush: (_orig(), sys.stdout.flush())
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
S3_BUCKET = "swarm-yrsn-datasets"
S3_OUTPUT_KEY = "rsct_curriculum/series_018/processed/noaa_storm_events_zcta.parquet"
S3_PROVENANCE_KEY = "rsct_curriculum/series_018/processed/noaa_storm_events_zcta_provenance.json"
S3_LONG_KEY = "rsct_curriculum/series_018/processed/noaa_storm_events_long.parquet"
S3_WIDE_KEY = "rsct_curriculum/series_018/processed/noaa_storm_events_wide.parquet"

NOAA_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

FLOOD_EVENT_TYPES = {"Flash Flood", "Flood", "Coastal Flood", "Lakeshore Flood"}

FIRST_YEAR = 1996
RECENT_CUTOFF = 2019
N_YEARS = 2024 - FIRST_YEAR + 1

DAMAGE_MULTIPLIERS = {"K": 1.0, "M": 1000.0, "B": 1_000_000.0}

# Temporal epochs keyed to major flood events
EPOCHS = {
    "e1": (1996, 2004),   # pre-Katrina baseline
    "e2": (2005, 2011),   # post-Katrina / pre-Sandy
    "e3": (2012, 2024),   # Sandy onward
}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def _s3():
    return boto3.client("s3")


def _s3_upload(local_path: str, key: str):
    try:
        _s3().upload_file(local_path, S3_BUCKET, key)
        log.info("  -> s3://%s/%s", S3_BUCKET, key)
    except Exception as e:
        log.warning("  S3 upload failed for %s: %s", key, e)


# ---------------------------------------------------------------------------
# NOAA download helpers
# ---------------------------------------------------------------------------
def list_detail_files(year_start: int, year_end: int) -> list[tuple[int, str]]:
    """Fetch NOAA directory listing and return (year, url) pairs for detail CSVs."""
    log.info("Fetching NOAA directory listing from %s", NOAA_BASE)
    resp = requests.get(NOAA_BASE, timeout=60)
    resp.raise_for_status()

    pattern = re.compile(
        r'StormEvents_details-ftp_v1\.0_d(\d{4})_c\d{8}\.csv\.gz'
    )
    seen_years = {}
    for match in pattern.finditer(resp.text):
        fname = match.group(0)
        year = int(match.group(1))
        if year_start <= year <= year_end:
            seen_years[year] = fname  # latest creation-date file wins

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
        return float(v) / 1000.0
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

    df["EVENT_TYPE"] = df.get("EVENT_TYPE", pd.Series(dtype=str)).str.strip().str.title()
    df = df[df["EVENT_TYPE"].isin(FLOOD_EVENT_TYPES)].copy()

    if df.empty:
        log.info("  %d: 0 flood events", year)
        return pd.DataFrame()

    if "CZ_TYPE" in df.columns:
        df = df[df["CZ_TYPE"].str.strip().str.upper() == "C"].copy()

    df["state_fips"] = df.get("STATE_FIPS", pd.Series(dtype=str)).astype(str).str.zfill(2)
    df["cz_fips"] = df.get("CZ_FIPS", pd.Series(dtype=str)).astype(str).str.zfill(3)
    df["county_fips"] = df["state_fips"] + df["cz_fips"]

    df["prop_dmg_k"] = df.get("DAMAGE_PROPERTY", pd.Series(dtype=str)).apply(parse_damage)
    df["crop_dmg_k"] = df.get("DAMAGE_CROPS", pd.Series(dtype=str)).apply(parse_damage)

    for col in ("DEATHS_DIRECT", "DEATHS_INDIRECT", "INJURIES_DIRECT", "INJURIES_INDIRECT"):
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    df["deaths"] = df["DEATHS_DIRECT"] + df["DEATHS_INDIRECT"]
    df["injuries"] = df["INJURIES_DIRECT"] + df["INJURIES_INDIRECT"]
    df["year"] = year

    log.info("  %d: %d flood events, %d counties",
             year, len(df), df["county_fips"].nunique())

    return df[["county_fips", "year", "prop_dmg_k", "crop_dmg_k",
               "deaths", "injuries"]].copy()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_to_county(events: pd.DataFrame) -> pd.DataFrame:
    log.info("Aggregating to county level (%d total events)...", len(events))

    by_county = events.groupby("county_fips").agg(
        flood_event_count=("year", "count"),
        flood_deaths=("deaths", "sum"),
        flood_injuries=("injuries", "sum"),
        flood_property_damage_k=("prop_dmg_k", "sum"),
        flood_crop_damage_k=("crop_dmg_k", "sum"),
    ).reset_index()

    recent = (
        events[events["year"] >= RECENT_CUTOFF]
        .groupby("county_fips")
        .size()
        .reset_index(name="flood_event_count_5y")
    )
    by_county = by_county.merge(recent, on="county_fips", how="left")
    by_county["flood_event_count_5y"] = by_county["flood_event_count_5y"].fillna(0).astype(int)
    by_county["flood_events_per_year"] = (by_county["flood_event_count"] / N_YEARS).round(3)

    log.info("  %d counties with flood history", len(by_county))
    return by_county


def county_to_zcta(county_df: pd.DataFrame, crosswalk_path: Path) -> pd.DataFrame:
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
    log.info("  %d ZCTAs enriched; %d (%.1f%%) have flood events",
             len(result),
             (result["flood_event_count"] > 0).sum(),
             100 * (result["flood_event_count"] > 0).mean())
    return result


# ---------------------------------------------------------------------------
# Temporal pipeline
# ---------------------------------------------------------------------------
def aggregate_to_county_by_year(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate flood events to county × year grain."""
    log.info("Aggregating to county x year (%d events)...", len(events))
    by_county_year = events.groupby(["county_fips", "year"]).agg(
        flood_events=("prop_dmg_k", "count"),
        deaths=("deaths", "sum"),
        injuries=("injuries", "sum"),
        property_damage_k=("prop_dmg_k", "sum"),
        crop_damage_k=("crop_dmg_k", "sum"),
    ).reset_index()
    log.info("  %d county x year rows (%d counties, %d years)",
             len(by_county_year),
             by_county_year["county_fips"].nunique(),
             by_county_year["year"].nunique())
    return by_county_year


def make_long_zcta(
    county_year: pd.DataFrame, crosswalk_path: Path
) -> pd.DataFrame:
    """Join county x year to ZCTA, zero-fill all zcta x year combinations.

    Returns long format: one row per (zcta_id, year) for all years in FIRST_YEAR..2024.
    """
    import numpy as np

    log.info("Building long format (zcta x year)...")
    xwalk = pd.read_parquet(crosswalk_path)[["zcta_id", "county_fips"]].copy()
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    xwalk["county_fips"] = xwalk["county_fips"].astype(str).str.zfill(5)

    merged = xwalk.merge(county_year, on="county_fips", how="left")

    # Build complete zcta x year index so every combination exists
    all_zctas = xwalk["zcta_id"].unique()
    all_years = np.arange(FIRST_YEAR, 2025)
    idx = pd.MultiIndex.from_product([all_zctas, all_years], names=["zcta_id", "year"])
    template = pd.DataFrame(index=idx).reset_index()

    long = template.merge(
        merged[["zcta_id", "year", "flood_events", "deaths",
                "injuries", "property_damage_k", "crop_damage_k"]],
        on=["zcta_id", "year"],
        how="left",
    )
    for col in ("flood_events", "deaths", "injuries"):
        long[col] = long[col].fillna(0).astype(int)
    for col in ("property_damage_k", "crop_damage_k"):
        long[col] = long[col].fillna(0.0)

    long = long.sort_values(["zcta_id", "year"]).reset_index(drop=True)
    log.info("  Long format: %d rows (%d ZCTAs x %d years)",
             len(long), len(all_zctas), len(all_years))
    return long


def make_wide_epochs(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long format to pre-computed epoch aggregates.

    Produces one row per zcta_id with e1/e2/e3 columns for each metric,
    plus total and peak-year features.
    """
    log.info("Building wide epoch format...")
    metrics = ["flood_events", "deaths", "property_damage_k", "crop_damage_k"]
    result = long[["zcta_id"]].drop_duplicates().copy()

    for epoch, (y0, y1) in EPOCHS.items():
        mask = long["year"].between(y0, y1)
        epoch_agg = (
            long[mask]
            .groupby("zcta_id")[metrics]
            .sum()
            .reset_index()
            .rename(columns={m: f"{m}_{epoch}" for m in metrics})
        )
        result = result.merge(epoch_agg, on="zcta_id", how="left")

    # Rolling totals and peak-year features
    totals = long.groupby("zcta_id")[metrics].sum().reset_index()
    totals = totals.rename(columns={m: f"{m}_total" for m in metrics})
    result = result.merge(totals, on="zcta_id", how="left")

    peak = (
        long.groupby("zcta_id")["property_damage_k"]
        .max()
        .reset_index(name="property_damage_k_peak_yr")
    )
    result = result.merge(peak, on="zcta_id", how="left")

    # Zero-fill
    for col in result.columns:
        if col != "zcta_id":
            result[col] = result[col].fillna(0)

    # Coerce integer columns
    int_cols = [c for c in result.columns
                if c.startswith("flood_events") or c.startswith("deaths")]
    for col in int_cols:
        result[col] = result[col].astype(int)

    log.info("  Wide format: %d rows, %d columns", len(result), len(result.columns))
    log.info("  Epoch columns: %s",
             [c for c in result.columns if any(f"_{e}" in c for e in EPOCHS)])
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build NOAA storm events per ZCTA (SageMaker)")
    parser.add_argument("--data-dir", default="/opt/ml/processing/input/data")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--year-start", type=int, default=FIRST_YEAR)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--temporal", action="store_true",
                        help="Also produce long (zcta x year) and wide (epoch) Parquet outputs")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve crosswalk
    crosswalk_path = Path(args.data_dir) / "zcta_county_crosswalk.parquet"
    if not crosswalk_path.exists():
        log.error("zcta_county_crosswalk.parquet not found at %s", crosswalk_path)
        sys.exit(1)
    log.info("Crosswalk: %s", crosswalk_path)

    # Fetch file list
    log.info("=== FETCHING NOAA DIRECTORY ===")
    all_files = list_detail_files(args.year_start, args.year_end)
    log.info("Processing %d years (%d-%d)", len(all_files), args.year_start, args.year_end)

    # Download and parse each year
    log.info("=== DOWNLOADING ANNUAL FILES ===")
    all_events = []
    for year, url in all_files:
        year_df = fetch_year(year, url)
        if not year_df.empty:
            all_events.append(year_df)

    if not all_events:
        log.error("No flood events found.")
        sys.exit(1)

    events = pd.concat(all_events, ignore_index=True)
    log.info("Total flood events: %d across %d years", len(events), len(all_events))

    # Aggregate to county then ZCTA
    log.info("=== AGGREGATING ===")
    county_df = aggregate_to_county(events)
    result = county_to_zcta(county_df, crosswalk_path)

    # Validation
    log.info("=== VALIDATION ===")
    n_with = (result["flood_event_count"] > 0).sum()
    log.info("ZCTAs total:           %d", len(result))
    log.info("ZCTAs with events:     %d (%.1f%%)", n_with, 100 * n_with / len(result))
    log.info("Total events:          %d", int(result["flood_event_count"].sum()))
    log.info("Total deaths:          %d", int(result["flood_deaths"].sum()))
    log.info("Total property dmg:    $%.1fB",
             result["flood_property_damage_k"].sum() / 1_000_000)

    # Sanity: Harris County TX (48201) should have many flood events
    harris = result[result["zcta_id"].str.startswith("770")]
    harris_events = harris["flood_event_count"].sum()
    log.info("Houston/Harris area spot check (770xx): %d total flood events", harris_events)
    if harris_events == 0:
        log.warning("VALIDATION WARN: Zero flood events in Houston area — check county FIPS join")
    else:
        log.info("  PASS: Houston area shows flood events as expected")

    if n_with < 1000:
        log.warning("VALIDATION WARN: Only %d ZCTAs with events — expected >5K", n_with)

    # Save and upload — standard aggregated output
    out_path = output_dir / "noaa_storm_events_zcta.parquet"
    result.to_parquet(out_path, index=False)
    log.info("Saved: %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)
    _s3_upload(str(out_path), S3_OUTPUT_KEY)

    provenance = {
        "operation": "build_noaa_storm_events",
        "timestamp": timestamp,
        "source": NOAA_BASE,
        "year_range": [args.year_start, args.year_end],
        "flood_event_types": sorted(FLOOD_EVENT_TYPES),
        "n_zctas": len(result),
        "n_zctas_with_events": int(n_with),
        "total_events": int(events.shape[0]),
        "total_deaths": int(result["flood_deaths"].sum()),
        "total_property_damage_k": float(result["flood_property_damage_k"].sum()),
        "temporal": args.temporal,
    }
    prov_path = output_dir / "noaa_storm_events_zcta_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    _s3_upload(str(prov_path), S3_PROVENANCE_KEY)

    # Temporal outputs — long and wide epoch
    if args.temporal:
        log.info("=== TEMPORAL PIPELINE ===")

        county_year = aggregate_to_county_by_year(events)

        log.info("Building long format...")
        long = make_long_zcta(county_year, crosswalk_path)
        long_path = output_dir / "noaa_storm_events_long.parquet"
        long.to_parquet(long_path, index=False)
        log.info("Saved: %s (%.1f KB)", long_path, long_path.stat().st_size / 1024)
        _s3_upload(str(long_path), S3_LONG_KEY)

        log.info("Building wide epoch format...")
        wide = make_wide_epochs(long)
        wide_path = output_dir / "noaa_storm_events_wide.parquet"
        wide.to_parquet(wide_path, index=False)
        log.info("Saved: %s (%.1f KB)", wide_path, wide_path.stat().st_size / 1024)
        _s3_upload(str(wide_path), S3_WIDE_KEY)

        log.info("=== TEMPORAL SUMMARY ===")
        log.info("Long rows:         %d (%d ZCTAs x %d years)",
                 len(long), long["zcta_id"].nunique(), long["year"].nunique())
        log.info("Wide rows:         %d", len(wide))
        for epoch, (y0, y1) in EPOCHS.items():
            col = f"flood_events_{epoch}"
            active = (wide[col] > 0).sum()
            log.info("  %s (%d-%d): %d ZCTAs with events", epoch, y0, y1, active)
        log.info("Peak damage ZCTA:  %s ($%.1fM damage in worst year)",
                 wide.loc[wide["property_damage_k_peak_yr"].idxmax(), "zcta_id"],
                 wide["property_damage_k_peak_yr"].max() / 1000)

    log.info("Done.")


if __name__ == "__main__":
    main()
