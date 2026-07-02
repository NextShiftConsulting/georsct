#!/usr/bin/env python3
"""
build_zcta_county_crosswalk.py -- Build ZCTA-to-county majority assignment.

Uses Census 2020 ZCTA-to-county relationship file to assign each ZCTA
to its majority county by land area overlap.

One ZCTA can touch multiple counties; this picks the county with the
largest area intersection (standard approach for stratified sampling).

Source:
  https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt

Output: zcta_county_crosswalk.parquet
  - zcta_id (str, 5-digit zero-padded)
  - county_fips (str, 5-digit, e.g. "48201" for Harris County, TX)
  - county_name (str, e.g. "Harris County")
  - state_fips (str, 2-digit)
  - state (str, 2-letter abbrev, e.g. "TX")

Usage:
    python build_zcta_county_crosswalk.py --output /tmp/zcta_county_crosswalk.parquet
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from io import StringIO
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

CENSUS_REL_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
    "zcta520/tab20_zcta520_county20_natl.txt"
)

# State FIPS to abbreviation (all 50 states + DC + territories)
STATE_FIPS_TO_ABBREV = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
    # Territories
    "60": "AS", "66": "GU", "69": "MP", "72": "PR", "78": "VI",
}

# Non-CONUS state FIPS codes (Alaska, Hawaii, territories)
# All downstream pipelines assume lower-48 + DC.
NON_CONUS_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}


def build_majority_crosswalk() -> pd.DataFrame:
    """Download Census relationship file and build majority-county assignment."""
    log.info("Downloading Census ZCTA-county relationship file...")
    log.info("  URL: %s", CENSUS_REL_URL)
    resp = requests.get(CENSUS_REL_URL, timeout=120)
    resp.raise_for_status()
    log.info("  Downloaded %.1f MB", len(resp.content) / 1e6)

    rel = pd.read_csv(
        StringIO(resp.text),
        sep="|",
        dtype=str,
        encoding="utf-8-sig",
    )
    log.info("  %d ZCTA-county relationship rows", len(rel))

    # Drop rows with missing ZCTA or county
    rel = rel.dropna(subset=["GEOID_ZCTA5_20", "GEOID_COUNTY_20"])
    rel["AREALAND_PART"] = pd.to_numeric(rel["AREALAND_PART"], errors="coerce")
    log.info("  %d valid relationships", len(rel))

    # Multi-county stats
    zcta_counts = rel.groupby("GEOID_ZCTA5_20").size()
    log.info("  %d unique ZCTAs", len(zcta_counts))
    log.info("  %d ZCTAs in 1 county", (zcta_counts == 1).sum())
    log.info("  %d ZCTAs in 2+ counties", (zcta_counts > 1).sum())

    # Majority county: largest AREALAND_PART
    majority = (
        rel.sort_values("AREALAND_PART", ascending=False)
        .drop_duplicates(subset="GEOID_ZCTA5_20")
    )

    # Clean up county name (remove " County", " Parish", etc. suffix -- actually keep it)
    result = pd.DataFrame({
        "zcta_id": majority["GEOID_ZCTA5_20"].str.zfill(5),
        "county_fips": majority["GEOID_COUNTY_20"].str.zfill(5),
        "county_name": majority["NAMELSAD_COUNTY_20"],
    })

    # Derive state from county FIPS
    result["state_fips"] = result["county_fips"].str[:2]
    result["state"] = result["state_fips"].map(STATE_FIPS_TO_ABBREV)

    unmapped = result["state"].isna().sum()
    if unmapped > 0:
        log.warning("  %d ZCTAs with unmapped state FIPS", unmapped)

    result = result.reset_index(drop=True)

    # Filter to CONUS (lower 48 + DC)
    n_before = len(result)
    non_conus = result["state_fips"].isin(NON_CONUS_STATE_FIPS)
    dropped_states = sorted(result.loc[non_conus, "state"].dropna().unique())
    dropped_count = non_conus.sum()
    result = result[~non_conus].reset_index(drop=True)
    log.info("  CONUS filter: dropped %d ZCTAs in %s", dropped_count, dropped_states)
    log.info("  %d ZCTAs retained (was %d)", len(result), n_before)

    log.info("  %d unique counties (FIPS)", result["county_fips"].nunique())
    log.info("  %d unique states", result["state"].nunique())

    return result


def main():
    parser = argparse.ArgumentParser(description="Build ZCTA-county crosswalk")
    parser.add_argument("--output", default="/tmp/zcta_county_crosswalk.parquet")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    result = build_majority_crosswalk()

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:    %d", len(result))
    log.info("Counties: %d", result["county_fips"].nunique())
    log.info("States:   %d", result["state"].nunique())
    log.info("")
    log.info("Top 5 counties by ZCTA count:")
    top = result.groupby(["county_fips", "county_name", "state"]).size().sort_values(ascending=False).head()
    for (fips, name, st), count in top.items():
        log.info("  %s %s, %s: %d ZCTAs", fips, name, st, count)

    # Save
    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
        BUCKET = "swarm-yrsn-datasets"
        KEY = "rsct_curriculum/series_018/processed/zcta_county_crosswalk.parquet"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.upload_file(args.output, BUCKET, KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, KEY)

        provenance = {
            "operation": "build_zcta_county_crosswalk",
            "timestamp": timestamp,
            "source": CENSUS_REL_URL,
            "method": "majority county by AREALAND_PART",
            "n_zctas": len(result),
            "n_counties": int(result["county_fips"].nunique()),
            "n_states": int(result["state"].nunique()),
        }
        s3.put_object(
            Bucket=BUCKET,
            Key="rsct_curriculum/series_018/processed/zcta_county_crosswalk_provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
