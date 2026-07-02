#!/usr/bin/env python3
"""
fetch_svi.py -- Fetch CDC/ATSDR Social Vulnerability Index (SVI) 2022.

Downloads SVI 2022 at census tract level, then aggregates to ZCTA using
the HUD USPS crosswalk (population-weighted average).

SVI columns extracted (percentile ranks, 0-1):
  - RPL_THEME1: Socioeconomic Status
  - RPL_THEME2: Household Characteristics & Disability
  - RPL_THEME3: Racial & Ethnic Minority Status
  - RPL_THEME4: Housing Type & Transportation
  - RPL_THEMES: Overall SVI

Output: svi_zcta.parquet
  - zcta_id (str, 5-digit zero-padded)
  - svi_socioeconomic (float64, 0-1)
  - svi_household_disability (float64, 0-1)
  - svi_minority_language (float64, 0-1)
  - svi_housing_transport (float64, 0-1)
  - svi_overall (float64, 0-1)

Usage:
    python fetch_svi.py --output /tmp/svi_zcta.parquet
    python fetch_svi.py --upload   # also upload to S3
"""

import argparse
import json
import logging
import sys
import zipfile
from datetime import datetime, timezone
from io import BytesIO
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

# SVI 2022 national CSV from CDC/ATSDR
# The download is a CSV with all US census tracts
SVI_URL = "https://svi.cdc.gov/Documents/Data/2022/csv/states/SVI_2022_US.csv"

# HUD USPS crosswalk: tract -> ZCTA (with residential ratios for weighting)
# API docs: https://www.huduser.gov/portal/dataset/uspszip-api.html
HUD_CROSSWALK_URL = "https://www.huduser.gov/hudapi/public/usps"

# SVI columns to extract
SVI_COLS = {
    "RPL_THEME1": "svi_socioeconomic",
    "RPL_THEME2": "svi_household_disability",
    "RPL_THEME3": "svi_minority_language",
    "RPL_THEME4": "svi_housing_transport",
    "RPL_THEMES": "svi_overall",
}


def fetch_svi_tracts() -> pd.DataFrame:
    """Download SVI 2022 tract-level data."""
    log.info("Downloading SVI 2022 from CDC/ATSDR...")
    log.info("  URL: %s", SVI_URL)
    resp = requests.get(SVI_URL, timeout=300)
    resp.raise_for_status()
    log.info("  Downloaded %.1f MB", len(resp.content) / 1e6)

    from io import StringIO
    svi = pd.read_csv(StringIO(resp.text), dtype={"FIPS": str})
    log.info("  %d tracts loaded", len(svi))

    # Keep only what we need
    keep = ["FIPS", "E_TOTPOP"] + list(SVI_COLS.keys())
    missing = [c for c in keep if c not in svi.columns]
    if missing:
        log.error("Missing SVI columns: %s", missing)
        log.info("Available columns: %s", sorted(svi.columns.tolist()))
        sys.exit(1)

    svi = svi[keep].copy()
    svi["FIPS"] = svi["FIPS"].astype(str).str.zfill(11)  # 11-digit tract FIPS

    # SVI uses -999 for missing values
    for col in SVI_COLS:
        svi[col] = pd.to_numeric(svi[col], errors="coerce")
        svi.loc[svi[col] < 0, col] = float("nan")

    log.info("  Valid tracts (non-null overall SVI): %d",
             svi["RPL_THEMES"].notna().sum())

    return svi


def fetch_hud_crosswalk() -> pd.DataFrame:
    """Fetch HUD USPS tract-to-ZCTA crosswalk.

    Falls back to a simple geographic crosswalk if the HUD API
    is unavailable (requires API token).
    """
    # Try loading a pre-built crosswalk from S3 first
    crosswalk_url = (
        "https://www.huduser.gov/portal/datasets/usps/TRACT_ZIP_032023.xlsx"
    )
    log.info("Downloading HUD tract-ZCTA crosswalk...")

    # Use the Census tract-to-ZCTA relationship file instead (no API key needed)
    census_url = (
        "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
        "zcta520/tab20_zcta520_tract20_natl.txt"
    )
    log.info("  Using Census 2020 tract-ZCTA relationship file")
    log.info("  URL: %s", census_url)

    resp = requests.get(census_url, timeout=120)
    resp.raise_for_status()
    log.info("  Downloaded %.1f MB", len(resp.content) / 1e6)

    from io import StringIO
    xwalk = pd.read_csv(
        StringIO(resp.text),
        sep="|",
        dtype={"GEOID_ZCTA5_20": str, "GEOID_TRACT_20": str},
    )

    # Key columns:
    # GEOID_ZCTA5_20: 5-digit ZCTA
    # GEOID_TRACT_20: 11-digit tract FIPS
    # AREALAND_PART: land area of the intersection
    # AREALAND_TRACT_20: total land area of the tract
    xwalk = xwalk.rename(columns={
        "GEOID_ZCTA5_20": "zcta_id",
        "GEOID_TRACT_20": "tract_fips",
    })

    # Compute area-based weight: fraction of tract land area in this ZCTA
    if "AREALAND_PART" in xwalk.columns and "AREALAND_TRACT_20" in xwalk.columns:
        xwalk["weight"] = (
            xwalk["AREALAND_PART"] / xwalk["AREALAND_TRACT_20"].replace(0, 1)
        )
    else:
        # Fallback: equal weight
        xwalk["weight"] = 1.0

    xwalk = xwalk[["zcta_id", "tract_fips", "weight"]].copy()
    xwalk["zcta_id"] = xwalk["zcta_id"].str.zfill(5)
    xwalk["tract_fips"] = xwalk["tract_fips"].str.zfill(11)

    log.info("  %d tract-ZCTA relationships loaded", len(xwalk))
    log.info("  %d unique ZCTAs, %d unique tracts",
             xwalk["zcta_id"].nunique(), xwalk["tract_fips"].nunique())

    return xwalk


def aggregate_to_zcta(
    svi: pd.DataFrame, xwalk: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate tract-level SVI to ZCTA using population-weighted average."""
    # Join SVI with crosswalk
    merged = xwalk.merge(svi, left_on="tract_fips", right_on="FIPS", how="inner")
    log.info("Crosswalk-SVI join: %d rows (%d tracts matched)",
             len(merged), merged["tract_fips"].nunique())

    # Population-weighted average per ZCTA
    # Weight = tract population * area fraction in this ZCTA
    merged["pop_weight"] = merged["E_TOTPOP"].fillna(0) * merged["weight"]

    result_rows = []
    svi_cols = list(SVI_COLS.keys())

    for zcta_id, group in merged.groupby("zcta_id"):
        row = {"zcta_id": zcta_id}
        total_weight = group["pop_weight"].sum()

        for svi_col, target_name in SVI_COLS.items():
            valid = group[group[svi_col].notna()]
            if len(valid) > 0 and total_weight > 0:
                w = valid["pop_weight"]
                row[target_name] = (valid[svi_col] * w).sum() / w.sum()
            else:
                row[target_name] = float("nan")

        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    result["zcta_id"] = result["zcta_id"].astype(str).str.zfill(5)

    log.info("Aggregated to %d ZCTAs", len(result))
    log.info("  Non-null overall SVI: %d", result["svi_overall"].notna().sum())

    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch CDC SVI 2022")
    parser.add_argument("--output", default="/tmp/svi_zcta.parquet")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to S3 after building")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    # Step 1: Fetch SVI tract data
    svi = fetch_svi_tracts()

    # Step 2: Fetch crosswalk
    xwalk = fetch_hud_crosswalk()

    # Step 3: Aggregate to ZCTA
    result = aggregate_to_zcta(svi, xwalk)

    # Validate
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:       %d", len(result))
    log.info("SVI columns: %d", len(SVI_COLS))
    for target_name in SVI_COLS.values():
        valid = result[target_name].notna().sum()
        if valid > 0:
            log.info("  %s: mean=%.3f, min=%.3f, max=%.3f (%d valid)",
                     target_name,
                     result[target_name].mean(),
                     result[target_name].min(),
                     result[target_name].max(),
                     valid)

    # Save
    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
        BUCKET = "swarm-yrsn-datasets"
        KEY = "rsct_curriculum/series_018/processed/svi_zcta.parquet"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.upload_file(args.output, BUCKET, KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, KEY)

        provenance = {
            "operation": "fetch_svi",
            "timestamp": timestamp,
            "svi_source": SVI_URL,
            "crosswalk_source": "Census 2020 tract-ZCTA relationship file",
            "aggregation_method": "population-weighted area average",
            "n_zctas": len(result),
            "svi_columns": list(SVI_COLS.values()),
            "coverage_non_null": int(result["svi_overall"].notna().sum()),
        }
        s3.put_object(
            Bucket=BUCKET,
            Key="rsct_curriculum/series_018/processed/svi_zcta_provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
