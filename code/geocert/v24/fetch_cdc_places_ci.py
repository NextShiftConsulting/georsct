#!/usr/bin/env python3
"""
fetch_cdc_places_ci.py -- Fetch CDC PLACES 2023 confidence intervals at ZCTA level.

Downloads crude prevalence + 95% CI for all 21 health measures from:
  https://data.cdc.gov/resource/c76y-7pzg.json

The ZCTA-level endpoint stores CIs as "(low, high)" text in *_crude95ci columns.
This script parses them into separate _ci_low and _ci_high float columns.

Output: cdc_places_ci.parquet
  - zcta_id (str, 5-digit zero-padded)
  - target_*_ci_low (float64) for each of 21 health measures
  - target_*_ci_high (float64) for each of 21 health measures
  - has_cdc_ci (bool) — True if all 21 CI pairs are present

Usage:
    python fetch_cdc_places_ci.py --output /tmp/cdc_places_ci.parquet
    python fetch_cdc_places_ci.py --upload   # also upload to S3
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# CDC PLACES 2023 ZCTA-level endpoint (Socrata)
API_BASE = "https://data.cdc.gov/resource/c76y-7pzg.json"
PAGE_SIZE = 50000  # Socrata max per request

# Map CDC PLACES column prefixes to our target names
# CDC uses short names like "diabetes_crudeprev" and "diabetes_crude95ci"
CDC_TO_TARGET = {
    "arthritis":    "target_arthritis",
    "binge":        "target_binge_drinking",
    "bphigh":       "target_high_blood_pressure",
    "bpmed":        "target_bp_medicated",
    "cancer":       "target_cancer",
    "casthma":      "target_asthma",
    "chd":          "target_coronary_heart_disease",
    "checkup":      "target_annual_checkup",
    "cholscreen":   "target_cholesterol_screening",
    "copd":         "target_copd",
    "csmoking":     "target_smoking",
    "dental":       "target_dental_visit",
    "diabetes":     "target_diabetes",
    "highchol":     "target_high_cholesterol",
    "kidney":       "target_chronic_kidney_disease",
    "lpa":          "target_physical_inactivity",
    "mhlth":        "target_mental_health_not_good",
    "obesity":      "target_obesity",
    "phlth":        "target_physical_health_not_good",
    "sleep":        "target_sleep_less_7hr",
    "stroke":       "target_stroke",
}

# Columns to fetch from the API
FETCH_COLS = ["zcta5"]
for cdc_name in CDC_TO_TARGET:
    FETCH_COLS.append(f"{cdc_name}_crudeprev")
    FETCH_COLS.append(f"{cdc_name}_crude95ci")


def parse_ci(ci_str: str) -> tuple:
    """Parse '(10.3, 13.2)' -> (10.3, 13.2). Returns (NaN, NaN) on failure."""
    if not ci_str or pd.isna(ci_str):
        return (float("nan"), float("nan"))
    match = re.match(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", str(ci_str))
    if match:
        return (float(match.group(1)), float(match.group(2)))
    return (float("nan"), float("nan"))


def fetch_all_zctas() -> pd.DataFrame:
    """Fetch all ZCTA records from CDC PLACES API with pagination."""
    select_clause = ",".join(FETCH_COLS)
    all_rows = []
    offset = 0

    while True:
        url = (
            f"{API_BASE}"
            f"?$select={select_clause}"
            f"&$limit={PAGE_SIZE}"
            f"&$offset={offset}"
            f"&$order=zcta5"
        )
        log.info("Fetching offset=%d ...", offset)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_rows.extend(data)
        log.info("  Got %d records (total: %d)", len(data), len(all_rows))

        if len(data) < PAGE_SIZE:
            break

        offset += PAGE_SIZE
        time.sleep(0.5)  # rate limit

    log.info("Total records fetched: %d", len(all_rows))
    return pd.DataFrame(all_rows)


def build_ci_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse raw CDC data into clean CI columns."""
    result = pd.DataFrame()
    result["zcta_id"] = raw["zcta5"].astype(str).str.zfill(5)

    ci_count = 0
    for cdc_name, target_name in CDC_TO_TARGET.items():
        ci_col = f"{cdc_name}_crude95ci"

        if ci_col not in raw.columns:
            log.warning("Missing CI column: %s", ci_col)
            result[f"{target_name}_ci_low"] = float("nan")
            result[f"{target_name}_ci_high"] = float("nan")
            continue

        parsed = raw[ci_col].apply(parse_ci)
        result[f"{target_name}_ci_low"] = parsed.apply(lambda x: x[0])
        result[f"{target_name}_ci_high"] = parsed.apply(lambda x: x[1])
        ci_count += 1

    log.info("Parsed CIs for %d/%d measures", ci_count, len(CDC_TO_TARGET))

    # Coverage flag: True if ALL 21 CI pairs are present
    ci_low_cols = [f"{t}_ci_low" for t in CDC_TO_TARGET.values()]
    result["has_cdc_ci"] = result[ci_low_cols].notna().all(axis=1)

    log.info("  has_cdc_ci=True:  %d", result["has_cdc_ci"].sum())
    log.info("  has_cdc_ci=False: %d", (~result["has_cdc_ci"]).sum())

    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch CDC PLACES CIs")
    parser.add_argument("--output", default="/tmp/cdc_places_ci.parquet")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to S3 after building")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    # Fetch
    raw = fetch_all_zctas()

    # Parse
    ci_df = build_ci_dataframe(raw)

    # Validate
    assert ci_df["zcta_id"].is_unique, "Duplicate ZCTA IDs"
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:     %d", len(ci_df))
    log.info("CI cols:   %d (21 low + 21 high)", 42)
    log.info("Coverage:  %d / %d have full CIs",
             ci_df["has_cdc_ci"].sum(), len(ci_df))

    # Sample CI widths
    log.info("")
    log.info("=== SAMPLE CI WIDTHS (diabetes) ===")
    if "target_diabetes_ci_low" in ci_df.columns:
        width = ci_df["target_diabetes_ci_high"] - ci_df["target_diabetes_ci_low"]
        log.info("  Mean CI width: %.2f pp", width.mean())
        log.info("  Median:        %.2f pp", width.median())
        log.info("  Max:           %.2f pp", width.max())
        log.info("  Min:           %.2f pp", width.min())

    # Save
    ci_df.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             __import__("pathlib").Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
        BUCKET = "swarm-yrsn-datasets"
        KEY = "rsct_curriculum/series_018/processed/cdc_places_ci.parquet"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.upload_file(args.output, BUCKET, KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, KEY)

        provenance = {
            "operation": "fetch_cdc_places_ci",
            "timestamp": timestamp,
            "source": API_BASE,
            "n_zctas": len(ci_df),
            "n_ci_columns": 42,
            "measures": list(CDC_TO_TARGET.keys()),
            "coverage_has_ci": int(ci_df["has_cdc_ci"].sum()),
            "coverage_missing_ci": int((~ci_df["has_cdc_ci"]).sum()),
        }
        s3.put_object(
            Bucket=BUCKET,
            Key="rsct_curriculum/series_018/processed/cdc_places_ci_provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
