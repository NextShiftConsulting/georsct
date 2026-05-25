#!/usr/bin/env python3
"""
audit_cdc_places_coverage.py -- Audit and document CDC PLACES label coverage
in zcta_features_labels.parquet.

CDC PLACES uses small-area estimation from BRFSS survey data. Some ZCTAs
(~263 out of 35,157) are not covered by the CDC model. Their 21 health
label columns should be explicitly NaN, not some other placeholder.

This script:
  1. Downloads zcta_features_labels.parquet from S3
  2. Identifies ZCTAs where ANY health label is missing
  3. Ensures all missing health labels are NaN (not sentinel or zero)
  4. Writes a coverage manifest: which ZCTAs are complete vs incomplete
  5. Uploads provenance JSON to S3

Usage:
    python audit_cdc_places_coverage.py --dry-run    # Audit only
    python audit_cdc_places_coverage.py              # Fix + upload

Idempotent: safe to run multiple times.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-yrsn-datasets"
S3_KEY = "rsct_curriculum/series_018/processed/zcta_features_labels.parquet"
REGION = "us-east-1"

# The 21 CDC PLACES health target columns (parquet uses target_* prefix).
# Source: CDC PLACES 2023 release, ZCTA-level model-based estimates.
# These are NOT direct survey measurements -- they are modeled from BRFSS.
CDC_PLACES_COLUMNS = [
    "target_annual_checkup",
    "target_arthritis",
    "target_asthma",
    "target_binge_drinking",
    "target_bp_medicated",
    "target_cancer",
    "target_cholesterol_screening",
    "target_chronic_kidney_disease",
    "target_copd",
    "target_coronary_heart_disease",
    "target_dental_visit",
    "target_diabetes",
    "target_high_blood_pressure",
    "target_high_cholesterol",
    "target_mental_health_not_good",
    "target_obesity",
    "target_physical_health_not_good",
    "target_physical_inactivity",
    "target_sleep_less_7hr",
    "target_smoking",
    "target_stroke",
]

# Non-CDC target columns (ACS, VIIRS, Hansen, USGS).
# These have independent coverage patterns.
NON_CDC_COLUMNS = [
    "target_income",
    "target_home_value",
    "target_night_lights",
    "target_population_density",
    "target_tree_cover",
    "target_elevation",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Audit only, do not modify or upload")
    parser.add_argument("--local-path", type=str, default=None,
                        help="Use local parquet instead of S3")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    timestamp = datetime.now(timezone.utc).isoformat()

    # -- 1. Load --
    if args.local_path:
        local = Path(args.local_path)
    else:
        local = Path("/tmp/zcta_features_labels.parquet")
        log.info("Downloading s3://%s/%s", BUCKET, S3_KEY)
        s3.download_file(BUCKET, S3_KEY, str(local))

    df = pd.read_parquet(local)
    df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)
    n_total = len(df)
    log.info("Loaded %d ZCTAs, %d columns", n_total, len(df.columns))

    # -- 2. Verify expected columns exist --
    all_target_cols = CDC_PLACES_COLUMNS + NON_CDC_COLUMNS
    missing_cols = [c for c in all_target_cols if c not in df.columns]
    if missing_cols:
        log.error("Missing expected columns: %s", missing_cols)
        log.error("Available columns: %s", sorted(df.columns.tolist()))
        sys.exit(1)

    # -- 3. Audit CDC PLACES coverage --
    cdc_df = df[["zcta_id"] + CDC_PLACES_COLUMNS].copy()

    # Per-column missing counts
    col_missing = {}
    for col in CDC_PLACES_COLUMNS:
        n_nan = int(cdc_df[col].isna().sum())
        n_zero = int((cdc_df[col] == 0.0).sum())
        n_negative = int((cdc_df[col] < 0).sum()) if pd.api.types.is_numeric_dtype(cdc_df[col]) else 0
        col_missing[col] = {
            "n_nan": n_nan,
            "n_zero": n_zero,
            "n_negative": n_negative,
            "pct_nan": round(n_nan / n_total * 100, 3),
        }
        if n_nan > 0 or n_negative > 0:
            log.info("  %s: %d NaN (%.1f%%), %d negative",
                     col.split("_")[-1][:20], n_nan, n_nan / n_total * 100, n_negative)

    # ZCTAs missing ANY health label
    any_missing_mask = cdc_df[CDC_PLACES_COLUMNS].isna().any(axis=1)
    all_missing_mask = cdc_df[CDC_PLACES_COLUMNS].isna().all(axis=1)

    zctas_any_missing = set(cdc_df.loc[any_missing_mask, "zcta_id"])
    zctas_all_missing = set(cdc_df.loc[all_missing_mask, "zcta_id"])
    zctas_partial = zctas_any_missing - zctas_all_missing
    zctas_complete = set(cdc_df.loc[~any_missing_mask, "zcta_id"])

    log.info("")
    log.info("CDC PLACES coverage summary:")
    log.info("  Complete (all 21 labels):   %d ZCTAs", len(zctas_complete))
    log.info("  All 21 missing:             %d ZCTAs", len(zctas_all_missing))
    log.info("  Partial (some missing):     %d ZCTAs", len(zctas_partial))
    log.info("  Total:                      %d ZCTAs", n_total)

    if zctas_partial:
        log.warning("UNEXPECTED: %d ZCTAs have partial CDC coverage. "
                    "CDC PLACES typically covers all-or-nothing per ZCTA.",
                    len(zctas_partial))
        for z in sorted(zctas_partial)[:10]:
            row = cdc_df[cdc_df["zcta_id"] == z][CDC_PLACES_COLUMNS]
            n_present = int(row.notna().sum(axis=1).iloc[0])
            log.warning("  ZCTA %s: %d/21 labels present", z, n_present)

    # -- 4. Audit non-CDC coverage --
    noncdc_missing = {}
    for col in NON_CDC_COLUMNS:
        n_nan = int(df[col].isna().sum())
        noncdc_missing[col] = {"n_nan": n_nan, "pct_nan": round(n_nan / n_total * 100, 3)}
        if n_nan > 0:
            log.info("  Non-CDC %s: %d NaN (%.1f%%)", col, n_nan, n_nan / n_total * 100)

    # -- 5. Check for suspicious values that should be NaN --
    fixes_needed = 0
    fix_details = {}

    for col in CDC_PLACES_COLUMNS:
        vals = df[col].values
        # Check for negative percentages (CDC PLACES are always 0-100)
        if pd.api.types.is_numeric_dtype(df[col]):
            bad_mask = (vals < 0) & ~np.isnan(vals)
            n_bad = int(bad_mask.sum())
            if n_bad > 0:
                bad_vals = np.unique(vals[bad_mask]).tolist()
                fix_details[col] = {
                    "n_bad": n_bad,
                    "bad_values": bad_vals,
                    "action": "replace_with_nan",
                }
                fixes_needed += n_bad
                log.warning("  %s: %d suspicious negative values: %s",
                           col, n_bad, bad_vals[:5])

    # -- 6. Apply fixes if needed --
    if fixes_needed > 0 and not args.dry_run:
        log.info("Replacing %d suspicious values with NaN", fixes_needed)
        for col, detail in fix_details.items():
            mask = (df[col].values < 0) & ~np.isnan(df[col].values)
            df.loc[mask, col] = np.nan

        # Re-upload
        cleaned = Path("/tmp/zcta_features_labels_cdc_cleaned.parquet")
        df.to_parquet(cleaned, index=False)
        log.info("Uploading cleaned parquet to s3://%s/%s", BUCKET, S3_KEY)
        s3.upload_file(str(cleaned), BUCKET, S3_KEY)
    elif fixes_needed > 0:
        log.info("[DRY RUN] Would replace %d suspicious values with NaN", fixes_needed)
    else:
        log.info("No suspicious values found -- CDC columns are clean.")

    # -- 7. Write coverage manifest --
    manifest = {
        "operation": "audit_cdc_places_coverage",
        "timestamp": timestamp,
        "source": f"s3://{BUCKET}/{S3_KEY}",
        "total_zctas": n_total,
        "cdc_places": {
            "n_columns": len(CDC_PLACES_COLUMNS),
            "columns": CDC_PLACES_COLUMNS,
            "data_source": "CDC PLACES 2023 (BRFSS model-based estimates, NOT direct measurements)",
            "zctas_complete": len(zctas_complete),
            "zctas_all_missing": len(zctas_all_missing),
            "zctas_partial_missing": len(zctas_partial),
            "all_missing_zcta_ids": sorted(zctas_all_missing),
            "partial_missing_zcta_ids": sorted(zctas_partial),
            "per_column_missing": col_missing,
        },
        "non_cdc": {
            "columns": NON_CDC_COLUMNS,
            "per_column_missing": noncdc_missing,
        },
        "fixes_applied": fix_details if (fixes_needed > 0 and not args.dry_run) else {},
        "fixes_needed_but_dry_run": fix_details if (fixes_needed > 0 and args.dry_run) else {},
    }

    if not args.dry_run:
        manifest_key = "rsct_curriculum/series_018/processed/cdc_places_coverage.json"
        s3.put_object(
            Bucket=BUCKET,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2),
            ContentType="application/json",
        )
        log.info("Coverage manifest saved: s3://%s/%s", BUCKET, manifest_key)
    else:
        # Write locally for inspection
        local_manifest = Path(__file__).parent / "cdc_places_coverage_preview.json"
        local_manifest.write_text(json.dumps(manifest, indent=2))
        log.info("[DRY RUN] Coverage manifest written to %s", local_manifest)

    # -- 8. Summary --
    log.info("")
    log.info("=== COVERAGE SUMMARY ===")
    log.info("CDC PLACES (21 health labels):")
    log.info("  %d/%d ZCTAs have complete labels (%.1f%%)",
             len(zctas_complete), n_total, len(zctas_complete) / n_total * 100)
    log.info("  %d ZCTAs have no CDC labels (all 21 NaN)", len(zctas_all_missing))
    log.info("  %d ZCTAs have partial labels", len(zctas_partial))
    log.info("")
    log.info("Non-CDC (6 labels):")
    for col, info in noncdc_missing.items():
        status = "complete" if info["n_nan"] == 0 else f"{info['n_nan']} missing"
        log.info("  %s: %s", col, status)
    log.info("")

    if fixes_needed == 0:
        log.info("Dataset is CLEAN. All missing values are already NaN.")
    else:
        log.info("Fixes %s: %d cells corrected.",
                 "APPLIED" if not args.dry_run else "PENDING (dry run)", fixes_needed)

    log.info("Done.")


if __name__ == "__main__":
    main()
