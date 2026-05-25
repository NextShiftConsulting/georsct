#!/usr/bin/env python3
"""
build_benchmark_splits.py -- Build the canonical GeoCert benchmark split index.

Extracts split assignments and coverage flags from zcta_features_labels.parquet
into a lightweight, standalone parquet for artifact release.

Output: geocert_splits.parquet
Columns:
    zcta_id             str   Zero-padded 5-digit ZCTA code
    split_imputation    str   County-holdout CV fold (valid1..valid5 | test)
    split_extrapolation str   State-holdout CV fold (valid1..valid4 | test)
    split_superres      str   Super-resolution split (valid | test)
    county              str   County name
    state               str   State abbreviation
    has_cdc_places      bool  True if all 21 CDC PLACES health labels present
    has_income          bool  True if target_income is not NaN
    has_home_value      bool  True if target_home_value is not NaN

This file is sufficient to reproduce all evaluation protocols without
downloading the full feature/label parquet.

Usage:
    python build_benchmark_splits.py --dry-run    # Preview locally
    python build_benchmark_splits.py              # Upload to S3
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-yrsn-datasets"
SOURCE_KEY = "rsct_curriculum/series_018/processed/zcta_features_labels.parquet"
OUTPUT_KEY = "rsct_curriculum/series_018/processed/geocert_splits.parquet"
PROVENANCE_KEY = "rsct_curriculum/series_018/processed/geocert_splits_provenance.json"
REGION = "us-east-1"

CDC_PLACES_COLUMNS = [
    "target_annual_checkup", "target_arthritis", "target_asthma",
    "target_binge_drinking", "target_bp_medicated", "target_cancer",
    "target_cholesterol_screening", "target_chronic_kidney_disease",
    "target_copd", "target_coronary_heart_disease", "target_dental_visit",
    "target_diabetes", "target_high_blood_pressure", "target_high_cholesterol",
    "target_mental_health_not_good", "target_obesity",
    "target_physical_health_not_good", "target_physical_inactivity",
    "target_sleep_less_7hr", "target_smoking", "target_stroke",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-path", type=str, default=None)
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    timestamp = datetime.now(timezone.utc).isoformat()

    # -- 1. Load source --
    if args.local_path:
        local = Path(args.local_path)
    else:
        local = Path("/tmp/zcta_features_labels.parquet")
        log.info("Downloading s3://%s/%s", BUCKET, SOURCE_KEY)
        s3.download_file(BUCKET, SOURCE_KEY, str(local))

    df = pd.read_parquet(local)
    df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)
    log.info("Loaded %d ZCTAs", len(df))

    # -- 2. Build splits index --
    splits = pd.DataFrame({
        "zcta_id": df["zcta_id"],
        "split_imputation": df["split_imputation"],
        "split_extrapolation": df["split_extrap"],
        "split_superres": df["split_superres"],
        "county": df["county_name"],
        "state": df["state"],
        "has_cdc_places": df[CDC_PLACES_COLUMNS].notna().all(axis=1),
        "has_income": df["target_income"].notna(),
        "has_home_value": df["target_home_value"].notna(),
    })

    # -- 3. Validate --
    assert len(splits) == len(df), "Row count mismatch"
    assert splits["zcta_id"].is_unique, "Duplicate ZCTA IDs"
    assert splits["split_imputation"].notna().all(), "NaN in imputation split"
    assert splits["split_extrapolation"].notna().all(), "NaN in extrapolation split"
    assert splits["split_superres"].notna().all(), "NaN in super-resolution split"

    # -- 4. Summary --
    log.info("")
    log.info("=== SPLIT SUMMARY ===")
    log.info("")
    log.info("Imputation (county holdout, 5-fold CV + test):")
    for val, cnt in splits["split_imputation"].value_counts().sort_index().items():
        log.info("  %s: %d ZCTAs", val, cnt)

    log.info("")
    log.info("Extrapolation (state holdout, 4-fold CV + test):")
    for val, cnt in splits["split_extrapolation"].value_counts().sort_index().items():
        log.info("  %s: %d ZCTAs", val, cnt)

    log.info("")
    log.info("Super-resolution (county train -> ZCTA predict):")
    for val, cnt in splits["split_superres"].value_counts().sort_index().items():
        log.info("  %s: %d ZCTAs", val, cnt)

    log.info("")
    log.info("Coverage flags:")
    log.info("  has_cdc_places=True:  %d", splits["has_cdc_places"].sum())
    log.info("  has_cdc_places=False: %d", (~splits["has_cdc_places"]).sum())
    log.info("  has_income=True:      %d", splits["has_income"].sum())
    log.info("  has_income=False:     %d", (~splits["has_income"]).sum())
    log.info("  has_home_value=True:  %d", splits["has_home_value"].sum())
    log.info("  has_home_value=False: %d", (~splits["has_home_value"]).sum())

    n_states = splits["state"].nunique()
    n_counties = splits["county"].nunique()
    log.info("")
    log.info("Geography: %d states, %d counties, %d ZCTAs", n_states, n_counties, len(splits))

    # -- 5. Write --
    local_out = Path("/tmp/geocert_splits.parquet")
    splits.to_parquet(local_out, index=False)
    size_kb = local_out.stat().st_size / 1024
    log.info("")
    log.info("Output: %s (%.1f KB)", local_out, size_kb)

    if not args.dry_run:
        s3.upload_file(str(local_out), BUCKET, OUTPUT_KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, OUTPUT_KEY)

        # Provenance
        provenance = {
            "operation": "build_benchmark_splits",
            "timestamp": timestamp,
            "source": f"s3://{BUCKET}/{SOURCE_KEY}",
            "output": f"s3://{BUCKET}/{OUTPUT_KEY}",
            "n_zctas": len(splits),
            "n_states": int(n_states),
            "n_counties": int(n_counties),
            "splits": {
                "imputation": splits["split_imputation"].value_counts().sort_index().to_dict(),
                "extrapolation": splits["split_extrapolation"].value_counts().sort_index().to_dict(),
                "super_resolution": splits["split_superres"].value_counts().sort_index().to_dict(),
            },
            "coverage": {
                "has_cdc_places": int(splits["has_cdc_places"].sum()),
                "missing_cdc_places": int((~splits["has_cdc_places"]).sum()),
                "missing_income": int((~splits["has_income"]).sum()),
                "missing_home_value": int((~splits["has_home_value"]).sum()),
            },
            "evaluation_protocols": {
                "imputation": "County-holdout 5-fold CV. For each fold, train on remaining folds, predict held-out county ZCTAs. Test fold is final evaluation.",
                "extrapolation": "State-holdout 4-fold CV. Entire states held out to test distribution shift. 9 test states held aside.",
                "super_resolution": "Train on county-aggregated labels, predict ZCTA-level labels. Tests within-county heterogeneity recovery.",
            },
        }
        s3.put_object(
            Bucket=BUCKET, Key=PROVENANCE_KEY,
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved: s3://%s/%s", BUCKET, PROVENANCE_KEY)
    else:
        log.info("[DRY RUN] Would upload to s3://%s/%s", BUCKET, OUTPUT_KEY)

    log.info("Done.")


if __name__ == "__main__":
    main()
