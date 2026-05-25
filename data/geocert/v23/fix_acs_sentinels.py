#!/usr/bin/env python3
"""
fix_acs_sentinels.py -- Replace ACS sentinel values with NaN in zcta_features_labels.parquet.

The US Census Bureau uses -666666666 (and similar large negatives) as
sentinel values for suppressed estimates. These are NOT NaN in the parquet,
so downstream median imputation misses them. StandardScaler and PCA then
produce garbage features for 22% of ZCTAs.

This script:
  1. Downloads zcta_features_labels.parquet from S3
  2. Replaces all values < -1e6 in ACS columns with NaN
  3. Backs up the original to s3://.../_backup/
  4. Uploads the cleaned version
  5. Writes a provenance log

Usage:
    python fix_acs_sentinels.py --dry-run    # Show what would change
    python fix_acs_sentinels.py              # Fix and upload

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
BACKUP_KEY = "rsct_curriculum/series_018/processed/_backup/zcta_features_labels.pre_sentinel_fix.parquet"
SENTINEL_THRESHOLD = -1e6
REGION = "us-east-1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-path", type=str, default=None,
                        help="Use local file instead of downloading from S3")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Load
    if args.local_path:
        local = Path(args.local_path)
        log.info(f"Reading local: {local}")
    else:
        local = Path("/tmp/zcta_features_labels.parquet")
        log.info(f"Downloading s3://{BUCKET}/{S3_KEY}")
        s3.download_file(BUCKET, S3_KEY, str(local))

    df = pd.read_parquet(local)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # 2. Identify ACS columns
    acs_cols = sorted(
        c for c in df.columns
        if c.startswith("acs_") and pd.api.types.is_numeric_dtype(df[c])
    )
    log.info(f"ACS columns: {len(acs_cols)}")

    # 3. Audit sentinels
    report = {}
    total_fixed = 0

    for col in acs_cols:
        vals = df[col].values
        sentinel_mask = vals < SENTINEL_THRESHOLD
        n_sentinel = int(sentinel_mask.sum())
        n_nan_before = int(np.isnan(vals).sum())

        if n_sentinel > 0:
            sentinel_values = np.unique(vals[sentinel_mask]).tolist()
            report[col] = {
                "n_sentinel": n_sentinel,
                "pct": round(n_sentinel / len(df) * 100, 2),
                "sentinel_values": sentinel_values,
                "n_nan_before": n_nan_before,
                "n_nan_after": n_nan_before + n_sentinel,
            }
            total_fixed += n_sentinel
            log.info(
                f"  {col}: {n_sentinel} sentinels "
                f"({n_sentinel / len(df) * 100:.1f}%), "
                f"values: {sentinel_values}"
            )

    zctas_affected = int((df[acs_cols].values < SENTINEL_THRESHOLD).any(axis=1).sum())

    log.info(f"\nTotal sentinel cells: {total_fixed}")
    log.info(f"ZCTAs affected: {zctas_affected}/{len(df)} ({zctas_affected / len(df) * 100:.1f}%)")
    log.info(f"Columns affected: {len(report)}/{len(acs_cols)}")

    if total_fixed == 0:
        log.info("No sentinels found -- dataset is already clean.")
        return

    if args.dry_run:
        log.info("\n[DRY RUN] Would replace the above sentinels with NaN.")
        return

    # 4. Fix: replace sentinels with NaN
    for col in report:
        mask = df[col].values < SENTINEL_THRESHOLD
        df.loc[mask, col] = np.nan

    # Verify
    remaining = (df[acs_cols].values < SENTINEL_THRESHOLD).sum()
    assert remaining == 0, f"Fix incomplete: {remaining} sentinels remain"
    log.info(f"Replaced {total_fixed} sentinel cells with NaN -- verified 0 remaining")

    # 5. Backup original
    log.info(f"Backing up original to s3://{BUCKET}/{BACKUP_KEY}")
    s3.copy_object(
        Bucket=BUCKET,
        CopySource={"Bucket": BUCKET, "Key": S3_KEY},
        Key=BACKUP_KEY,
    )

    # 6. Write cleaned parquet
    cleaned_local = Path("/tmp/zcta_features_labels_cleaned.parquet")
    df.to_parquet(cleaned_local, index=False)
    log.info(f"Uploading cleaned parquet to s3://{BUCKET}/{S3_KEY}")
    s3.upload_file(str(cleaned_local), BUCKET, S3_KEY)

    # 7. Write provenance
    provenance = {
        "operation": "fix_acs_sentinels",
        "timestamp": timestamp,
        "source": f"s3://{BUCKET}/{S3_KEY}",
        "backup": f"s3://{BUCKET}/{BACKUP_KEY}",
        "sentinel_threshold": SENTINEL_THRESHOLD,
        "total_cells_fixed": total_fixed,
        "zctas_affected": zctas_affected,
        "zctas_total": len(df),
        "columns_affected": report,
    }
    prov_key = "rsct_curriculum/series_018/processed/sentinel_fix_provenance.json"
    s3.put_object(
        Bucket=BUCKET,
        Key=prov_key,
        Body=json.dumps(provenance, indent=2),
        ContentType="application/json",
    )
    log.info(f"Provenance saved: s3://{BUCKET}/{prov_key}")

    log.info("\nDone. All downstream experiments will now get clean features.")


if __name__ == "__main__":
    main()
