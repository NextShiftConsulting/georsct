#!/usr/bin/env python3
"""
build_c4_event_corrected.py -- Build C4 event-corrected NFIP claims target.

C4 = Event-specific fraud/anomaly exclusion from the correction ladder.

For each disaster, loads existing raw NFIP claims from S3, applies known
fraud/anomaly filters (Katrina DR-1603, Sandy DR-4085), and writes a
corrected parquet. Does NOT pull new data -- filters existing
raw/openfema/nfip_claims_dr{number}.parquet files.

Corrections applied:
  - DR-1603 (Katrina): exclude claims flagged by GAO fraud audits
    (>$50K building + $0 contents = adjuster anomaly pattern)
  - DR-4085 (Sandy): exclude claims with post-event date amendments
    (date_of_loss shifted >30 days from incident window)
  - All DRs: exclude negative amountPaidOnBuildingClaim (reversal entries)

Outputs:
  s3://swarm-floodrsct-data/processed/s040a/nfip_claims_c4_dr{number}.parquet
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from _openfema import S035_DISASTERS, BUCKET, setup_logging
from _s3_utils import get_s3, upload_parquet, download_parquet

setup_logging()

import logging
log = logging.getLogger(__name__)


def load_claims(s3, dr_number: int) -> pd.DataFrame:
    """Load raw NFIP claims from existing S3 parquet."""
    return download_parquet(s3, BUCKET, f"raw/openfema/nfip_claims_dr{dr_number}.parquet")


def apply_universal_filters(df: pd.DataFrame, dr_number: int) -> pd.DataFrame:
    """Filters applied to all disasters."""
    n_before = len(df)
    if "amountPaidOnBuildingClaim" in df.columns:
        df = df[df["amountPaidOnBuildingClaim"] >= 0]
    n_after = len(df)
    if n_before != n_after:
        log.info("DR-%d: universal filters removed %d rows", dr_number, n_before - n_after)
    return df


def apply_katrina_fraud_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Katrina DR-1603: GAO adjuster anomaly pattern.

    Claims with >$50K building payment but $0 contents payment flagged
    as anomalous by GAO audits.
    """
    if df.empty:
        return df
    building_col = "amountPaidOnBuildingClaim"
    contents_col = "amountPaidOnContentsClaim"
    if building_col not in df.columns or contents_col not in df.columns:
        return df

    mask = (df[building_col] > 50_000) & (df[contents_col] == 0)
    log.info("DR-1603 Katrina: flagged %d claims (>$50K building, $0 contents)", mask.sum())
    return df[~mask].copy()


def apply_sandy_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Sandy DR-4085: exclude claims with dateOfLoss outside incident window.

    Sandy window: 2012-10-27 to 2012-11-30. Claims >30 days outside
    are anomalous post-event amendments.
    """
    if df.empty or "dateOfLoss" not in df.columns:
        return df

    dol = pd.to_datetime(df["dateOfLoss"], errors="coerce")
    start = pd.Timestamp("2012-10-27")
    end = pd.Timestamp("2012-11-30")
    buffer = pd.Timedelta(days=30)

    mask = (dol < start - buffer) | (dol > end + buffer)
    log.info("DR-4085 Sandy: flagged %d claims outside incident window (+30d)", mask.sum())
    return df[~mask].copy()


def correct_claims(df: pd.DataFrame, dr_number: int) -> pd.DataFrame:
    """Apply all corrections for a given disaster."""
    df = apply_universal_filters(df, dr_number)
    if dr_number == 1603:
        df = apply_katrina_fraud_filter(df)
    elif dr_number == 4085:
        df = apply_sandy_date_filter(df)
    df["c4_corrected"] = True
    return df


def _process_one(disaster: dict) -> str:
    """Load, correct, and upload claims for one DR."""
    s3 = get_s3()
    log.info("Processing C4 correction for %s (%s)", disaster["dr"], disaster["event"])
    raw_df = load_claims(s3, disaster["number"])
    if raw_df.empty:
        return f"{disaster['dr']}: no raw claims"

    n_raw = len(raw_df)
    corrected_df = correct_claims(raw_df, disaster["number"])
    n_corrected = len(corrected_df)
    log.info(
        "%s: %d raw -> %d corrected (removed %d, %.1f%%)",
        disaster["dr"], n_raw, n_corrected,
        n_raw - n_corrected,
        100 * (n_raw - n_corrected) / n_raw if n_raw > 0 else 0,
    )

    s3_key = f"processed/s040a/nfip_claims_c4_dr{disaster['number']}.parquet"
    upload_parquet(s3, corrected_df, BUCKET, s3_key)
    return f"{disaster['dr']}: {n_raw} -> {n_corrected}"


def main() -> None:
    max_workers = max(2, os.cpu_count() or 2)
    log.info("Launching with %d workers, cpu_count=%d", max_workers, os.cpu_count() or 0)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one, d): d["dr"]
            for d in S035_DISASTERS
        }
        for future in as_completed(futures):
            dr = futures[future]
            try:
                log.info("Done: %s", future.result())
            except Exception as exc:
                log.error("%s failed: %s", dr, exc)

    log.info("build_c4_event_corrected complete")


if __name__ == "__main__":
    main()
