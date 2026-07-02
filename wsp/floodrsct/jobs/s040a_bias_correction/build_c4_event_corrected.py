#!/usr/bin/env python3
"""
build_c4_event_corrected.py -- Build C4 event-corrected NFIP claims target.

C4 = Event-specific fraud/anomaly exclusion from the correction ladder.

For each disaster, loads existing raw NFIP claims from S3, applies known
fraud/anomaly filters (Katrina DR-1603, Sandy DR-4085), and writes a
corrected parquet. This does NOT pull new data -- it filters existing
raw/openfema/nfip_claims_dr{number}.parquet files.

Corrections applied:
  - DR-1603 (Katrina): exclude claims flagged by GAO fraud audits
    (>$50K building + $0 contents = adjuster anomaly pattern)
  - DR-4085 (Sandy): exclude claims with post-event date amendments
    (date_of_loss shifted >30 days from incident window)
  - All DRs: exclude negative amountPaidOnBuildingClaim (reversal entries)

Outputs:
  s3://swarm-floodrsct-data/processed/s040a/nfip_claims_c4_dr{number}.parquet

NOTE: Reads from raw/openfema/ (existing), writes to processed/s040a/ (new).
"""

import logging
import sys
from io import BytesIO
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"

# Same DR list as fetch_openfema_event.py
S035_DISASTERS = [
    {"dr": "DR-4332-TX", "number": 4332, "event": "Harvey 2017"},
    {"dr": "DR-4466-TX", "number": 4466, "event": "Imelda 2019"},
    {"dr": "DR-4781-TX", "number": 4781, "event": "Beryl 2024"},
    {"dr": "DR-1603-LA", "number": 1603, "event": "Katrina 2005"},
    {"dr": "DR-4080-LA", "number": 4080, "event": "Isaac 2012"},
    {"dr": "DR-4458-LA", "number": 4458, "event": "Barry 2019"},
    {"dr": "DR-4611-LA", "number": 4611, "event": "Ida 2021 LA"},
    {"dr": "DR-4085-NY", "number": 4085, "event": "Sandy 2012"},
    {"dr": "DR-4615-NY", "number": 4615, "event": "Ida 2021 NY"},
    {"dr": "DR-4755-NY", "number": 4755, "event": "NYC Flooding Sep 2023"},
    {"dr": "DR-4673-FL", "number": 4673, "event": "Ian 2022"},
    {"dr": "DR-4828-FL", "number": 4828, "event": "Helene 2024"},
    {"dr": "DR-4834-FL", "number": 4834, "event": "Milton 2024"},
    {"dr": "DR-4699-CA", "number": 4699, "event": "Hilary 2023"},
]


def get_s3_client():
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    return boto3.client("s3", region_name="us-east-1", **_aws)


def load_claims(s3, dr_number: int) -> pd.DataFrame:
    """Load raw NFIP claims from existing S3 parquet."""
    key = f"raw/openfema/nfip_claims_dr{dr_number}.parquet"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        log.info("Loaded %d raw claims from %s", len(df), key)
        return df
    except s3.exceptions.NoSuchKey:
        log.warning("No raw claims file: %s", key)
        return pd.DataFrame()


def apply_universal_filters(df: pd.DataFrame, dr_number: int) -> pd.DataFrame:
    """Filters applied to all disasters."""
    n_before = len(df)

    # Exclude reversal entries (negative building claim amounts)
    if "amountPaidOnBuildingClaim" in df.columns:
        df = df[df["amountPaidOnBuildingClaim"] >= 0]

    n_after = len(df)
    if n_before != n_after:
        log.info("DR-%d: universal filters removed %d rows", dr_number, n_before - n_after)
    return df


def apply_katrina_fraud_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Katrina DR-1603: GAO adjuster anomaly pattern.

    Claims with >$50K building payment but $0 contents payment flagged
    as anomalous by GAO audits -- adjuster pattern where building claims
    inflated while contents ignored.
    """
    if df.empty:
        return df
    building_col = "amountPaidOnBuildingClaim"
    contents_col = "amountPaidOnContentsClaim"
    if building_col not in df.columns or contents_col not in df.columns:
        return df

    mask = (df[building_col] > 50_000) & (df[contents_col] == 0)
    n_flagged = mask.sum()
    log.info("DR-1603 Katrina: flagged %d claims (>$50K building, $0 contents)", n_flagged)
    df = df[~mask].copy()
    return df


def apply_sandy_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Sandy DR-4085: exclude claims with shifted dateOfLoss.

    Sandy incident window: 2012-10-27 to 2012-11-30.
    Claims with dateOfLoss >30 days outside this window are anomalous
    (post-event amendments that inflate counts).
    """
    if df.empty:
        return df
    if "dateOfLoss" not in df.columns:
        return df

    df["_dol"] = pd.to_datetime(df["dateOfLoss"], errors="coerce")
    incident_start = pd.Timestamp("2012-10-27")
    incident_end = pd.Timestamp("2012-11-30")
    buffer = pd.Timedelta(days=30)

    mask = (df["_dol"] < incident_start - buffer) | (df["_dol"] > incident_end + buffer)
    n_flagged = mask.sum()
    log.info("DR-4085 Sandy: flagged %d claims outside incident window (+30d)", n_flagged)
    df = df[~mask].copy()
    df = df.drop(columns=["_dol"])
    return df


def correct_claims(df: pd.DataFrame, dr_number: int) -> pd.DataFrame:
    """Apply all corrections for a given disaster."""
    df = apply_universal_filters(df, dr_number)

    if dr_number == 1603:
        df = apply_katrina_fraud_filter(df)
    elif dr_number == 4085:
        df = apply_sandy_date_filter(df)

    df["c4_corrected"] = True
    return df


def upload(s3, df: pd.DataFrame, s3_key: str) -> None:
    local = f"/tmp/{Path(s3_key).name}"
    df.to_parquet(local, index=False)
    s3.upload_file(local, BUCKET, s3_key)
    log.info("Uploaded %d rows to s3://%s/%s", len(df), BUCKET, s3_key)


def main() -> None:
    s3 = get_s3_client()

    for disaster in S035_DISASTERS:
        log.info("Processing C4 correction for %s (%s)", disaster["dr"], disaster["event"])
        raw_df = load_claims(s3, disaster["number"])
        if raw_df.empty:
            log.warning("Skipping %s -- no raw claims", disaster["dr"])
            continue

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
        upload(s3, corrected_df, s3_key)

    log.info("build_c4_event_corrected complete")


if __name__ == "__main__":
    main()
