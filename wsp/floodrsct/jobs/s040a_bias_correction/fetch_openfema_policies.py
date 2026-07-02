#!/usr/bin/env python3
"""
fetch_openfema_policies.py -- Pull NFIP policy counts for IPW penetration rates.

Used by C2 (Inverse Probability Weighting): penetration_rate = claims / policies
gives the propensity score for IPW correction of selection bias in NFIP claims.

Pulls:
  FimaNfipPolicies (v1)
    - Active policy counts by ZIP + flood zone
    - policyCount, crsClassCode, floodZone, propertyState
    - Filtered to states in s035 scenarios

Outputs:
  s3://swarm-floodrsct-data/raw/openfema/s040a/nfip_policies_{state}.parquet

NOTE: New directory (s040a/) -- does NOT overwrite existing raw/openfema/ data.
"""

import logging
import sys
import time
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
OPENFEMA_BASE = "https://www.fema.gov/api/open/v1"
PAGE_SIZE = 10_000
RETRY_DELAY = 10
MAX_RETRIES = 3

# States from s035 scenarios
S035_STATES = ["TX", "LA", "NY", "FL", "CA"]

POLICY_SELECT_FIELDS = (
    "propertyState,reportedZipCode,floodZone,"
    "policyCount,crsClassCode,policyEffectiveDate,"
    "policyTerminationDate,occupancyType,"
    "totalBuildingInsuranceCoverage,totalContentsInsuranceCoverage,"
    "deductibleAmountInBuildingCoverage"
)


def get_json(url: str, params: dict) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"All retries exhausted for {url}")


def fetch_policies(state: str) -> pd.DataFrame:
    """Paginated pull of NFIP policies for a given state."""
    url = f"{OPENFEMA_BASE}/FimaNfipPolicies"
    filter_str = f"propertyState eq '{state}'"
    offset = 0
    all_records = []

    while True:
        params = {
            "$filter": filter_str,
            "$top": PAGE_SIZE,
            "$skip": offset,
            "$format": "json",
            "$select": POLICY_SELECT_FIELDS,
        }
        data = get_json(url, params)
        dataset_key = "FimaNfipPolicies"
        records = data.get(dataset_key, [])
        if not records:
            break
        all_records.extend(records)
        log.info(
            "%s policies: fetched %d so far (offset %d)",
            state, len(all_records), offset,
        )
        if len(records) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.5)

    if not all_records:
        log.warning("No NFIP policies for state %s", state)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Rename ZIP to zcta_id for downstream join compatibility
    if "reportedZipCode" in df.columns:
        df = df.rename(columns={"reportedZipCode": "zcta_id"})
        df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)

    log.info("%s: %d total NFIP policies", state, len(df))
    return df


def upload(df: pd.DataFrame, s3_key: str) -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    local = f"/tmp/{Path(s3_key).name}"
    df.to_parquet(local, index=False)
    s3.upload_file(local, BUCKET, s3_key)
    log.info("Uploaded %d rows to s3://%s/%s", len(df), BUCKET, s3_key)


def main() -> None:
    for state in S035_STATES:
        log.info("Fetching NFIP policies for %s", state)
        pol_df = fetch_policies(state)
        if not pol_df.empty:
            s3_key = f"raw/openfema/s040a/nfip_policies_{state}.parquet"
            upload(pol_df, s3_key)

    log.info("fetch_openfema_policies complete")


if __name__ == "__main__":
    main()
