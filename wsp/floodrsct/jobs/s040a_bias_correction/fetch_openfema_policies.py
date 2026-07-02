#!/usr/bin/env python3
"""
fetch_openfema_policies.py -- Pull NFIP policy counts for IPW penetration rates.

Used by C2 (Inverse Probability Weighting): penetration_rate = claims / policies
gives the propensity score for IPW correction of selection bias in NFIP claims.

Outputs:
  s3://swarm-floodrsct-data/raw/openfema/s040a/nfip_policies_{state}.parquet
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from _openfema import S035_STATES, BUCKET, fetch_paginated, setup_logging
from _s3_utils import get_s3, upload_parquet

setup_logging()

import logging
log = logging.getLogger(__name__)

POLICY_ENDPOINT = "FimaNfipPolicies"
POLICY_SELECT_FIELDS = (
    "propertyState,reportedZipCode,floodZone,"
    "policyCount,crsClassCode,policyEffectiveDate,"
    "policyTerminationDate,occupancyType,"
    "totalBuildingInsuranceCoverage,totalContentsInsuranceCoverage,"
    "deductibleAmountInBuildingCoverage"
)


def fetch_policies(state: str) -> pd.DataFrame:
    """Pull NFIP policies for one state."""
    records = fetch_paginated(
        endpoint=POLICY_ENDPOINT,
        filter_str=f"propertyState eq '{state}'",
        select_fields=POLICY_SELECT_FIELDS,
    )
    if not records:
        log.warning("No NFIP policies for state %s", state)
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if "reportedZipCode" in df.columns:
        df = df.rename(columns={"reportedZipCode": "zcta_id"})
        df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)

    log.info("%s: %d total NFIP policies", state, len(df))
    return df


def _fetch_and_upload(state: str) -> str:
    """Fetch NFIP policies for one state and upload to S3."""
    log.info("Fetching NFIP policies for %s", state)
    df = fetch_policies(state)
    if not df.empty:
        s3_key = f"raw/openfema/s040a/nfip_policies_{state}.parquet"
        upload_parquet(get_s3(), df, BUCKET, s3_key)
        return f"{state}: {len(df)} rows"
    return f"{state}: empty"


def main() -> None:
    max_workers = max(2, os.cpu_count() or 2)
    log.info("Launching with %d workers, cpu_count=%d", max_workers, os.cpu_count() or 0)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_and_upload, s): s
            for s in S035_STATES
        }
        for future in as_completed(futures):
            state = futures[future]
            try:
                log.info("Done: %s", future.result())
            except Exception as exc:
                log.error("%s failed: %s", state, exc)

    log.info("fetch_openfema_policies complete")


if __name__ == "__main__":
    main()
