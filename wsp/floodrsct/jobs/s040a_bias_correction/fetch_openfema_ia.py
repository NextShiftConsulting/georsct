#!/usr/bin/env python3
"""
fetch_openfema_ia.py -- Pull FEMA Individual Assistance registrations for s040a.

Pulls IndividualAssistanceHousingRegistrantsLargeDisasters for all s035
disaster numbers. Used by C7 (Bayesian multi-source fusion).

Outputs:
  s3://swarm-floodrsct-data/raw/openfema/s040a/ia_registrations_dr{number}.parquet
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from _openfema import S035_DISASTERS, BUCKET, fetch_paginated, setup_logging
from _s3_utils import get_s3, upload_parquet

setup_logging()

import logging
log = logging.getLogger(__name__)

IA_ENDPOINT = "IndividualAssistanceHousingRegistrantsLargeDisasters"
IA_SELECT_FIELDS = (
    "disasterNumber,damagedZipCode,censusBlockId,"
    "damagedCity,damagedStateAbbreviation,"
    "floodDamage,foundationDamage,roofDamage,destroyed,"
    "waterLevel,highWaterLocation,"
    "rentalAssistanceAmount,repairAmount,replacementAmount,"
    "floodInsurance,homeOwnersInsurance,"
    "ownRent,residenceType,inspected,"
    "habitabilityRepairsRequired,renterDamageLevel"
)


def fetch_ia_registrations(dr_number: int) -> pd.DataFrame:
    """Pull IA registrations for one disaster number."""
    records = fetch_paginated(
        endpoint=IA_ENDPOINT,
        filter_str=f"disasterNumber eq {dr_number}",
        select_fields=IA_SELECT_FIELDS,
    )
    if not records:
        log.warning("No IA registrations for DR-%d", dr_number)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["dr_number"] = dr_number

    if "damagedZipCode" in df.columns:
        df = df.rename(columns={"damagedZipCode": "zcta_id"})
        df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)

    log.info("DR-%d: %d total IA registrations", dr_number, len(df))
    return df


def _fetch_and_upload(disaster: dict) -> str:
    """Fetch IA registrations for one DR and upload to S3."""
    log.info("Fetching IA for %s (%s)", disaster["dr"], disaster["event"])
    df = fetch_ia_registrations(disaster["number"])
    if not df.empty:
        s3_key = f"raw/openfema/s040a/ia_registrations_dr{disaster['number']}.parquet"
        upload_parquet(get_s3(), df, BUCKET, s3_key)
        return f"{disaster['dr']}: {len(df)} rows"
    return f"{disaster['dr']}: empty"


def main() -> None:
    max_workers = max(2, os.cpu_count() or 2)
    log.info("Launching with %d workers, cpu_count=%d", max_workers, os.cpu_count() or 0)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_and_upload, d): d["dr"]
            for d in S035_DISASTERS
        }
        for future in as_completed(futures):
            dr = futures[future]
            try:
                log.info("Done: %s", future.result())
            except Exception as exc:
                log.error("%s failed: %s", dr, exc)

    log.info("fetch_openfema_ia complete")


if __name__ == "__main__":
    main()
