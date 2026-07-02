#!/usr/bin/env python3
"""
fetch_openfema_ia.py -- Pull FEMA Individual Assistance registrations for s040a.

Mirrors fetch_openfema_event.py pattern but pulls IndividualAssistanceHousing-
RegistrantsLargeDisasters instead of FimaNfipClaims.  This is the primary
alternative to NFIP claims for C7 (Bayesian multi-source fusion).

Pulls:
  IndividualAssistanceHousingRegistrantsLargeDisasters (v1)
    - 6.4M registrant records nationally
    - ZIP + census block resolution
    - floodDamage, repairAmount, replacementAmount, destroyed, floodInsurance
    - Filtered to same DR numbers as s035

Outputs:
  s3://swarm-floodrsct-data/raw/openfema/s040a/ia_registrations_dr{number}.parquet

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

# Same DR list as fetch_openfema_event.py (s035 disasters)
S035_DISASTERS = [
    {"dr": "DR-4332-TX", "number": 4332, "state": "TX", "event": "Harvey 2017"},
    {"dr": "DR-4466-TX", "number": 4466, "state": "TX", "event": "Imelda 2019"},
    {"dr": "DR-4781-TX", "number": 4781, "state": "TX", "event": "Beryl 2024"},
    {"dr": "DR-1603-LA", "number": 1603, "state": "LA", "event": "Katrina 2005"},
    {"dr": "DR-4080-LA", "number": 4080, "state": "LA", "event": "Isaac 2012"},
    {"dr": "DR-4458-LA", "number": 4458, "state": "LA", "event": "Barry 2019"},
    {"dr": "DR-4611-LA", "number": 4611, "state": "LA", "event": "Ida 2021 LA"},
    {"dr": "DR-4085-NY", "number": 4085, "state": "NY", "event": "Sandy 2012"},
    {"dr": "DR-4615-NY", "number": 4615, "state": "NY", "event": "Ida 2021 NY"},
    {"dr": "DR-4755-NY", "number": 4755, "state": "NY", "event": "NYC Flooding Sep 2023"},
    {"dr": "DR-4673-FL", "number": 4673, "state": "FL", "event": "Ian 2022"},
    {"dr": "DR-4828-FL", "number": 4828, "state": "FL", "event": "Helene 2024"},
    {"dr": "DR-4834-FL", "number": 4834, "state": "FL", "event": "Milton 2024"},
    {"dr": "DR-4699-CA", "number": 4699, "state": "CA", "event": "Hilary 2023"},
]

# Fields to select from IA registrations
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


def fetch_ia_registrations(dr_number: int) -> pd.DataFrame:
    """Paginated pull of IA registrations for a given disaster number."""
    url = f"{OPENFEMA_BASE}/IndividualAssistanceHousingRegistrantsLargeDisasters"
    filter_str = f"disasterNumber eq {dr_number}"
    offset = 0
    all_records = []

    while True:
        params = {
            "$filter": filter_str,
            "$top": PAGE_SIZE,
            "$skip": offset,
            "$format": "json",
            "$select": IA_SELECT_FIELDS,
        }
        data = get_json(url, params)
        dataset_key = "IndividualAssistanceHousingRegistrantsLargeDisasters"
        records = data.get(dataset_key, [])
        if not records:
            break
        all_records.extend(records)
        log.info(
            "DR-%d IA: fetched %d registrations so far (offset %d)",
            dr_number, len(all_records), offset,
        )
        if len(records) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.5)

    if not all_records:
        log.warning("No IA registrations for DR-%d", dr_number)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["dr_number"] = dr_number

    # Rename ZIP to zcta_id for downstream join compatibility
    if "damagedZipCode" in df.columns:
        df = df.rename(columns={"damagedZipCode": "zcta_id"})
        df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)

    log.info("DR-%d: %d total IA registrations", dr_number, len(df))
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
    for disaster in S035_DISASTERS:
        log.info("Fetching IA registrations for %s (%s)", disaster["dr"], disaster["event"])
        ia_df = fetch_ia_registrations(dr_number=disaster["number"])
        if not ia_df.empty:
            s3_key = f"raw/openfema/s040a/ia_registrations_dr{disaster['number']}.parquet"
            upload(ia_df, s3_key)

    log.info("fetch_openfema_ia complete")


if __name__ == "__main__":
    main()
