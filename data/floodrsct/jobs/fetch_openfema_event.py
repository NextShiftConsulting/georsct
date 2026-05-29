#!/usr/bin/env python3
"""
fetch_openfema_event.py -- SageMaker job: pull FEMA OpenFEMA disaster declarations
and event-specific NFIP claims for s035 disaster registrations.

Pulls:
  1. DisasterDeclarationsSummaries — all DRs for the 5 s035 events
  2. FimaNfipClaims — paginated, filtered by incidentBeginDate within each DR window

Outputs:
  s3://swarm-floodrsct-data/raw/openfema/disaster_declarations.parquet
  s3://swarm-floodrsct-data/raw/openfema/nfip_claims_dr{number}.parquet  (one per DR)
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
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
OPENFEMA_BASE = "https://www.fema.gov/api/open/v2"
PAGE_SIZE = 10_000
RETRY_DELAY = 10
MAX_RETRIES = 3

# s035 disaster registrations
S035_DISASTERS = [
    {"dr": "DR-4332-TX", "number": 4332, "state": "TX", "start": "2017-08-17", "end": "2017-09-30", "event": "Harvey 2017"},
    {"dr": "DR-4466-TX", "number": 4466, "state": "TX", "start": "2019-09-17", "end": "2019-10-31", "event": "Imelda 2019"},
    {"dr": "DR-4781-TX", "number": 4781, "state": "TX", "start": "2024-07-08", "end": "2024-08-31", "event": "Beryl 2024"},
    {"dr": "DR-4611-LA", "number": 4611, "state": "LA", "start": "2021-08-26", "end": "2021-10-31", "event": "Ida 2021 LA"},
    {"dr": "DR-4615-NY", "number": 4615, "state": "NY", "start": "2021-09-01", "end": "2021-10-31", "event": "Ida 2021 NY"},
    # Southwest Florida
    {"dr": "DR-4673-FL", "number": 4673, "state": "FL", "start": "2022-09-23", "end": "2022-10-15", "event": "Ian 2022"},
    {"dr": "DR-4828-FL", "number": 4828, "state": "FL", "start": "2024-09-24", "end": "2024-10-15", "event": "Helene 2024"},
    {"dr": "DR-4834-FL", "number": 4834, "state": "FL", "start": "2024-10-07", "end": "2024-11-01", "event": "Milton 2024"},
    # Riverside-Coachella: Hilary 2023 (CA DR)
    {"dr": "DR-4699-CA", "number": 4699, "state": "CA", "start": "2023-08-20", "end": "2023-09-30", "event": "Hilary 2023"},
]


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


def fetch_declarations() -> pd.DataFrame:
    """Fetch disaster declarations for all s035 DR numbers."""
    dr_numbers = [d["number"] for d in S035_DISASTERS]
    filter_str = " or ".join(f"disasterNumber eq {n}" for n in dr_numbers)
    url = f"{OPENFEMA_BASE}/DisasterDeclarationsSummaries"
    params = {
        "$filter": filter_str,
        "$format": "json",
        "$top": PAGE_SIZE,
    }
    data = get_json(url, params)
    records = data.get("DisasterDeclarationsSummaries", [])
    df = pd.DataFrame(records)
    log.info("Fetched %d disaster declaration records", len(df))
    return df


def fetch_nfip_claims(state: str, start: str, end: str, dr_number: int) -> pd.DataFrame:
    """Paginated pull of NFIP claims for a given state + incident date window."""
    url = f"{OPENFEMA_BASE}/FimaNfipClaims"
    # Filter: state abbreviation + incident date within event window
    filter_str = (
        f"state eq '{state}' and "
        f"dateOfLoss ge '{start}' and "
        f"dateOfLoss le '{end}'"
    )
    offset = 0
    all_records = []

    while True:
        params = {
            "$filter": filter_str,
            "$top": PAGE_SIZE,
            "$skip": offset,
            "$format": "json",
            "$select": (
                "reportedZipCode,dateOfLoss,amountPaidOnBuildingClaim,"
                "amountPaidOnContentsClaim,totalBuildingInsuranceCoverage,"
                "ratedFloodZone,occupancyType,numberOfFloorsInTheInsuredBuilding,"
                "basementEnclosureCrawlspaceType"
            ),
        }
        data = get_json(url, params)
        records = data.get("FimaNfipClaims", [])
        if not records:
            break
        all_records.extend(records)
        log.info("DR-%d: fetched %d claims so far (offset %d)", dr_number, len(all_records), offset)
        if len(records) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.5)

    if not all_records:
        log.warning("No NFIP claims for DR-%d (%s, %s to %s)", dr_number, state, start, end)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["dr_number"] = dr_number
    # Rename ZIP to zcta_id for downstream join compatibility
    if "reportedZipCode" in df.columns:
        df = df.rename(columns={"reportedZipCode": "zcta_id"})
        df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)
    log.info("DR-%d: %d total claims", dr_number, len(df))
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
    # Disaster declarations
    decl_df = fetch_declarations()
    upload(decl_df, "raw/openfema/disaster_declarations.parquet")

    # Event-specific NFIP claims
    for disaster in S035_DISASTERS:
        log.info("Fetching NFIP claims for %s", disaster["dr"])
        claims_df = fetch_nfip_claims(
            state=disaster["state"],
            start=disaster["start"],
            end=disaster["end"],
            dr_number=disaster["number"],
        )
        if not claims_df.empty:
            s3_key = f"raw/openfema/nfip_claims_dr{disaster['number']}.parquet"
            upload(claims_df, s3_key)

    log.info("fetch_openfema_event complete")


if __name__ == "__main__":
    main()
