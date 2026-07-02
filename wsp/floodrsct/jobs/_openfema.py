"""
_openfema.py -- Shared OpenFEMA API client and disaster registry.

Centralizes the paginated fetch pattern, retry logic, disaster metadata,
and S3 constants used by all OpenFEMA job scripts (fetch_openfema_event,
fetch_openfema_ia, fetch_openfema_policies, build_c4_event_corrected).

Usage:
    from _openfema import (
        S035_DISASTERS, S035_STATES, BUCKET,
        fetch_paginated, setup_logging,
    )

    df = fetch_paginated(
        endpoint="FimaNfipClaims",
        filter_str="state eq 'TX'",
        select_fields="reportedZipCode,dateOfLoss,amountPaidOnBuildingClaim",
    )
"""

import logging
import sys
import time

import requests

BUCKET = "swarm-floodrsct-data"
OPENFEMA_V1 = "https://www.fema.gov/api/open/v1"
OPENFEMA_V2 = "https://www.fema.gov/api/open/v2"
PAGE_SIZE = 10_000
RETRY_DELAY = 10
MAX_RETRIES = 3

# s035 disaster registrations -- single source of truth for all OpenFEMA jobs.
S035_DISASTERS = [
    {"dr": "DR-4332-TX", "number": 4332, "state": "TX", "start": "2017-08-17", "end": "2017-09-30", "event": "Harvey 2017"},
    {"dr": "DR-4466-TX", "number": 4466, "state": "TX", "start": "2019-09-17", "end": "2019-10-31", "event": "Imelda 2019"},
    {"dr": "DR-4781-TX", "number": 4781, "state": "TX", "start": "2024-07-08", "end": "2024-08-31", "event": "Beryl 2024"},
    {"dr": "DR-1603-LA", "number": 1603, "state": "LA", "start": "2005-08-29", "end": "2006-02-28", "event": "Katrina 2005"},
    {"dr": "DR-4080-LA", "number": 4080, "state": "LA", "start": "2012-08-26", "end": "2012-09-30", "event": "Isaac 2012"},
    {"dr": "DR-4458-LA", "number": 4458, "state": "LA", "start": "2019-07-10", "end": "2019-08-15", "event": "Barry 2019"},
    {"dr": "DR-4611-LA", "number": 4611, "state": "LA", "start": "2021-08-26", "end": "2021-10-31", "event": "Ida 2021 LA"},
    {"dr": "DR-4085-NY", "number": 4085, "state": "NY", "start": "2012-10-27", "end": "2012-11-30", "event": "Sandy 2012"},
    {"dr": "DR-4615-NY", "number": 4615, "state": "NY", "start": "2021-09-01", "end": "2021-10-31", "event": "Ida 2021 NY"},
    {"dr": "DR-4755-NY", "number": 4755, "state": "NY", "start": "2023-09-28", "end": "2023-10-31", "event": "NYC Flooding Sep 2023"},
    {"dr": "DR-4673-FL", "number": 4673, "state": "FL", "start": "2022-09-23", "end": "2022-10-15", "event": "Ian 2022"},
    {"dr": "DR-4828-FL", "number": 4828, "state": "FL", "start": "2024-09-24", "end": "2024-10-15", "event": "Helene 2024"},
    {"dr": "DR-4834-FL", "number": 4834, "state": "FL", "start": "2024-10-07", "end": "2024-11-01", "event": "Milton 2024"},
    {"dr": "DR-4699-CA", "number": 4699, "state": "CA", "start": "2023-08-20", "end": "2023-09-30", "event": "Hilary 2023"},
]

# Unique states from s035 disasters
S035_STATES = sorted(set(d["state"] for d in S035_DISASTERS))

log = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure root logging for SageMaker container output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


def get_json(url: str, params: dict) -> dict:
    """HTTP GET with retry. Returns parsed JSON."""
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


def fetch_paginated(
    endpoint: str,
    filter_str: str,
    select_fields: str,
    *,
    api_version: int = 1,
    page_delay: float = 0.5,
) -> list[dict]:
    """Paginated pull from any OpenFEMA dataset.

    Args:
        endpoint: Dataset name (e.g. "FimaNfipClaims",
            "IndividualAssistanceHousingRegistrantsLargeDisasters").
        filter_str: OData $filter expression.
        select_fields: Comma-separated field names for $select.
        api_version: 1 or 2 (IA/policies use v1, claims use v2).
        page_delay: Seconds between pages to avoid rate limits.

    Returns:
        List of record dicts (empty list if no data).
    """
    base = OPENFEMA_V2 if api_version == 2 else OPENFEMA_V1
    url = f"{base}/{endpoint}"
    offset = 0
    all_records = []

    while True:
        params = {
            "$filter": filter_str,
            "$top": PAGE_SIZE,
            "$skip": offset,
            "$format": "json",
            "$select": select_fields,
        }
        data = get_json(url, params)
        records = data.get(endpoint, [])
        if not records:
            break
        all_records.extend(records)
        log.info(
            "%s: fetched %d records so far (offset %d)",
            endpoint, len(all_records), offset,
        )
        if len(records) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(page_delay)

    return all_records
