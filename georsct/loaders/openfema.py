"""
openfema.py -- OpenFEMA API client with pagination and exponential backoff.

Canonical implementation for all FEMA data fetches across georsct and
yrsn-experiments. Three consumers, one source of truth.

API version notes (verified 2026-07-03):
  - FimaNfipClaims:    v2 only
  - FimaNfipPolicies:  v2 only (v1 returns 404)
  - IndividualAssistanceHousingRegistrantsLargeDisasters: v1 only (v2 returns 503)

Rate limiting:
  - Government API; no published rate limit but returns 503 under load.
  - All callers MUST fetch sequentially with REQUEST_GAP between items.
  - NEVER use ThreadPoolExecutor or concurrent requests against OpenFEMA.

Usage:
    from georsct.loaders.openfema import fetch_paginated, get_record_count

    records = fetch_paginated(
        endpoint="FimaNfipClaims",
        filter_str="reportedState eq 'TX'",
        select_fields="reportedZipCode,yearOfLoss,amountPaidOnBuildingClaim",
        api_version=2,
    )
"""

import logging
import time

import requests

log = logging.getLogger(__name__)

OPENFEMA_V1 = "https://www.fema.gov/api/open/v1"
OPENFEMA_V2 = "https://www.fema.gov/api/open/v2"

# Endpoint -> required API version (verified 2026-07-03)
ENDPOINT_VERSIONS = {
    "FimaNfipClaims": 2,
    "FimaNfipPolicies": 2,
    "IndividualAssistanceHousingRegistrantsLargeDisasters": 1,
}

PAGE_SIZE = 10_000
RETRY_BASE_DELAY = 15  # seconds; doubles each attempt (15, 30, 60, 120, 240)
MAX_RETRIES = 5
REQUEST_GAP = 1.0  # seconds between sequential fetches (items, not pages)
PAGE_DELAY = 0.5  # seconds between pages within one item


def _base_url(api_version: int) -> str:
    return OPENFEMA_V2 if api_version == 2 else OPENFEMA_V1


def _validate_version(endpoint: str, api_version: int) -> None:
    """Warn if using a known-bad API version for an endpoint."""
    required = ENDPOINT_VERSIONS.get(endpoint)
    if required is not None and api_version != required:
        log.warning(
            "Endpoint '%s' requires v%d but v%d was requested. "
            "This will likely return %s.",
            endpoint, required, api_version,
            "404" if required == 2 else "503",
        )


def get_json(url: str, params: dict) -> dict:
    """HTTP GET with exponential backoff. Returns parsed JSON."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning(
                "Attempt %d/%d failed (backoff %ds): %s",
                attempt, MAX_RETRIES, delay, e,
            )
            if attempt < MAX_RETRIES:
                time.sleep(delay)
    raise RuntimeError(f"All {MAX_RETRIES} retries exhausted for {url}")


def get_record_count(
    endpoint: str,
    filter_str: str | None = None,
    api_version: int = 2,
) -> int:
    """Get total record count from OpenFEMA metadata endpoint."""
    _validate_version(endpoint, api_version)
    base = _base_url(api_version)
    params: dict = {"$top": 1, "$inlinecount": "allpages", "$format": "json"}
    if filter_str:
        params["$filter"] = filter_str
    try:
        data = get_json(f"{base}/{endpoint}", params)
        return data.get("metadata", {}).get("count", 0)
    except Exception as exc:
        log.warning("Could not get count for %s: %s", endpoint, exc)
        return 0


def fetch_paginated(
    endpoint: str,
    filter_str: str,
    select_fields: str,
    *,
    api_version: int | None = None,
    page_delay: float = PAGE_DELAY,
) -> list[dict]:
    """Paginated pull from any OpenFEMA dataset.

    Args:
        endpoint: Dataset name (e.g. "FimaNfipClaims").
        filter_str: OData $filter expression.
        select_fields: Comma-separated field names for $select.
        api_version: 1 or 2. If None, auto-detected from ENDPOINT_VERSIONS.
        page_delay: Seconds between pages to avoid rate limits.

    Returns:
        List of record dicts (empty list if no data).

    Raises:
        RuntimeError: If all retries exhausted on any page.
    """
    if api_version is None:
        api_version = ENDPOINT_VERSIONS.get(endpoint, 2)
        log.info("Auto-detected API v%d for %s", api_version, endpoint)

    _validate_version(endpoint, api_version)
    base = _base_url(api_version)
    url = f"{base}/{endpoint}"
    offset = 0
    all_records: list[dict] = []

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
