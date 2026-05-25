#!/usr/bin/env python3
"""
build_nfip_claims.py -- FEMA NFIP flood insurance claims per ZCTA.

Downloads FEMA OpenFEMA NFIP Redacted Claims v2 dataset and aggregates
to ZCTA level. NFIP claims use reportedZipCode which maps directly to
ZCTA (ZCTAs ≈ ZIP codes for residential areas; non-residential ZIPs excluded).

Separate from flood zone exposure (flood_zones_zcta.parquet):
  - Flood zones = structural risk (what FEMA maps say *could* flood)
  - NFIP claims = realized risk (what actually flooded and was insured)
  High zone_a + zero claims = levee protected, sparse, or underinsured.
  Low zone_a + high claims = rising risk not yet captured in maps.

Output: nfip_claims_zcta.parquet
  - zcta_id                   (str, 5-digit)
  - nfip_claim_count          (int)   total paid claims
  - nfip_total_building_loss  (float) total building payout $USD
  - nfip_total_contents_loss  (float) total contents payout $USD
  - nfip_total_loss           (float) total combined payout $USD
  - nfip_mean_loss_per_claim  (float) average payout per claim
  - nfip_has_claims           (bool)  any historical claim
  - nfip_claim_density        (float) claims per 1000 housing units (if ACS available)

Source: FEMA OpenFEMA v2
  https://www.fema.gov/api/open/v2/fimaNfipClaims.csv
  Max 10000 records per page; ~2.4M total claims nationally.

Usage:
    python build_nfip_claims.py --dry-run
    python build_nfip_claims.py --upload
    python build_nfip_claims.py --state TX    # subset by state for testing
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-yrsn-datasets"
PREFIX = "rsct_curriculum/series_018/processed"

OPENFEMA_BASE = "https://www.fema.gov/api/open/v2/fimaNfipClaims"
PAGE_SIZE = 10_000
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

# OpenFEMA columns to request (reduces payload size)
SELECT_COLS = [
    "reportedZipCode",
    "yearOfLoss",
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "numberOfFloorsInTheInsuredBuilding",
    "primaryResidence",
]


def fetch_page(offset: int, state: str | None = None) -> pd.DataFrame:
    """Fetch one page of NFIP claims from OpenFEMA."""
    params = {
        "$format": "csv",
        "$top": PAGE_SIZE,
        "$skip": offset,
        "$select": ",".join(SELECT_COLS),
        "$orderby": "reportedZipCode asc",
    }
    if state:
        params["$filter"] = f"reportedState eq '{state}'"

    url = f"{OPENFEMA_BASE}.csv"
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text), dtype=str, low_memory=False)
            return df
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                log.error("Page at offset %d failed: %s", offset, exc)
                return pd.DataFrame()
            log.warning("  Retry %d for offset %d: %s", attempt + 1, offset, exc)
            time.sleep(RETRY_DELAY * (attempt + 1))

    return pd.DataFrame()


def get_total_count(state: str | None = None) -> int:
    """Get total record count from OpenFEMA metadata endpoint."""
    params = {"$format": "json", "$top": 1, "$inlinecount": "allpages"}
    if state:
        params["$filter"] = f"reportedState eq '{state}'"
    try:
        resp = requests.get(f"{OPENFEMA_BASE}.json", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("metadata", {}).get("count", 0)
    except Exception as exc:
        log.warning("Could not get count: %s — will paginate until empty", exc)
        return 0


def fetch_all_claims(state: str | None = None) -> pd.DataFrame:
    """Paginate through all NFIP claims. National = ~2.4M records."""
    total = get_total_count(state)
    if total:
        n_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
        log.info("Total claims: %d (~%d pages)", total, n_pages)
    else:
        log.info("Unknown total; paginating until empty page.")
        n_pages = None

    all_pages = []
    offset = 0
    page_num = 0

    while True:
        page_num += 1
        if n_pages:
            log.info("  Page %d/%d (offset=%d)", page_num, n_pages, offset)
        else:
            log.info("  Page %d (offset=%d)", page_num, offset)

        page = fetch_page(offset, state)

        if page.empty:
            log.info("  Empty page at offset %d — done.", offset)
            break

        all_pages.append(page)
        offset += len(page)

        if len(page) < PAGE_SIZE:
            log.info("  Partial page (%d records) — done.", len(page))
            break

        # Throttle to avoid rate limiting
        time.sleep(0.5)

    if not all_pages:
        log.error("No claims fetched.")
        return pd.DataFrame()

    df = pd.concat(all_pages, ignore_index=True)
    log.info("Total fetched: %d claims", len(df))
    return df


def clean_and_aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean NFIP claims and aggregate to ZCTA."""
    log.info("Cleaning and aggregating %d records...", len(raw))

    # Normalize zip code to 5-digit ZCTA
    raw["zcta_id"] = (
        raw["reportedZipCode"]
        .astype(str)
        .str.strip()
        .str.replace(r"\D", "", regex=True)  # strip non-digits
        .str[:5]                              # take first 5 digits
        .str.zfill(5)
    )

    # Drop clearly invalid zip codes
    raw = raw[raw["zcta_id"].str.match(r"^\d{5}$")].copy()
    raw = raw[raw["zcta_id"] != "00000"].copy()

    # Parse financial columns
    for col in ("amountPaidOnBuildingClaim", "amountPaidOnContentsClaim"):
        raw[col] = pd.to_numeric(raw.get(col, 0), errors="coerce").fillna(0.0)

    raw["total_loss"] = raw["amountPaidOnBuildingClaim"] + raw["amountPaidOnContentsClaim"]

    # Keep only paid claims (total_loss > 0 OR any positive component)
    # NFIP file includes denied/closed-without-payment records
    paid = raw[raw["total_loss"] > 0].copy()
    log.info("  Paid claims: %d of %d (%.1f%%)",
             len(paid), len(raw), 100 * len(paid) / max(len(raw), 1))

    # Aggregate to ZCTA
    by_zcta = paid.groupby("zcta_id").agg(
        nfip_claim_count=("total_loss", "count"),
        nfip_total_building_loss=("amountPaidOnBuildingClaim", "sum"),
        nfip_total_contents_loss=("amountPaidOnContentsClaim", "sum"),
        nfip_total_loss=("total_loss", "sum"),
    ).reset_index()

    by_zcta["nfip_mean_loss_per_claim"] = (
        by_zcta["nfip_total_loss"] / by_zcta["nfip_claim_count"]
    ).round(2)
    by_zcta["nfip_has_claims"] = True

    log.info("  %d ZCTAs with paid claims", len(by_zcta))
    return by_zcta


def fill_zero_zctas(
    claims: pd.DataFrame, crosswalk_path: Path
) -> pd.DataFrame:
    """Ensure all ZCTAs in the crosswalk appear in output (zero-fill)."""
    xwalk = pd.read_parquet(crosswalk_path)[["zcta_id"]].drop_duplicates()
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)

    result = xwalk.merge(claims, on="zcta_id", how="left")
    result["nfip_claim_count"] = result["nfip_claim_count"].fillna(0).astype(int)
    result["nfip_total_building_loss"] = result["nfip_total_building_loss"].fillna(0.0)
    result["nfip_total_contents_loss"] = result["nfip_total_contents_loss"].fillna(0.0)
    result["nfip_total_loss"] = result["nfip_total_loss"].fillna(0.0)
    result["nfip_mean_loss_per_claim"] = result["nfip_mean_loss_per_claim"].fillna(0.0)
    result["nfip_has_claims"] = result["nfip_has_claims"].fillna(False)

    log.info("Final ZCTA coverage: %d rows", len(result))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build FEMA NFIP claims features per ZCTA"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Build locally, skip S3 upload")
    parser.add_argument("--upload", action="store_true",
                        help="Upload result to S3")
    parser.add_argument("--output", default="/tmp/nfip_claims_zcta.parquet")
    parser.add_argument("--crosswalk", default=None,
                        help="Path to zcta_county_crosswalk.parquet (local)")
    parser.add_argument("--state", default=None,
                        help="Filter to one state (2-letter abbrev, e.g. TX) for testing")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    # Resolve crosswalk
    if args.crosswalk:
        crosswalk_path = Path(args.crosswalk)
    else:
        here = Path(__file__).parent
        crosswalk_path = here / "zcta_county_crosswalk.parquet"
        if not crosswalk_path.exists():
            log.error("Crosswalk not found. Pass --crosswalk or run "
                      "build_zcta_county_crosswalk.py first.")
            sys.exit(1)

    if args.state:
        log.info("STATE FILTER: %s only", args.state.upper())

    # Fetch
    raw = fetch_all_claims(state=args.state.upper() if args.state else None)
    if raw.empty:
        sys.exit(1)

    # Clean and aggregate
    claims = clean_and_aggregate(raw)

    # Zero-fill all ZCTAs
    result = fill_zero_zctas(claims, crosswalk_path)

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs total:              %d", len(result))
    log.info("ZCTAs with claims:        %d (%.1f%%)",
             result["nfip_has_claims"].sum(),
             100 * result["nfip_has_claims"].mean())
    log.info("Total paid claims:        %d", int(result["nfip_claim_count"].sum()))
    log.info("Total losses:             $%.1fB",
             result["nfip_total_loss"].sum() / 1e9)
    log.info("Mean loss per claim:      $%.0f",
             result[result["nfip_claim_count"] > 0]["nfip_mean_loss_per_claim"].mean())

    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
from swarm_auth import get_aws_credentials
        key = f"{PREFIX}/nfip_claims_zcta.parquet"
        _aws = get_aws_credentials()
        s3 = boto3.client("s3", **_aws)
        s3.upload_file(args.output, BUCKET, key)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)

        provenance = {
            "operation": "build_nfip_claims",
            "timestamp": timestamp,
            "source": OPENFEMA_BASE,
            "state_filter": args.state,
            "n_zctas": len(result),
            "n_zctas_with_claims": int(result["nfip_has_claims"].sum()),
            "total_paid_claims": int(result["nfip_claim_count"].sum()),
            "total_loss_usd": float(result["nfip_total_loss"].sum()),
        }
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{PREFIX}/nfip_claims_zcta_provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
