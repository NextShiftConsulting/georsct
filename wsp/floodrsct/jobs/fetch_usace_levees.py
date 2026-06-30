#!/usr/bin/env python3
"""
fetch_usace_levees.py -- SageMaker job: pull USACE National Levee Database records.

Queries the USACE NLD public API for levee system records and inspection
ratings in the New Orleans and NYC/NJ scenarios.

API: https://levees.sec.usace.army.mil/api-local/

Outputs:
  s3://swarm-floodrsct-data/raw/usace_levees/no_levees.parquet
  s3://swarm-floodrsct-data/raw/usace_levees/nyc_levees.parquet
"""

import argparse
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

# ArcGIS Online FeatureServer — NLD2 public dataset (Leveed_Areas = layer 16)
# Original API at levees.sec.usace.army.mil/api-local went 404 circa May 2026.
NLD_FS_BASE = (
    "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services"
    "/NLD2_PUBLIC_v1/FeatureServer/16/query"
)
PAGE_SIZE = 1000

# State name filter (STATES field contains full names, e.g. "Louisiana")
SCENARIO_STATES = {
    "new_orleans": ["Louisiana"],
    "nyc": ["New York", "New Jersey"],
    "houston": ["Texas"],
    "southwest_florida": ["Florida"],
}

# USACE district / county keywords for filtering to scenario metro area
SCENARIO_KEYWORDS = {
    "new_orleans": ["Orleans", "Jefferson", "St. Bernard", "Plaquemines", "St. Tammany",
                    "New Orleans", "Lake Pontchartrain"],
    "nyc": ["Jamaica", "Newtown", "Coney Island", "Hudson", "Passaic", "Hackensack",
            "New York", "New Jersey"],
    "houston": ["Harris", "Fort Bend", "Montgomery", "Galveston", "Brazoria", "Chambers",
                "Addicks", "Barker", "Brays Bayou", "Buffalo Bayou", "Houston"],
    "southwest_florida": ["Lee", "Collier", "Charlotte", "Sarasota", "Manatee", "DeSoto",
                          "Caloosahatchee", "Fort Myers", "Cape Coral"],
}

# Fields to retrieve from NLD2 Leveed_Areas
OUT_FIELDS = (
    "SYSTEM_ID,SYSTEM_NAME,LEVEED_ID,STATES,COUNTIES,DISTRICTS,"
    "FEMA_ACCREDITATION_RATING,LEVEED_AREA_SQ_MI,SPONSORS,RESPONSIBLE_ORGANIZATION"
)


def fetch_leveed_areas(state_name: str) -> list[dict]:
    """Paginated ArcGIS FeatureServer query for leveed areas in a state."""
    all_records = []
    offset = 0
    while True:
        params = {
            "where": f"STATES LIKE '%{state_name}%'",
            "outFields": OUT_FIELDS,
            "resultRecordCount": PAGE_SIZE,
            "resultOffset": offset,
            "f": "json",
            "returnGeometry": "false",
        }
        log.info("Fetching NLD2 leveed areas for %s, offset %d", state_name, offset)
        try:
            resp = requests.get(NLD_FS_BASE, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            log.warning("NLD2 request failed for %s: %s", state_name, e)
            break
        if "error" in data:
            log.warning("NLD2 query error for %s: %s", state_name, data["error"])
            break
        features = data.get("features", [])
        if not features:
            break
        all_records.extend(f["attributes"] for f in features)
        if len(features) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.3)
    return all_records


def build_levee_df(scenario: str) -> pd.DataFrame:
    state_names = SCENARIO_STATES[scenario]
    keywords = SCENARIO_KEYWORDS[scenario]
    keywords_lower = [k.lower() for k in keywords]

    all_records = []
    for state_name in state_names:
        records = fetch_leveed_areas(state_name)
        all_records.extend(records)

    log.info("Scenario %s: %d raw leveed areas from NLD2", scenario, len(all_records))

    if not all_records:
        log.warning("No leveed areas found for %s", scenario)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Filter to scenario-relevant systems by county/district/name keywords
    text_cols = ["SYSTEM_NAME", "COUNTIES", "DISTRICTS"]
    existing_cols = [c for c in text_cols if c in df.columns]
    if existing_cols:
        combined = df[existing_cols].fillna("").apply(
            lambda row: " ".join(row).lower(), axis=1
        )
        mask = combined.apply(lambda x: any(k in x for k in keywords_lower))
        df_filtered = df[mask].copy()
        log.info("Filtered to %d systems matching scenario keywords", len(df_filtered))
    else:
        df_filtered = df.copy()

    # Derive condition_rating from FEMA_ACCREDITATION_RATING
    if "FEMA_ACCREDITATION_RATING" in df_filtered.columns:
        df_filtered["condition_rating"] = df_filtered["FEMA_ACCREDITATION_RATING"]

    df_filtered["scenario"] = scenario
    return df_filtered.reset_index(drop=True)


def upload(df: pd.DataFrame, s3_key: str) -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    local = f"/tmp/{Path(s3_key).name}"
    df.to_parquet(local, index=False)
    s3.upload_file(local, BUCKET, s3_key)
    log.info("Uploaded %d rows to s3://%s/%s", len(df), BUCKET, s3_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                        choices=list(SCENARIO_STATES.keys()) + ["all"])
    args = parser.parse_args()

    scenarios = list(SCENARIO_STATES.keys()) if args.scenario == "all" else [args.scenario]

    for scenario in scenarios:
        df = build_levee_df(scenario)
        if df.empty:
            log.error("No levee data for scenario %s", scenario)
            continue
        s3_key = f"raw/usace_levees/{scenario}_levees.parquet"
        upload(df, s3_key)

    log.info("fetch_usace_levees complete")


if __name__ == "__main__":
    main()
