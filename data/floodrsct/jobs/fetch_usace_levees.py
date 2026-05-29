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
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
NLD_BASE = "https://levees.sec.usace.army.mil/api-local"

# State FIPS codes for spatial filter
SCENARIO_STATES = {
    "new_orleans": ["22"],      # Louisiana
    "nyc": ["36", "34"],        # New York, New Jersey
}

# USACE districts by scenario (for filtering)
SCENARIO_KEYWORDS = {
    "new_orleans": ["Orleans", "Jefferson", "St. Bernard", "Plaquemines", "St. Tammany",
                    "New Orleans", "Lake Pontchartrain"],
    "nyc": ["Jamaica", "Newtown", "Coney Island", "Hudson", "Passaic", "Hackensack",
            "New York", "New Jersey"],
}


def fetch_systems_by_state(state_fips: str) -> list[dict]:
    """Fetch all levee systems in a state from NLD."""
    url = f"{NLD_BASE}/levee-systems"
    params = {
        "state": state_fips,
        "format": "json",
    }
    log.info("Fetching levee systems for state FIPS %s", state_fips)
    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # NLD response may be under 'features' (GeoJSON) or direct list
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "features" in data:
            return [f.get("properties", f) for f in data["features"]]
        return []
    except requests.RequestException as e:
        log.warning("NLD request failed for state %s: %s", state_fips, e)
        return []


def fetch_inspections(system_id: str) -> list[dict]:
    """Fetch inspection records for a levee system."""
    url = f"{NLD_BASE}/levee-systems/{system_id}/inspections"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json() if isinstance(resp.json(), list) else []
    except requests.RequestException as e:
        log.debug("Could not fetch inspections for system %s: %s", system_id, e)
        return []


def build_levee_df(scenario: str) -> pd.DataFrame:
    states = SCENARIO_STATES[scenario]
    keywords = SCENARIO_KEYWORDS[scenario]
    keywords_lower = [k.lower() for k in keywords]

    all_systems = []
    for state_fips in states:
        systems = fetch_systems_by_state(state_fips)
        all_systems.extend(systems)
        time.sleep(0.5)

    log.info("Scenario %s: %d raw systems from NLD", scenario, len(all_systems))

    if not all_systems:
        log.warning("No levee systems found for %s — NLD API may have changed", scenario)
        return pd.DataFrame()

    df = pd.DataFrame(all_systems)

    # Filter to scenario-relevant systems by name keyword
    name_col = next((c for c in df.columns if "name" in c.lower()), None)
    if name_col:
        mask = df[name_col].str.lower().apply(
            lambda x: any(k in str(x) for k in keywords_lower)
        )
        df_filtered = df[mask].copy()
        log.info("Filtered to %d systems matching scenario keywords", len(df_filtered))
    else:
        log.warning("No name column found in NLD response; using all %d systems", len(df))
        df_filtered = df.copy()

    # Enrich with inspection records
    id_col = next((c for c in df_filtered.columns if "system_id" in c.lower() or c == "id"), None)
    if id_col and len(df_filtered) <= 200:
        inspection_rows = []
        for _, row in df_filtered.iterrows():
            sys_id = row[id_col]
            inspections = fetch_inspections(str(sys_id))
            if inspections:
                latest = max(inspections, key=lambda x: x.get("inspection_date", ""))
                inspection_rows.append({
                    id_col: sys_id,
                    "latest_inspection_date": latest.get("inspection_date"),
                    "condition_rating": latest.get("overall_system_rating",
                                                   latest.get("condition_rating")),
                    "inspection_type": latest.get("inspection_type"),
                })
            time.sleep(0.2)
        if inspection_rows:
            insp_df = pd.DataFrame(inspection_rows)
            df_filtered = pd.merge(df_filtered, insp_df, on=id_col, how="left")
            log.info("Merged inspection data for %d systems", len(insp_df))

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
    parser.add_argument("--scenario", required=True, choices=["new_orleans", "nyc", "all"])
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
