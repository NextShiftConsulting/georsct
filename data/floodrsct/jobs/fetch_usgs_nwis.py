#!/usr/bin/env python3
"""
fetch_usgs_nwis.py -- SageMaker job: pull USGS NWIS gauge timeseries.

Queries USGS waterservices.usgs.gov for instantaneous (15-min) values of
streamflow (00060) and gauge height (00065) for all active gauges in the
target county, across each event window defined in the scenario config.

Outputs one parquet per event to S3:
  s3://swarm-floodrsct-data/raw/usgs_nwis/{scenario}_{event}.parquet

Usage (inside SageMaker container):
    python fetch_usgs_nwis.py --scenario houston
    python fetch_usgs_nwis.py --scenario new_orleans
    python fetch_usgs_nwis.py --scenario nyc
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import pandas as pd
import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
NWIS_BASE = "https://waterservices.usgs.gov/nwis/iv/"
PARAM_CODES = "00060,00065"  # discharge (cfs), gauge height (ft)
PAGE_SLEEP = 1.0  # seconds between requests

# Config files are uploaded alongside this script by the launcher
# Launcher uploads configs to the same S3 prefix as code, so they land
# in /opt/ml/processing/input/code/ (not a separate /config/ channel).
CODE_DIR = Path("/opt/ml/processing/input/code")


def load_config(scenario: str) -> dict:
    path = CODE_DIR / f"{scenario}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_site_list(county_fips: str) -> list[str]:
    """Return all active USGS sites in county via site service."""
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format": "rdb",
        "countyCd": county_fips,
        "siteType": "ST",
        "siteStatus": "all",
        "hasDataTypeCd": "iv",
    }
    log.info("Querying site list for county %s", county_fips)
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=300)
            if resp.status_code == 404:
                log.warning("No USGS stream gauges in county %s (404)", county_fips)
                return []
            resp.raise_for_status()
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            log.warning("Site list request failed (attempt %d/3): %s", attempt + 1, type(e).__name__)
            if attempt == 2:
                raise
            time.sleep(10)

    sites = []
    for line in resp.text.splitlines():
        if line.startswith("#") or line.startswith("agency_cd") or line.startswith("5s"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] == "USGS":
            sites.append(parts[1].strip())
    log.info("Found %d sites in county %s", len(sites), county_fips)
    return sites


def fetch_iv_data(sites: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch instantaneous values for list of sites over a date range.

    NWIS IV service has a ~300-site limit per request; chunk accordingly.
    """
    chunk_size = 25  # smaller chunks reduce SSL/timeout risk on large queries
    frames = []

    for i in range(0, len(sites), chunk_size):
        chunk = sites[i : i + chunk_size]
        params = {
            "format": "json",
            "sites": ",".join(chunk),
            "parameterCd": PARAM_CODES,
            "startDT": start,
            "endDT": end,
        }
        log.info("Fetching IV data: sites %d-%d, %s to %s", i, i + len(chunk), start, end)
        for attempt in range(3):
            try:
                resp = requests.get(NWIS_BASE, params=params, timeout=300)
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                log.warning("IV data request failed (attempt %d/3): %s", attempt + 1, type(e).__name__)
                if attempt == 2:
                    raise
                time.sleep(10)
        if resp.status_code == 400:
            log.warning("Bad request for chunk %d; skipping. Response: %s", i, resp.text[:200])
            time.sleep(PAGE_SLEEP)
            continue
        resp.raise_for_status()

        data = resp.json()
        time_series = data.get("value", {}).get("timeSeries", [])
        for ts in time_series:
            site_code = ts["sourceInfo"]["siteCode"][0]["value"]
            param = ts["variable"]["variableCode"][0]["value"]
            unit = ts["variable"]["unit"]["unitCode"]
            values = ts["values"][0]["value"] if ts.get("values") else []
            for v in values:
                frames.append({
                    "site_no": site_code,
                    "datetime_utc": v["dateTime"],
                    "param_code": param,
                    "value": float(v["value"]) if v["value"] != "-999999" else None,
                    "unit": unit,
                    "quality_code": v.get("qualifiers", [""])[0],
                })
        time.sleep(PAGE_SLEEP)

    if not frames:
        log.warning("No IV data retrieved for this window")
        return pd.DataFrame(columns=["site_no", "datetime_utc", "param_code", "value", "unit", "quality_code"])

    df = pd.DataFrame(frames)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df


def pivot_params(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot param_code into flow_cfs and stage_ft columns."""
    if df.empty:
        return df
    flow = df[df["param_code"] == "00060"][["site_no", "datetime_utc", "value"]].rename(
        columns={"value": "flow_cfs"}
    )
    stage = df[df["param_code"] == "00065"][["site_no", "datetime_utc", "value"]].rename(
        columns={"value": "stage_ft"}
    )
    merged = pd.merge(flow, stage, on=["site_no", "datetime_utc"], how="outer")
    return merged.sort_values(["site_no", "datetime_utc"]).reset_index(drop=True)


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
                        choices=["houston", "new_orleans", "nyc",
                                 "riverside_coachella", "southwest_florida"])
    args = parser.parse_args()

    cfg = load_config(args.scenario)

    # Determine site list — support both county_fips (str/list) and county_fips_list
    county_fips = cfg.get("county_fips_list") or cfg.get("county_fips") or []
    if isinstance(county_fips, str):
        county_fips = [county_fips]
    anchor_sites = cfg.get("usgs_anchor_sites", [])

    sites = set(anchor_sites)
    for fips in county_fips:
        # NWIS site service takes 5-char county FIPS (state+county)
        county_sites = fetch_site_list(fips)
        sites.update(county_sites)
    sites = list(sites)

    if not sites:
        log.error("No USGS sites identified for scenario %s", args.scenario)
        sys.exit(1)

    log.info("Total sites for %s: %d", args.scenario, len(sites))

    for event_name, event_cfg in cfg["events"].items():
        log.info("Fetching event: %s", event_name)
        df = fetch_iv_data(sites, event_cfg["start_date"], event_cfg["end_date"])
        df = pivot_params(df)
        df["event"] = event_name
        df["scenario"] = args.scenario

        s3_key = f"raw/usgs_nwis/{args.scenario}_{event_name}.parquet"
        upload(df, s3_key)

    log.info("fetch_usgs_nwis complete for scenario: %s", args.scenario)


if __name__ == "__main__":
    main()
