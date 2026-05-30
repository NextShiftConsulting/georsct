#!/usr/bin/env python3
"""
validate_data_readiness.py -- Machine-readable data readiness matrix.

Checks every (event, dataset) cell against S3, counts records,
and emits a status: missing | launched | fetched | validated | no_data_available.

Usage:
    python validate_data_readiness.py
    python validate_data_readiness.py --format csv
    python validate_data_readiness.py --format json
    python validate_data_readiness.py --upload   # writes matrix to S3
"""

import argparse
import csv
import io
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import boto3
from swarm_auth import get_aws_credentials

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"

# ---------------------------------------------------------------------------
# Scenario -> event mapping
# ---------------------------------------------------------------------------
SCENARIOS = {
    "houston": ["harvey2017", "imelda2019", "beryl2024"],
    "new_orleans": ["ida2021_nola"],
    "nyc": ["ida2021_nyc", "henri2021"],
    "southwest_florida": ["ian2022", "helene2024", "milton2024"],
    "riverside_coachella": ["hilary2023", "ar_flood_2023"],
}

# ---------------------------------------------------------------------------
# Dataset specifications: (s3_prefix_template, required_scenarios, is_event_level)
# For event-level datasets, {event} is substituted.
# For static datasets, the prefix is checked once.
# ---------------------------------------------------------------------------
EVENT_DATASETS = {
    "mrms": {
        "prefix": "raw/noaa_mrms/{event}/",
        "scenarios": ["houston", "new_orleans", "nyc", "southwest_florida", "riverside_coachella"],
    },
    "hrrr": {
        "prefix": "raw/noaa_hrrr/{event}/",
        "scenarios": ["houston", "new_orleans", "nyc", "southwest_florida", "riverside_coachella"],
    },
    "tides": {
        "prefix": "raw/noaa_tides/{event}/",
        "scenarios": ["houston", "new_orleans", "southwest_florida", "nyc"],
    },
    "gpm_imerg": {
        "prefix": "raw/gpm_imerg/daily/{event}/",
        "scenarios": ["houston", "new_orleans", "nyc", "southwest_florida", "riverside_coachella"],
    },
    "smap": {
        "prefix": "raw/smap_soil_moisture/v008/{event}/",
        "scenarios": ["houston", "new_orleans", "nyc", "southwest_florida", "riverside_coachella"],
    },
    "hwm": {
        "prefix": "raw/surge_estimates/{event}/",
        "scenarios": ["houston", "southwest_florida", "nyc", "riverside_coachella"],
    },
    "hurdat2": {
        "prefix": "raw/hurdat2/",
        "scenarios": ["houston", "southwest_florida", "riverside_coachella"],
        "static": True,
    },
}

STATIC_DATASETS = {
    "dem": {
        "prefix": "raw/dem/3dep/",
        "scenarios": ["southwest_florida", "new_orleans"],
    },
    "nlcd": {
        "prefix": "raw/nlcd/impervious_2021/",
        "scenarios": ["houston", "nyc"],
    },
    "nhdplus": {
        "prefix": "raw/nhdplus/catchments/",
        "scenarios": ["riverside_coachella", "houston"],
    },
    "mtbs": {
        "prefix": "raw/mtbs/perimeters/",
        "scenarios": ["riverside_coachella"],
    },
    "openfema": {
        "prefix": "raw/openfema/",
        "scenarios": ["houston", "new_orleans", "nyc", "southwest_florida", "riverside_coachella"],
    },
    "houston_311": {
        "prefix": "raw/houston_311/",
        "scenarios": ["houston"],
    },
    "nyc_311": {
        "prefix": "raw/nyc_311/",
        "scenarios": ["nyc"],
    },
    "tiger_coastline": {
        "prefix": "raw/tiger/coastline/",
        "scenarios": ["southwest_florida"],
    },
    "usace_levees": {
        "prefix": "raw/usace_levees/",
        "scenarios": ["new_orleans", "nyc"],
    },
    "usgs_subsidence": {
        "prefix": "raw/usgs_subsidence/",
        "scenarios": ["new_orleans"],
    },
    "nyc_sewersheds": {
        "prefix": "raw/nyc_sewersheds/",
        "scenarios": ["nyc"],
    },
    "mta_stations": {
        "prefix": "raw/mta/",
        "scenarios": ["nyc"],
    },
    "osm_canals": {
        "prefix": "raw/osm/new_orleans_canals/",
        "scenarios": ["new_orleans"],
    },
    "hcfcd": {
        "prefix": "raw/hcfcd/",
        "scenarios": ["houston"],
    },
    "slosh_mom": {
        "prefix": "raw/noaa_slosh/mom_national/",
        "scenarios": ["southwest_florida"],
    },
}

# Known no-data cases (STN confirmed zero HWMs)
NO_DATA = {
    ("beryl2024", "hwm"): "USGS STN event 342 confirmed zero HWMs",
    ("ar_flood_2023", "tides"): "Inland atmospheric river; tidal stations not relevant",
    ("ar_flood_2023", "hurdat2"): "Atmospheric river, not tropical cyclone",
}


@dataclass
class CellStatus:
    scenario: str
    event: str
    dataset: str
    s3_prefix: str
    status: str  # missing | fetched | no_data_available
    record_count: int
    missing_reason: str


def count_s3_objects(s3, prefix: str) -> int:
    """Count objects under a prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        count += len(page.get("Contents", []))
    return count


def check_event_dataset(s3, scenario: str, event: str, dataset: str, spec: dict) -> CellStatus:
    """Check one (event, dataset) cell."""
    # Check no-data registry
    no_data_key = (event, dataset)
    if no_data_key in NO_DATA:
        prefix = spec["prefix"].format(event=event)
        return CellStatus(
            scenario=scenario, event=event, dataset=dataset,
            s3_prefix=f"s3://{BUCKET}/{prefix}",
            status="no_data_available", record_count=0,
            missing_reason=NO_DATA[no_data_key],
        )

    # Static datasets (not event-level)
    if spec.get("static"):
        prefix = spec["prefix"]
    else:
        prefix = spec["prefix"].format(event=event)

    count = count_s3_objects(s3, prefix)

    if count == 0:
        return CellStatus(
            scenario=scenario, event=event, dataset=dataset,
            s3_prefix=f"s3://{BUCKET}/{prefix}",
            status="missing", record_count=0,
            missing_reason="No objects found at prefix",
        )

    return CellStatus(
        scenario=scenario, event=event, dataset=dataset,
        s3_prefix=f"s3://{BUCKET}/{prefix}",
        status="fetched", record_count=count,
        missing_reason="",
    )


def check_static_dataset(s3, scenario: str, dataset: str, spec: dict) -> CellStatus:
    """Check one static dataset for a scenario."""
    prefix = spec["prefix"]
    count = count_s3_objects(s3, prefix)

    status = "fetched" if count > 0 else "missing"
    return CellStatus(
        scenario=scenario, event="(static)", dataset=dataset,
        s3_prefix=f"s3://{BUCKET}/{prefix}",
        status=status, record_count=count,
        missing_reason="" if count > 0 else "No objects found at prefix",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table")
    parser.add_argument("--upload", action="store_true", help="Upload matrix to S3")
    args = parser.parse_args()

    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    results: list[CellStatus] = []

    # Event-level datasets
    for dataset, spec in EVENT_DATASETS.items():
        if spec.get("static"):
            # Static but listed in event datasets (e.g. hurdat2)
            for scenario in spec["scenarios"]:
                for event in SCENARIOS[scenario]:
                    results.append(check_event_dataset(s3, scenario, event, dataset, spec))
            continue

        for scenario in spec["scenarios"]:
            for event in SCENARIOS[scenario]:
                results.append(check_event_dataset(s3, scenario, event, dataset, spec))

    # Static datasets
    for dataset, spec in STATIC_DATASETS.items():
        for scenario in spec["scenarios"]:
            results.append(check_static_dataset(s3, scenario, dataset, spec))

    # Sort by scenario, event, dataset
    results.sort(key=lambda r: (r.scenario, r.event, r.dataset))

    # Output
    if args.format == "json":
        print(json.dumps([asdict(r) for r in results], indent=2))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=[
            "scenario", "event", "dataset", "s3_prefix",
            "status", "record_count", "missing_reason",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    else:
        # Table format
        _print_table(results)

    # Summary
    total = len(results)
    fetched = sum(1 for r in results if r.status == "fetched")
    missing = sum(1 for r in results if r.status == "missing")
    no_data = sum(1 for r in results if r.status == "no_data_available")
    print(f"\n--- Summary: {fetched}/{total} fetched, {missing} missing, {no_data} no_data_available ---")

    if missing > 0:
        print("\n--- MISSING ---")
        for r in results:
            if r.status == "missing":
                print(f"  {r.scenario}/{r.event}/{r.dataset}: {r.s3_prefix}")

    # Upload
    if args.upload:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = f"meta/data_readiness_{ts}.json"
        body = json.dumps([asdict(r) for r in results], indent=2)
        s3.put_object(Bucket=BUCKET, Key=key, Body=body.encode())
        log.info("Uploaded readiness matrix to s3://%s/%s", BUCKET, key)

        # Also write latest
        s3.put_object(Bucket=BUCKET, Key="meta/data_readiness_latest.json", Body=body.encode())
        log.info("Updated s3://%s/meta/data_readiness_latest.json", BUCKET)


def _print_table(results: list[CellStatus]) -> None:
    """Print a formatted table grouped by scenario."""
    current_scenario = None
    for r in results:
        if r.scenario != current_scenario:
            current_scenario = r.scenario
            print(f"\n{'='*70}")
            print(f"  SCENARIO: {current_scenario}")
            print(f"{'='*70}")
            print(f"  {'Event':<20} {'Dataset':<15} {'Status':<20} {'Count':>6}  {'Reason'}")
            print(f"  {'-'*20} {'-'*15} {'-'*20} {'-'*6}  {'-'*20}")

        status_marker = {
            "fetched": "FETCHED",
            "missing": "** MISSING **",
            "no_data_available": "NO DATA (OK)",
        }.get(r.status, r.status)

        print(f"  {r.event:<20} {r.dataset:<15} {status_marker:<20} {r.record_count:>6}  {r.missing_reason}")


if __name__ == "__main__":
    main()
