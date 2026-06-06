#!/usr/bin/env python3
"""
launch_fetch_noaa_mrms.py -- Launch NOAA Stage IV hourly precip download.

Harvey 2017 is ~408 hourly files (~500 MB); use ml.m5.xlarge for bandwidth.

Usage:
    python launch_fetch_noaa_mrms.py --event harvey2017 --dry-run
    python launch_fetch_noaa_mrms.py --event harvey2017
    python launch_fetch_noaa_mrms.py --event imelda2019
    python launch_fetch_noaa_mrms.py --event beryl2024
    python launch_fetch_noaa_mrms.py --event ida2021_nyc
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

EVENT_INSTANCES = {
    "harvey2017":  "ml.m5.xlarge",  # ~17 days, ~408 files
    "imelda2019":  "ml.m5.large",
    "beryl2024":   "ml.m5.large",
    "ida2021_nola": "ml.m5.large",   # ~8 days
    "ida2021_nyc": "ml.m5.large",
    "ian2022":     "ml.m5.xlarge",  # ~9 days
    "helene2024":  "ml.m5.large",
    "milton2024":  "ml.m5.large",
    "hilary2023":  "ml.m5.large",
    "henri2021":   "ml.m5.large",
    "ar_flood_2023": "ml.m5.xlarge",  # ~22 days
    "isaac2012_nola": "ml.m5.large",
    "barry2019_nola": "ml.m5.large",
    "sandy2012":   "ml.m5.large",   # ~6 days, Oct 2012 (MRMS just operational)
    "nyc_flood_2023": "ml.m5.large",  # ~4 days
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, choices=list(EVENT_INSTANCES.keys()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"fetch-mrms-{args.event.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_noaa_mrms_v2.py",
        job_args=["--event", args.event],
        instance_type=EVENT_INSTANCES[args.event],
        volume_size_gb=100,   # grib2 files accumulate
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
