#!/usr/bin/env python3
"""launch_fetch_noaa_hrrr.py -- Launch HRRR 3-km QPF fetch for a single event."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

# ar_flood_2023 is 22 days; all others <= 10 days
_LARGE_EVENTS = {"harvey2017", "ar_flood_2023"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event",
        required=True,
        choices=[
            "harvey2017", "imelda2019", "beryl2024", "ida2021_nyc",
            "ian2022", "helene2024", "milton2024", "hilary2023", "ar_flood_2023",
        ],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    instance_type = "ml.m5.xlarge" if args.event in _LARGE_EVENTS else "ml.m5.large"

    job_name = make_job_name(f"fetch-hrrr-{args.event.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_noaa_hrrr.py",
        job_args=["--event", args.event],
        instance_type=instance_type,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
