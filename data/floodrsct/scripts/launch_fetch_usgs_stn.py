#!/usr/bin/env python3
"""launch_fetch_usgs_stn.py -- Launch USGS STN high-water mark pull.

Usage:
    python launch_fetch_usgs_stn.py --event ida2021_nola --dry-run
    python launch_fetch_usgs_stn.py --event beryl2024
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

ALL_EVENTS = [
    "harvey2017", "imelda2019", "ian2022",
    "ida2021_nola", "ida2021_nyc",
    "beryl2024", "hilary2023",
    "helene2024", "milton2024",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, choices=ALL_EVENTS + ["all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    events = ALL_EVENTS if args.event == "all" else [args.event]

    for event in events:
        job_name = make_job_name(f"fetch-hwm-{event.replace('_', '-')}")
        launch_processing_job(
            job_name=job_name,
            job_script="fetch_surge_hwm.py",
            job_args=["--event", event],
            instance_type="ml.m5.large",
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
