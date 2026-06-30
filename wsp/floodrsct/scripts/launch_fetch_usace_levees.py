#!/usr/bin/env python3
"""launch_fetch_usace_levees.py -- Launch USACE National Levee Database pull."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                        choices=["new_orleans", "nyc", "houston",
                                 "southwest_florida", "all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"fetch-usace-levees-{args.scenario.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_usace_levees.py",
        job_args=["--scenario", args.scenario],
        instance_type="ml.m5.large",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
