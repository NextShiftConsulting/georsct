#!/usr/bin/env python3
"""
launch_fetch_usgs_nwis.py -- Launch USGS NWIS gauge pull on SageMaker.

Usage:
    python launch_fetch_usgs_nwis.py --scenario houston --dry-run
    python launch_fetch_usgs_nwis.py --scenario houston
    python launch_fetch_usgs_nwis.py --scenario new_orleans
    python launch_fetch_usgs_nwis.py --scenario nyc
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["houston", "new_orleans", "nyc"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"fetch-nwis-{args.scenario.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_usgs_nwis.py",
        job_args=["--scenario", args.scenario],
        instance_type="ml.m5.large",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
