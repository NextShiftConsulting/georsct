#!/usr/bin/env python3
"""launch_fetch_nsi_structures.py -- Fetch NSI 2.0 building inventory from USACE API.

Downloads building structures for FAST depth-damage analysis (DOE Change 11).
Public API, no authentication. ~50-80 MB per scenario.

Usage:
    python launch_fetch_nsi_structures.py --scenario houston --dry-run
    python launch_fetch_nsi_structures.py --scenario houston
    python launch_fetch_nsi_structures.py --all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = ["houston", "nyc", "southwest_florida"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        parser.error("Specify --scenario or --all")

    job_args = ["--all"] if args.all else ["--scenario", args.scenario]
    suffix = "all" if args.all else args.scenario.replace("_", "-")

    job_name = make_job_name(f"fetch-nsi-{suffix}")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_nsi_structures.py",
        job_args=job_args,
        instance_type="ml.m5.large",
        pip_packages="geopandas",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
