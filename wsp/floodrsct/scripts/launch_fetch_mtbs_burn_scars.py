#!/usr/bin/env python3
"""launch_fetch_mtbs_burn_scars.py -- Launch USGS MTBS burn perimeter fetch."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("fetch-mtbs-burn-scars")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_mtbs_burn_scars.py",
        job_args=[],
        instance_type="ml.m5.large",
        volume_size_gb=20,
        pip_packages="geopandas pyogrio",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
