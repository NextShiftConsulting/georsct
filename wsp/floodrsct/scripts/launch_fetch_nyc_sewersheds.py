#!/usr/bin/env python3
"""launch_fetch_nyc_sewersheds.py -- Launch NYC DEP sewer-shed polygon download."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("fetch-nyc-sewersheds")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_nyc_sewersheds.py",
        job_args=[],
        instance_type="ml.m5.large",
        pip_packages="geopandas pyogrio",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
