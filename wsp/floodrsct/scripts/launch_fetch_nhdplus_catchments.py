#!/usr/bin/env python3
"""launch_fetch_nhdplus_catchments.py -- Launch NHDPlus V2 catchment fetch."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("fetch-nhdplus-catchments")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_nhdplus_catchments.py",
        job_args=[],
        instance_type="ml.m5.xlarge",  # 7z extraction is CPU-heavy
        volume_size_gb=50,
        pip_packages="geopandas pyogrio py7zr",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
