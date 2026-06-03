#!/usr/bin/env python3
"""launch_build_hawaii_reference.py -- Extract Hawaii ZCTA reference data.

Filters national geocertdb2026 tables to Hawaii ZCTAs (968xx), computes
centroids from geometry, and uploads to s3://swarm-floodrsct-data/raw/hawaii/.

Resource: ml.m5.large (2 vCPU, 8 GB). Reads national parquets, writes 3 small
Hawaii-specific files. Needs geopandas+shapely for centroid computation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("build-hawaii-ref")

    launch_processing_job(
        job_name=job_name,
        job_script="build_hawaii_reference.py",
        job_args=[],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="geopandas shapely",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
