#!/usr/bin/env python3
"""
launch_fetch_smap_soil_moisture.py -- Launch SMAP L4 SPL4SMGP v008 antecedent
soil moisture download via SageMaker Processing.

Downloads 7-day antecedent windows for all 4 FloodRSCT storm events and
uploads to s3://swarm-floodrsct-data/raw/smap_soil_moisture/v008/.

Option A (NASA Harmony bbox subsetting) is tried first; Option B (direct
NSIDC HDF5, ~142 MB each) is the automatic fallback.

Usage:
    python launch_fetch_smap_soil_moisture.py --dry-run
    python launch_fetch_smap_soil_moisture.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import DATA_BUCKET, launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch SMAP L4 soil moisture download job"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without launching",
    )
    args = parser.parse_args()

    # EARTHDATA_TOKEN must be set in the environment before launching.
    # It is passed to the container as an environment variable via the
    # SageMaker job config below.
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not token:
        print(
            "ERROR: EARTHDATA_TOKEN is not set.\n"
            "Export it first:\n"
            "  export EARTHDATA_TOKEN=<your-token>",
            file=sys.stderr,
        )
        sys.exit(1)

    job_name = make_job_name("fetch-smap-soil-moisture")

    # We need requests; boto3 + pyyaml + pandas are in _BASE_PACKAGES.
    # h5py is included so the container can validate downloaded HDF5 files.
    pip_packages = "requests h5py netCDF4"

    # ml.m5.large is enough: Harmony subsets are tiny; direct HDF5 files
    # are ~142 MB each but streaming uploads keep RAM low.
    # Total worst-case: 4 events x 7 days x 142 MB = ~4 GB -- fits on 30 GB EBS.
    instance_type = "ml.m5.large"
    volume_size_gb = 30

    launch_processing_job(
        job_name=job_name,
        job_script="fetch_smap_soil_moisture.py",
        job_args=[],          # script uses env var EARTHDATA_TOKEN, no CLI args
        instance_type=instance_type,
        volume_size_gb=volume_size_gb,
        pip_packages=pip_packages,
        env_overrides={"EARTHDATA_TOKEN": token},
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
