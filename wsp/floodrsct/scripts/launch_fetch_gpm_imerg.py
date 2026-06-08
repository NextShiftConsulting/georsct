#!/usr/bin/env python3
"""launch_fetch_gpm_imerg.py -- Launch GPM IMERG daily precipitation download.

Usage:
    python launch_fetch_gpm_imerg.py --dry-run
    python launch_fetch_gpm_imerg.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # EARTHDATA_TOKEN is read at runtime inside the container via
    # swarm_auth -> AWS Secrets Manager (token is 681 chars, exceeds
    # SageMaker's 256-char env var limit).
    job_name = make_job_name("fetch-gpm-imerg")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_gpm_imerg_daily.py",
        job_args=[],
        instance_type="ml.m5.large",
        volume_size_gb=50,
        pip_packages="requests netCDF4",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
