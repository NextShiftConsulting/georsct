#!/usr/bin/env python3
"""launch_fetch_nlcd_impervious.py -- Launch NLCD 2021 impervious surface fetch."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("fetch-nlcd-impervious")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_nlcd_impervious.py",
        job_args=[],
        instance_type="ml.m5.large",
        volume_size_gb=20,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
