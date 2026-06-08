#!/usr/bin/env python3
"""launch_fetch_houston_311.py -- Launch Houston 311 flood report pull."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("fetch-houston-311")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_houston_311.py",
        job_args=[],
        instance_type="ml.m5.large",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
