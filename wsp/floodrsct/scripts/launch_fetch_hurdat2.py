#!/usr/bin/env python3
"""
launch_fetch_hurdat2.py -- Launch NHC HURDAT2 storm track pull on SageMaker.

Usage:
    python launch_fetch_hurdat2.py --dry-run
    python launch_fetch_hurdat2.py
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

    job_name = make_job_name("fetch-hurdat2")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_hurdat2.py",
        job_args=[],
        instance_type="ml.m5.large",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
