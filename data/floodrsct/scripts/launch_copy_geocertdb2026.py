#!/usr/bin/env python3
"""launch_copy_geocertdb2026.py -- Copy reusable ZCTA features from geocertdb2026."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("copy-geocertdb2026")
    launch_processing_job(
        job_name=job_name,
        job_script="copy_geocertdb2026.py",
        job_args=[],
        instance_type="ml.m5.large",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
