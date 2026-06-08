#!/usr/bin/env python3
"""
launch_fetch_openfema_event.py -- Launch FEMA OpenFEMA event-specific pull.

Fetches disaster declarations + NFIP claims for all 5 s035 DRs.

Usage:
    python launch_fetch_openfema_event.py --dry-run
    python launch_fetch_openfema_event.py
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

    job_name = make_job_name("fetch-openfema")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_openfema_event.py",
        job_args=[],
        instance_type="ml.m5.large",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
