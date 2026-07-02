#!/usr/bin/env python3
"""
launch_s040a_build_c4.py -- Launch C4 event-corrected NFIP claims build.

Reads existing raw/openfema/nfip_claims_dr*.parquet, applies fraud/anomaly
filters (Katrina, Sandy, universal), writes processed/s040a/ corrected files.

Usage:
    python launch_s040a_build_c4.py --dry-run
    python launch_s040a_build_c4.py
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

    job_name = make_job_name("s040a-build-c4")
    launch_processing_job(
        job_name=job_name,
        job_script="s040a_bias_correction/build_c4_event_corrected.py",
        job_args=[],
        instance_type="ml.m5.large",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
