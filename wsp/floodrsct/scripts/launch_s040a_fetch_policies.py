#!/usr/bin/env python3
"""
launch_s040a_fetch_policies.py -- Launch NFIP policy pull for s040a IPW.

Fetches FimaNfipPolicies for all s035 states (TX, LA, NY, FL, CA).
Output lands in s3://swarm-floodrsct-data/raw/openfema/s040a/.

Usage:
    python launch_s040a_fetch_policies.py --dry-run
    python launch_s040a_fetch_policies.py
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

    job_name = make_job_name("s040a-fetch-policies")
    launch_processing_job(
        job_name=job_name,
        job_script="s040a_bias_correction/fetch_openfema_policies.py",
        job_args=[],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        phase_id="data_fetch_policies",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
