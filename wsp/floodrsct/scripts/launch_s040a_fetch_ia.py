#!/usr/bin/env python3
"""
launch_s040a_fetch_ia.py -- Launch FEMA IA registrations pull for s040a.

Fetches IndividualAssistanceHousingRegistrantsLargeDisasters for all
s035 disaster numbers. Output lands in s3://swarm-floodrsct-data/raw/openfema/s040a/.

Usage:
    python launch_s040a_fetch_ia.py --dry-run
    python launch_s040a_fetch_ia.py
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

    job_name = make_job_name("s040a-fetch-ia")
    launch_processing_job(
        job_name=job_name,
        job_script="s040a_bias_correction/fetch_openfema_ia.py",
        job_args=[],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        phase_id="data_fetch_ia",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
