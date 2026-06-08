#!/usr/bin/env python3
"""
launch_fetch_sen1floods11.py -- Launch Sen1Floods11 benchmark download on SageMaker.

Downloads Sen1Floods11 (tries cloudtostreet/sen1floods11, sen1floods11/sen1floods11,
torchgeo/sen1floods11 in order), saves metadata-only parquets (skips SAR arrays),
and uploads to s3://swarm-floodrsct-data/raw/sen1floods11/

Role: benchmark dataset for flood segmentation / Prithvi fine-tuning reference.
Placed in whitepaper appendix — benchmark comparison only, not a primary source.

Usage:
    python scripts/launch_fetch_sen1floods11.py --dry-run
    python scripts/launch_fetch_sen1floods11.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name, log


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Sen1Floods11 benchmark download")
    parser.add_argument("--dry-run", action="store_true", help="Print config without launching")
    args = parser.parse_args()

    job_name = make_job_name("fetch-sen1floods11")

    launch_processing_job(
        job_name=job_name,
        job_script="fetch_sen1floods11.py",
        job_args=[],
        instance_type="ml.m5.large",     # CPU; metadata-only parquets, no SAR arrays
        volume_size_gb=50,
        pip_packages=None,   # base packages (requests, boto3) cover everything
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        log.info(
            "Monitor: aws logs tail /aws/sagemaker/ProcessingJobs "
            "--log-stream-name-prefix %s --follow",
            job_name,
        )
        log.info(
            "Results: aws s3 ls s3://swarm-floodrsct-data/raw/sen1floods11/"
        )
        log.info(
            "Summary: aws s3 cp "
            "s3://swarm-floodrsct-data/raw/sen1floods11/summary.json -"
        )


if __name__ == "__main__":
    main()
