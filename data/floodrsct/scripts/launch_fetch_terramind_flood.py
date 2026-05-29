#!/usr/bin/env python3
"""
launch_fetch_terramind_flood.py -- Launch TerraMind-base-Flood download on SageMaker.

Downloads ibm-esa-geospatial/TerraMind-base-Flood (~673 MB weights) via
huggingface_hub snapshot_download and runs a torch.load smoke test.

Outputs to s3://swarm-floodrsct-data/model/terramind_flood/

Usage:
    python scripts/launch_fetch_terramind_flood.py --dry-run
    python scripts/launch_fetch_terramind_flood.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name, log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch TerraMind-base-Flood download + smoke test"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without launching")
    args = parser.parse_args()

    job_name = make_job_name("fetch-terramind-flood")

    launch_processing_job(
        job_name=job_name,
        job_script="fetch_terramind_flood.py",
        job_args=[],
        instance_type="ml.m5.xlarge",   # CPU; 673 MB weights, torch.load smoke test
        volume_size_gb=20,              # weights 673 MB + tmp overhead
        pip_packages="huggingface_hub",  # torch already in base image
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        log.info(
            "Monitor: aws logs tail /aws/sagemaker/ProcessingJobs "
            "--log-stream-name-prefix %s --follow",
            job_name,
        )
        log.info(
            "Smoke test: aws s3 cp "
            "s3://swarm-floodrsct-data/model/terramind_flood/smoke_test/smoke_test_result.json -"
        )


if __name__ == "__main__":
    main()
