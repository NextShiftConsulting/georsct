#!/usr/bin/env python3
"""
launch_fetch_prithvi_eo2.py -- Launch Prithvi-EO-2.0 download + smoke test on SageMaker.

Downloads ibm-nasa-geospatial/Prithvi-EO-2.0 from HuggingFace Hub, uploads weights
to S3, and runs a forward-pass smoke test on a dummy 6-band 224x224 tile.

Requires GPU instance for the smoke test (CUDA forward pass).
Outputs to s3://swarm-floodrsct-data/model/prithvi_eo2/

Usage:
    python scripts/launch_fetch_prithvi_eo2.py --dry-run
    python scripts/launch_fetch_prithvi_eo2.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name, log


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Prithvi-EO-2.0 download + smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Print config without launching")
    args = parser.parse_args()

    job_name = make_job_name("fetch-prithvi-eo2")

    launch_processing_job(
        job_name=job_name,
        job_script="fetch_prithvi_eo2.py",
        job_args=[],
        instance_type="ml.m5.4xlarge",   # g5.2xlarge quota=0; CPU smoke test is slower but valid
        volume_size_gb=100,              # ~30 GB weights + workspace
        # Pin transformers>=4.45 — TimmWrapperConfig __strict_setattr__ compat fix
        pip_packages="huggingface_hub transformers>=4.45.0 timm datasets",
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        log.info(
            "Monitor: aws logs tail /aws/sagemaker/ProcessingJobs "
            "--log-stream-name-prefix %s --follow",
            job_name,
        )
        log.info(
            "Results: aws s3 ls s3://swarm-floodrsct-data/model/prithvi_eo2/"
        )


if __name__ == "__main__":
    main()
