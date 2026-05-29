#!/usr/bin/env python3
"""
launch_fetch_floodsimbench.py -- Launch FloodSimBench GeoTIFF download on SageMaker.

Downloads 6hr_max MaxDepth GeoTIFFs from chrimerss/FloodSimBench (HuggingFace
file repo, CC-BY-4.0) for Houston and NYC tiles via direct HTTPS + stream to S3.
No HF Datasets / load_dataset() — files are served as Git LFS blobs.

Outputs to s3://swarm-floodrsct-data/raw/floodsimbench/

Usage:
    python scripts/launch_fetch_floodsimbench.py --dry-run
    python scripts/launch_fetch_floodsimbench.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name, log


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FloodSimBench download + scenario clip")
    parser.add_argument("--dry-run", action="store_true", help="Print config without launching")
    args = parser.parse_args()

    job_name = make_job_name("fetch-floodsimbench")

    launch_processing_job(
        job_name=job_name,
        job_script="fetch_floodsimbench.py",
        job_args=[],
        instance_type="ml.m5.xlarge",    # direct HTTPS->S3 stream; no dataset materialisation
        volume_size_gb=50,               # tmp files only; streamed directly to S3
        pip_packages=None,               # base packages (requests, boto3) cover everything
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        log.info(
            "Monitor: aws logs tail /aws/sagemaker/ProcessingJobs "
            "--log-stream-name-prefix %s --follow",
            job_name,
        )
        log.info(
            "Results: aws s3 ls s3://swarm-floodrsct-data/raw/floodsimbench/"
        )
        log.info(
            "Summary: aws s3 cp "
            "s3://swarm-floodrsct-data/raw/floodsimbench/summary.json -"
        )


if __name__ == "__main__":
    main()
