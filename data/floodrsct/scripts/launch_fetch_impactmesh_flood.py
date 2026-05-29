#!/usr/bin/env python3
"""
launch_fetch_impactmesh_flood.py -- Launch ImpactMesh-Flood benchmark download on SageMaker.

Downloads masks + DEMs + split lists from ibm-esa-geospatial/ImpactMesh-Flood
via direct HTTPS (Git LFS blobs). Skips S1RTC/S2L2A imagery tars (~100+ GB).

What we get:
  - split/*.txt      (scene lists, tiny)
  - {train,val,test}/MASK.tar   (~307 MB total)
  - {train,val,test}/DEM.tar    (~333 MB total)
Total: ~640 MB

Outputs to s3://swarm-floodrsct-data/raw/impactmesh_flood/

Usage:
    python scripts/launch_fetch_impactmesh_flood.py --dry-run
    python scripts/launch_fetch_impactmesh_flood.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name, log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch ImpactMesh-Flood benchmark download"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without launching")
    args = parser.parse_args()

    job_name = make_job_name("fetch-impactmesh-flood")

    launch_processing_job(
        job_name=job_name,
        job_script="fetch_impactmesh_flood.py",
        job_args=[],
        instance_type="ml.m5.xlarge",   # direct HTTPS -> S3 stream; ~640 MB total
        volume_size_gb=10,              # streamed directly to S3, minimal tmp usage
        pip_packages=None,              # base packages (requests, boto3) cover everything
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        log.info(
            "Monitor: aws logs tail /aws/sagemaker/ProcessingJobs "
            "--log-stream-name-prefix %s --follow",
            job_name,
        )
        log.info(
            "Summary: aws s3 cp "
            "s3://swarm-floodrsct-data/raw/impactmesh_flood/summary.json -"
        )


if __name__ == "__main__":
    main()
