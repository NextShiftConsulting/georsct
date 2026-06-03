#!/usr/bin/env python3
"""launch_floodcaster_kappa_spatial.py -- S036 H4: Floodcaster kappa spatial probe.

Computes Moran's I on floodcaster damage residuals vs NFIP claims to test
whether prediction errors exhibit spatial autocorrelation (coastal
discontinuities, reef effects, elevation breaks).

Resource: ml.m5.large (2 vCPU, 8 GB). Reads floodcaster parquet + geocertdb2026
reference tables, computes kappa_spatial, uploads evidence JSON.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-id",
        default="1f3ba5fedaaa",
        help="Floodcaster job ID to analyze (default: Oahu coastal surge)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("floodcaster-kappa-spatial")

    launch_processing_job(
        job_name=job_name,
        job_script="run_floodcaster_kappa_spatial.py",
        job_args=[],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="matplotlib",
        env_overrides={
            "JOB_ID": args.job_id,
            "S3_OUTPUT_PREFIX": "geocert-experiments/s036/floodcaster_spatial",
            "LOCAL_OUTPUT_DIR": "outputs/floodcaster_spatial",
        },
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
