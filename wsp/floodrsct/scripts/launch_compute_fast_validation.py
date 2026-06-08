#!/usr/bin/env python3
"""launch_compute_fast_validation.py -- Phase 7b: FAST external validation.

Compares R0/R1/R2 predictions against FEMA FAST engineering damage
estimates via Spearman rho across scenarios and return periods.

Resource: ml.m5.large (2 vCPU, 8 GB). Reads predictions + FAST
parquets from S3, computes correlations.
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

    job_name = make_job_name("fast-validation")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_fast_validation.py",
        job_args=["--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy",
        dry_run=args.dry_run,
        phase_id="fast_validation",
    )


if __name__ == "__main__":
    main()
