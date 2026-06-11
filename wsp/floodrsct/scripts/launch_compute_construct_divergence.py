#!/usr/bin/env python3
"""launch_compute_construct_divergence.py -- DOE-C1: FAST vs NFIP construct divergence.

Computes spatial correlation and quadrant mass distribution between FAST
engineering damage estimates and NFIP administrative claims per (scenario, event).

Resource: ml.m5.large (2 vCPU, 8 GB). Reads FAST + event features
parquets from S3, computes correlations, uploads per-scenario + summary JSONs.
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

    job_name = make_job_name("construct-divergence")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_construct_divergence.py",
        job_args=["--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy",
        dry_run=args.dry_run,
        phase_id="construct_divergence",
    )


if __name__ == "__main__":
    main()
