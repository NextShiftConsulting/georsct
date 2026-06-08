#!/usr/bin/env python3
"""launch_compute_certificates.py -- Phase 4.5a/4.5b/4.5c: RSCT certificates.

Builds RSCT certificates (R, S_sup, N, kappa, sigma) for all cells at
one representation level. Requires rsct wheel (vendored in jobs/).

Resource: ml.m5.large (2 vCPU, 8 GB). Reads results + diagnostics
from S3, computes simplex + bootstrap CIs.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

LEVELS = ["r0", "r1", "r2"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True, choices=LEVELS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"certificates-{args.level}")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_certificates.py",
        job_args=["--level", args.level, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy",
        dry_run=args.dry_run,
        phase_id=f"certificates_{args.level}",
    )


if __name__ == "__main__":
    main()
