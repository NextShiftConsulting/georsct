#!/usr/bin/env python3
"""launch_permutation_null.py -- Permutation test for construct exchangeability.

Post-processing job: reads existing DOE-C1 artifacts from S3, permutes
construct labels, and tests the null that constructs are exchangeable.

AC-1: >= 3 pairs reject permutation null at p < 0.005.

Resource: ml.m5.large (2 vCPU, 8 GB). Lightweight — pure numpy on
precomputed certificate vectors.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch permutation null test (AC-1)"
    )
    parser.add_argument("--n-perm", type=int, default=10000,
                        help="Number of permutations (default: 10000)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("perm-null")

    job_args = ["--all", "--upload", "--n-perm", str(args.n_perm)]

    launch_processing_job(
        job_name=job_name,
        job_script="compute_permutation_null.py",
        job_args=job_args,
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="numpy",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
