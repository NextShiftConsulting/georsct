#!/usr/bin/env python3
"""launch_compute_diagnostics.py -- Phase 4a/4b/4c: kappa diagnostics.

Computes per-cell kappa proxies at a given representation level.
Run once per level (r0, r1, r2), each time BEFORE the next level's
training starts (pre-registration).

Resource: ml.m5.large (2 vCPU, 8 GB). Reads results JSON + adjacency,
computes spatial autocorrelation metrics.
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

    job_name = make_job_name(f"diagnostics-{args.level}")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_diagnostics.py",
        job_args=["--level", args.level, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy",
        dry_run=args.dry_run,
        phase_id=f"diagnostics_{args.level}",
    )


if __name__ == "__main__":
    main()
