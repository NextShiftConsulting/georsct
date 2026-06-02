#!/usr/bin/env python3
"""launch_compute_geometry_kappa.py -- Phase 0.5: geometry-only kappa.

Computes kappa_geom from problem geometry BEFORE any model training.
Must run before Phase 1 (R0 baseline) to establish independence from
RSN coordinates.

Resource: ml.m5.large (2 vCPU, 8 GB). Reads adjacency + feature
coverage from S3, no heavy compute.
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

    job_name = make_job_name("geometry-kappa")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_geometry_kappa.py",
        job_args=["--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy",
        dry_run=args.dry_run,
        phase_id="geometry_kappa",
    )


if __name__ == "__main__":
    main()
