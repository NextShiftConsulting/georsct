#!/usr/bin/env python3
"""launch_compute_dgm_routing.py -- Phase 6: DGM routing proof-of-concept.

Applies certificate-driven routing decision tree per cell, compares
recommended arm vs actual best. Requires yrsn wheel (vendored in
jobs/) for MorphType enum.

Resource: ml.m5.large (2 vCPU, 8 GB). Reads certificates + results
from S3, no heavy compute.
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

    job_name = make_job_name("dgm-routing")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_dgm_routing.py",
        job_args=["--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy",
        dry_run=args.dry_run,
        phase_id="dgm_routing",
    )


if __name__ == "__main__":
    main()
