#!/usr/bin/env python3
"""launch_compute_r3_block_admission.py -- Phase R3_1c: DGM block admission.

Applies SequentialGatekeeper with GEOSPATIAL_CONUS27 preset to block
certificates. Produces (EnforcementDecision, GearState) per block.

Prerequisite: kappa pipeline bug (P1) must be fixed.
Requires: yrsn + yrsn-controlplane wheels.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Gate evaluation, lightweight.
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

    job_name = make_job_name("r3-block-admission")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_r3_block_admission.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=30,
        pip_packages="scipy",
        dry_run=args.dry_run,
        phase_id="r3_block_admission",
    )


if __name__ == "__main__":
    main()
