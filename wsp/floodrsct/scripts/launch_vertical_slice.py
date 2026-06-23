#!/usr/bin/env python3
"""launch_vertical_slice.py -- Launch V8 vertical slice on SageMaker.

Resource: ml.m5.2xlarge (8 vCPU, 32 GB).
TJEPA pretraining per fold + Gate 4-GC permutation null.
Estimated runtime: ~60 min.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name, PYTORCH_CPU

SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch V8 vertical slice on SageMaker"
    )
    parser.add_argument("--scenario", default="houston", choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tag = args.scenario[:8].replace("_", "-")
    job_name = make_job_name(f"vslice-{tag}")

    job_name = launch_processing_job(
        job_name=job_name,
        job_script="run_vertical_slice.py",
        job_args=["--scenario", args.scenario, "--upload"],
        instance_type="ml.m5.2xlarge",
        volume_size_gb=30,
        pip_packages="scipy scikit-learn",
        image_uri=PYTORCH_CPU,  # TJEPA/MAE need torch (pre-installed)
        dry_run=args.dry_run,
        phase_id=None,  # Vertical slice is a one-off, not a registered phase
        scenario=args.scenario,
    )

    print()
    print("=" * 60)
    print("VERTICAL SLICE: Launched %s job: %s" % (args.scenario, job_name))
    print("=" * 60)


if __name__ == "__main__":
    main()
