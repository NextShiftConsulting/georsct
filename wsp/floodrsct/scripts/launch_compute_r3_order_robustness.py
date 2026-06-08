#!/usr/bin/env python3
"""launch_compute_r3_order_robustness.py -- Phase R3_1b: Order robustness.

Permutation test: does block admission order affect R3 outcome?
Forward + reverse + 20 random orderings.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Model fitting across orderings.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"r3-order-robust-{args.scenario.replace('_', '-')}")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_r3_order_robustness.py",
        job_args=["--scenario", args.scenario, "--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=30,
        pip_packages="scipy scikit-learn",
        dry_run=args.dry_run,
        phase_id="r3_order_robustness",
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
