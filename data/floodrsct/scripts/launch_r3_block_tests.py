#!/usr/bin/env python3
"""launch_r3_block_tests.py -- Launch R3_1a block tests for one or all scenarios.

Runs compute_r3_block_tests.py on SageMaker. Trains HistGBDT + Ridge across
6 blocks x 3 targets x 3 splits x N folds per scenario.

Resource assumptions
--------------------
CPU-bound sklearn fits. Largest scenario (houston) is ~few thousand rows x
~60 features. Memory is not a concern.

  ml.m5.xlarge:  4 vCPU, 16 GB RAM  -> sufficient for all scenarios

Usage:
    python launch_r3_block_tests.py --scenario houston --dry-run
    python launch_r3_block_tests.py --scenario houston
    python launch_r3_block_tests.py --all --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]


def launch_one(scenario: str, dry_run: bool) -> str:
    job_name = make_job_name(f"r3-block-tests-{scenario.replace('_', '-')}")
    return launch_processing_job(
        job_name=job_name,
        job_script="compute_r3_block_tests.py",
        job_args=["--scenario", scenario, "--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=30,
        pip_packages="scikit-learn scipy",
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=SCENARIOS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios = SCENARIOS if args.all else [args.scenario]
    for s in scenarios:
        launch_one(s, args.dry_run)


if __name__ == "__main__":
    main()
