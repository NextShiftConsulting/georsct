#!/usr/bin/env python3
"""launch_adversarial_reconstruct.py -- Adversarial kappa_reconstruct harness.

Runs the graded W-matrix corruption ladder per scenario:
  baseline -> 10% -> 25% -> 50% -> 75% -> 100% scramble
with discriminant tests (intercept or ladder mode).

Resource: ml.m5.2xlarge (8 vCPU, 32 GB).  HistGBDT refitting across
5 corruption levels x 25 seeds = 125 refit cycles per scenario.

Usage:
    python launch_adversarial_reconstruct.py --scenario houston --dry-run
    python launch_adversarial_reconstruct.py --scenario houston
    python launch_adversarial_reconstruct.py --all --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def launch_one(scenario: str, dry_run: bool, n_permutations: int) -> str:
    job_name = make_job_name(f"adversarial-{scenario[:8]}")

    return launch_processing_job(
        job_name=job_name,
        job_script="adversarial_reconstruct.py",
        job_args=[
            "--scenario", scenario,
            "--n-permutations", str(n_permutations),
            "--corruption-levels", "0.1", "0.25", "0.5", "0.75", "1.0",
            "--test-mode", "ladder",
            "--n-bootstrap", "2000",
            "--upload",
        ],
        instance_type="ml.m5.2xlarge",
        volume_size_gb=30,
        pip_packages="scipy scikit-learn",
        dry_run=dry_run,
        phase_id="kappa_reconstruct",
        scenario=scenario,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch adversarial kappa_reconstruct harness on SageMaker"
    )
    parser.add_argument("--scenario", choices=SCENARIOS, default=None,
                        help="Single scenario to launch")
    parser.add_argument("--all", action="store_true",
                        help="Launch all scenarios")
    parser.add_argument("--n-permutations", type=int, default=25,
                        help="Seeds per corruption level (default: 25)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        parser.error("Provide --scenario or --all")

    scenarios = SCENARIOS if args.all else [args.scenario]

    for scenario in scenarios:
        launch_one(scenario, args.dry_run, args.n_permutations)


if __name__ == "__main__":
    main()
