#!/usr/bin/env python3
"""launch_omega_bootstrap.py -- DOE-C2a: Omega bootstrap per construct.

Launches SageMaker jobs to compute distributional reliability (omega)
for each construct via block bootstrap resampling of DOE-C1 certification.

Supports --all to launch all 5 scenarios as separate SageMaker jobs.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Each job runs B=50 bootstrap
iterations per available construct (~3-5 constructs per scenario).
Expected runtime: ~2-3 hours per scenario.

S3 output convention:
  results/s035/doe_c2a/omega_bootstrap_{scenario}.json
  results/s035/doe_c2a/cache/bootstrap_samples_{scenario}.parquet
  results/s035/doe_c2a/cache/omega_table_{scenario}.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def _launch_one(
    scenario: str,
    n_bootstrap: int,
    dry_run: bool,
) -> str:
    """Launch a single scenario job."""
    tag = scenario[:8].replace("_", "-")
    job_name = make_job_name("omega-boot-%s" % tag)

    job_args = [
        "--scenario", scenario,
        "--n-bootstrap", str(n_bootstrap),
        "--upload",
    ]

    return launch_processing_job(
        job_name=job_name,
        job_script="run_omega_bootstrap.py",
        job_args=job_args,
        instance_type="ml.m5.xlarge",
        volume_size_gb=20,
        pip_packages="scipy scikit-learn",
        extra_files=["run_five_construct_divergence.py"],
        dry_run=dry_run,
        phase_id="omega_bootstrap",
        scenario=scenario,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch DOE-C2a: Omega bootstrap per construct"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=SCENARIOS,
                       help="Run a single scenario")
    group.add_argument("--all", action="store_true",
                       help="Run ALL 5 scenarios (separate jobs)")
    parser.add_argument("--n-bootstrap", type=int, default=50,
                        help="Bootstrap iterations per construct (default: 50)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        jobs = []
        for scenario in SCENARIOS:
            job_name = _launch_one(scenario, args.n_bootstrap, args.dry_run)
            jobs.append((scenario, job_name))
        print()
        print("=" * 60)
        print("DOE-C2a: Launched %d omega bootstrap jobs (B=%d)" % (len(jobs), args.n_bootstrap))
        print("=" * 60)
        for scenario, jn in jobs:
            print("  %-25s  %s" % (scenario, jn))
        print("=" * 60)
    else:
        _launch_one(args.scenario, args.n_bootstrap, args.dry_run)


if __name__ == "__main__":
    main()
