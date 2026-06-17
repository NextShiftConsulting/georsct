#!/usr/bin/env python3
"""launch_temporal_prior.py -- DOE-C2b: Sequential certification with P16 hint.

Launches SageMaker jobs for temporal prior experiment.
Houston (3 events) and NYC (4 events).

Resource: ml.m5.large (2 vCPU, 8 GB). Small experiment (~30 certs total).
Expected runtime: ~15 min per scenario.

S3 output convention:
  results/s035/doe_c2b/temporal_prior_{scenario}.json
  results/s035/doe_c2b/cache/sequential_certificates_{scenario}.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


SCENARIOS = ["houston", "nyc"]


def _launch_one(
    scenario: str,
    dry_run: bool,
) -> str:
    """Launch a single scenario job."""
    tag = scenario[:8].replace("_", "-")
    job_name = make_job_name("temp-prior-%s" % tag)

    job_args = [
        "--scenario", scenario,
        "--upload",
    ]

    return launch_processing_job(
        job_name=job_name,
        job_script="run_temporal_prior.py",
        job_args=job_args,
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy scikit-learn joblib",
        extra_files=["run_five_construct_divergence.py"],
        dry_run=dry_run,
        phase_id="temporal_prior",
        scenario=scenario,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch DOE-C2b: Temporal prior -- sequential certification"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=SCENARIOS,
                       help="Run a single scenario")
    group.add_argument("--all", action="store_true",
                       help="Run both scenarios (separate jobs)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        jobs = []
        for scenario in SCENARIOS:
            job_name = _launch_one(scenario, args.dry_run)
            jobs.append((scenario, job_name))
        print()
        print("=" * 60)
        print("DOE-C2b: Launched %d temporal prior jobs" % len(jobs))
        print("=" * 60)
        for scenario, jn in jobs:
            print("  %-25s  %s" % (scenario, jn))
        print("=" * 60)
    else:
        _launch_one(args.scenario, args.dry_run)


if __name__ == "__main__":
    main()
