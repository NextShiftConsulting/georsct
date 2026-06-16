#!/usr/bin/env python3
"""launch_five_construct_divergence.py -- Five-construct divergence harness.

Certifies the same geography under each of five flood constructs and
computes pairwise certificate distance to produce a 5x5 divergence matrix.

Supports --all to launch all 5 scenarios as separate SageMaker jobs.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Fits 5 HistGBDT models per run,
computes Gabriel graphs for kappa_reconstruct per construct.

S3 output convention:
  results/s035/doe_c1/five_construct_{scenario}.json
  results/s035/doe_c1/cache/certificates_{scenario}.parquet
  results/s035/doe_c1/cache/pairwise_{scenario}.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def _launch_one(scenario: str, event: str | None, dry_run: bool) -> str:
    """Launch a single scenario job."""
    job_name = make_job_name(f"5construct-{scenario[:8]}")

    job_args = ["--scenario", scenario, "--upload"]
    if event:
        job_args.extend(["--event", event])

    return launch_processing_job(
        job_name=job_name,
        job_script="run_five_construct_divergence.py",
        job_args=job_args,
        instance_type="ml.m5.xlarge",
        volume_size_gb=20,
        pip_packages="scipy scikit-learn",
        dry_run=dry_run,
        phase_id="five_construct_divergence",
        scenario=scenario,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch five-construct divergence (DOE-C1)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=SCENARIOS,
                       help="Run a single scenario")
    group.add_argument("--all", action="store_true",
                       help="Run ALL 5 scenarios (separate jobs)")
    parser.add_argument("--event", default=None,
                        help="Optional event filter (single-scenario only)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all and args.event:
        parser.error("--event cannot be used with --all")

    if args.all:
        jobs = []
        for scenario in SCENARIOS:
            job_name = _launch_one(scenario, event=None, dry_run=args.dry_run)
            jobs.append((scenario, job_name))
        print()
        print("=" * 60)
        print("DOE-C1: Launched %d scenario jobs" % len(jobs))
        print("=" * 60)
        for scenario, jn in jobs:
            print("  %-25s  %s" % (scenario, jn))
        print("=" * 60)
    else:
        _launch_one(args.scenario, args.event, args.dry_run)


if __name__ == "__main__":
    main()
