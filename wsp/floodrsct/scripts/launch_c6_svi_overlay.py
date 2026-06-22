#!/usr/bin/env python3
"""launch_c6_svi_overlay.py -- C6 artifact: SVI-disagreement overlay.

Re-runs DOE-C1 fit_predict per construct saving per-ZCTA predictions,
computes disagreement, joins with SVI/HIFLD, computes Spearman correlations.

Supports --all to run all 5 scenarios as separate SageMaker jobs.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Fits 5 HistGBDT models per
construct per scenario -- same footprint as DOE-C1.

S3 output:
  results/s035/doe_c1/c6_svi_overlay_{scenario}.json
  results/s035/doe_c1/c6_svi_overlay_{scenario}.parquet
  results/s035/doe_c1/c6_svi_overlay_all.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def _launch_one(scenario: str, dry_run: bool) -> str:
    tag = scenario[:8].replace("_", "-")
    job_name = make_job_name(f"c6-svi-{tag}")

    return launch_processing_job(
        job_name=job_name,
        job_script="run_c6_svi_overlay.py",
        job_args=["--scenario", scenario, "--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=20,
        pip_packages="scipy scikit-learn",
        dry_run=dry_run,
        phase_id=None,  # C6 is an artifact, not a registered experiment phase
        scenario=scenario,
    )


def _launch_all(dry_run: bool) -> str:
    """Launch a single job that runs all 5 scenarios sequentially."""
    job_name = make_job_name("c6-svi-all")

    return launch_processing_job(
        job_name=job_name,
        job_script="run_c6_svi_overlay.py",
        job_args=["--all", "--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=20,
        pip_packages="scipy scikit-learn",
        dry_run=dry_run,
        phase_id=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch C6 SVI-disagreement overlay"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=SCENARIOS,
                       help="Run a single scenario")
    group.add_argument("--all", action="store_true",
                       help="Run ALL 5 scenarios in one job")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        job_name = _launch_all(args.dry_run)
        print()
        print("=" * 60)
        print("C6: Launched all-scenario job: %s" % job_name)
        print("=" * 60)
    else:
        job_name = _launch_one(args.scenario, args.dry_run)
        print()
        print("=" * 60)
        print("C6: Launched %s job: %s" % (args.scenario, job_name))
        print("=" * 60)


if __name__ == "__main__":
    main()
