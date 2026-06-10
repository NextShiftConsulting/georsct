"""
launch_compute_deployment_alignment.py -- SageMaker launcher for Phase 4e.

Deployment-aligned validation (TWCV): reweights CV losses to match
deployment task distribution. Runs AFTER all training levels complete.

Usage:
    python launch_compute_deployment_alignment.py --all-scenarios --dry-run
    python launch_compute_deployment_alignment.py --all-scenarios
    python launch_compute_deployment_alignment.py --scenario houston
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

MODELABLE = [
    "houston",
    "southwest_florida",
    "nyc",
    "riverside_coachella",
    "new_orleans",
]

JOB_SCRIPT = "compute_deployment_alignment.py"


def main():
    parser = argparse.ArgumentParser(
        description="Launch deployment alignment (TWCV) on SageMaker"
    )
    parser.add_argument("--scenario", choices=MODELABLE)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.scenario and not args.all_scenarios:
        parser.error("Specify --scenario or --all-scenarios")

    # Build CLI arguments for the job script
    script_args = []
    if args.all_scenarios:
        script_args.append("--all-scenarios")
    else:
        script_args.extend(["--scenario", args.scenario])
    script_args.append("--all-levels")
    script_args.append("--upload")

    suffix = "all" if args.all_scenarios else args.scenario
    job_name = make_job_name(f"deploy-align-{suffix}")

    launch_processing_job(
        job_name=job_name,
        job_script=JOB_SCRIPT,
        job_args=script_args,
        instance_type="ml.m5.large",
        volume_size_gb=20,
        pip_packages="scikit-learn scipy",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
