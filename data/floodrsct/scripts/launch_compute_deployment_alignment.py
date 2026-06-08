"""
launch_compute_deployment_alignment.py -- SageMaker launcher for Phase 4e.

Deployment-aligned validation (TWCV): reweights CV losses to match
deployment task distribution. Runs AFTER all training levels complete.

Usage:
    python launch_compute_deployment_alignment.py --scenario houston
    python launch_compute_deployment_alignment.py --all-scenarios
    python launch_compute_deployment_alignment.py --all-scenarios --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_common import launch_processing_job

MODELABLE = [
    "houston",
    "southwest_florida",
    "nyc",
    "riverside_coachella",
    "new_orleans",
]

JOB_SCRIPT = "compute_deployment_alignment.py"
EXTRA_FILES = [
    "_coverage_common.py",
]


def main():
    parser = argparse.ArgumentParser(
        description="Launch deployment alignment (TWCV) on SageMaker"
    )
    parser.add_argument("--scenario", choices=MODELABLE)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upload", action="store_true", default=True)
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
    if args.upload:
        script_args.append("--upload")

    job_name_suffix = "all" if args.all_scenarios else args.scenario

    launch_processing_job(
        job_name=f"s035-deploy-align-{job_name_suffix}",
        script=JOB_SCRIPT,
        extra_files=EXTRA_FILES,
        script_args=script_args,
        instance_type="ml.m5.large",
        volume_size_gb=20,
        pip_packages="georsct scikit-learn scipy pandas numpy pyarrow swarm-auth",
        pre_install_cmd=None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
