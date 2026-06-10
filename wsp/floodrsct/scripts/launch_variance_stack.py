"""
launch_variance_stack.py -- SageMaker launcher for the 5-step variance stack.

Runs the full decomposition: unweighted -> coverage gate -> product ->
raking -> raking+shrinkage. Produces per-cell ladder tables and the
multi-panel figure.

Usage:
    python launch_variance_stack.py --dry-run
    python launch_variance_stack.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


JOB_SCRIPT = "run_variance_stack.py"


def main():
    parser = argparse.ArgumentParser(
        description="Launch 5-step variance stack decomposition on SageMaker"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_args = ["--upload"]

    job_name = make_job_name("variance-stack")

    launch_processing_job(
        job_name=job_name,
        job_script=JOB_SCRIPT,
        job_args=script_args,
        instance_type="ml.m5.xlarge",
        volume_size_gb=20,
        pip_packages="scikit-learn scipy matplotlib",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
