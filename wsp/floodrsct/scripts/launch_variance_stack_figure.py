"""
launch_variance_stack_figure.py -- SageMaker launcher for variance stack figure.

Reads variance_stack_results.json from S3 and renders fig_variance_stack.pdf.

Usage:
    python launch_variance_stack_figure.py --dry-run
    python launch_variance_stack_figure.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


JOB_SCRIPT = "run_variance_stack_figure.py"


def main():
    parser = argparse.ArgumentParser(
        description="Launch variance stack figure generation on SageMaker"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_args = ["--upload"]

    job_name = make_job_name("vstack-figure")

    launch_processing_job(
        job_name=job_name,
        job_script=JOB_SCRIPT,
        job_args=script_args,
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scikit-learn matplotlib",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
