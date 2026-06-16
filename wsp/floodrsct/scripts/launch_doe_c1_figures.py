#!/usr/bin/env python3
"""launch_doe_c1_figures.py -- Render DOE-C1 paper figures on SageMaker.

Reads existing DOE-C1 result JSONs from S3, renders matplotlib figures,
uploads PDFs back to S3.

Resource: ml.m5.large (2 vCPU, 8 GB). Lightweight rendering job.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch DOE-C1 figure rendering"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("doe-c1-figs")

    launch_processing_job(
        job_name=job_name,
        job_script="render_doe_c1_figures.py",
        job_args=["--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="matplotlib numpy",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
