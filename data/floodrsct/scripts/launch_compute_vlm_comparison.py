#!/usr/bin/env python3
"""launch_compute_vlm_comparison.py -- Phase R4.5: VLM comparison.

Compares VLM risk scores against observed NFIP claims and across
VLMs. Produces R4 money table extension.

Resource: ml.m5.large (2 vCPU, 8 GB). Reads parquets, computes
Spearman correlations. Lightweight.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("vlm-comparison")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_vlm_comparison.py",
        job_args=["--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
