#!/usr/bin/env python3
"""launch_compute_r3_money_table.py -- Phase R3_3: R3 money table.

Produces extended money table (R0-R3) + H5-H8 hypothesis tests with
bootstrap CIs. Pure computation on results JSONs.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Statistics computation only.
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

    job_name = make_job_name("r3-money-table")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_r3_money_table.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=30,
        pip_packages="scipy scikit-learn",
        dry_run=args.dry_run,
        phase_id="r3_money_table",
    )


if __name__ == "__main__":
    main()
