#!/usr/bin/env python3
"""launch_cross_analysis_a6.py -- A6: Coverage Gap Geographic Overlap.

Fisher's exact test on Prithvi/hydrology gap overlap per scenario.
Jaccard index and descriptive geographic characterization.

Resource: ml.m5.large (2 vCPU, 8 GB). No model training -- pure join +
contingency table. Loads 5 parquets + metadata JSONs. < 5 min total.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("cross-a6-coverage")

    launch_processing_job(
        job_name=job_name,
        job_script="cross_analysis_a6_coverage.py",
        job_args=["--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy",
        dry_run=args.dry_run,
        phase_id="cross_analysis_a6",
    )


if __name__ == "__main__":
    main()
