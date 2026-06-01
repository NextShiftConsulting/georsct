#!/usr/bin/env python3
"""launch_train_r1_hydrology.py -- Launch R1 hydrology training for one scenario.

Loads R0 folds + R1 supplement parquet, trains HistGBDT + Ridge on the
expanded spatial feature set. Same folds/solvers/targets as R0.

Resource assumptions
--------------------
Bottleneck: sklearn training on ~400 rows x ~60 features, 3 targets x 3 splits.
CPU-bound, finishes in <10 min. Instance needs headroom for R1 supplement load.

  ml.m5.xlarge:  4 vCPU, 16 GB RAM  -> sufficient

Usage:
    python launch_train_r1_hydrology.py --scenario houston --dry-run
    python launch_train_r1_hydrology.py --scenario houston
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"r1-hydrology-{args.scenario.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="train_r1_hydrology.py",
        job_args=["--scenario", args.scenario, "--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="scikit-learn scipy",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
