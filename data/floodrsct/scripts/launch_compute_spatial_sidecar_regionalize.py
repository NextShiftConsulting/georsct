#!/usr/bin/env python3
"""launch_compute_spatial_sidecar_regionalize.py -- Regionalization robustness (S3).

Re-runs the locked pipeline under Skater/MaxP regionalization instead of
county blocking. Compares qualitative findings (tab:robustness in paper).

Deployment Resource Review
--------------------------
1. Memory:    Assembled parquet ~20 MB. Adjacency edge list ~1 MB.
              Skater/MaxP on ~600 ZCTAs with 3 features. Training same as R0/R1.
              Peak <4 GB (no geometry loading).
2. S3 cache:  Reads assembled parquet, folds, adjacency, boundaries.
3. Threads:   Single-threaded (subprocess calls to train_r0/r1 are sequential).
4. Image:     PyTorch CPU. Needs spopt + libpysal + geopandas + scikit-learn.
5. Instance:  ml.m5.xlarge (4 vCPU, 16 GB). Training is light but Skater
              needs enough RAM for the Queen weights matrix.
6. Volume:    10 GB.
7. pip:       spopt libpysal geopandas scikit-learn scipy.
8. pre_install: None.
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario", choices=SCENARIOS)
    group.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all_scenarios:
        job_args = ["--all-scenarios", "--upload"]
        label = "region-all"
    else:
        job_args = ["--scenario", args.scenario, "--upload"]
        label = f"region-{args.scenario.replace('_', '-')}"

    job_name = make_job_name(label)

    launch_processing_job(
        job_name=job_name,
        job_script="compute_spatial_sidecar_regionalize.py",
        job_args=job_args,
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="'numpy<2.3' spopt libpysal scikit-learn scipy networkx",
        extra_files=[
            "compute_residual_lisa.py",
            "train_r0_baseline.py",
            "train_r1_hydrology.py",
        ],
        dry_run=args.dry_run,
        scenario=args.scenario if args.scenario else "all",
    )


if __name__ == "__main__":
    main()
