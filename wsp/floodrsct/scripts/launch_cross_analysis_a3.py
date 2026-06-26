#!/usr/bin/env python3
"""launch_cross_analysis_a3.py -- A3: Cross-Scenario Transfer Matrix.

Trains R0 HistGBDT on each source scenario, predicts on all targets.
Builds 5x5 transfer matrix. Computes partial Spearman correlation with
Wasserstein distance controlling for source sample size.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Trains 5 models + 20 predictions +
baselines. CPU-bound sklearn on ~200-5000 rows per scenario. < 15 min total.
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

    job_name = make_job_name("cross-a3-transfer")

    launch_processing_job(
        job_name=job_name,
        job_script="cross_analysis_a3_transfer.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="scikit-learn scipy",
        dry_run=args.dry_run,
        phase_id="cross_analysis_a3",
    )


if __name__ == "__main__":
    main()
