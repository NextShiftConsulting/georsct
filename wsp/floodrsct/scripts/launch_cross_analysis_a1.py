#!/usr/bin/env python3
"""launch_cross_analysis_a1.py -- A1: Prithvi vs Tabular Predictive Utility.

PCA-reduces Prithvi embeddings, trains R0-only vs R0+Prithvi HistGBDT with
spatial-blocked 5-fold CV. Paired t-test on per-fold R2 delta.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). 5 scenarios x 5 folds x 2 arms = 50
model fits. Prithvi parquet load (~1024 cols) needs ~2 GB. < 20 min total.
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

    job_name = make_job_name("cross-a1-prithvi")

    launch_processing_job(
        job_name=job_name,
        job_script="cross_analysis_a1_prithvi.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="scikit-learn scipy",
        dry_run=args.dry_run,
        phase_id="cross_analysis_a1",
    )


if __name__ == "__main__":
    main()
