#!/usr/bin/env python3
"""launch_cross_analysis_a4.py -- A4: Feature Importance Stability.

Retrains R0 HistGBDT per scenario with spatial-blocked 5-fold CV,
extracts feature_importances_, computes pairwise Kendall's tau.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). 5 scenarios x 5 folds = 25 model fits.
CPU-bound sklearn on ~200-5000 rows. < 10 min total.
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

    job_name = make_job_name("cross-a4-importance")

    launch_processing_job(
        job_name=job_name,
        job_script="cross_analysis_a4_importance.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="scikit-learn scipy",
        dry_run=args.dry_run,
        phase_id="cross_analysis_a4",
    )


if __name__ == "__main__":
    main()
