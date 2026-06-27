#!/usr/bin/env python3
"""launch_cross_analysis_a7.py -- A7: NFIP Feature Ablation.

Retrains R0 HistGBDT with and without NFIP features. Compares:
  - Within-scenario R2 (measures NFIP dependence)
  - Feature importance stability (Kendall tau, full vs ablated)
  - Cross-scenario transfer (5x5 matrix, full vs ablated)

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Trains 2 arms x 5 scenarios x
5 folds (within) + 2 arms x 20 transfer pairs = 90 model fits total.
CPU-bound sklearn on ~200-5000 rows. < 30 min total.
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

    job_name = make_job_name("cross-a7-nfip-ablation")

    launch_processing_job(
        job_name=job_name,
        job_script="cross_analysis_a7_nfip_ablation.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="scikit-learn scipy",
        dry_run=args.dry_run,
        phase_id="cross_analysis_a7",
    )


if __name__ == "__main__":
    main()
