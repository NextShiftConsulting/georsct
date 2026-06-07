#!/usr/bin/env python3
"""launch_compute_verdicts.py -- Phase 8: Six-geometry sufficiency verdicts.

Reads existing mmar-input JSONs + prediction parquets from S3 to produce
per-geometry verdict table for the paper.

Deployment Resource Review
--------------------------
1. Memory:    Prediction parquets ~5 MB each x 10 (5 scenarios x 2 levels).
              Local JSONs <1 MB each. Peak <2 GB.
2. S3 cache:  Reads prediction parquets + fast_validation.json. No heavy writes.
3. Threads:   Single-threaded. Kendall tau-b is O(n log n) on ~500 ZCTAs.
4. Image:     PyTorch CPU. Needs scipy + pandas.
5. Instance:  ml.m5.large (2 vCPU, 8 GB). Lightweight post-processing.
6. Volume:    10 GB. Outputs are JSON <1 MB.
7. pip:       scipy.
8. pre_install: None.
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

    job_name = make_job_name("verdicts")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_verdicts.py",
        job_args=["--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy",
        dry_run=args.dry_run,
        phase_id="verdicts",
    )


if __name__ == "__main__":
    main()
