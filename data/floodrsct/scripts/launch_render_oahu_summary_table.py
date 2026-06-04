#!/usr/bin/env python3
"""launch_render_oahu_summary_table.py -- Render Oahu per-ZCTA summary table.

LaTeX table + JSON from existing kappa_spatial probe output and Floodcaster
building results. Appendix companion to the HAZUS depth-damage figure.

Deployment Resource Review
--------------------------
1. Memory:    Oahu parquet ~5 MB, centroids <1 KB, residuals CSV <3 KB. Trivial.
2. S3 cache:  Reads pre-computed kappa_spatial output (evidence + residuals CSV).
3. Threads:   Single-threaded. Haversine assignment is vectorized numpy.
4. Image:     PyTorch CPU. Only needs pandas + numpy (included in base image).
5. Instance:  ml.m5.xlarge (4 vCPU, 16 GB).
6. Volume:    10 GB. No large downloads.
7. pip:       None beyond base image (pandas, numpy, boto3 already present).
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

    job_name = make_job_name("render-oahu-summary-table")

    launch_processing_job(
        job_name=job_name,
        job_script="render_oahu_summary_table.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
