#!/usr/bin/env python3
"""launch_compute_uplift_table.py -- Phase 5: money table + hypothesis tests.

Loads R0/R1/R2 results + kappa diagnostics, produces the money table,
Wilcoxon signed-rank tests (H2a), and exploratory cell-level associations.
Requires rsct wheel (vendored in jobs/).

Resource: ml.m5.large (2 vCPU, 8 GB). Pure computation on results JSONs.
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

    job_name = make_job_name("uplift-table")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_uplift_table.py",
        job_args=["--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy scikit-learn",
        dry_run=args.dry_run,
        phase_id="uplift_table",
    )


if __name__ == "__main__":
    main()
