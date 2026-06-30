#!/usr/bin/env python3
"""launch_rsct_certificates.py -- RSCT certificate generation for GeoRSCT.

Three modes (mutually exclusive):
  --collapse    Score-collapse demo: R/S/N vs geometry kappa gradient
  --certificate Certificate sidecar per (scenario, target)
  --sweep       Inflation-vs-sigma sweep across all cells

Resource: ml.m5.large (2 vCPU, 8 GB). Ridge CV is cheap; data fits in
~200 MB per scenario parquet. No GPU needed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collapse", action="store_true")
    mode.add_argument("--certificate", action="store_true")
    mode.add_argument("--sweep", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.collapse:
        mode_flag = "--collapse"
        mode_label = "collapse"
    elif args.certificate:
        mode_flag = "--certificate"
        mode_label = "certificate"
    else:
        mode_flag = "--sweep"
        mode_label = "sweep"

    job_name = make_job_name(f"rsct-cert-{mode_label}")

    extra = [str(Path(__file__).parent.parent / "jobs" / "generate_folds.py")]

    launch_processing_job(
        job_name=job_name,
        job_script="compute_rsct_certificates.py",
        job_args=[mode_flag, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scipy scikit-learn",
        extra_files=extra,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
