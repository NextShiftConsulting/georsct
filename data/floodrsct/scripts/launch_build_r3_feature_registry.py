#!/usr/bin/env python3
"""launch_build_r3_feature_registry.py -- Phase R3_0: Feature registry.

Collects R0-R2 certificates + diagnostics into per-block feature registry
and candidate graph. Pure metadata assembly, no model training.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). S3 reads only.
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

    job_name = make_job_name("r3-feature-registry")

    launch_processing_job(
        job_name=job_name,
        job_script="build_r3_feature_registry.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=30,
        dry_run=args.dry_run,
        phase_id="r3_feature_registry",
    )


if __name__ == "__main__":
    main()
