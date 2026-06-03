#!/usr/bin/env python3
"""launch_kappa_concordance.py -- Concordance test: probe vs yrsn Moran's I.

Verifies the S036 probe's self-contained compute_kappa_spatial() matches
the canonical yrsn implementation on synthetic + edge-case inputs.

Resource: ml.m5.large. Needs yrsn wheel (mounted via _launcher_base).
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

    job_name = make_job_name("kappa-concordance")

    launch_processing_job(
        job_name=job_name,
        job_script="test_kappa_spatial_concordance.py",
        job_args=[],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
