#!/usr/bin/env python3
"""launch_build_nfip_historical.py -- Build temporally-gated NFIP historical features.

Aggregates all NFIP claims with dateOfLoss < event start date per ZCTA,
producing nfip_historical_frequency and nfip_historical_severity.  Must run
BEFORE any R0/R1/R2 training jobs (the training scripts load the output).

Resource: ml.m5.large (2 vCPU, 8 GB). Job is pandas groupby on ~50K claims,
finishes in <2 min.

Usage:
    python launch_build_nfip_historical.py --scenario houston --dry-run
    python launch_build_nfip_historical.py --scenario houston
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"build-nfip-hist-{args.scenario.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="build_nfip_historical.py",
        job_args=["--scenario", args.scenario, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
