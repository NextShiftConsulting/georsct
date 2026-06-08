#!/usr/bin/env python3
"""launch_build_zcta_evidence.py -- Phase R4.2: ZCTA text evidence.

Assembles structured text per ZCTA from event features, crosswalk,
and SVI data. One job per scenario. Output feeds VLM assessment.

Resource: ml.m5.large (2 vCPU, 8 GB). Reads parquets, writes text
files to S3. Lightweight.
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

    job_name = make_job_name(f"zcta-evidence-{args.scenario.replace('_', '-')}")

    launch_processing_job(
        job_name=job_name,
        job_script="build_zcta_evidence.py",
        job_args=["--scenario", args.scenario, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
