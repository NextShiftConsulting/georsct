#!/usr/bin/env python3
"""launch_train_surrogate.py -- Launch surrogate model training (fallback if MaxFloodCast unavailable).

Decision gate: May 30. If Lee et al. (lipai.huang@tamu.edu) have not responded
by end-of-day May 30, run this script for each scenario.

Instance: ml.g5.2xlarge (1x A10G GPU, 32 GB RAM, ~$1.01/hr on-demand).
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
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_args = ["--scenario", args.scenario]
    if args.skip_lstm:
        job_args.append("--skip-lstm")

    job_name = make_job_name(f"train-surrogate-{args.scenario.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="train_surrogate.py",
        job_args=job_args,
        instance_type="ml.g5.2xlarge",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
