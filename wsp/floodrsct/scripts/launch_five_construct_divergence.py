#!/usr/bin/env python3
"""launch_five_construct_divergence.py -- Five-construct divergence harness.

Certifies the same geography under each of five flood constructs and
computes pairwise certificate distance to produce a 5x5 divergence matrix.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Fits 5 HistGBDT models per run,
computes Gabriel graphs for kappa_reconstruct per construct.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--event", default=None,
                        help="Optional event filter (e.g., harvey2017)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"5construct-{args.scenario[:8]}")

    job_args = ["--scenario", args.scenario, "--upload"]
    if args.event:
        job_args.extend(["--event", args.event])

    launch_processing_job(
        job_name=job_name,
        job_script="run_five_construct_divergence.py",
        job_args=job_args,
        instance_type="ml.m5.xlarge",
        volume_size_gb=20,
        pip_packages="scipy scikit-learn",
        dry_run=args.dry_run,
        phase_id="five_construct_divergence",
    )


if __name__ == "__main__":
    main()
