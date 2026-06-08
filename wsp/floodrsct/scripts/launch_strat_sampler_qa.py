#!/usr/bin/env python3
"""launch_strat_sampler_qa.py -- Run stratified sampler QA as a SageMaker job.

Validates assembled event parquets across scenarios with random sampling.
Each seed produces a different sample fingerprint for independent verification.

Usage:
    python launch_strat_sampler_qa.py                         # seed=42
    python launch_strat_sampler_qa.py --seed 123              # different sample
    python launch_strat_sampler_qa.py --seed 456 --n-samples 5
    python launch_strat_sampler_qa.py --scenario new_orleans  # single scenario
    python launch_strat_sampler_qa.py --dry-run
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
    parser.add_argument("--scenario", default=None, choices=SCENARIOS,
                        help="Single scenario (default: all available)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stratified sampling")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Number of sample rows per scenario")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_args = ["--seed", str(args.seed), "--n-samples", str(args.n_samples),
                "--verbose"]
    if args.scenario:
        job_args.extend(["--scenario", args.scenario])

    # Write evidence to S3 keyed by seed for reproducibility
    evidence_key = f"evidence/qa/strat_sampler_seed{args.seed}.json"
    job_args.extend(["--json-out", evidence_key])

    suffix = f"seed{args.seed}"
    if args.scenario:
        suffix = f"{args.scenario.replace('_', '-')}-{suffix}"

    job_name = make_job_name(f"qa-strat-{suffix}")
    launch_processing_job(
        job_name=job_name,
        job_script="strat_sampler_qa.py",
        job_args=job_args,
        instance_type="ml.m5.large",  # QA is I/O-bound, small instance fine
        volume_size_gb=20,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
