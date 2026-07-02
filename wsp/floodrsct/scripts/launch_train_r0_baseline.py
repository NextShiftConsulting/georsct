#!/usr/bin/env python3
"""launch_train_r0_baseline.py -- Launch R0 baseline experiment (folds + training).

Single SageMaker job: generates folds, trains HistGBDT + Ridge on R0 features
across 3 targets x 3 splits. Uploads folds parquet + results JSON.

Supports --ablation flag for R0 feature sub-group ablations:
  full (default), no-fema, no-r0-hydro

Resource: ml.m5.large (2 vCPU, 8 GB). Job is CPU-bound sklearn on ~400 rows,
finishes in <5 min. No parallelism needed at this scale.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]
ABLATIONS = ["full", "no-fema", "no-r0-hydro"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--ablation", default="full", choices=ABLATIONS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ablation_slug = args.ablation.replace("-", "")
    if args.ablation == "full":
        job_name = make_job_name(f"r0-baseline-{args.scenario.replace('_', '-')}")
    else:
        job_name = make_job_name(
            f"r0-{ablation_slug}-{args.scenario.replace('_', '-')}"
        )

    # generate_folds.py is imported by train_r0_baseline.py but not
    # underscore-prefixed, so it isn't auto-included by upload_code.
    extra = [str(Path(__file__).parent.parent / "jobs" / "generate_folds.py")]

    job_args = ["--scenario", args.scenario, "--ablation", args.ablation, "--upload"]
    phase_id = "r0_baseline"

    launch_processing_job(
        job_name=job_name,
        job_script="train_r0_baseline.py",
        job_args=job_args,
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scikit-learn scipy",
        extra_files=extra,
        dry_run=args.dry_run,
        phase_id=phase_id,
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
