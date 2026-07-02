#!/usr/bin/env python3
"""launch_train_r2_temporal.py -- Launch R2 temporal training for one scenario.

Loads R0 folds + R1 supplement + R2 supplement parquets, trains HistGBDT +
Ridge on the full feature set (R0 + spatial + temporal). Same folds/solvers/
targets as R0/R1.

Supports --ablation flag for temporal feature sub-group ablations:
  full (default), no-storm-track, no-rainfall, temporal-only

Resource assumptions
--------------------
Bottleneck: sklearn training on ~400 rows x ~70 features, 3 targets x 3 splits.
CPU-bound, finishes in <10 min. Instance needs headroom for R1+R2 supplement load.

  ml.m5.large:  2 vCPU, 8 GB RAM  -> sufficient

Usage:
    python launch_train_r2_temporal.py --scenario houston --dry-run
    python launch_train_r2_temporal.py --scenario houston
    python launch_train_r2_temporal.py --scenario houston --ablation no-storm-track
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]
ABLATIONS = ["full", "no-storm-track", "no-rainfall", "temporal-only"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--ablation", default="full", choices=ABLATIONS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ablation_slug = args.ablation.replace("-", "")
    job_name = make_job_name(
        f"r2-{ablation_slug}-{args.scenario.replace('_', '-')}"
    )

    job_args = ["--scenario", args.scenario, "--ablation", args.ablation, "--upload"]
    phase_id = "r2_temporal"

    launch_processing_job(
        job_name=job_name,
        job_script="train_r2_temporal.py",
        job_args=job_args,
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="scikit-learn scipy",
        dry_run=args.dry_run,
        phase_id=phase_id,
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
