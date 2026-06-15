#!/usr/bin/env python3
"""launch_topology_necessity.py -- DOE-P1: Topology Necessity ablation test.

3-arm ablation (baseline / no_topology / shuffled_topology) per scenario.
Tests whether HAND, GFI, TWI, SPI are load-bearing for flood prediction.

Deployment Resource Review (9 dimensions):
  1. Memory:    3 x HGB fits per fold x ~5 folds. Each HGB with max_bins=32
                on ~300-1500 rows is small (<200 MB per model). Peak during
                data loading + join: ~500 MB. Bootstrap CI: negligible.
                Budget: ~1.5 GB peak per scenario. 8 GB instance sufficient.
  2. Cache:     Reads processed/shared/zcta_hydrology.parquet (shared cache).
                No cache writes. Reads folds/{scenario}_folds.parquet.
  3. Threads:   HGB uses all cores internally. 3 sequential arms, each with
                sequential folds. 4 vCPU sufficient.
  4. Image:     PYTORCH_CPU (default). No GPU needed.
  5. Instance:  ml.m5.large (2 vCPU, 8 GB). Lightweight tabular job.
  6. Volume:    20 GB. Parquet files are small (<50 MB total).
  7. Pip:       scikit-learn numpy pandas pyarrow scipy
  8. Pre-inst:  None (base image sufficient).
  9. Timeout:   3600s (1h). Expected runtime 5-15 min per scenario.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def _launch_one(scenario: str, dry_run: bool) -> str:
    job_name = make_job_name(f"doep1-{scenario[:8].replace('_', '-')}")
    return launch_processing_job(
        job_name=job_name,
        job_script="run_topology_necessity.py",
        job_args=["--scenario", scenario, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=20,
        pip_packages="scikit-learn numpy pandas pyarrow scipy",
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch DOE-P1 topology necessity ablation (SageMaker)"
    )
    parser.add_argument("--scenario", default=None, choices=SCENARIOS,
                        help="Single scenario (required unless --all)")
    parser.add_argument("--all", action="store_true",
                        help="Launch all 5 scenarios as parallel SageMaker jobs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        parser.error("Specify --scenario or --all")

    if args.all:
        for scenario in SCENARIOS:
            _launch_one(scenario, args.dry_run)
    else:
        _launch_one(args.scenario, args.dry_run)


if __name__ == "__main__":
    main()
