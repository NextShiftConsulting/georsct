#!/usr/bin/env python3
"""launch_build_event_dataset.py -- Launch the event dataset assembler for one scenario.

This is the final processing step before Data Lock A/B. Run only after all
raw data pulls have completed for the target scenario.

Data Lock A (June 1): --scenario houston
Data Lock B (June 2): all remaining scenarios

Instance: ml.m5.2xlarge (8 vCPU, 32 GB RAM) — MRMS spatial aggregation is
the bottleneck; cfgrib loads entire grib2 grids into memory.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]

# Larger instance for Houston (3 events × ~18 days MRMS) and SW Florida
_LARGE_SCENARIOS = {"houston", "new_orleans", "southwest_florida", "ar_flood_2023"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    instance_type = (
        "ml.m5.4xlarge" if args.scenario in _LARGE_SCENARIOS else "ml.m5.2xlarge"
    )

    job_name = make_job_name(f"build-events-{args.scenario.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="build_event_dataset.py",
        job_args=["--scenario", args.scenario],
        instance_type=instance_type,
        volume_size_gb=100,  # grib2 scratch space
        pip_packages="geopandas pyogrio rasterio cfgrib xarray eccodes scikit-learn xgboost",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
