#!/usr/bin/env python3
"""launch_build_r2_features.py -- Launch R2 temporal feature builder for one scenario.

Computes per-ZCTA temporal statistics from MRMS grib2, HRRR QPF grib2,
NOAA tide gauges, and HURDAT2 storm tracks. Produces the R2 supplement
parquet that the training scripts merge with the assembled event features.

Resource assumptions
--------------------
Bottleneck: MRMS grib2 decode via ProcessPoolExecutor. Each CONUS grid
decompresses to ~100 MB. HRRR grids are ~100 MB each.

  ml.m5.2xlarge:  8 vCPU, 32 GB RAM  -> 4 workers x 100 MB = safe
  ml.m5.4xlarge: 16 vCPU, 64 GB RAM  -> 8 workers (use for large scenarios)

Usage:
    python launch_build_r2_features.py --scenario houston --dry-run
    python launch_build_r2_features.py --scenario houston --upload
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]

_LARGE_SCENARIOS = {"houston", "southwest_florida", "riverside_coachella"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    instance_type = (
        "ml.m5.4xlarge" if args.scenario in _LARGE_SCENARIOS else "ml.m5.2xlarge"
    )

    job_name = make_job_name(f"build-r2-{args.scenario.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="build_r2_features.py",
        job_args=["--scenario", args.scenario, "--upload"],
        instance_type=instance_type,
        volume_size_gb=50,
        pip_packages="cfgrib xarray eccodes geopandas",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
