#!/usr/bin/env python3
"""launch_build_r1_features.py -- Launch R1 spatial feature builder for one scenario.

Computes per-ZCTA spatial hydrology features from NHDPlus catchments,
USACE levees, and other infrastructure layers. Produces the R1 supplement
parquet that the training scripts merge with the assembled event features.

Resource assumptions
--------------------
Primary operation: spatial joins (NHDPlus catchments + ZCTA centroids).
NHDPlus catchment GeoParquet can be 200-500 MB per VPU. Spatial joins
are CPU-bound but not parallelized (single-threaded geopandas sjoin).

  ml.m5.xlarge:  4 vCPU, 16 GB RAM  -> sufficient for most scenarios
  ml.m5.2xlarge: 8 vCPU, 32 GB RAM  -> for large NHDPlus VPUs

Usage:
    python launch_build_r1_features.py --scenario houston --dry-run
    python launch_build_r1_features.py --scenario houston
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

    job_name = make_job_name(f"build-r1-{args.scenario.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="build_r1_features.py",
        job_args=["--scenario", args.scenario, "--upload"],
        instance_type="ml.m5.2xlarge",
        volume_size_gb=50,
        pip_packages="geopandas pyogrio rasterio",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
