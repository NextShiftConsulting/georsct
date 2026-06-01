#!/usr/bin/env python3
"""launch_run_fast_zcta.py -- Run FEMA FAST depth-damage analysis per ZCTA.

Loads NSI structures + depth rasters, runs Hazus via sphere (through
floodcaster.engine), aggregates to ZCTA. Produces 6 FAST features per ZCTA.

Prerequisites (must be on S3 before launch):
  1. NSI structures: run launch_fetch_nsi_structures.py --all
  2. ZCTA boundaries: run launch_stage_zcta_geometry.py
  3. FloodSimBench/SLOSH rasters: already staged

Resource: ml.m5.xlarge (CPU-bound Hazus, ~10 min per scenario per return period).
Volume: 50 GB (multi-tile raster merge + NSI structures).

NOTE: floodcaster + sphere must be available as vendored wheels in jobs/*.whl
or pip-installable. The launcher installs geopandas + rasterio for spatial ops.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = ["houston", "nyc", "southwest_florida"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--return-period",
                        help="e.g. 100yr, cat3 (default: all)")
    parser.add_argument("--all-return-periods", action="store_true",
                        help="Run all return periods (default for paper)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.return_period and not args.all_return_periods:
        args.all_return_periods = True

    suffix = args.scenario.replace("_", "-")
    if args.return_period:
        suffix += f"-{args.return_period}"
    job_name = make_job_name(f"fast-zcta-{suffix}")

    job_args = ["--scenario", args.scenario, "--upload"]
    if args.all_return_periods:
        job_args.append("--all-return-periods")
    elif args.return_period:
        job_args.extend(["--return-period", args.return_period])

    launch_processing_job(
        job_name=job_name,
        job_script="run_fast_zcta.py",
        job_args=job_args,
        instance_type="ml.m5.xlarge",
        volume_size_gb=50,
        pip_packages="geopandas rasterio",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
