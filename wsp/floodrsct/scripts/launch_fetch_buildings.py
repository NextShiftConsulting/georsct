#!/usr/bin/env python3
"""launch_fetch_buildings.py -- Extract Overture building footprints per ZCTA.

Downloads building polygons via open-buildings (DuckDB+S3), spatially joins
with ZCTA boundaries, aggregates building_count + total_footprint_area_m2.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). DuckDB Overture queries are
memory-intensive for large bounding boxes; 16 GB handles ~200 ZCTA
scenarios comfortably.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None, choices=SCENARIOS,
                        help="Single scenario (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    suffix = args.scenario[:8] if args.scenario else "all"
    job_name = make_job_name(f"buildings-{suffix}")

    job_args = ["--upload"]
    if args.scenario:
        job_args.extend(["--scenario", args.scenario])

    launch_processing_job(
        job_name=job_name,
        job_script="run_fetch_buildings.py",
        job_args=job_args,
        instance_type="ml.m5.xlarge",
        volume_size_gb=50,
        pip_packages="open-buildings geopandas duckdb shapely pyarrow",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
