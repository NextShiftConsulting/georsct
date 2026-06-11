#!/usr/bin/env python3
"""launch_fetch_depressions.py -- Depression delineation from Copernicus DEM.

Fetches Copernicus GLO-30 DEM per scenario bbox from Planetary Computer,
runs lidar depression delineation (whitebox-tools), aggregates per-ZCTA
depression volume and depth statistics.

Resource: ml.m5.2xlarge (8 vCPU, 32 GB). Rasterio DEM merge + whitebox
depression delineation on city-scale DEMs (~5000x5000 px per scenario)
benefits from extra memory. Whitebox engine is single-threaded per tile,
but we process scenarios sequentially.

Pre-install: whitebox-tools binary needs to be downloaded on first use;
the lidar package handles this automatically. Rasterio requires GDAL
system libraries which the PyTorch SageMaker image provides.
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
    job_name = make_job_name(f"depressions-{suffix}")

    job_args = ["--upload"]
    if args.scenario:
        job_args.extend(["--scenario", args.scenario])

    launch_processing_job(
        job_name=job_name,
        job_script="run_fetch_depressions.py",
        job_args=job_args,
        instance_type="ml.m5.2xlarge",
        volume_size_gb=100,
        pip_packages=(
            "lidar whitebox geopandas rasterio "
            "planetary-computer pystac-client shapely"
        ),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
