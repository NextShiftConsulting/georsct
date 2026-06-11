#!/usr/bin/env python3
"""
launch_fetch_jrc_deltares.py -- Fetch JRC + Deltares shared features.

Launches a SageMaker Processing job that fetches static flood-hazard
features from Microsoft Planetary Computer for all ~794 ZCTA centroids:

  - JRC Global Surface Water occurrence (1984-2020)
  - Deltares Global Flood Maps depth at RP 10/50/100

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_jrc_water_occurrence_pct.parquet
  s3://swarm-floodrsct-data/processed/shared/zcta_deltares_depth.parquet

Resource review:
  - Memory: ~794 centroids, single-tile raster sampling per source.
    JRC does 1 STAC call (full bbox), Deltares does 3 (per RP).
    Peak memory ~500 MB (one raster tile in memory). 8 GB is ample.
  - Cache: No prior cache exists on S3 (first run).
  - Threads: Single-threaded (STAC calls are I/O-bound but sequential
    within floodcaster; parallelism at centroid-loop level is in-process).
  - Image: PYTORCH_CPU (default). No GPU needed.
  - Instance: ml.m5.large (2 vCPU, 8 GB). Point sampling, no heavy GIS.
  - Volume: 20 GB. No local raster storage; STAC reads via HTTP.
  - pip_packages: rasterio + planetary-computer + pystac-client + duckdb +
    geopandas + shapely (floodcaster transitive deps not in wheel).
  - pre_install_cmd: None. rasterio works without system GDAL on PyTorch image.
  - Timeout: Default (7200s). Expect ~10-20 min for 794 centroids.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch JRC + Deltares shared feature fetch"
    )
    parser.add_argument("--jrc-only", action="store_true",
                        help="Fetch JRC only")
    parser.add_argument("--deltares-only", action="store_true",
                        help="Fetch Deltares only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without launching")
    args = parser.parse_args()

    job_args = []
    suffix = "jrc-deltares"
    if args.jrc_only:
        job_args.append("--jrc-only")
        suffix = "jrc"
    elif args.deltares_only:
        job_args.append("--deltares-only")
        suffix = "deltares"

    job_name = make_job_name(f"fetch-{suffix}")

    launch_processing_job(
        job_name=job_name,
        job_script="fetch_jrc_deltares.py",
        job_args=job_args,
        instance_type="ml.m5.large",
        volume_size_gb=20,
        pip_packages=(
            "geopandas rasterio planetary-computer pystac-client "
            "duckdb shapely pyogrio"
        ),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
