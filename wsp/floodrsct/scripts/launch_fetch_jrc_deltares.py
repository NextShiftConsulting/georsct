#!/usr/bin/env python3
"""
launch_fetch_jrc_deltares.py -- Fetch JRC + Deltares shared features.

Launches a SageMaker Processing job that fetches static flood-hazard
features from Microsoft Planetary Computer for the s035 ZCTA universe:

  - JRC Global Surface Water occurrence (1984-2020)
  - Deltares Global Flood Maps depth at RP 10/50/100

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_jrc_water_occurrence_pct.parquet
  s3://swarm-floodrsct-data/processed/shared/zcta_deltares_depth.parquet

Resource review:
  - Memory: ~794 centroids batched per scenario (~200 each). JRC does
    1 STAC call per scenario bbox, Deltares does 3 (per RP). Peak
    memory ~500 MB. 8 GB is ample.
  - Cache: No prior cache exists on S3 (first run).
  - Threads: Sequential per scenario. STAC calls are I/O-bound.
  - Image: PYTORCH_CPU (default). No GPU needed.
  - Instance: ml.m5.large (2 vCPU, 8 GB). Point sampling, not heavy GIS.
  - Volume: 20 GB. No local raster storage; STAC reads via HTTP.
  - pip_packages: PyPI deps + floodcaster/sphere from local wheels via
    --find-links. This avoids the bootstrap glob (*.whl || true) which
    swallows install errors silently.
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

    # Install floodcaster + sphere from local wheels via --find-links
    # in the SAME pip resolution pass as PyPI deps. The default bootstrap
    # glob (pip install *.whl 2>/dev/null || true) swallows errors.
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_jrc_deltares.py",
        job_args=job_args,
        instance_type="ml.m5.large",
        volume_size_gb=20,
        pip_packages=(
            "geopandas rasterio planetary-computer pystac-client "
            "duckdb shapely pyogrio "
            "--find-links /opt/ml/processing/input/wheels/ "
            "sphere-core sphere-data sphere-flood floodcaster"
        ),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
