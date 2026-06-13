#!/usr/bin/env python3
"""launch_fetch_depressions.py -- Depression delineation from Copernicus DEM.

Fetches Copernicus GLO-30 DEM per scenario bbox from Planetary Computer,
runs whitebox-tools depression delineation (depth_in_sink + connected-component
labeling), aggregates per-ZCTA depression volume and depth statistics.

Deployment Resource Review (9 dimensions):
  1. Memory:    DEM tile merge is the bottleneck. Each COP-30 tile is
                3601x3601 float32 = ~50 MB. Houston bbox spans ~8 tiles,
                merge -> 10800x10800 = ~445 MB. Whitebox depth_in_sink
                loads DEM + filled DEM + output = ~1.3 GB internal.
                scipy.ndimage.label on depth raster = ~450 MB.
                Budget: ~3 GB peak. 16 GB instance required.
  2. Cache:     Cache-first on s3://swarm-floodrsct-data/processed/shared/
                zcta_depressions.parquet. ZCTAs already in cache are skipped.
  3. Threads:   Whitebox engine is single-threaded per DEM. rasterio merge
                is single-threaded. No parallelism benefit beyond 2 vCPU.
  4. Image:     PYTORCH_CPU (default). rasterio bundles its own GDAL.
  5. Instance:  ml.m5.xlarge (4 vCPU, 16 GB). Whitebox depth_in_sink
                needs ~3x DEM size in memory. 8 GB OOM-killed on Houston
                (10800x10800 merged DEM). 16 GB provides safe headroom.
  6. Volume:    50 GB. DEM tiles ~50 MB each x 8 = 400 MB. Whitebox creates
                intermediate rasters (fill, depth) = ~2x DEM size.
                Budget ~2 GB per scenario on disk.
  7. Pip:       whitebox geopandas rasterio scipy
                planetary-computer pystac-client shapely
  8. Pre-inst:  None. rasterio wheel bundles GDAL. whitebox-tools binary
                auto-downloads on first call.
  9. Timeout:   7200s (2h). DEM fetch ~2-5 min, delineation ~5-15 min
                per scenario. Cache-first means re-runs are fast.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def _launch_one(scenario: str, dry_run: bool) -> str:
    job_name = make_job_name(f"depress-{scenario[:8].replace('_', '-')}")
    return launch_processing_job(
        job_name=job_name,
        job_script="run_fetch_depressions.py",
        job_args=["--scenario", scenario, "--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=50,
        pip_packages=(
            "whitebox geopandas rasterio scipy "
            "planetary-computer pystac-client shapely"
        ),
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
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
