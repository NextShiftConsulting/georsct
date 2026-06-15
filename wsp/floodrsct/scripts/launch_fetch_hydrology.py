#!/usr/bin/env python3
"""launch_fetch_hydrology.py -- Extract DEM-derived hydrology features per ZCTA.

Fetches Copernicus GLO-30 DEM from Planetary Computer and computes HAND, TWI,
GFI, SPI within ~1 km of each ZCTA centroid.

IMPORTANT: Run scenarios SEQUENTIALLY (not --all), same as buildings.
The shared cache at processed/shared/zcta_hydrology.parquet has a
read-modify-write pattern that is not safe under parallel writes.

Deployment Resource Review (9 dimensions):
  1. Memory:    DEM tile for a metro bbox (~1-2 deg at 30m GLO-30) is
                3600-7200 px per side. Houston (largest, ~1x1.5 deg) =
                ~20M pixels x 8 bytes = ~150 MB. Flow accumulation +
                4 derived grids (HAND/TWI/GFI/SPI) = ~6 arrays x 150 MB
                = ~900 MB peak. 8 GB (ml.m5.large) is sufficient.
  2. Cache:     Reads/writes processed/shared/zcta_hydrology.parquet.
                Reads raw/geocertdb2026/zcta_features_labels.parquet.
                Cache-first: skips ZCTAs already present.
  3. Threads:   D8 flow accumulation is single-threaded (numpy). 2 vCPU ok.
  4. Image:     PYTORCH_CPU (default). No GPU needed.
  5. Instance:  ml.m5.large (2 vCPU, 8 GB). DEM peak < 1 GB.
  6. Volume:    20 GB. One temp GeoTIFF per scenario (~150 MB).
  7. Pip:       PyPI: geopandas rasterio planetary-computer pystac-client.
                Wheels (mounted from S3): floodcaster, sphere-*.
                NOTE: floodcaster is NOT on PyPI; comes from ecosystem wheels.
  8. Pre-inst:  None. SageMaker PyTorch image includes GDAL.
  9. Timeout:   7200s (default). STAC download + D8 accumulation: expected
                10-30 min per scenario. Houston largest (~30 min).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def _launch_one(scenario: str, dry_run: bool) -> str:
    job_name = make_job_name(f"hydro-{scenario[:8].replace('_', '-')}")
    return launch_processing_job(
        job_name=job_name,
        job_script="run_fetch_hydrology.py",
        job_args=["--scenario", scenario, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=20,
        pip_packages="geopandas rasterio planetary-computer pystac-client",
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch hydrology feature extraction (SageMaker)"
    )
    parser.add_argument("--scenario", default=None, choices=SCENARIOS,
                        help="Single scenario (run one at a time!)")
    parser.add_argument("--all", action="store_true",
                        help="Launch all 5 scenarios SEQUENTIALLY (waits between each)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        parser.error("Specify --scenario or --all")

    if args.all:
        print("WARNING: --all launches sequentially. Use --scenario for single runs.")
        for scenario in SCENARIOS:
            _launch_one(scenario, args.dry_run)
    else:
        _launch_one(args.scenario, args.dry_run)


if __name__ == "__main__":
    main()
