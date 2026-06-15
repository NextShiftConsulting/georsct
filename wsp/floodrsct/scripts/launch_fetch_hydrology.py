#!/usr/bin/env python3
"""launch_fetch_hydrology.py -- Extract DEM-derived hydrology features per ZCTA.

Fetches Copernicus GLO-30 DEM from Planetary Computer and computes HAND, TWI,
GFI, SPI within ~1 km of each ZCTA centroid.

IMPORTANT: Run scenarios SEQUENTIALLY (not --all), same as buildings.
The shared cache at processed/shared/zcta_hydrology.parquet has a
read-modify-write pattern that is not safe under parallel writes.

Deployment Resource Review (9 dimensions):
  1. Memory:    DEM tile for a metro bbox (~1 deg x 1 deg at 30m) is
                ~12,000 x 12,000 pixels = ~1.1 GB float64. Flow accumulation
                doubles this. Peak ~3 GB for large metros (Houston 6-county).
                Budget: 8 GB for small metros, 16 GB for Houston/NYC.
  2. Cache:     Reads/writes processed/shared/zcta_hydrology.parquet.
                Reads raw/geocertdb2026/zcta_features_labels.parquet.
  3. Threads:   D8 flow accumulation is single-threaded (numpy). 2 vCPU ok.
  4. Image:     PYTORCH_CPU (default). No GPU needed.
  5. Instance:  ml.m5.xlarge (4 vCPU, 16 GB). Needed for DEM tile in memory.
  6. Volume:    30 GB. DEM temp files can be large.
  7. Pip:       floodcaster planetary-computer pystac-client rasterio
                numpy pandas pyarrow scipy scikit-image
  8. Pre-inst:  GDAL (included in SageMaker base image).
  9. Timeout:   7200s (2h). DEM download + D8 accumulation can be slow for
                large bboxes. Houston (largest) expected ~30-60 min.
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
        instance_type="ml.m5.xlarge",
        volume_size_gb=30,
        pip_packages="floodcaster planetary-computer pystac-client rasterio numpy pandas pyarrow scipy scikit-image",
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
