#!/usr/bin/env python3
"""launch_build_event_dataset.py -- Launch the event dataset assembler for one scenario.

This is the final processing step before Data Lock A/B. Run only after all
raw data pulls have completed for the target scenario.

Data Lock A (June 1): --scenario houston
Data Lock B (June 2): all remaining scenarios

Deployment Resource Review (8 dimensions)
------------------------------------------
1. Memory:    64 GB (4xlarge) / 32 GB (2xlarge). MRMS: 4-8 workers x 100 MB.
              NLCD rasters read via rasterio.sample() at centroids (~1 MB).
              SLOSH MOM: sampled, no full-raster load.
2. S3 cache:  cropland_pct cached at processed/shared/zcta_cropland_pct.parquet.
              impervious_pct cached at processed/shared/zcta_impervious_pct.parquet.
              First run computes from raster; subsequent runs hit cache.
3. Threads:   MRMS: ProcessPoolExecutor, vCPU/2 workers. Raster sampling:
              single-threaded per-centroid loop.
4. Image:     PyTorch 2.5.1 CPU (SageMaker-managed). Spatial packages
              (rasterio, geopandas, cfgrib) pip-installed at boot (~3 min).
5. Instance:  ml.m5.4xlarge for large MRMS scenarios (houston, new_orleans,
              southwest_florida, riverside_coachella). ml.m5.2xlarge for nyc.
6. Volume:    50 GB. Grib2 streamed from S3. NLCD .tif: 1.15 GB download.
              Parquet output: <100 MB. Peak ~3 GB.
7. pip:       geopandas pyogrio rasterio cfgrib xarray eccodes scikit-learn xgboost
              pystac-client planetary-computer.
              rasterio required for NLCD raster read (cropland_pct, impervious_pct).
              pystac-client + planetary-computer required for floodcaster STAC
              extraction (Deltares depth, JRC water, DEM hydrology, Sentinel-1 SAR).
8. pre_install: None when NLCD .tif exists on S3. gdal-bin only needed if
              falling back to .img format (--nlcd-img-fallback flag).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]

# ml.m5.4xlarge (16 vCPU, 64 GB) for scenarios with high MRMS file counts:
#   houston:             3 events x ~540 grib2 files
#   southwest_florida:   3 events x ~525 grib2 files
#   new_orleans:         1 event but large HRRR grid overlay
#   ar_flood_2023:       21-day window = ~504 grib2 files
_LARGE_SCENARIOS = {"houston", "new_orleans", "southwest_florida", "riverside_coachella"}

# Scenarios that sample NLCD impervious surface raster.
# The pre-converted .tif (812 MB) on S3 is preferred over the raw .img (26 GB).
# gdal-bin is only needed if the .tif is missing and we fall back to .img.
# With .tif present, volume can stay at 50 GB and gdal-bin install is skipped.
_NLCD_SCENARIOS = {"houston", "nyc", "new_orleans", "riverside_coachella", "southwest_florida"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    instance_type = (
        "ml.m5.4xlarge" if args.scenario in _LARGE_SCENARIOS else "ml.m5.2xlarge"
    )

    # NLCD: prefer pre-converted .tif (812 MB) over raw .img (26 GB).
    # gdal-bin is only needed if the .tif is missing and we fall back to .img
    # conversion. With .tif present, skip gdal-bin and use 50 GB volume.
    # Set --nlcd-img-fallback to force .img path (adds gdal-bin + 100 GB volume).
    needs_img_fallback = getattr(args, "nlcd_img_fallback", False)
    pre_install = None
    if needs_img_fallback:
        pre_install = (
            "apt-get update -qq && apt-get install -y -qq gdal-bin > /dev/null 2>&1"
        )

    job_name = make_job_name(f"build-events-{args.scenario.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="build_event_dataset.py",
        job_args=["--scenario", args.scenario],
        instance_type=instance_type,
        volume_size_gb=100 if needs_img_fallback else 50,
        pip_packages="geopandas pyogrio rasterio cfgrib xarray eccodes scikit-learn xgboost netCDF4 pystac-client planetary-computer",
        pre_install_cmd=pre_install,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
