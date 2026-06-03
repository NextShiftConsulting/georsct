#!/usr/bin/env python3
"""launch_build_event_dataset.py -- Launch the event dataset assembler for one scenario.

This is the final processing step before Data Lock A/B. Run only after all
raw data pulls have completed for the target scenario.

Data Lock A (June 1): --scenario houston
Data Lock B (June 2): all remaining scenarios

Resource assumptions and calculations
--------------------------------------
Bottleneck: MRMS grib2 decode (ProcessPoolExecutor, N concurrent grids).
Each CONUS grib2 grid decompresses to ~100 MB in memory.

  ml.m5.2xlarge:  8 vCPU, 32 GB RAM  -> 4 workers x 100 MB = 0.4 GB concurrent
  ml.m5.4xlarge: 16 vCPU, 64 GB RAM  -> 8 workers x 100 MB = 0.8 GB concurrent

Large scenarios (houston: 3 events x ~18 days, sw_florida: 3 events x ~7 days)
produce 400-500 grib2 files total. At 8 workers, wall-clock is ~15 min/event
for MRMS alone.

SLOSH MOM GeoTIFF: 318K x 224K pixels (1.2 GB on disk, 66 GB uncompressed).
Sampled via rasterio.sample() at ZCTA centroids -- no full-raster load.
Memory cost: negligible (~1 MB for coordinate arrays).

Volume: 50 GB is sufficient. Grib2 files are streamed from S3, decoded in
memory, accumulated into a running sum, then discarded. Only one grid plus
the running sum array (~200 MB) live simultaneously per worker.

Image: PyTorch 2.5.1 CPU (SageMaker-managed). Spatial packages (rasterio,
geopandas, cfgrib) pip-installed at boot (~3 min). No custom image needed
at current job frequency.
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

# Scenarios that sample NLCD .img (Erdas Imagine HFA format).
# pip-installed rasterio lacks the HFA driver; install libgdal-dev + GDAL
# python bindings so the osgeo.gdal fallback path in build_event_dataset.py works.
_NLCD_SCENARIOS = {"houston", "nyc", "new_orleans", "riverside_coachella", "southwest_florida"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    instance_type = (
        "ml.m5.4xlarge" if args.scenario in _LARGE_SCENARIOS else "ml.m5.2xlarge"
    )

    # NLCD .img (Erdas Imagine HFA) needs gdal_translate to convert to GeoTIFF
    # before rasterio can read it. The apt gdal-bin on Ubuntu 22.04 includes
    # HFA read support -- but the python GDAL bindings crash (segfault) when
    # reading .img directly. Converting to .tif via gdal_translate avoids this:
    # the subprocess reads HFA, writes GeoTIFF, and rasterio reads the .tif.
    pre_install = None
    if args.scenario in _NLCD_SCENARIOS:
        pre_install = (
            "apt-get update -qq && apt-get install -y -qq gdal-bin > /dev/null 2>&1"
        )

    job_name = make_job_name(f"build-events-{args.scenario.replace('_', '-')}")
    launch_processing_job(
        job_name=job_name,
        job_script="build_event_dataset.py",
        job_args=["--scenario", args.scenario],
        instance_type=instance_type,
        # NLCD raster is 26 GB; need headroom for download + open. 100 GB safe.
        volume_size_gb=100 if args.scenario in _NLCD_SCENARIOS else 50,
        pip_packages="geopandas pyogrio rasterio cfgrib xarray eccodes scikit-learn xgboost",
        pre_install_cmd=pre_install,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
