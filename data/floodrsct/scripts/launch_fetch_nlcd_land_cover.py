#!/usr/bin/env python3
"""launch_fetch_nlcd_land_cover.py -- Launch NLCD 2021 Land Cover fetch.

Downloads NLCD 2021 Land Cover raster (16-class categorical, 30m, CONUS).
Used for cropland_pct feature (classes 81=Pasture/Hay, 82=Cultivated Crops).

Resource assumptions:
  - Download: ~2.5 GB zip from ScienceBase/MRLC
  - Extract: ~26 GB .img raster
  - Convert: gdal_translate .img -> .tif (~1 GB compressed)
  - Volume: 50 GB covers zip + extracted + converted with headroom
  - Instance: ml.m5.large (2 vCPU, 8 GB) -- download + convert is I/O bound
  - Pre-install: gdal-bin for .img -> .tif conversion
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("fetch-nlcd-land-cover")
    launch_processing_job(
        job_name=job_name,
        job_script="fetch_nlcd_land_cover.py",
        job_args=[],
        instance_type="ml.m5.large",
        volume_size_gb=50,
        pre_install_cmd="apt-get update -qq && apt-get install -y -qq gdal-bin > /dev/null 2>&1",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
