#!/usr/bin/env python3
"""launch_stage_nlcd_geotiff.py -- Convert NLCD .img to GeoTIFF on S3.

One-time job. Downloads 26 GB .img, converts via gdal_translate, uploads .tif.
Unblocks impervious_pct in build_event_dataset.py.

Resource: ml.m5.xlarge (needs 60 GB disk for .img + .tif simultaneously).
Duration: ~15 min (download + convert + upload).
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

    job_name = make_job_name("stage-nlcd-geotiff")
    launch_processing_job(
        job_name=job_name,
        job_script="stage_nlcd_geotiff.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=80,
        pip_packages="rasterio",
        pre_install_cmd="conda install -y -c conda-forge rasterio",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
