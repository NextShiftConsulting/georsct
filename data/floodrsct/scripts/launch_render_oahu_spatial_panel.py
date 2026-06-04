#!/usr/bin/env python3
"""launch_render_oahu_spatial_panel.py -- Oahu H4 spatial setup figure.

Two-panel publication figure: ZCTA choropleth + residual bar chart.
Reads real geometry data from S3 (boundaries, adjacency, reef raster).

Deployment Resource Review
--------------------------
1. Memory:    ZCTA boundaries ~800 MB, filtered to 20 Hawaii ZCTAs <2 MB.
              Reef raster ~50 MB, reprojected in memory. Floodcaster parquet
              ~5 MB. Peak <4 GB.
2. S3 cache:  Reads boundaries, adjacency, centroids, floodcaster results,
              reef raster, residuals CSV.
3. Threads:   Single-threaded matplotlib render.
4. Image:     PyTorch CPU. Needs geopandas + rasterio + pyproj + matplotlib.
5. Instance:  ml.m5.xlarge (4 vCPU, 16 GB). Reef raster reproject needs RAM.
6. Volume:    10 GB. Output PDF+SVG <2 MB.
7. pip:       geopandas matplotlib rasterio pyproj shapely.
8. pre_install: None.
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

    job_name = make_job_name("render-oahu-spatial-panel")

    launch_processing_job(
        job_name=job_name,
        job_script="render_oahu_spatial_panel.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="geopandas matplotlib rasterio pyproj shapely",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
