#!/usr/bin/env python3
"""launch_render_oahu_spatial.py -- Render Oahu H4 spatial panel from real geometry.

Produces publication-quality PDF/SVG with ZCTA choropleth, building scatter,
adjacency edges, and reef contour. All geometry from actual data sources.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Boundary parquet is 800 MB but
filtered to 20 Hawaii ZCTAs immediately. Peak memory < 2 GB.
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

    job_name = make_job_name("render-oahu-spatial")

    launch_processing_job(
        job_name=job_name,
        job_script="render_oahu_spatial_panel.py",
        job_args=[],
        instance_type="ml.m5.xlarge",
        volume_size_gb=30,
        pip_packages="geopandas pyogrio rasterio shapely matplotlib",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
