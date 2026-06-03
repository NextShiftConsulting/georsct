#!/usr/bin/env python3
"""launch_render_oahu_dashboard.py -- Render interactive Oahu flood dashboard.

Produces a self-contained HTML dashboard (JHU COVID-19 style) with
Leaflet map, KPI cards, and Chart.js comparison charts.

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

    job_name = make_job_name("render-oahu-dashboard")

    launch_processing_job(
        job_name=job_name,
        job_script="render_oahu_dashboard.py",
        job_args=[],
        instance_type="ml.m5.xlarge",
        volume_size_gb=30,
        pip_packages="geopandas pyogrio folium",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
