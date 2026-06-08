#!/usr/bin/env python3
"""launch_render_oahu_paper_figure.py -- Render static paper figure (PDF+SVG).

Dark-theme matplotlib figure replacing oahu_h4_results.pdf in Appendix B.
Three-panel layout: metrics strip | ZCTA map | normalized comparison chart.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). Needs geopandas + rasterio + matplotlib.
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

    job_name = make_job_name("render-oahu-paper-figure")

    launch_processing_job(
        job_name=job_name,
        job_script="render_oahu_paper_figure.py",
        job_args=[],
        instance_type="ml.m5.xlarge",
        volume_size_gb=30,
        pip_packages="geopandas pyogrio rasterio matplotlib pyproj",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
