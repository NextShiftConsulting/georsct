#!/usr/bin/env python3
"""launch_render_zcta_maps.py -- Phase R4.1: ZCTA map rendering.

Renders choropleth map PNG per ZCTA for VLM input. Needs geopandas
for GeoParquet boundaries (800MB file). One job per scenario.

Resource: ml.m5.xlarge (4 vCPU, 16 GB). 800MB GeoParquet + matplotlib
rendering. Extra memory for boundary GeoDataFrame.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"zcta-maps-{args.scenario.replace('_', '-')}")

    launch_processing_job(
        job_name=job_name,
        job_script="render_zcta_maps.py",
        job_args=["--scenario", args.scenario, "--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=20,
        pip_packages="geopandas matplotlib",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
