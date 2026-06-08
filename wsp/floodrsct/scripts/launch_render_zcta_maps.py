#!/usr/bin/env python3
"""launch_render_zcta_maps.py -- Phase R4.1: ZCTA map rendering.

Renders choropleth map PNG per ZCTA for VLM input. Needs geopandas
for GeoParquet boundaries (800MB file). One job per scenario.

ThreadPoolExecutor avoids copying the 800MB GeoDataFrame across worker
processes. Each thread renders independent figures using the Agg backend
and closes figures after upload. Passes --skip-existing to avoid
re-rendering maps already present on S3.

Instance sizing is scenario-aware: large scenarios (200+ ZCTAs) use
ml.m5.2xlarge (32 GB) for matplotlib headroom.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]

LARGE_SCENARIOS = {"houston", "nyc", "southwest_florida"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"zcta-maps-{args.scenario.replace('_', '-')}")
    instance_type = "ml.m5.2xlarge" if args.scenario in LARGE_SCENARIOS else "ml.m5.xlarge"

    launch_processing_job(
        job_name=job_name,
        job_script="render_zcta_maps.py",
        job_args=["--scenario", args.scenario, "--upload", "--skip-existing"],
        instance_type=instance_type,
        volume_size_gb=20,
        pip_packages="geopandas matplotlib",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
