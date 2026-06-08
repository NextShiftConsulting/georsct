#!/usr/bin/env python3
"""launch_stage_zcta_geometry.py -- Stage ZCTA boundaries + adjacency to S3.

One-time job. Downloads Census TIGER ZCTA5 shapefile (~500 MB), converts to
GeoParquet, builds Queen's contiguity adjacency, uploads both.

Unblocks: W-matrix features in build_event_dataset.py, aggregate_by_zcta()
in run_fast_zcta.py.

Resource: ml.m5.xlarge (needs ~4 GB RAM for national ZCTA shapefile).
Duration: ~15 min (download + adjacency build).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-adjacency", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name("stage-zcta-geometry")
    job_args = ["--upload"]
    if args.skip_adjacency:
        job_args.append("--skip-adjacency")

    launch_processing_job(
        job_name=job_name,
        job_script="stage_zcta_geometry.py",
        job_args=job_args,
        instance_type="ml.m5.xlarge",
        volume_size_gb=20,
        pip_packages="geopandas pyproj",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
