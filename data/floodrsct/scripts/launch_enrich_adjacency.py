#!/usr/bin/env python3
"""launch_enrich_adjacency.py -- Enrich adjacency with degree + centroid distance.

One-time job. Reads zcta_adjacency.parquet + zcta5_boundaries.parquet,
adds degree_1, degree_2, centroid_distance_km columns.

Resource: ml.m5.xlarge (4 vCPU, 16 GB) -- boundaries parquet is ~760 MB,
needs memory for shapely centroid computation.

Usage:
    python launch_enrich_adjacency.py --dry-run
    python launch_enrich_adjacency.py
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

    job_name = make_job_name("enrich-adjacency")
    launch_processing_job(
        job_name=job_name,
        job_script="enrich_adjacency.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=20,
        pip_packages="shapely",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
