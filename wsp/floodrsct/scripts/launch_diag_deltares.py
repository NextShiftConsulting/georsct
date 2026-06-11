#!/usr/bin/env python3
"""One-shot diagnostic for Deltares coordinate triage."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    launch_processing_job(
        job_name=make_job_name("diag-deltares"),
        job_script="diag_deltares.py",
        job_args=[],
        instance_type="ml.m5.large",
        volume_size_gb=20,
        pip_packages=(
            "geopandas rasterio planetary-computer pystac-client "
            "duckdb shapely pyogrio"
        ),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
