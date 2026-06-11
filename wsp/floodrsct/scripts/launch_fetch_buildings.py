#!/usr/bin/env python3
"""launch_fetch_buildings.py -- Extract Overture building footprints per ZCTA.

Downloads building polygons via open-buildings (DuckDB+S3), spatially joins
with ZCTA boundaries, aggregates building_count + total_footprint_area_m2.

Deployment Resource Review (9 dimensions):
  1. Memory:    DuckDB Overture query streams from remote parquet but
                materializes result locally. Largest scenario (houston,
                ~130 ZCTAs, ~6-county bbox) yields ~500k-1M buildings.
                GeoDataFrame ~200-400 MB. Spatial join doubles it briefly.
                Budget: ~2 GB peak per scenario. 8 GB instance sufficient.
  2. Cache:     Cache-first on s3://swarm-floodrsct-data/processed/shared/
                zcta_buildings.parquet. ZCTAs already in cache are skipped.
  3. Threads:   DuckDB uses all cores for Overture query. geopandas sjoin
                is single-threaded. n_jobs=1 is fine; I/O-bound.
  4. Image:     PYTORCH_CPU (default). No GPU needed.
  5. Instance:  ml.m5.large (2 vCPU, 8 GB). I/O-bound (HTTP to Overture S3).
                DuckDB peak ~1 GB, sjoin peak ~1 GB. 8 GB is sufficient.
  6. Volume:    30 GB. Overture parquet download ~500 MB per scenario +
                ZCTA boundaries ~50 MB + temp files.
  7. Pip:       open-buildings geopandas duckdb shapely pyarrow
  8. Pre-inst:  None (base image sufficient).
  9. Timeout:   7200s (2h). Overture query ~5-15 min per scenario.
                Cache-first means re-runs are fast.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


def _launch_one(scenario: str, dry_run: bool) -> str:
    job_name = make_job_name(f"buildings-{scenario[:8]}")
    return launch_processing_job(
        job_name=job_name,
        job_script="run_fetch_buildings.py",
        job_args=["--scenario", scenario, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=30,
        pip_packages="open-buildings geopandas duckdb shapely pyarrow",
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None, choices=SCENARIOS,
                        help="Single scenario (required unless --all)")
    parser.add_argument("--all", action="store_true",
                        help="Launch all 5 scenarios as parallel SageMaker jobs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        parser.error("Specify --scenario or --all")

    if args.all:
        for scenario in SCENARIOS:
            _launch_one(scenario, args.dry_run)
    else:
        _launch_one(args.scenario, args.dry_run)


if __name__ == "__main__":
    main()
