#!/usr/bin/env python3
"""launch_render_fig6_gwr_local_r2.py -- SIGSPATIAL Figure 6: GWR local R2.

Single-panel choropleth of GWR local R2 + nonstationarity inset bars.
Requires sidecar GWR outputs to exist on S3.

Deployment Resource Review
--------------------------
1. Memory:    ZCTA boundaries parquet ~800 MB, filtered to one scenario <5 MB.
              GWR sidecar parquet + JSON <1 MB. Peak <2 GB.
2. S3 cache:  Reads sidecar GWR parquet + JSON + boundaries.
3. Threads:   Single-threaded matplotlib render.
4. Image:     PyTorch CPU. Needs geopandas + matplotlib.
5. Instance:  ml.m5.xlarge (4 vCPU, 16 GB). Upgraded from large: 800 MB boundary parquet expands to ~4 GB as GeoDataFrame.
6. Volume:    10 GB. Output PDF+SVG <2 MB.
7. pip:       geopandas matplotlib pyproj.
8. pre_install: None.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name, wait_for_job

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="houston", choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for job to complete and sync figures to overleaf")
    args = parser.parse_args()

    job_name = make_job_name(
        f"fig6-gwr-{args.scenario.replace('_', '-')}"
    )

    launch_processing_job(
        job_name=job_name,
        job_script="render_fig6_gwr_local_r2.py",
        job_args=["--scenario", args.scenario, "--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="geopandas matplotlib pyproj",
        dry_run=args.dry_run,
    )

    if not args.dry_run and args.wait:
        status = wait_for_job(job_name)
        if status == "Completed":
            import subprocess as _sp
            _sp.run([sys.executable, str(Path(__file__).resolve().parents[2] /
                     "V6-SIGSPATIAL" / "render_figures.py"), "--sync"])
    elif not args.dry_run:
        print("\n--- POST-JOB: sync figures to overleaf ---")
        print("After job completes, run:")
        print("  python V6-SIGSPATIAL/render_figures.py --sync")


if __name__ == "__main__":
    main()
