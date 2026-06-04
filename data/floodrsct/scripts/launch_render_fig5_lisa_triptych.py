#!/usr/bin/env python3
"""launch_render_fig5_lisa_triptych.py -- SIGSPATIAL Figure 5: LISA triptych.

Three-panel choropleth (R0/R1/R2 LISA clusters) for one scenario.
Requires sidecar LISA outputs to exist on S3.

Deployment Resource Review
--------------------------
1. Memory:    ZCTA boundaries parquet ~800 MB, filtered to one scenario <5 MB.
              LISA parquets <1 MB each. Peak <2 GB.
2. S3 cache:  Reads sidecar LISA parquets + boundaries.
3. Threads:   Single-threaded matplotlib render.
4. Image:     PyTorch CPU. Needs geopandas + matplotlib.
5. Instance:  ml.m5.large (2 vCPU, 8 GB).
6. Volume:    10 GB. Output PDF+SVG <2 MB.
7. pip:       geopandas matplotlib pyproj.
8. pre_install: None.
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
    parser.add_argument("--scenario", default="houston", choices=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(
        f"fig5-lisa-{args.scenario.replace('_', '-')}"
    )

    launch_processing_job(
        job_name=job_name,
        job_script="render_fig5_lisa_triptych.py",
        job_args=["--scenario", args.scenario, "--upload"],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="geopandas matplotlib pyproj",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
