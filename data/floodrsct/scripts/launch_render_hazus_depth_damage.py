#!/usr/bin/env python3
"""launch_render_hazus_depth_damage.py -- Render HAZUS depth-damage figure.

Three-panel figure (Appendix): structural curves, content curves, Oahu depth
band summary. Reads HAZUS tables from sphere.data and Floodcaster Oahu output
from S3.

Deployment Resource Review
--------------------------
1. Memory:    Oahu parquet ~5 MB in memory, HAZUS CSVs <1 MB. Trivial.
2. S3 cache:  None. Reads raw Floodcaster output each time.
3. Threads:   Single-threaded matplotlib render.
4. Image:     PyTorch CPU. sphere-flood pip-installed at boot.
5. Instance:  ml.m5.xlarge (4 vCPU, 16 GB). Overkill but matches render jobs.
6. Volume:    10 GB. Outputs are PDF+SVG <100 KB each.
7. pip:       sphere-flood matplotlib. sphere-flood bundles the HAZUS CSVs.
8. pre_install: None.
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

    job_name = make_job_name("render-hazus-depth-damage")

    launch_processing_job(
        job_name=job_name,
        job_script="render_hazus_depth_damage.py",
        job_args=["--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="sphere-flood matplotlib",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
