#!/usr/bin/env python3
"""launch_compute_spatial_sidecar.py -- Spatial diagnostics sidecar (LISA/GWR/Geary).

Non-locking sidecar analysis: LISA residual clusters (S1), GWR/MGWR
non-stationarity probe (S2), and Geary's C companion (S4).
Produces inputs for Fig 5 (LISA triptych) and Fig 6 (GWR local R2).

Deployment Resource Review
--------------------------
1. Memory:    Predictions parquet <5 MB, boundaries parquet ~800 MB but
              filtered to one scenario's ZCTAs (<2 MB). GWR fits on ~600 rows
              x 8 features. Peak <4 GB.
2. S3 cache:  Reads prediction parquets, adjacency, boundaries. No write cache.
3. Threads:   Single-threaded. mgwr Sel_BW is CPU-heavy but single-core.
4. Image:     PyTorch CPU. Needs esda + libpysal + mgwr + geopandas + shapely.
5. Instance:  ml.m5.xlarge (4 vCPU, 16 GB). GWR bandwidth search benefits
              from headroom; mgwr can spike to 8 GB on large datasets.
6. Volume:    10 GB. Outputs are JSON + parquet <5 MB total.
7. pip:       esda libpysal mgwr geopandas shapely scipy pyproj.
8. pre_install: None.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name

SECTIONS = ["lisa", "gwr", "geary", "all"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", required=True, choices=SECTIONS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(f"sidecar-{args.section}")

    launch_processing_job(
        job_name=job_name,
        job_script="compute_spatial_sidecar.py",
        job_args=["--section", args.section, "--upload"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=10,
        pip_packages="numpy<2.3 esda libpysal mgwr spreg geopandas shapely scipy pyproj",
        extra_files=["compute_residual_lisa.py"],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
