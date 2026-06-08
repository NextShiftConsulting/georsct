#!/usr/bin/env python3
"""launch_compute_residual_lisa.py -- Local Moran's I cluster maps on residuals.

Per-ZCTA LISA classification (HH/HL/LH/LL/NS) for one scenario+level.
Feeds into compute_spatial_sidecar.py (section lisa) and render_fig5.

Deployment Resource Review
--------------------------
1. Memory:    Predictions parquet <5 MB, adjacency <1 MB. Trivial.
2. S3 cache:  Reads prediction parquets + adjacency. No write cache.
3. Threads:   Single-threaded (esda permutation test is vectorized numpy).
4. Image:     PyTorch CPU. Needs esda + libpysal pip-installed.
5. Instance:  ml.m5.large (2 vCPU, 8 GB). LISA on ~600 ZCTAs is instant.
6. Volume:    10 GB. Outputs are JSON + parquet <1 MB each.
7. pip:       esda libpysal scipy (esda needs libpysal, scipy for p-values).
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
LEVELS = ["r0", "r1", "r2"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--level", required=True, choices=LEVELS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job_name = make_job_name(
        f"lisa-{args.level}-{args.scenario.replace('_', '-')}"
    )

    launch_processing_job(
        job_name=job_name,
        job_script="compute_residual_lisa.py",
        job_args=[
            "--level", args.level,
            "--scenario", args.scenario,
            "--upload",
        ],
        instance_type="ml.m5.large",
        volume_size_gb=10,
        pip_packages="esda libpysal scipy",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
