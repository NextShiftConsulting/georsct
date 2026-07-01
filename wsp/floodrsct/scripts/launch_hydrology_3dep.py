#!/usr/bin/env python3
"""launch_hydrology_3dep.py -- Compute HAND/TWI/SPI/GFI from 3DEP tiles.

Replaces the STAC-based hydrology extraction with local 3DEP DEM processing.
This fixes two issues:
  1. GDAL/curl segfault (rasterio VSICURL on PyTorch base image)
  2. Sensory bias from 1km centroid buffers (too small for HAND)

Deployment Resource Review (9 dimensions)
------------------------------------------
1. Memory:    ml.m5.4xlarge (64 GB). Per tile peak: ~5.4 GB (DEM 900 MB +
              flow_acc 900 MB + 4 metric arrays 3.6 GB). 8 parallel workers
              = ~43 GB peak. Fits with headroom.
2. Cache:     No S3 cache dependency. Writes new cache on completion.
3. Threads:   ProcessPoolExecutor with 8 workers (64 GB / 5.4 GB per tile).
              Each worker runs single-threaded numpy flow accumulation.
4. Image:     PyTorch 2.5.1 CPU. rasterio for local .tif read only (no
              remote VSICURL). floodcaster.hydrology for compute (numpy).
5. Instance:  ml.m5.4xlarge (16 vCPU, 64 GB). Need memory headroom for
              large tiles (SWFL has 47 tiles, processed serially).
6. Volume:    50 GB. Largest tile ~500 MB. One tile downloaded at a time,
              deleted after processing. Peak disk: ~1 GB.
7. pip:       rasterio (local .tif read), geopandas, pyogrio.
              floodcaster wheel provides hydrology module.
8. pre_install: None.
9. Timeout:   7200s (2h). Per-scenario launch (47 tiles x ~2 min = ~1.5h max).
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
    parser.add_argument("--scenario", required=True,
                        choices=SCENARIOS + ["all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]

    for scenario in scenarios:
        job_name = make_job_name(f"hydro-3dep-{scenario.replace('_', '-')}")

        launch_processing_job(
            job_name=job_name,
            job_script="run_fetch_hydrology_3dep.py",
            job_args=["--scenario", scenario, "--upload"],
            instance_type="ml.m5.4xlarge",
            volume_size_gb=50,
            pip_packages="rasterio geopandas pyogrio",
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
