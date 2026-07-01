#!/usr/bin/env python3
"""launch_hydrology_3dep.py -- Compute HAND/TWI/SPI/GFI from 3DEP tiles.

Replaces the STAC-based hydrology extraction with local 3DEP DEM processing.
This fixes two issues:
  1. GDAL/curl segfault (rasterio VSICURL on PyTorch base image)
  2. Sensory bias from 1km centroid buffers (too small for HAND)

Deployment Resource Review (9 dimensions)
------------------------------------------
1. Memory:    ml.m7i.8xlarge (128 GB). Per tile peak: ~8 GB (DEM 900 MB +
              flow_dir 900 MB + flow_acc 900 MB + 4 metrics 3.6 GB +
              Python/numpy overhead). 8 workers = ~64 GB peak. Fits
              with 50% headroom. (m5.4xlarge OOM'd on SWFL with 8 workers.)
2. Cache:     No S3 cache dependency. Writes new cache on completion.
3. Threads:   ProcessPoolExecutor with 8 workers (128 GB / 8 GB per tile).
              Each worker runs single-threaded numpy flow accumulation.
4. Image:     PyTorch 2.5.1 CPU. rasterio for local .tif read only (no
              remote VSICURL). floodcaster.hydrology for compute (numpy).
5. Instance:  ml.m7i.8xlarge (32 vCPU, 128 GB). Upgraded from m5.4xlarge
              after OOM on SWFL (47 tiles). 8 workers saturate 8/32 cores.
              m7i chosen over m5: processing quota=5, m5.8xlarge had 0.
6. Volume:    50 GB. All tiles downloaded up front (~500 MB each, 47 tiles
              for SWFL = ~23 GB). Fits within 50 GB.
7. pip:       rasterio (local .tif read), geopandas, pyogrio.
              floodcaster wheel provides hydrology module.
8. pre_install: None.
9. Timeout:   28800s (8h). GFI dominates at 60-100 min/tile. SWFL worst
              case: 47 tiles, ~20 active, ceil(20/8)=3 batches x 105 min
              = ~5.25h. 8h provides margin for GFI outlier tiles.
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
            instance_type="ml.m7i.8xlarge",
            volume_size_gb=50,
            pip_packages="rasterio geopandas pyogrio planetary-computer pystac-client",
            timeout_s=28800,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
