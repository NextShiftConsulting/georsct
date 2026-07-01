#!/usr/bin/env python3
"""launch_hydrology_owp_hand.py -- Fetch NOAA OWP pre-computed HAND per ZCTA.

Downloads production-grade HAND rasters from NOAA Office of Water Prediction,
computes zonal statistics per ZCTA, and merges with existing TWI/SPI/GFI from
the 3DEP pipeline.

Deployment Resource Review (9 dimensions)
------------------------------------------
1. Memory:    ml.m5.xlarge (16 GB). PyTorch base image uses ~4 GB +
              pip install overhead ~2 GB. Peak workload: mosaic of ~5-15
              HUC8 rasters (~86 MB each int16 = ~1.3 GB) + ZCTA polygons
              (~200 MB) + zonal stats working arrays (~500 MB) = ~2-3 GB.
              Total ~8-9 GB peak. 16 GB provides headroom.
              ml.m5.large (8 GB) OOM'd during pip install on PyTorch image.
2. Cache:     WBD HUC8 polygons cached at raw/reference/wbd_hu8_conus.parquet.
              ZCTA polygons cached at raw/reference/zcta_boundaries_2020.parquet.
              Both are static reference data -- no staleness concern.
3. Threads:   ThreadPoolExecutor(4) for parallel OWP HAND downloads
              (I/O-bound). Zonal stats is sequential (rasterio window reads).
              4 vCPU ml.m5.xlarge matches download parallelism.
4. Image:     PyTorch 2.5.1 CPU. rasterio for local .tif read + merge.
              geopandas for spatial join. floodcaster.hydrology for zonal
              stats (read-only import, no modifications to frozen codebase).
5. Instance:  ml.m5.xlarge (4 vCPU, 16 GB). Downloads are parallel
              (ThreadPool, requester-pays bucket). Memory is the constraint:
              PyTorch image ~4 GB + pip ~2 GB + workload ~3 GB = ~9 GB.
              ml.m5.large (8 GB) OOM'd on pip install.
6. Volume:    30 GB. WBD download (~1.5 GB zip) + ZCTA download (~800 MB)
              + OWP HAND tiles (15 x 86 MB = ~1.3 GB) + mosaic (~1 GB)
              + pip install (~2 GB). Total ~7 GB. 30 GB provides 4x margin.
7. pip:       rasterio (raster I/O + merge), geopandas (spatial join),
              pyogrio (geopandas file backend), pyarrow (parquet I/O).
              floodcaster wheel provides zonal_hydro_stats.
8. pre_install: None.
9. Timeout:   3600s (1h). Downloads: ~5 min (15 HUC8 x 86 MB at ~50 MB/s).
              WBD download: ~2 min (cached after first run). ZCTA download:
              ~1 min (cached). Zonal stats: ~5 min per scenario. Total: ~15
              min per scenario. 1h provides 4x margin for 5 scenarios.
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
        job_name = make_job_name(f"hydro-owp-{scenario.replace('_', '-')}")

        launch_processing_job(
            job_name=job_name,
            job_script="run_fetch_hydrology_owp_hand.py",
            job_args=["--scenario", scenario, "--upload"],
            instance_type="ml.m5.xlarge",
            volume_size_gb=30,
            pip_packages="rasterio geopandas pyogrio pyarrow planetary-computer pystac-client",
            timeout_s=3600,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
