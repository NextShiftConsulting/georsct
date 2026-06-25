#!/usr/bin/env python3
"""
launch_extract_prithvi_embeddings.py -- Launch Prithvi-EO-2.0 embedding extraction
on SageMaker.

For each ZCTA in a scenario, fetches HLS satellite imagery from NASA LP DAAC,
runs Prithvi encoder, and outputs a (n_zctas, 1024) embedding parquet.

Usage:
    python scripts/launch_extract_prithvi_embeddings.py --scenario houston --dry-run
    python scripts/launch_extract_prithvi_embeddings.py --scenario houston
    python scripts/launch_extract_prithvi_embeddings.py --all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _launcher_base import launch_processing_job, make_job_name, log

SCENARIOS = [
    "houston",
    "new_orleans",
    "nyc",
    "riverside_coachella",
    "southwest_florida",
]

# ---------------------------------------------------------------
# 9-dim pre-launch resource audit
# ---------------------------------------------------------------
# 1. Memory:  Prithvi encoder ~1.3 GB + 6-band chips for ~500 ZCTAs
#             (500 * 6 * 224 * 224 * 4B = ~720 MB) + embeddings ~2 MB
#             Total peak ~3 GB. 16 GB instance = safe.
#
# 2. Cache:   No S3 cache. HLS tiles downloaded fresh per run.
#
# 3. Threads: HLS download is I/O-bound but sequential (CMR rate limits).
#             Inference is CPU-bound, single-threaded (batch_size=8).
#             No parallelism needed.
#
# 4. Image:   PyTorch 2.x (for Prithvi model). Standard SageMaker pytorch image.
#
# 5. Instance: ml.m5.xlarge (4 vCPU, 16 GB). CPU sufficient -- Prithvi
#              inference on 500 ZCTAs takes ~10 min on CPU.
#              GPU not needed (batch is small, not training).
#
# 6. Volume:  Prithvi weights 1.3 GB + HLS tiles ~5 GB + workspace = ~10 GB.
#             30 GB volume = safe margin.
#
# 7. pip:     rasterio (HLS GeoTIFF reading), pyproj (coordinate transform),
#             timm (Prithvi architecture), einops (tensor rearrange),
#             requests (CMR API + HLS download)
#
# 8. pre_install: libgdal-dev for rasterio HFA driver (GeoTIFF should work
#                 without it, but safer to include)
#
# 9. Timeout: ~45 min per scenario (HLS download dominates).
#             7200s (2h) timeout = safe.
# ---------------------------------------------------------------


def launch_one(scenario: str, dry_run: bool) -> None:
    job_name = make_job_name(f"prithvi-embed-{scenario.replace('_', '-')}")

    launch_processing_job(
        job_name=job_name,
        job_script="extract_prithvi_embeddings.py",
        job_args=["--scenario", scenario, "--max-cloud", "30", "--batch-size", "8"],
        instance_type="ml.m5.xlarge",
        volume_size_gb=30,
        pip_packages="rasterio pyproj timm einops requests",
        pre_install_cmd="apt-get update -qq && apt-get install -y -qq libgdal-dev > /dev/null 2>&1",
        dry_run=dry_run,
    )

    if not dry_run:
        log.info(
            "Monitor: MSYS_NO_PATHCONV=1 aws logs tail "
            '"/aws/sagemaker/ProcessingJobs" '
            "--log-stream-name-prefix %s --follow "
            "--profile nsc-swarm --region us-east-1",
            job_name,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Prithvi-EO-2.0 embedding extraction"
    )
    parser.add_argument("--scenario", choices=SCENARIOS,
                        help="Single scenario to process")
    parser.add_argument("--all", action="store_true",
                        help="Process all 5 scenarios sequentially")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without launching")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        parser.error("Specify --scenario or --all")

    scenarios = SCENARIOS if args.all else [args.scenario]
    for scenario in scenarios:
        log.info("=== Launching Prithvi embedding extraction: %s ===", scenario)
        launch_one(scenario, args.dry_run)


if __name__ == "__main__":
    main()
