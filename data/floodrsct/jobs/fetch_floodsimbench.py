"""
fetch_floodsimbench.py -- SageMaker Processing job: download FloodSimBench
maximum-depth GeoTIFFs for Houston and NYC tiles.

FloodSimBench: chrimerss/FloodSimBench (HuggingFace file repo, CC-BY-4.0)
  1m-resolution urban flood inundation benchmark. 10 US cities; files served
  as GeoTIFFs via Git LFS at huggingface.co/datasets/chrimerss/FloodSimBench.
  NOT a tabular HF dataset — use direct HTTPS, not load_dataset().

What we download:
  - cities_rainfall.json  (tile/rainfall metadata, 5.7 kB)
  - 6hr_max/{tile}_{mm}mm_MaxDepth.tif  for HOU and NYC tiles only
    (max inundation depth composite per rainfall scenario)
  Total: ~3 GB (70 HOU + ~8 NYC files x ~40 MB each)

We skip the 72-frame time-series tiles (~2 GB per tile) — build_event_dataset.py
only needs the max-depth surface for ZCTA-level inundation fraction.

Outputs:
  s3://swarm-floodrsct-data/raw/floodsimbench/cities_rainfall.json
  s3://swarm-floodrsct-data/raw/floodsimbench/6hr_max/{tile}_{mm}mm_MaxDepth.tif
  s3://swarm-floodrsct-data/raw/floodsimbench/summary.json
"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
import requests

sys.path.insert(0, "/opt/ml/processing/input/code")
from _s3_stream import s3_key_exists, stream_download_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
HF_BASE = "https://huggingface.co/datasets/chrimerss/FloodSimBench/resolve/main"
OUTPUT_PREFIX = "raw/floodsimbench"

# Tiles relevant to s035 scenarios
SCENARIO_TILES = {
    "houston": ["HOU001", "HOU002", "HOU003", "HOU004", "HOU005", "HOU006", "HOU007"],
    "nyc_nj":  ["NYC001", "NYC002"],
}
ALL_TILES = [t for tiles in SCENARIO_TILES.values() for t in tiles]


def fetch_rainfall_metadata() -> dict:
    """Download cities_rainfall.json and return as dict."""
    url = f"{HF_BASE}/cities_rainfall.json"
    s3_key = f"{OUTPUT_PREFIX}/cities_rainfall.json"
    s3 = boto3.client("s3", region_name="us-east-1")

    # Upload to S3 if not already there
    if not s3_key_exists(s3, BUCKET, s3_key):
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        s3.put_object(
            Bucket=BUCKET, Key=s3_key,
            Body=resp.content, ContentType="application/json",
        )
        log.info("Uploaded cities_rainfall.json to s3://%s/%s", BUCKET, s3_key)
    else:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

    return data


def parse_rainfall_meta(rainfall_meta: list) -> dict[str, list[int]]:
    """Convert cities_rainfall.json list to {tile_id: [mm, ...]} dict.

    Each entry: {"City ID": "HOU001", "100-yr": "123 mm", "50-yr": "110 mm", ...}
    Returns:    {"HOU001": [48, 57, 70, 82, 98, 110, 123, 138, 162, 181], ...}
    """
    result = {}
    skip_keys = {"City ID", "Desc"}
    for entry in rainfall_meta:
        tile = entry.get("City ID", "").strip()
        if not tile:
            continue
        mm_values = []
        for k, v in entry.items():
            if k in skip_keys:
                continue
            try:
                mm_values.append(int(str(v).replace("mm", "").strip()))
            except ValueError:
                pass
        result[tile] = sorted(set(mm_values))
    return result


def build_file_list(rainfall_meta: list) -> list[dict]:
    """Build list of 6hr_max files to download for relevant tiles."""
    tile_mm = parse_rainfall_meta(rainfall_meta)
    files = []
    for tile in ALL_TILES:
        mm_values = tile_mm.get(tile)
        if not mm_values:
            log.warning("No rainfall metadata for tile %s — using default set", tile)
            mm_values = [10, 25, 50, 100, 200, 500, 1000]
        for mm in mm_values:
            fname = f"{tile}_{mm}mm_MaxDepth.tif"
            files.append({
                "tile": tile,
                "mm": mm,
                "url": f"{HF_BASE}/6hr_max/{fname}",
                "s3_key": f"{OUTPUT_PREFIX}/6hr_max/{fname}",
            })
    log.info("Built file list: %d files across %d tiles", len(files), len(ALL_TILES))
    return files


def download_one(item: dict) -> dict:
    """Download one MaxDepth GeoTIFF to S3 with checkpointing. Returns status dict."""
    ok = stream_download_to_s3(
        None, item["url"], BUCKET, item["s3_key"], timeout=300, retries=3
    )
    return {"tile": item["tile"], "mm": item["mm"],
            "s3_key": item["s3_key"], "ok": ok}


def main() -> None:
    log.info("fetch_floodsimbench: downloading 6hr_max tiles for Houston + NYC")

    rainfall_meta = fetch_rainfall_metadata()
    log.info("Rainfall metadata: %d tile entries", len(rainfall_meta))

    files = build_file_list(rainfall_meta)

    # Parallel download — 8 workers; each uses thread-local boto3 client
    results = []
    n_ok = 0
    n_skip = 0
    n_fail = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_one, item): item for item in files}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            if res["ok"]:
                n_ok += 1
            else:
                n_fail += 1
            if i % 10 == 0:
                log.info("Progress: %d/%d  ok=%d fail=%d", i, len(files), n_ok, n_fail)
            sys.stdout.flush()

    # Write summary
    summary = {
        "dataset": "chrimerss/FloodSimBench",
        "tiles_requested": ALL_TILES,
        "files_attempted": len(files),
        "files_ok": n_ok,
        "files_failed": n_fail,
        "scenario_tiles": SCENARIO_TILES,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTPUT_PREFIX}/summary.json",
        Body=json.dumps(summary, indent=2).encode(),
        ContentType="application/json",
    )
    log.info("Summary: %d ok, %d failed out of %d files", n_ok, n_fail, len(files))
    log.info("fetch_floodsimbench complete")


if __name__ == "__main__":
    main()
