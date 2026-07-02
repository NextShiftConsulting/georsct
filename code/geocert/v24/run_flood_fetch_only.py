#!/usr/bin/env python3
"""
Fast FEMA NFHL fetch-only job. Downloads flood zone polygons for all CONUS
counties and uploads raw JSON to S3. No overlay, no TIGER projection,
minimal memory. Designed to run on a cheap instance with max parallelism.

The companion run_flood_zones.py reads these cached fetch files from S3
on startup and skips the fetch phase entirely.

Output: s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/flood_fetch/{fips}.json
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
for _h in logging.root.handlers:
    _h.flush = lambda _orig=_h.flush: (_orig(), sys.stdout.flush())
log = logging.getLogger(__name__)

# FEMA NFHL ArcGIS REST endpoint (layer 28 = S_Fld_Haz_Ar)
NFHL_QUERY_URL = (
    "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
)
PAGE_SIZE = 2000
MAX_RETRIES = 3
INTER_REQUEST_SLEEP = 0.02
MAX_COUNTY_BBOX_DEG = 2.0
MIN_CELL_DEG = 0.1

S3_BUCKET = "swarm-yrsn-datasets"
S3_FETCH_PREFIX = "rsct_curriculum/series_018/processed/flood_fetch/"

NON_CONUS_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}


def _s3_upload(local_path: str, key: str):
    try:
        import boto3
        boto3.client("s3").upload_file(local_path, S3_BUCKET, key)
    except Exception as e:
        log.warning("S3 upload failed %s: %s", key, e)


def _s3_exists(key: str) -> bool:
    try:
        import boto3
        boto3.client("s3").head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


def _fetch_page(bbox_str: str, offset: int = 0):
    params = {
        "where": "1=1",
        "geometry": bbox_str,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF",
        "returnGeometry": "true",
        "f": "json",
        "resultRecordCount": PAGE_SIZE,
        "resultOffset": offset,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(NFHL_QUERY_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                return None, False
            return data.get("features", []), data.get("exceededTransferLimit", False)
        except (requests.RequestException, json.JSONDecodeError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None, False


def fetch_nfhl_for_bbox(xmin, ymin, xmax, ymax):
    bbox_str = f"{xmin},{ymin},{xmax},{ymax}"
    all_features = []
    offset = 0
    while True:
        features, exceeded = _fetch_page(bbox_str, offset)
        if features is None:
            if (xmax - xmin) > MIN_CELL_DEG and (ymax - ymin) > MIN_CELL_DEG:
                return _fetch_quadtree(xmin, ymin, xmax, ymax)
            return all_features
        if not features:
            break
        all_features.extend(features)
        if len(features) < PAGE_SIZE and not exceeded:
            break
        offset += len(features)
        time.sleep(INTER_REQUEST_SLEEP)
    return all_features


def _fetch_quadtree(xmin, ymin, xmax, ymax):
    mx, my = (xmin + xmax) / 2, (ymin + ymax) / 2
    result = []
    for bx in [(xmin, mx), (mx, xmax)]:
        for by in [(ymin, my), (my, ymax)]:
            result.extend(fetch_nfhl_for_bbox(bx[0], by[0], bx[1], by[1]))
            time.sleep(INTER_REQUEST_SLEEP)
    return result


def _split_bbox_to_tiles(xmin, ymin, xmax, ymax, max_deg):
    import math
    nx = max(1, math.ceil((xmax - xmin) / max_deg))
    ny = max(1, math.ceil((ymax - ymin) / max_deg))
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    tiles = []
    for i in range(nx):
        for j in range(ny):
            tiles.append((
                xmin + i * dx, ymin + j * dy,
                xmin + (i + 1) * dx, ymin + (j + 1) * dy,
            ))
    return tiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crosswalk", default="/opt/ml/processing/input/data/zcta_county_crosswalk.parquet")
    parser.add_argument("--tiger-dir", default="/opt/ml/processing/input/tiger")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip counties already on S3")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fetch_dir = output_dir / "fetch_cache"
    fetch_dir.mkdir(parents=True, exist_ok=True)

    # Load TIGER just for bounding boxes (lightweight — no projection needed)
    import geopandas as gpd
    import pandas as pd

    tiger_path = Path(args.tiger_dir) / "tl_2022_us_zcta520.shp"
    log.info("Loading TIGER: %s", tiger_path)
    zcta_geo = gpd.read_file(tiger_path)
    zcta_geo = zcta_geo.rename(columns={"ZCTA5CE20": "zcta_id"})
    log.info("  %d ZCTAs loaded", len(zcta_geo))

    # Load crosswalk for county assignment
    xwalk = pd.read_parquet(args.crosswalk)
    zcta_counties = dict(zip(xwalk["zcta_id"].astype(str), xwalk["county_fips"].astype(str)))
    zcta_geo["county_fips"] = zcta_geo["zcta_id"].map(zcta_counties).fillna("unknown")

    # CONUS filter
    state_fips = zcta_geo["county_fips"].str[:2]
    conus_mask = ~state_fips.isin(NON_CONUS_STATE_FIPS) & (zcta_geo["county_fips"] != "unknown")
    zcta_geo = zcta_geo[conus_mask].reset_index(drop=True)
    log.info("  CONUS: %d ZCTAs", len(zcta_geo))

    # Build county manifest with bounding boxes
    manifest = []
    for county_fips, idx in zcta_geo.groupby("county_fips").groups.items():
        idx_list = list(idx)
        bounds = zcta_geo.iloc[idx_list].total_bounds  # EPSG:4326
        pad = 0.01
        manifest.append({
            "county_fips": county_fips,
            "n_zctas": len(idx_list),
            "bbox": (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad),
        })
    manifest.sort(key=lambda m: m["n_zctas"])
    log.info("County manifest: %d counties", len(manifest))

    # Check which counties already exist on S3
    to_fetch = []
    skipped = 0
    if args.skip_existing:
        log.info("Checking S3 for existing fetch files...")
        # Batch check via list_objects
        try:
            import boto3
            s3 = boto3.client("s3")
            existing_keys = set()
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_FETCH_PREFIX):
                for obj in page.get("Contents", []):
                    # Extract FIPS from key: .../flood_fetch/12345.json -> 12345
                    fname = obj["Key"].split("/")[-1]
                    if fname.endswith(".json"):
                        existing_keys.add(fname[:-5])
            log.info("  Found %d existing fetch files on S3", len(existing_keys))
            for m in manifest:
                if m["county_fips"] in existing_keys:
                    skipped += 1
                else:
                    to_fetch.append(m)
        except Exception as e:
            log.warning("  S3 check failed, fetching all: %s", e)
            to_fetch = manifest
    else:
        to_fetch = manifest

    log.info("To fetch: %d counties (%d skipped, already on S3)", len(to_fetch), skipped)

    if not to_fetch:
        log.info("All counties already fetched. Done.")
        return

    # Fetch with max parallelism
    fetched = 0
    failed = 0

    def _fetch_one(entry):
        fips = entry["county_fips"]
        xmin, ymin, xmax, ymax = entry["bbox"]
        dx, dy = xmax - xmin, ymax - ymin
        if dx > MAX_COUNTY_BBOX_DEG or dy > MAX_COUNTY_BBOX_DEG:
            tiles = _split_bbox_to_tiles(xmin, ymin, xmax, ymax, MAX_COUNTY_BBOX_DEG)
            features = []
            for tile in tiles:
                features.extend(fetch_nfhl_for_bbox(*tile))
                time.sleep(INTER_REQUEST_SLEEP)
        else:
            features = fetch_nfhl_for_bbox(xmin, ymin, xmax, ymax)
        # Save locally + S3
        fetch_file = fetch_dir / f"{fips}.json"
        with open(fetch_file, "w") as f:
            json.dump(features, f)
        _s3_upload(str(fetch_file), f"{S3_FETCH_PREFIX}{fips}.json")
        # Delete local to save disk
        fetch_file.unlink(missing_ok=True)
        return fips, len(features)

    log.info("=== FETCHING %d COUNTIES (%d threads) ===", len(to_fetch), args.threads)
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(_fetch_one, m): m["county_fips"] for m in to_fetch}
        for fut in as_completed(futures):
            fips = futures[fut]
            try:
                fips, n_feat = fut.result()
                fetched += 1
                if fetched % 25 == 0 or fetched == len(to_fetch):
                    elapsed = time.time() - t_start
                    rate = fetched / (elapsed / 60) if elapsed > 0 else 0
                    eta_min = (len(to_fetch) - fetched) / rate if rate > 0 else 0
                    log.info("  [%d/%d] %s: %d features (%.1f/min, ETA %.0f min)",
                             fetched, len(to_fetch), fips, n_feat, rate, eta_min)
                    sys.stdout.flush()
            except Exception as e:
                log.warning("  FAILED %s: %s", fips, e)
                failed += 1
                fetched += 1

    elapsed = time.time() - t_start
    log.info("=== DONE ===")
    log.info("  Fetched: %d, Failed: %d, Skipped: %d", fetched - failed, failed, skipped)
    log.info("  Elapsed: %.1f min", elapsed / 60)

    # Write summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fetched": fetched - failed,
        "failed": failed,
        "skipped": skipped,
        "total_counties": len(manifest),
        "elapsed_sec": round(elapsed, 1),
        "threads": args.threads,
    }
    summary_path = output_dir / "fetch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _s3_upload(str(summary_path), f"{S3_FETCH_PREFIX}fetch_summary.json")
    log.info("Summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
