#!/usr/bin/env python3
"""
run_comid_zcta.py -- SageMaker run script: COMID-to-ZCTA spatial crosswalk.

Builds comid_zcta_crosswalk.parquet via pynhd WaterData WFS (USGS api.water.usgs.gov).
EDCINTL (original NHDPlus V2 host) is defunct as of 2025 — this uses the USGS
WaterData WFS endpoint instead.

Strategy:
  1. Load TIGER ZCTAs, reproject to EPSG:5070, save for workers
  2. Get all HUC8 bounding boxes via WaterData("wbd08") for CONUS HUC2 01-18
  3. ProcessPoolExecutor: each worker queries WaterData("catchmentsp").bybox()
     for one HUC8, gets NHDPlus V2 catchment polygons with COMID (FEATUREID)
  4. For each HUC8 batch: spatial overlay with ZCTAs, compute area fractions
  5. Concatenate, deduplicate, validate, upload

Instance: ml.m5.12xlarge (48 vCPU, 192 GB RAM)
Runtime:  ~30-50 min (HUC8-tiled WFS queries + parallel overlays)
Cost:     ~$1.20

Output: s3://swarm-yrsn-datasets/rsct_curriculum/geo/comid_zcta_crosswalk.parquet
  - COMID         (str)
  - zcta_id       (str, 5-digit)
  - area_fraction (float)
"""

import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
for _h in logging.root.handlers:
    _h.flush = lambda _orig=_h.flush: (_orig(), sys.stdout.flush())
log = logging.getLogger(__name__)

S3_BUCKET = "swarm-yrsn-datasets"
S3_OUTPUT_KEY = "rsct_curriculum/geo/comid_zcta_crosswalk.parquet"
S3_PROVENANCE_KEY = "rsct_curriculum/geo/comid_zcta_crosswalk_provenance.json"
TARGET_CRS = "EPSG:5070"

# CONUS HUC2 regions (01-18)
CONUS_HUC2 = [f"{i:02d}" for i in range(1, 19)]


def _s3():
    return boto3.client("s3")

def _s3_upload(local_path: str, key: str):
    try:
        _s3().upload_file(local_path, S3_BUCKET, key)
        log.info("  -> s3://%s/%s", S3_BUCKET, key)
    except Exception as e:
        log.warning("  S3 upload failed for %s: %s", key, e)


# ---------------------------------------------------------------------------
# Fetch HUC8 bounding boxes (runs in main process)
# ---------------------------------------------------------------------------
def get_huc8_bboxes(huc2_codes: list[str]) -> list[dict]:
    """Return list of {huc8, minx, miny, maxx, maxy} for all HUC8s in given HUC2s."""
    from pynhd import WaterData

    log.info("Fetching HUC8 boundaries for %d HUC2 regions...", len(huc2_codes))
    wd = WaterData("wbd08")
    all_huc8 = []

    for huc2 in huc2_codes:
        try:
            # Filter by huc8 prefix — "huc2" is not a valid field in wbd08
            gdf = wd.byfilter(f"huc8 LIKE '{huc2}%'")
            if gdf is None or gdf.empty:
                log.warning("  HUC2 %s returned no HUC8s", huc2)
                continue
            for _, row in gdf.iterrows():
                bbox = row.geometry.bounds  # (minx, miny, maxx, maxy)
                all_huc8.append({
                    "huc2": huc2,
                    "huc8": str(row.get("huc8", row.get("HUC8", ""))),
                    "minx": bbox[0], "miny": bbox[1],
                    "maxx": bbox[2], "maxy": bbox[3],
                })
            log.info("  HUC2 %s: %d HUC8s", huc2, len(gdf))
        except Exception as exc:
            log.warning("  HUC2 %s failed: %s", huc2, exc)

    log.info("Total HUC8s to query: %d", len(all_huc8))
    return all_huc8


# ---------------------------------------------------------------------------
# Per-HUC8 worker: fetch catchments + overlay with ZCTAs
# ---------------------------------------------------------------------------
def process_huc8(args: tuple) -> tuple[str, str | None]:
    """Fetch catchment polygons for one HUC8, overlay with ZCTAs, return pairs parquet path."""
    huc8, bbox_dict, zctas_parquet, work_dir_str, attempt_n = args
    work_dir = Path(work_dir_str)
    work_dir.mkdir(parents=True, exist_ok=True)

    from pynhd import WaterData
    from shapely.validation import make_valid as _make_valid

    logger = logging.getLogger(f"huc8.{huc8}")

    # --- Fetch catchments from WFS ---
    bbox = (bbox_dict["minx"], bbox_dict["miny"], bbox_dict["maxx"], bbox_dict["maxy"])
    try:
        wd = WaterData("catchmentsp")
        catchments = wd.bybox(bbox)
    except Exception as exc:
        logger.warning("[%s] WFS query failed: %s", huc8, exc)
        if attempt_n < 2:
            time.sleep(10 * (attempt_n + 1))
            return process_huc8((huc8, bbox_dict, zctas_parquet, work_dir_str, attempt_n + 1))
        return huc8, None

    if catchments is None or catchments.empty:
        return huc8, None

    # Normalize COMID column
    for col in catchments.columns:
        if col.lower() in ("featureid", "comid"):
            catchments = catchments.rename(columns={col: "COMID"})
            break

    if "COMID" not in catchments.columns:
        logger.warning("[%s] No COMID column — columns: %s", huc8, list(catchments.columns))
        return huc8, None

    catchments = catchments[["COMID", "geometry"]].copy()
    catchments["COMID"] = catchments["COMID"].astype(str).str.strip()
    catchments = catchments[catchments.geometry.notna()]
    catchments["geometry"] = catchments["geometry"].apply(_make_valid)

    # --- Reproject ---
    catchments = catchments.to_crs(TARGET_CRS)

    # --- Load TIGER ZCTAs (pre-saved parquet) and clip to HUC8 extent ---
    zctas = gpd.read_parquet(zctas_parquet)
    bbox_albers = catchments.total_bounds
    zctas_clip = zctas.cx[bbox_albers[0]:bbox_albers[2], bbox_albers[1]:bbox_albers[3]]

    if zctas_clip.empty:
        return huc8, None

    # --- Spatial overlay ---
    catchments["catchment_area"] = catchments.geometry.area
    try:
        inter = gpd.overlay(
            catchments, zctas_clip[["zcta_id", "geometry"]],
            how="intersection", keep_geom_type=False,
        )
    except Exception:
        catchments["geometry"] = catchments.geometry.buffer(0)
        zctas_clip = zctas_clip.copy()
        zctas_clip["geometry"] = zctas_clip.geometry.buffer(0)
        try:
            inter = gpd.overlay(
                catchments, zctas_clip[["zcta_id", "geometry"]],
                how="intersection", keep_geom_type=False,
            )
        except Exception as exc2:
            logger.warning("[%s] Overlay failed: %s", huc8, exc2)
            return huc8, None

    if inter.empty:
        return huc8, None

    inter["area_fraction"] = (inter.geometry.area / inter["catchment_area"]).clip(0, 1)
    inter = inter[inter["area_fraction"] > 0.001].copy()

    af_sum = inter.groupby("COMID")["area_fraction"].sum()
    inter = inter.join(af_sum.rename("af_sum"), on="COMID")
    inter["area_fraction"] = (inter["area_fraction"] / inter["af_sum"]).clip(0, 1).round(6)

    result = inter[["COMID", "zcta_id", "area_fraction"]].copy()
    out_path = work_dir / "pairs.parquet"
    result.to_parquet(out_path, index=False)

    logger.info("[%s] %d catchments → %d COMID-ZCTA pairs", huc8, len(catchments), len(result))
    return huc8, str(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiger-dir", default="/opt/ml/processing/input/tiger")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--cache-dir", default="/tmp/nhdplus_cache")
    parser.add_argument("--huc2", nargs="+", default=None,
                        help="Subset of HUC2 codes (default: all 18 CONUS)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_base = Path("/tmp/huc8_work")
    work_base.mkdir(parents=True, exist_ok=True)

    huc2_codes = args.huc2 or CONUS_HUC2
    n_cpu = os.cpu_count() or 16
    n_workers = args.workers or min(n_cpu, 48)
    log.info("HUC2 regions: %s | Workers: %d | CPUs: %d", huc2_codes, n_workers, n_cpu)

    # --- Load and reproject TIGER ZCTAs ---
    log.info("=== LOADING TIGER ZCTAs ===")
    tiger_dir = Path(args.tiger_dir)
    shp_files = list(tiger_dir.glob("*.shp"))
    if not shp_files:
        log.error("No shapefile in %s", tiger_dir)
        sys.exit(1)

    zctas = gpd.read_file(shp_files[0], engine="pyogrio")
    zcta_col = next(
        (c for c in zctas.columns if "ZCTA5" in c.upper() or "GEOID" in c.upper()), None
    )
    if zcta_col:
        zctas = zctas.rename(columns={zcta_col: "zcta_id"})
    zctas["zcta_id"] = zctas["zcta_id"].astype(str).str.zfill(5)
    zctas = zctas[["zcta_id", "geometry"]][zctas.geometry.notna()].copy()
    zctas["geometry"] = zctas["geometry"].apply(make_valid)
    zctas = zctas.to_crs(TARGET_CRS)
    log.info("ZCTAs: %d rows in %s", len(zctas), TARGET_CRS)

    zctas_parquet = cache_dir / "zctas_albers.parquet"
    zctas.to_parquet(zctas_parquet)
    log.info("TIGER parquet saved: %s", zctas_parquet)

    # --- Get HUC8 bboxes ---
    log.info("=== FETCHING HUC8 BOUNDARIES ===")
    huc8_list = get_huc8_bboxes(huc2_codes)
    if not huc8_list:
        log.error("No HUC8s retrieved")
        sys.exit(1)
    log.info("Total HUC8 work units: %d", len(huc8_list))

    # --- Parallel HUC8 processing ---
    log.info("=== PARALLEL HUC8 PROCESSING (%d workers) ===", n_workers)
    worker_args = [
        (
            item["huc8"],
            item,
            str(zctas_parquet),
            str(work_base / f"huc8_{item['huc8']}"),
            0,
        )
        for item in huc8_list
    ]

    results = {}
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(process_huc8, wa): wa[0] for wa in worker_args}
        for future in as_completed(futures):
            huc8 = futures[future]
            try:
                huc8_out, path = future.result()
                results[huc8_out] = path
                n_done = sum(1 for p in results.values() if p)
                n_fail = sum(1 for p in results.values() if p is None)
                remaining = len(huc8_list) - len(results)
                if len(results) % 50 == 0:
                    log.info("Progress: done=%d failed=%d remaining=%d", n_done, n_fail, remaining)
            except Exception as exc:
                log.warning("HUC8 %s exception: %s", huc8, exc)
                results[huc8] = None

    elapsed = time.time() - t0
    log.info("All HUC8s finished in %.1f min", elapsed / 60)

    # --- Assemble ---
    log.info("=== ASSEMBLING CROSSWALK ===")
    parts = [pd.read_parquet(p) for p in results.values() if p and Path(p).exists()]

    if not parts:
        log.error("No HUC8 pairs produced")
        sys.exit(1)

    result = pd.concat(parts, ignore_index=True)
    result = result.drop_duplicates(subset=["COMID", "zcta_id"])
    log.info("Total: %d pairs | %d COMIDs | %d ZCTAs",
             len(result), result["COMID"].nunique(), result["zcta_id"].nunique())

    af_sum = result.groupby("COMID")["area_fraction"].sum()
    pct_ok = (af_sum > 0.5).mean() * 100
    log.info("COMIDs af_sum > 0.5: %.1f%% | HUC8s succeeded: %d/%d",
             pct_ok, sum(1 for p in results.values() if p), len(huc8_list))

    # --- Save and upload ---
    out_path = output_dir / "comid_zcta_crosswalk.parquet"
    result.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info("Saved: %s (%.1f MB)", out_path, size_mb)
    _s3_upload(str(out_path), S3_OUTPUT_KEY)

    provenance = {
        "operation": "build_comid_zcta_crosswalk",
        "timestamp": timestamp,
        "source": "USGS WaterData WFS (pynhd WaterData catchmentsp)",
        "tiger_source": shp_files[0].name,
        "crs": TARGET_CRS,
        "huc2_regions": huc2_codes,
        "huc8_attempted": len(huc8_list),
        "huc8_succeeded": sum(1 for p in results.values() if p),
        "n_pairs": len(result),
        "n_comids": int(result["COMID"].nunique()),
        "n_zctas": int(result["zcta_id"].nunique()),
        "elapsed_min": round(elapsed / 60, 1),
        "n_workers": n_workers,
        "file_size_mb": round(size_mb, 1),
    }
    prov_path = output_dir / "comid_zcta_crosswalk_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    _s3_upload(str(prov_path), S3_PROVENANCE_KEY)
    log.info("Done.")


if __name__ == "__main__":
    main()
