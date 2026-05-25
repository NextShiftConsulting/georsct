#!/usr/bin/env python3
"""
run_comid_zcta.py -- SageMaker run script: COMID-to-ZCTA spatial crosswalk.

Builds comid_zcta_crosswalk.parquet — spatial join between NHDPlus V2
catchment polygons (COMID) and Census 2020 ZCTAs with area fraction weights.

Strategy:
  1. Load TIGER ZCTAs, reproject to EPSG:5070, save as parquet for workers
  2. ProcessPoolExecutor: all 21 VPUs in parallel (one worker per VPU)
     Each worker: download 7zip from EDCINTL → extract → overlay → parquet
     VPU parquets cached to S3 so reruns skip download entirely
  3. Concatenate, deduplicate, validate, upload

Instance: ml.m5.8xlarge (32 vCPU, 128 GB RAM, $1.845/hr)
Runtime:  ~20-30 min (all 21 VPUs parallel, bounded by slowest VPU)
Cost:     ~$0.90

Output: s3://swarm-yrsn-datasets/rsct_curriculum/geo/comid_zcta_crosswalk.parquet
  - COMID         (str)
  - zcta_id       (str, 5-digit)
  - area_fraction (float, fraction of COMID area inside ZCTA)
"""

import io
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
import geopandas as gpd
import pandas as pd
import py7zr
import requests
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
S3_VPU_CACHE_PREFIX = "rsct_curriculum/geo/nhdplus_vpu_cache"
TARGET_CRS = "EPSG:5070"

EDCINTL_BASE = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/NHDPlusV21/Data/"
)

VPU_MAP = {
    "01":  ("NHDPlusNE", "NE"),
    "02":  ("NHDPlusMA", "MA"),
    "03N": ("NHDPlusSA", "SA"),
    "03S": ("NHDPlusSA", "SA"),
    "03W": ("NHDPlusSA", "SA"),
    "04":  ("NHDPlusGL", "GL"),
    "05":  ("NHDPlusMS", "MS"),
    "06":  ("NHDPlusMS", "MS"),
    "07":  ("NHDPlusMS", "MS"),
    "08":  ("NHDPlusMS", "MS"),
    "09":  ("NHDPlusMS", "MS"),
    "10L": ("NHDPlusMO", "MO"),
    "10U": ("NHDPlusMO", "MO"),
    "11":  ("NHDPlusAR", "AR"),
    "12":  ("NHDPlusTX", "TX"),
    "13":  ("NHDPlusRG", "RG"),
    "14":  ("NHDPlusRG", "RG"),
    "15":  ("NHDPlusRG", "RG"),
    "16":  ("NHDPlusPN", "PN"),
    "17":  ("NHDPlusPN", "PN"),
    "18":  ("NHDPlusCA", "CA"),
}


# ---------------------------------------------------------------------------
# S3 helpers (no profile — uses IAM role inside SageMaker)
# ---------------------------------------------------------------------------
def _s3():
    return boto3.client("s3")

def _s3_upload(local_path: str, key: str):
    try:
        _s3().upload_file(local_path, S3_BUCKET, key)
        log.info("  -> s3://%s/%s", S3_BUCKET, key)
    except Exception as e:
        log.warning("  S3 upload failed for %s: %s", key, e)

def _s3_exists(key: str) -> bool:
    try:
        _s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False

def _s3_download_to(key: str, local_path: Path) -> bool:
    try:
        _s3().download_file(S3_BUCKET, key, str(local_path))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-VPU worker function (runs in subprocess)
# ---------------------------------------------------------------------------
def process_vpu(args: tuple) -> tuple[str, str | None]:
    """Download, extract, overlay one VPU. Returns (vpu, output_parquet_path | None)."""
    vpu, zctas_parquet_path, work_dir_str, cache_dir_str = args
    work_dir = Path(work_dir_str) / f"vpu{vpu}"
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir_str)

    # Subprocess gets its own logger
    logger = logging.getLogger(f"vpu{vpu}")

    # --- Load TIGER ZCTAs (pre-saved as parquet by main process) ---
    zctas = gpd.read_parquet(zctas_parquet_path)

    # --- Get catchments (S3 cache or EDCINTL download) ---
    dir_name, reg_code = VPU_MAP[vpu]
    cache_key = f"{S3_VPU_CACHE_PREFIX}/catchments_vpu{vpu}.parquet"
    cache_local = cache_dir / f"catchments_vpu{vpu}.parquet"

    catchments = None

    if _s3_exists(cache_key):
        logger.info("[%s] Loading catchments from S3 cache", vpu)
        if _s3_download_to(cache_key, cache_local):
            catchments = gpd.read_parquet(cache_local)
            logger.info("[%s] %d catchments from cache", vpu, len(catchments))

    if catchments is None:
        candidate_urls = [
            f"{EDCINTL_BASE}{dir_name}/NHDPlusV21_{reg_code}_{vpu}_NHDPlusCatchment.7z",
            f"{EDCINTL_BASE}{dir_name}/NHDPlusV21_{reg_code}NHDPlusCatchment.7z",
            f"{EDCINTL_BASE}{dir_name}/NHDPlusV21_{vpu}_NHDPlusCatchment.7z",
        ]
        archive = work_dir / f"vpu{vpu}.7z"
        shp_dir = work_dir / "shp"
        shp_dir.mkdir(exist_ok=True)

        downloaded = False
        for url in candidate_urls:
            for attempt in range(3):
                try:
                    resp = requests.get(url, timeout=300, stream=True)
                    if resp.status_code == 404:
                        break
                    resp.raise_for_status()
                    with open(archive, "wb") as f:
                        for chunk in resp.iter_content(1024 * 1024):
                            f.write(chunk)
                    downloaded = True
                    break
                except Exception as exc:
                    if attempt == 2:
                        pass
                    else:
                        time.sleep(5 * (attempt + 1))
            if downloaded:
                break

        if not downloaded:
            logger.warning("[%s] All download URLs failed — skipping", vpu)
            return vpu, None

        try:
            with py7zr.SevenZipFile(archive, mode="r") as z:
                z.extractall(path=shp_dir)
            archive.unlink()
        except Exception as exc:
            logger.warning("[%s] 7zip extraction failed: %s", vpu, exc)
            return vpu, None

        shp_files = list(shp_dir.rglob("*.shp"))
        shp_files = [f for f in shp_files
                     if "catchment" in f.name.lower()] or shp_files
        if not shp_files:
            logger.warning("[%s] No shapefile found", vpu)
            return vpu, None

        try:
            gdf = gpd.read_file(shp_files[0], engine="pyogrio")
        except Exception as exc:
            logger.warning("[%s] Shapefile read failed: %s", vpu, exc)
            return vpu, None

        comid_col = next(
            (c for c in gdf.columns if c.upper() in ("FEATUREID", "COMID")), None
        )
        if comid_col is None:
            logger.warning("[%s] No COMID column", vpu)
            return vpu, None

        catchments = gdf[[comid_col, "geometry"]].rename(columns={comid_col: "COMID"})
        catchments["COMID"] = catchments["COMID"].astype(str).str.strip()
        catchments = catchments[catchments.geometry.notna()].copy()
        catchments["geometry"] = catchments["geometry"].apply(make_valid)

        # Cache to S3
        catchments.to_parquet(cache_local)
        _s3_upload(str(cache_local), cache_key)
        shutil.rmtree(shp_dir, ignore_errors=True)

    # --- Reproject both to equal-area CRS ---
    catchments = catchments.to_crs(TARGET_CRS)

    # Clip ZCTAs to VPU extent
    bbox = catchments.total_bounds
    zctas_clip = zctas.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    if zctas_clip.empty:
        logger.info("[%s] No ZCTAs in extent — skipping", vpu)
        return vpu, None

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
        inter = gpd.overlay(
            catchments, zctas_clip[["zcta_id", "geometry"]],
            how="intersection", keep_geom_type=False,
        )

    if inter.empty:
        return vpu, None

    inter["area_fraction"] = inter.geometry.area / inter["catchment_area"]
    inter = inter[inter["area_fraction"] > 0.001].copy()

    # Normalize per COMID
    af_sum = inter.groupby("COMID")["area_fraction"].sum()
    inter = inter.join(af_sum.rename("af_sum"), on="COMID")
    inter["area_fraction"] = (inter["area_fraction"] / inter["af_sum"]).clip(0, 1).round(6)

    result = inter[["COMID", "zcta_id", "area_fraction"]].copy()
    out_path = work_dir / "pairs.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("[%s] Done: %d pairs", vpu, len(result))
    return vpu, str(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiger-dir", default="/opt/ml/processing/input/tiger")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--cache-dir", default="/tmp/nhdplus_cache")
    parser.add_argument("--vpus", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: min(cpu_count, n_vpus))")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/tmp/vpu_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    vpus = args.vpus or list(VPU_MAP.keys())
    n_cpu = os.cpu_count() or 8
    n_workers = args.workers or min(n_cpu, len(vpus))
    log.info("VPUs: %d | Workers: %d | CPUs: %d", len(vpus), n_workers, n_cpu)

    # --- Load TIGER ZCTAs once, reproject, save as parquet for workers ---
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
    zctas = zctas[["zcta_id", "geometry"]].copy()
    zctas = zctas[zctas.geometry.notna()]
    zctas["geometry"] = zctas["geometry"].apply(make_valid)
    zctas = zctas.to_crs(TARGET_CRS)
    log.info("ZCTAs: %d rows, reprojected to %s", len(zctas), TARGET_CRS)

    zctas_parquet = cache_dir / "zctas_albers.parquet"
    zctas.to_parquet(zctas_parquet)
    log.info("TIGER parquet saved for workers: %s", zctas_parquet)

    # --- Parallel VPU processing ---
    log.info("=== PARALLEL VPU PROCESSING (%d workers) ===", n_workers)
    worker_args = [
        (vpu, str(zctas_parquet), str(work_dir), str(cache_dir))
        for vpu in vpus
    ]

    results = {}
    t0 = time.time()

    # Use spawn to avoid geopandas fork issues
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(process_vpu, wargs): wargs[0] for wargs in worker_args}
        for future in as_completed(futures):
            vpu = futures[future]
            try:
                vpu_out, path = future.result()
                results[vpu_out] = path
                done = sum(1 for v in results if results[v] is not None)
                failed = sum(1 for v in results if results[v] is None)
                log.info("VPU %s complete | done=%d failed=%d remaining=%d",
                         vpu_out, done, failed, len(vpus) - len(results))
            except Exception as exc:
                log.warning("VPU %s raised exception: %s", vpu, exc)
                results[vpu] = None

    elapsed = time.time() - t0
    log.info("All VPUs finished in %.1f min", elapsed / 60)

    # --- Assemble ---
    log.info("=== ASSEMBLING CROSSWALK ===")
    parts = []
    for vpu, path in results.items():
        if path and Path(path).exists():
            parts.append(pd.read_parquet(path))

    if not parts:
        log.error("No VPU pairs produced — all workers failed")
        sys.exit(1)

    result = pd.concat(parts, ignore_index=True)
    result = result.drop_duplicates(subset=["COMID", "zcta_id"])
    log.info("Total: %d pairs | %d COMIDs | %d ZCTAs",
             len(result), result["COMID"].nunique(), result["zcta_id"].nunique())

    # Validation
    af_sum = result.groupby("COMID")["area_fraction"].sum()
    pct_ok = (af_sum > 0.5).mean() * 100
    log.info("COMIDs with af_sum > 0.5: %.1f%%", pct_ok)
    log.info("VPUs succeeded: %d / %d", sum(1 for p in results.values() if p), len(vpus))

    # --- Save and upload ---
    out_path = output_dir / "comid_zcta_crosswalk.parquet"
    result.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info("Saved: %s (%.1f MB)", out_path, size_mb)
    _s3_upload(str(out_path), S3_OUTPUT_KEY)

    provenance = {
        "operation": "build_comid_zcta_crosswalk",
        "timestamp": timestamp,
        "tiger_source": shp_files[0].name,
        "nhdplus_source": EDCINTL_BASE,
        "crs": TARGET_CRS,
        "vpus_attempted": len(vpus),
        "vpus_succeeded": sum(1 for p in results.values() if p),
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
