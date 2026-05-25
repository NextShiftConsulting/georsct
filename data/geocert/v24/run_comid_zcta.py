#!/usr/bin/env python3
"""
run_comid_zcta.py -- SageMaker run script: COMID-to-ZCTA spatial crosswalk.

Builds comid_zcta_crosswalk.parquet — the spatial join between NHDPlus V2
catchment polygons (COMID) and Census 2020 ZCTAs with area fraction weights.

Required for production-quality TWI aggregation (build_twi_features.py).
Without this, TWI falls back to national medians (useless).

Strategy:
  1. Load TIGER ZCTA polygons from /opt/ml/processing/input/tiger/
  2. For each NHDPlus VPU (21 CONUS regions):
     a. Download catchment shapefile from EDCINTL (7zip)
     b. Reproject to EPSG:5070 (Albers Equal Area CONUS)
     c. geopandas overlay (intersection) with ZCTAs
     d. Compute area_fraction = intersection_area / catchment_area
  3. Concatenate all VPUs, deduplicate
  4. Upload parquet to S3

Output: comid_zcta_crosswalk.parquet
  - COMID         (str)   NHDPlus V2 catchment identifier
  - zcta_id       (str)   5-digit ZCTA
  - area_fraction (float) fraction of COMID area inside ZCTA [0,1]

Instance: ml.m5.4xlarge (16 vCPU, 64 GB RAM). Runtime: ~90-150 min.

S3 output:
  s3://swarm-yrsn-datasets/rsct_curriculum/geo/comid_zcta_crosswalk.parquet
"""

import io
import json
import logging
import sys
import time
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
S3_BUCKET = "swarm-yrsn-datasets"
S3_OUTPUT_KEY = "rsct_curriculum/geo/comid_zcta_crosswalk.parquet"
S3_PROVENANCE_KEY = "rsct_curriculum/geo/comid_zcta_crosswalk_provenance.json"
S3_CACHE_PREFIX = "rsct_curriculum/geo/nhdplus_vpu_cache"

# Equal-area CRS for accurate area calculations (CONUS Albers)
TARGET_CRS = "EPSG:5070"

EDCINTL_BASE = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/NHDPlusV21/Data/"
)

# VPU → (directory name, file region code)
# Directory names match EDCINTL folder structure; region codes match filenames
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
# S3 helpers
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
# NHDPlus download
# ---------------------------------------------------------------------------
def download_vpu_catchments(vpu: str, cache_dir: Path) -> gpd.GeoDataFrame | None:
    """Download and extract NHDPlus V2 catchment shapefile for one VPU.

    Checks S3 cache first (parquet), then EDCINTL, then tries alternate URL patterns.
    Returns GeoDataFrame with COMID + geometry columns, or None on failure.
    """
    # Check S3 parquet cache
    cache_key = f"{S3_CACHE_PREFIX}/catchments_vpu{vpu}.parquet"
    cache_local = cache_dir / f"catchments_vpu{vpu}.parquet"

    if _s3_exists(cache_key):
        log.info("  VPU %s: loading from S3 cache", vpu)
        if _s3_download_to(cache_key, cache_local):
            gdf = gpd.read_parquet(cache_local)
            log.info("  VPU %s: %d catchments from cache", vpu, len(gdf))
            return gdf

    dir_name, reg_code = VPU_MAP[vpu]

    # Try multiple URL patterns (EDCINTL naming is inconsistent across VPUs)
    candidate_urls = [
        f"{EDCINTL_BASE}{dir_name}/NHDPlusV21_{reg_code}_{vpu}_NHDPlusCatchment.7z",
        f"{EDCINTL_BASE}{dir_name}/NHDPlusV21_{reg_code}NHDPlusCatchment.7z",
        f"{EDCINTL_BASE}{dir_name}/NHDPlusV21_{vpu}_NHDPlusCatchment.7z",
        f"{EDCINTL_BASE}{dir_name}/NHDPlusCatchment/NHDPlusV21_{reg_code}_{vpu}_Catchment.7z",
    ]

    archive_local = cache_dir / f"vpu{vpu}_catchment.7z"
    shp_dir = cache_dir / f"vpu{vpu}_shp"
    shp_dir.mkdir(exist_ok=True)

    downloaded = False
    for url in candidate_urls:
        log.info("  VPU %s: trying %s", vpu, url.split("/")[-1])
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=300, stream=True)
                if resp.status_code == 404:
                    break  # wrong URL, try next
                resp.raise_for_status()
                with open(archive_local, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                downloaded = True
                log.info("  VPU %s: downloaded %.1f MB", vpu,
                         archive_local.stat().st_size / (1024 * 1024))
                break
            except Exception as exc:
                if attempt == 2:
                    log.warning("  VPU %s: attempt failed for %s: %s", vpu, url, exc)
                else:
                    time.sleep(5 * (attempt + 1))
        if downloaded:
            break

    if not downloaded:
        log.warning("  VPU %s: all URLs failed — skipping", vpu)
        return None

    # Extract 7zip
    try:
        with py7zr.SevenZipFile(archive_local, mode="r") as z:
            z.extractall(path=shp_dir)
        archive_local.unlink()  # free space
    except Exception as exc:
        log.warning("  VPU %s: 7zip extraction failed: %s", vpu, exc)
        return None

    # Find the catchment shapefile
    shp_files = list(shp_dir.rglob("*.shp"))
    catchment_shps = [f for f in shp_files
                      if "catchment" in f.name.lower() or "Catchment" in f.name]
    if not catchment_shps:
        catchment_shps = shp_files  # take whatever is there

    if not catchment_shps:
        log.warning("  VPU %s: no shapefile found after extraction", vpu)
        return None

    shp_path = catchment_shps[0]
    log.info("  VPU %s: reading %s", vpu, shp_path.name)

    try:
        gdf = gpd.read_file(shp_path, engine="pyogrio")
    except Exception as exc:
        log.warning("  VPU %s: shapefile read failed: %s", vpu, exc)
        return None

    # Normalize COMID column name
    comid_col = next(
        (c for c in gdf.columns if c.upper() in ("FEATUREID", "COMID")), None
    )
    if comid_col is None:
        log.warning("  VPU %s: no COMID/FEATUREID column — columns: %s",
                    vpu, list(gdf.columns))
        return None

    gdf = gdf[[comid_col, "geometry"]].rename(columns={comid_col: "COMID"})
    gdf["COMID"] = gdf["COMID"].astype(str).str.strip()
    gdf = gdf[gdf.geometry.notna()].copy()

    # Fix invalid geometries
    gdf["geometry"] = gdf["geometry"].apply(make_valid)
    log.info("  VPU %s: %d catchments loaded", vpu, len(gdf))

    # Cache to S3 parquet for future runs
    cache_local.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache_local)
    _s3_upload(str(cache_local), cache_key)

    # Clean up extracted files
    import shutil
    shutil.rmtree(shp_dir, ignore_errors=True)

    return gdf


# ---------------------------------------------------------------------------
# Spatial overlay: one VPU × all ZCTAs
# ---------------------------------------------------------------------------
def overlay_vpu(
    catchments: gpd.GeoDataFrame,
    zctas: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Compute COMID → ZCTA area fractions for one VPU.

    Both inputs must be in the same CRS (TARGET_CRS = EPSG:5070).
    Returns DataFrame: COMID, zcta_id, area_fraction.
    """
    # Compute catchment areas before overlay (in m²)
    catchments = catchments.copy()
    catchments["catchment_area"] = catchments.geometry.area

    log.info("  Overlay: %d catchments × %d ZCTAs", len(catchments), len(zctas))
    t0 = time.time()

    # Clip to VPU extent first (avoids touching all 33K ZCTAs for each VPU)
    vpu_bbox = catchments.total_bounds  # [minx, miny, maxx, maxy]
    zctas_clip = zctas.cx[vpu_bbox[0]:vpu_bbox[2], vpu_bbox[1]:vpu_bbox[3]]
    log.info("  ZCTAs in VPU extent: %d", len(zctas_clip))

    if zctas_clip.empty:
        log.info("  No ZCTAs in this VPU extent — skipping")
        return pd.DataFrame(columns=["COMID", "zcta_id", "area_fraction"])

    # Spatial overlay
    try:
        inter = gpd.overlay(catchments, zctas_clip[["zcta_id", "geometry"]],
                            how="intersection", keep_geom_type=False)
    except Exception as exc:
        log.warning("  Overlay failed: %s — trying with buffer(0) fix", exc)
        catchments["geometry"] = catchments.geometry.buffer(0)
        zctas_clip = zctas_clip.copy()
        zctas_clip["geometry"] = zctas_clip.geometry.buffer(0)
        inter = gpd.overlay(catchments, zctas_clip[["zcta_id", "geometry"]],
                            how="intersection", keep_geom_type=False)

    elapsed = time.time() - t0
    log.info("  Overlay complete: %d intersections in %.1fs", len(inter), elapsed)

    if inter.empty:
        return pd.DataFrame(columns=["COMID", "zcta_id", "area_fraction"])

    inter["intersection_area"] = inter.geometry.area
    inter["area_fraction"] = inter["intersection_area"] / inter["catchment_area"]

    # Keep only meaningful overlaps (>0.1% of catchment area)
    inter = inter[inter["area_fraction"] > 0.001].copy()

    # Normalize so area fractions sum to ≤1 per COMID
    comid_total = inter.groupby("COMID")["area_fraction"].sum()
    inter = inter.join(comid_total.rename("af_sum"), on="COMID")
    inter["area_fraction"] = (inter["area_fraction"] / inter["af_sum"]).clip(0, 1)

    result = inter[["COMID", "zcta_id", "area_fraction"]].copy()
    result["area_fraction"] = result["area_fraction"].round(6)
    log.info("  %d COMID-ZCTA pairs (area_fraction > 0.001)", len(result))
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build COMID-ZCTA spatial crosswalk (SageMaker)"
    )
    parser.add_argument("--tiger-dir", default="/opt/ml/processing/input/tiger")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--cache-dir", default="/tmp/nhdplus_cache")
    parser.add_argument("--vpus", nargs="+", default=None,
                        help="Subset of VPUs to process (default: all 21)")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    vpus_to_run = args.vpus if args.vpus else list(VPU_MAP.keys())
    log.info("VPUs to process: %s", vpus_to_run)

    # --- Load TIGER ZCTAs ---
    log.info("=== LOADING TIGER ZCTAs ===")
    tiger_dir = Path(args.tiger_dir)
    shp_files = list(tiger_dir.glob("*.shp"))
    if not shp_files:
        log.error("No shapefile in %s", tiger_dir)
        sys.exit(1)
    tiger_shp = shp_files[0]
    log.info("Loading %s", tiger_shp.name)

    zctas = gpd.read_file(tiger_shp, engine="pyogrio")
    # Normalize ZCTA column (Census uses ZCTA5CE20 or GEOID20)
    zcta_col = next(
        (c for c in zctas.columns if "ZCTA5" in c.upper() or "GEOID" in c.upper()), None
    )
    if zcta_col:
        zctas = zctas.rename(columns={zcta_col: "zcta_id"})
    zctas["zcta_id"] = zctas["zcta_id"].astype(str).str.zfill(5)
    zctas = zctas[["zcta_id", "geometry"]].copy()
    zctas = zctas[zctas.geometry.notna()]
    zctas["geometry"] = zctas["geometry"].apply(make_valid)
    log.info("ZCTAs loaded: %d", len(zctas))

    # Reproject to equal-area CRS
    log.info("Reprojecting ZCTAs to %s", TARGET_CRS)
    zctas = zctas.to_crs(TARGET_CRS)

    # --- Process each VPU ---
    log.info("=== PROCESSING VPUs ===")
    all_pairs = []
    vpus_done = 0
    vpus_failed = 0

    for vpu in vpus_to_run:
        log.info("\n--- VPU %s ---", vpu)
        catchments = download_vpu_catchments(vpu, cache_dir)

        if catchments is None:
            vpus_failed += 1
            continue

        # Reproject catchments
        catchments = catchments.to_crs(TARGET_CRS)

        pairs = overlay_vpu(catchments, zctas)
        if not pairs.empty:
            pairs["vpu"] = vpu
            all_pairs.append(pairs)

        vpus_done += 1
        log.info("VPU %s done. Total pairs so far: %d",
                 vpu, sum(len(p) for p in all_pairs))

        # Checkpoint to S3 every 5 VPUs
        if vpus_done % 5 == 0 and all_pairs:
            partial = pd.concat(all_pairs, ignore_index=True)
            ckpt_path = output_dir / "comid_zcta_partial.parquet"
            partial.to_parquet(ckpt_path, index=False)
            _s3_upload(str(ckpt_path), "rsct_curriculum/geo/comid_zcta_partial.parquet")
            log.info("Checkpoint saved (%d VPUs done, %d pairs)", vpus_done, len(partial))

    # --- Assemble final result ---
    log.info("\n=== ASSEMBLING FINAL CROSSWALK ===")
    if not all_pairs:
        log.error("No VPU pairs produced — all downloads failed")
        sys.exit(1)

    result = pd.concat(all_pairs, ignore_index=True)
    result = result.drop_duplicates(subset=["COMID", "zcta_id"])
    log.info("Final: %d COMID-ZCTA pairs, %d unique COMIDs, %d unique ZCTAs",
             len(result), result["COMID"].nunique(), result["zcta_id"].nunique())

    # Drop helper column
    result = result[["COMID", "zcta_id", "area_fraction"]]

    # --- Validation ---
    log.info("=== VALIDATION ===")
    # Each COMID's area fractions should sum to ~1.0 (some edge COMIDs may be < 1)
    af_sum = result.groupby("COMID")["area_fraction"].sum()
    log.info("Area fraction per COMID: mean=%.3f, min=%.3f, max=%.3f",
             af_sum.mean(), af_sum.min(), af_sum.max())
    pct_ok = (af_sum > 0.5).mean() * 100
    log.info("COMIDs with >50%% area accounted for: %.1f%%", pct_ok)

    if pct_ok < 70:
        log.warning("VALIDATION WARN: <70%% of COMIDs have af_sum > 0.5 — "
                    "possible CRS mismatch or incomplete VPU coverage")

    # --- Save and upload ---
    out_path = output_dir / "comid_zcta_crosswalk.parquet"
    result.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info("Saved: %s (%.1f MB)", out_path, size_mb)

    _s3_upload(str(out_path), S3_OUTPUT_KEY)

    provenance = {
        "operation": "build_comid_zcta_crosswalk",
        "timestamp": timestamp,
        "tiger_source": "tl_2022_us_zcta520.shp",
        "nhdplus_source": EDCINTL_BASE,
        "crs": TARGET_CRS,
        "vpus_attempted": len(vpus_to_run),
        "vpus_succeeded": vpus_done,
        "vpus_failed": vpus_failed,
        "n_pairs": len(result),
        "n_comids": int(result["COMID"].nunique()),
        "n_zctas": int(result["zcta_id"].nunique()),
        "file_size_mb": round(size_mb, 1),
    }
    prov_path = output_dir / "comid_zcta_crosswalk_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    _s3_upload(str(prov_path), S3_PROVENANCE_KEY)

    log.info("Done. VPUs: %d/%d succeeded.", vpus_done, len(vpus_to_run))


if __name__ == "__main__":
    main()
