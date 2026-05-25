#!/usr/bin/env python3
"""
Combined Stage 2+3: Extract FEMA shapefiles from raw zips AND overlay
against TIGER ZCTA boundaries in one pass. No intermediate JSON.

Pipeline per county:
  S3 zip -> download -> unzip -> fiona read S_FLD_HAZ_AR -> classify zones
  -> reproject EPSG:5070 -> intersect with ZCTA boundaries -> accumulate areas

Output: flood_zones_zcta.parquet (per-ZCTA area fractions)

Instance: ml.m5.12xlarge (48 vCPU, 192 GB, 10 Gbps)
Threads:  48 (1x CPU — overlay is CPU-bound, not I/O-bound like pure download)
"""

import argparse
import gc
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

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
S3_RAW_PREFIX = "rsct_curriculum/series_018/processed/flood_raw/"
S3_OUTPUT_KEY = "rsct_curriculum/series_018/processed/flood_zones_zcta.parquet"
S3_PROVENANCE_KEY = "rsct_curriculum/series_018/processed/flood_zones_provenance.json"

NON_CONUS_PREFIXES = {"02", "15", "60", "66", "69", "72", "78"}
CONUS_5070_BOUNDS = (-2_500_000, 100_000, 2_400_000, 3_300_000)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def _s3_client():
    import boto3
    return boto3.client("s3")


def _s3_upload(s3, local_path: str, key: str):
    s3.upload_file(local_path, S3_BUCKET, key)
    log.info("  -> s3://%s/%s", S3_BUCKET, key)


def _s3_list_prefix(s3, prefix: str, suffix: str = "") -> list:
    results = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            fname = obj["Key"].split("/")[-1]
            if suffix and not fname.endswith(suffix):
                continue
            results.append((obj["Key"], fname))
    return results


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _signed_area(ring: list) -> float:
    n = len(ring)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ring[i][0] * ring[j][1]
        area -= ring[j][0] * ring[i][1]
    return area / 2.0


def _rings_to_shapely(rings: list):
    """Convert Esri-style rings to shapely polygon."""
    if not rings:
        return None
    try:
        shells, holes = [], []
        for ring in rings:
            if len(ring) < 4:
                continue
            sa = _signed_area(ring)
            if sa < 0:
                shells.append(ring)
            elif sa > 0:
                holes.append(ring)
            else:
                shells.append(ring)

        if not shells:
            if holes:
                shells = holes
                holes = []
            else:
                return None

        if len(shells) == 1:
            poly = Polygon(shells[0], holes) if holes else Polygon(shells[0])
        else:
            polys = [Polygon(s) for s in shells]
            poly = MultiPolygon([(p.exterior.coords, []) for p in polys if p.is_valid])
            if poly.is_empty:
                poly = Polygon(shells[0], holes) if holes else Polygon(shells[0])

        return poly.buffer(0) if not poly.is_valid else poly
    except Exception:
        return None


def _geom_to_rings(geom: dict) -> list:
    """Convert fiona/GeoJSON geometry to Esri-style rings."""
    if geom is None:
        return []
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if gtype == "Polygon":
        return [list(ring) for ring in coords]
    elif gtype == "MultiPolygon":
        rings = []
        for polygon_coords in coords:
            for ring in polygon_coords:
                rings.append(list(ring))
        return rings
    return []


def _rings_to_shapely_checked(rings: list, vstats: dict):
    """Like _rings_to_shapely but tracks Check 6 (area conservation on repair)."""
    if not rings:
        return None
    try:
        shells, holes = [], []
        for ring in rings:
            if len(ring) < 4:
                continue
            sa = _signed_area(ring)
            if sa < 0:
                shells.append(ring)
            elif sa > 0:
                holes.append(ring)
            else:
                shells.append(ring)

        if not shells:
            if holes:
                shells = holes
                holes = []
            else:
                return None

        if len(shells) == 1:
            poly = Polygon(shells[0], holes) if holes else Polygon(shells[0])
        else:
            polys = [Polygon(s) for s in shells]
            poly = MultiPolygon([(p.exterior.coords, []) for p in polys if p.is_valid])
            if poly.is_empty:
                poly = Polygon(shells[0], holes) if holes else Polygon(shells[0])

        if not poly.is_valid:
            original_area = poly.area
            poly = poly.buffer(0)
            vstats["repaired"] += 1
            # Check 6: area conservation (allow +/- 10%)
            if original_area > 0:
                ratio = poly.area / original_area
                if ratio < 0.9 or ratio > 1.1:
                    vstats["area_ratio_bad"] += 1

        return poly
    except Exception:
        return None


def classify_zone(fld_zone: str, zone_subty: str) -> str:
    if not fld_zone:
        return "X"
    zone = str(fld_zone).strip().upper()
    subty = str(zone_subty).strip().upper() if zone_subty else ""
    if zone in ("A", "AE", "AH", "AO", "AR", "A99", "VE", "V"):
        return "A"
    if "FLOODWAY" in zone:
        return "A"
    if zone == "X" and ("500" in subty or "0.2" in subty or "SHADED" in subty):
        return "X500"
    if "0.2 PCT" in zone:
        return "X500"
    return "X"


# ---------------------------------------------------------------------------
# Per-county: download zip, extract shapefile, overlay, return area dicts
# ---------------------------------------------------------------------------
def process_county(s3, zip_key: str, dfirm_id: str, fips: str,
                   county_zcta_indices: list, zcta_proj,
                   work_dir: Path) -> dict:
    """Download zip, extract S_FLD_HAZ_AR, overlay against ZCTAs.

    Returns {"fips": str, "n_features": int, "zone_a": {idx: area}, "zone_x500": {idx: area},
             "vstats": dict, "error": str|None}
    """
    import threading
    tid = threading.get_ident()
    zip_path = work_dir / f"{dfirm_id}_{tid}.zip"
    extract_dir = work_dir / f"{dfirm_id}_{tid}_ext"

    _empty = {"fips": fips, "n_features": 0, "zone_a": {}, "zone_x500": {},
              "vstats": {}, "error": None}

    try:
        # Download from S3
        s3.download_file(S3_BUCKET, zip_key, str(zip_path))

        # Validate zip
        if not zipfile.is_zipfile(zip_path):
            return {**_empty, "error": "corrupt zip"}
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                return {**_empty, "error": f"bad entry: {bad}"}
            zf.extractall(extract_dir)

        # Find shapefile
        import fiona
        shp_paths = [p for p in extract_dir.rglob("*.shp")
                     if p.stem.upper() == "S_FLD_HAZ_AR"]
        gdb_paths = list(extract_dir.rglob("*.gdb"))

        if shp_paths:
            data_path = str(shp_paths[0])
            layer_arg = {}
        elif gdb_paths:
            data_path = str(gdb_paths[0])
            layers = fiona.listlayers(data_path)
            if "S_Fld_Haz_Ar" not in layers:
                return {**_empty, "error": "No S_Fld_Haz_Ar layer"}
            layer_arg = {"layer": "S_Fld_Haz_Ar"}
        else:
            return {**_empty, "error": "No .shp or .gdb found"}

        # Validation stats (Checks 3, 4, 6)
        vstats = {
            "null_geom": 0, "degenerate": 0, "unclosed": 0,
            "cw_outer": 0, "ccw_outer": 0, "holes": 0,
            "repaired": 0, "area_ratio_bad": 0,
        }

        # Read shapefile, classify zones, build geometry lists
        zone_a_polys = []
        zone_x500_polys = []
        n_features = 0

        with fiona.open(data_path, **layer_arg) as src:
            for feat in src:
                geom = feat.get("geometry", {})
                rings = _geom_to_rings(geom)
                if not rings:
                    vstats["null_geom"] += 1
                    continue
                n_features += 1

                # Check 4: winding order (sample first ring)
                if rings and len(rings[0]) >= 4:
                    sa = _signed_area(rings[0])
                    if sa < 0:
                        vstats["cw_outer"] += 1
                    elif sa > 0:
                        vstats["ccw_outer"] += 1
                for i, ring in enumerate(rings):
                    if len(ring) < 4:
                        vstats["degenerate"] += 1
                    elif ring[0] != ring[-1]:
                        vstats["unclosed"] += 1
                    if i > 0:
                        vstats["holes"] += 1

                props = dict(feat.get("properties", {}))
                zclass = classify_zone(
                    props.get("FLD_ZONE", ""),
                    props.get("ZONE_SUBTY", ""),
                )
                if zclass not in ("A", "X500"):
                    continue

                # Build polygon with Check 6: area conservation on repair
                poly = _rings_to_shapely_checked(rings, vstats)
                if poly is None:
                    continue
                (zone_a_polys if zclass == "A" else zone_x500_polys).append(poly)

        # Overlay against county's ZCTAs
        zone_a_areas = {}
        zone_x500_areas = {}

        for polys, out_dict in [(zone_a_polys, zone_a_areas), (zone_x500_polys, zone_x500_areas)]:
            if not polys:
                continue
            try:
                zone_gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_crs("EPSG:5070")
                if hasattr(zone_gdf.geometry, 'union_all'):
                    zone_union = zone_gdf.geometry.union_all()
                else:
                    zone_union = zone_gdf.geometry.unary_union
                for pos_idx in county_zcta_indices:
                    zcta_geom = zcta_proj.iloc[pos_idx].geometry
                    intersection = zcta_geom.intersection(zone_union)
                    if not intersection.is_empty:
                        out_dict[pos_idx] = intersection.area
            except Exception as e:
                log.warning("  Overlay error %s: %s", fips, e)

        return {"fips": fips, "n_features": n_features,
                "zone_a": zone_a_areas, "zone_x500": zone_x500_areas,
                "vstats": vstats, "error": None}

    except Exception as e:
        return {**_empty, "error": str(e)}
    finally:
        if zip_path.exists():
            zip_path.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiger-dir", default="/opt/ml/processing/input/tiger")
    parser.add_argument("--data-dir", default="/opt/ml/processing/input/data")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="nfhl_combined_"))

    s3 = _s3_client()
    t_job_start = time.time()

    # ===================================================================
    # 1. List raw zips on S3, dedup by FIPS (latest date)
    # ===================================================================
    log.info("=== LISTING RAW ZIPS ===")
    raw_files = _s3_list_prefix(s3, S3_RAW_PREFIX, suffix=".zip")
    log.info("  Found %d raw zips", len(raw_files))

    by_fips = {}
    for key, fname in raw_files:
        dfirm_id = fname.split("_")[0] if "_" in fname else fname[:-4]
        fips = dfirm_id[:5]
        if fips[:2] in NON_CONUS_PREFIXES:
            continue
        m = re.search(r"_(\d{8})\.", fname)
        entry_date = m.group(1) if m else "00000000"
        if fips not in by_fips or entry_date > by_fips[fips]["date"]:
            by_fips[fips] = {"key": key, "dfirm_id": dfirm_id, "date": entry_date}

    log.info("  Dedup: %d unique CONUS FIPS", len(by_fips))

    # ===================================================================
    # 2. Load TIGER + crosswalk, project to EPSG:5070
    # ===================================================================
    log.info("\n=== LOADING TIGER ===")
    tiger_path = Path(args.tiger_dir)
    shp_files = list(tiger_path.glob("*.shp"))
    zcta_geo = gpd.read_file(shp_files[0], engine="pyogrio")
    zcta_geo = zcta_geo[["ZCTA5CE20", "geometry"]].rename(columns={"ZCTA5CE20": "zcta_id"})
    zcta_geo["zcta_id"] = zcta_geo["zcta_id"].astype(str).str.zfill(5)
    zcta_geo = zcta_geo.to_crs("EPSG:4326")
    log.info("  %d ZCTAs loaded", len(zcta_geo))

    xwalk = pd.read_parquet(Path(args.data_dir) / "zcta_county_crosswalk.parquet")
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    zcta_counties = dict(zip(xwalk["zcta_id"], xwalk["county_fips"].astype(str)))
    zcta_geo["county_fips"] = zcta_geo["zcta_id"].map(zcta_counties).fillna("unknown")

    # CONUS filter
    state_fips = zcta_geo["county_fips"].str[:2]
    conus_mask = ~state_fips.isin(NON_CONUS_PREFIXES) & (zcta_geo["county_fips"] != "unknown")
    zcta_geo = zcta_geo[conus_mask].reset_index(drop=True)
    log.info("  CONUS: %d ZCTAs", len(zcta_geo))

    log.info("  Projecting to EPSG:5070...")
    zcta_proj = zcta_geo.to_crs("EPSG:5070")
    n = len(zcta_proj)
    zcta_areas = zcta_proj.geometry.area.values

    # Validation: CRS + bounds
    log.info("\n=== VALIDATION ===")
    crs = zcta_proj.crs
    if crs is None or crs.to_epsg() != 5070:
        log.error("Check 1 FAIL: TIGER CRS is not EPSG:5070")
        sys.exit(1)
    log.info("  Check 1 PASS: TIGER CRS is EPSG:5070")

    xmin, ymin, xmax, ymax = zcta_proj.total_bounds
    cxmin, cymin, cxmax, cymax = CONUS_5070_BOUNDS
    if xmin > cxmin and ymin > cymin and xmax < cxmax and ymax < cymax:
        log.info("  Check 2 PASS: bounds within CONUS EPSG:5070")
    else:
        log.warning("  Check 2 WARN: bounds outside CONUS range")

    # ===================================================================
    # 3. Build county -> ZCTA index mapping
    # ===================================================================
    log.info("\n=== BUILDING COUNTY MANIFEST ===")
    county_zcta_map = {}
    for county_fips, idx in zcta_geo.groupby("county_fips").groups.items():
        county_zcta_map[county_fips] = list(idx)

    # Match raw zips to ZCTA counties
    to_process = []
    matched = 0
    for fips, info in by_fips.items():
        if fips in county_zcta_map:
            to_process.append((fips, info, county_zcta_map[fips]))
            matched += 1
        # else: FEMA has data but no ZCTAs in this county (rare)

    log.info("  %d counties with both raw zip and ZCTAs", matched)
    log.info("  %d counties with ZCTAs but no FEMA data",
             len(county_zcta_map) - matched)

    # ===================================================================
    # 4. Process in parallel: extract + overlay
    # ===================================================================
    n_cpus = multiprocessing.cpu_count()
    n_threads = n_cpus  # 1x CPU — overlay is CPU-bound
    log.info("\n=== PROCESSING %d COUNTIES (%d threads, %d CPUs) ===",
             len(to_process), n_threads, n_cpus)

    zone_a_area = np.zeros(n, dtype=np.float64)
    zone_x500_area = np.zeros(n, dtype=np.float64)
    area_lock = Lock()

    t_start = time.time()
    done = 0
    failed = 0
    total_features = 0
    counties_with_flood = 0
    empty_counties = []

    # Aggregate validation stats (Checks 3, 4, 6)
    agg_vstats = {
        "null_geom": 0, "degenerate": 0, "unclosed": 0,
        "cw_outer": 0, "ccw_outer": 0, "holes": 0,
        "repaired": 0, "area_ratio_bad": 0,
    }

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(process_county, s3, info["key"], info["dfirm_id"],
                            fips, zcta_indices, zcta_proj, work_dir): fips
            for fips, info, zcta_indices in to_process
        }
        for fut in as_completed(futures):
            fips = futures[fut]
            try:
                result = fut.result()
                done += 1

                if result["error"]:
                    log.warning("  SKIP %s: %s", fips, result["error"])
                    failed += 1
                else:
                    total_features += result["n_features"]
                    if result["n_features"] == 0:
                        empty_counties.append(fips)

                    # Accumulate validation stats
                    for k in agg_vstats:
                        agg_vstats[k] += result.get("vstats", {}).get(k, 0)

                    has_flood = False
                    with area_lock:
                        for idx, area in result["zone_a"].items():
                            zone_a_area[idx] += area
                            has_flood = True
                        for idx, area in result["zone_x500"].items():
                            zone_x500_area[idx] += area
                            has_flood = True
                    if has_flood:
                        counties_with_flood += 1

                if done % 50 == 0 or done == len(to_process):
                    elapsed = time.time() - t_start
                    rate = done / (elapsed / 60) if elapsed > 0 else 0
                    eta = (len(to_process) - done) / rate if rate > 0 else 0
                    log.info("  [%d/%d] %.0f/min, ETA %.0f min, %d failed, %d features",
                             done, len(to_process), rate, eta, failed, total_features)
                    sys.stdout.flush()

            except Exception as e:
                log.warning("  ERROR %s: %s", fips, e)
                failed += 1
                done += 1

    elapsed_overlay = time.time() - t_start

    # ===================================================================
    # 5. Compute percentages
    # ===================================================================
    log.info("\n=== COMPUTING PERCENTAGES ===")
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_a = np.where(zcta_areas > 0,
                         np.minimum(zone_a_area / zcta_areas * 100, 100.0), 0.0)
        pct_x500 = np.where(zcta_areas > 0,
                            np.minimum(zone_x500_area / zcta_areas * 100, 100.0), 0.0)
    pct_x = np.maximum(100.0 - pct_a - pct_x500, 0.0)

    result_df = pd.DataFrame({
        "zcta_id": zcta_geo["zcta_id"].values,
        "flood_pct_zone_a": np.round(pct_a, 2),
        "flood_pct_zone_x500": np.round(pct_x500, 2),
        "flood_pct_zone_x": np.round(pct_x, 2),
        "flood_sfha": pct_a > 0,
    })

    # ===================================================================
    # 6. Validation: Check 5 (Harris) + Check 7 (distribution)
    # ===================================================================
    log.info("\n=== VALIDATION ===")

    # Check 3: Geometry stats
    log.info("  Check 3 - Geometry:")
    log.info("    null_geom=%d  degenerate_rings=%d  unclosed_rings=%d",
             agg_vstats["null_geom"], agg_vstats["degenerate"], agg_vstats["unclosed"])

    # Check 4: Winding order
    log.info("  Check 4 - Winding:")
    log.info("    CW outer (Esri-correct)=%d  CCW outer (reversed)=%d  holes=%d",
             agg_vstats["cw_outer"], agg_vstats["ccw_outer"], agg_vstats["holes"])
    cw_total = agg_vstats["cw_outer"] + agg_vstats["ccw_outer"]
    if cw_total > 0:
        cw_pct = agg_vstats["cw_outer"] / cw_total * 100
        log.info("    %.1f%% CW (expected: >95%% for Esri shapefiles)", cw_pct)

    # Check 5: Harris County
    harris = result_df[result_df["zcta_id"].isin(
        zcta_geo[zcta_geo["county_fips"] == "48201"]["zcta_id"]
    )]
    if len(harris) > 0 and harris["flood_pct_zone_a"].sum() > 0:
        log.info("  Check 5 PASS: Harris County has Zone A coverage")
    else:
        log.warning("  Check 5 FAIL: Harris County has no Zone A coverage")

    # Check 7: Distribution
    log.info("  Check 7 - Distribution:")
    for col_name, arr in [("zone_a", pct_a), ("zone_x500", pct_x500), ("zone_x", pct_x)]:
        log.info("    %s: min=%.4f max=%.2f mean=%.2f median=%.2f zero=%.1f%%",
                 col_name, np.min(arr), np.max(arr), np.mean(arr), np.median(arr),
                 np.mean(arr == 0) * 100)

    overflow = int(np.sum((pct_a + pct_x500) > 100.01))
    if overflow > 0:
        log.warning("    %d ZCTAs have zone_a + zone_x500 > 100%%", overflow)
    else:
        log.info("    No overflow (zone_a + zone_x500 <= 100%%)")

    # Check 6: Area conservation on repair
    log.info("  Check 6 - Area conservation:")
    log.info("    repaired=%d  area_ratio_bad (>10%% change)=%d",
             agg_vstats["repaired"], agg_vstats["area_ratio_bad"])
    if agg_vstats["repaired"] > 0:
        bad_pct = agg_vstats["area_ratio_bad"] / agg_vstats["repaired"] * 100
        log.info("    %.1f%% of repairs exceeded 10%% area change", bad_pct)

    zone_a_coverage = float(np.mean(pct_a > 0) * 100)

    # ===================================================================
    # 7. Save output + provenance
    # ===================================================================
    out_path = output_dir / "flood_zones_zcta.parquet"
    result_df.to_parquet(out_path, index=False)
    log.info("\nSaved: %s (%d rows)", out_path, len(result_df))
    _s3_upload(s3, str(out_path), S3_OUTPUT_KEY)

    elapsed_total = time.time() - t_job_start
    n_cpus_actual = multiprocessing.cpu_count()

    prov = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "FEMA NFHL bulk county shapefiles (S3 flood_raw/)",
        "method": "combined_extract_overlay",
        "counties_processed": done - failed,
        "counties_failed": failed,
        "counties_with_flood": counties_with_flood,
        "total_features": total_features,
        "n_zctas": len(result_df),
        "zone_a_coverage_pct": round(zone_a_coverage, 2),
        "compute": {
            "n_cpus": n_cpus_actual,
            "n_threads": n_threads,
            "overlay_elapsed_sec": round(elapsed_overlay, 1),
            "overlay_elapsed_min": round(elapsed_overlay / 60, 1),
            "total_elapsed_sec": round(elapsed_total, 1),
            "total_elapsed_min": round(elapsed_total / 60, 1),
        },
        "validation": {
            "check_3_geometry": {
                "null_geom": agg_vstats["null_geom"],
                "degenerate_rings": agg_vstats["degenerate"],
                "unclosed_rings": agg_vstats["unclosed"],
            },
            "check_4_winding": {
                "cw_outer": agg_vstats["cw_outer"],
                "ccw_outer": agg_vstats["ccw_outer"],
                "holes": agg_vstats["holes"],
            },
            "check_6_area_conservation": {
                "repaired": agg_vstats["repaired"],
                "area_ratio_bad": agg_vstats["area_ratio_bad"],
            },
        },
    }
    prov_path = output_dir / "flood_zones_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2))
    _s3_upload(s3, str(prov_path), S3_PROVENANCE_KEY)

    log.info("\n=== COMPLETE ===")
    log.info("  Counties: %d processed, %d failed, %d with flood",
             done - failed, failed, counties_with_flood)
    log.info("  Features: %d total", total_features)
    log.info("  ZCTAs: %d, Zone A coverage: %.1f%%", len(result_df), zone_a_coverage)
    log.info("  Overlay time: %.1f min", elapsed_overlay / 60)
    log.info("  Total time: %.1f min", elapsed_total / 60)

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
