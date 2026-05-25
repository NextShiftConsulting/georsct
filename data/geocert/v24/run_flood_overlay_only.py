#!/usr/bin/env python3
"""
Overlay-only flood zones job (Lane 3).

Reads pre-fetched county JSONs from S3, runs spatial overlay against
TIGER ZCTA boundaries in EPSG:5070, produces flood_zones_zcta.parquet.

No FEMA API calls. All data comes from S3 flood_fetch/ prefix
(populated by run_flood_fetch_only.py or run_flood_zones.py).

Instance: ml.m5.2xlarge (8 vCPU, 32 GB — overlay is memory-bound)
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
S3_FETCH_PREFIX = "rsct_curriculum/series_018/processed/flood_fetch/"
S3_OUTPUT_KEY = "rsct_curriculum/series_018/processed/flood_zones_zcta.parquet"
S3_PROVENANCE_KEY = "rsct_curriculum/series_018/processed/flood_zones_provenance.json"

NON_CONUS_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}

CONUS_5070_BOUNDS = (-2_500_000, 100_000, 2_400_000, 3_300_000)


def _s3_upload(local_path: str, key: str):
    try:
        import boto3
        boto3.client("s3").upload_file(local_path, S3_BUCKET, key)
        log.info("  -> s3://%s/%s", S3_BUCKET, key)
    except Exception as e:
        log.warning("  S3 upload failed for %s: %s", key, e)


def _s3_download(key: str, local_path: str, quiet: bool = False) -> bool:
    try:
        import boto3
        boto3.client("s3").download_file(S3_BUCKET, key, local_path)
        if not quiet:
            log.info("  <- s3://%s/%s", S3_BUCKET, key)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Esri JSON -> Shapely (same as run_flood_zones.py)
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


def _esri_to_shapely(geometry: dict):
    rings = geometry.get("rings", [])
    if not rings:
        return None
    try:
        shells, holes = [], []
        for ring in rings:
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


def classify_zone(fld_zone: str, zone_subty: str) -> str:
    if pd.isna(fld_zone):
        return "X"
    zone = str(fld_zone).strip().upper()
    subty = str(zone_subty).strip().upper() if pd.notna(zone_subty) else ""
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
# Overlay
# ---------------------------------------------------------------------------
def overlay_county(county_fips, raw_features, zcta_proj, idx_list):
    result = {"zone_a": {}, "zone_x500": {}}
    if not raw_features:
        return result

    zone_a_polys, zone_x500_polys = [], []
    for f in raw_features:
        zclass = classify_zone(
            f["attributes"].get("FLD_ZONE", ""),
            f["attributes"].get("ZONE_SUBTY", ""),
        )
        if zclass not in ("A", "X500"):
            continue
        poly = _esri_to_shapely(f["geometry"])
        if poly is None:
            continue
        (zone_a_polys if zclass == "A" else zone_x500_polys).append(poly)

    for polys, key in [(zone_a_polys, "zone_a"), (zone_x500_polys, "zone_x500")]:
        if not polys:
            continue
        try:
            zone_gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_crs("EPSG:5070")
            zone_union = zone_gdf.geometry.union_all()
            for pos_idx in idx_list:
                zcta_geom = zcta_proj.iloc[pos_idx].geometry
                intersection = zcta_geom.intersection(zone_union)
                if not intersection.is_empty:
                    result[key][pos_idx] = intersection.area
        except Exception as e:
            log.warning("  Overlay error county %s %s: %s", county_fips, key, e)

    return result


# ---------------------------------------------------------------------------
# Validation (subset — CRS + bounds only, no FEMA API calls)
# ---------------------------------------------------------------------------
def validate(zcta_proj):
    crs = zcta_proj.crs
    if crs is None or crs.to_epsg() != 5070:
        log.error("VALIDATION FAIL: TIGER CRS is not EPSG:5070")
        return False
    log.info("  Check 1 PASS: TIGER CRS is EPSG:5070")

    xmin, ymin, xmax, ymax = zcta_proj.total_bounds
    cxmin, cymin, cxmax, cymax = CONUS_5070_BOUNDS
    if xmin > cxmin and ymin > cymin and xmax < cxmax and ymax < cymax:
        log.info("  Check 2 PASS: TIGER bounds within CONUS EPSG:5070 range")
    else:
        log.warning("  Check 2 WARN: bounds outside CONUS range")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiger-dir", default="/opt/ml/processing/input/tiger")
    parser.add_argument("--data-dir", default="/opt/ml/processing/input/data")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fetch_dir = output_dir / "fetch_cache"
    fetch_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Download all fetch files from S3
    # ---------------------------------------------------------------
    log.info("=== DOWNLOADING FETCH FILES FROM S3 ===")
    import boto3
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    s3_keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_FETCH_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key.split("/")[-1]
            if fname.endswith(".json") and fname != "fetch_summary.json":
                s3_keys.append((key, fname))

    log.info("  Found %d county fetch files on S3", len(s3_keys))

    downloaded = 0
    for key, fname in s3_keys:
        local = fetch_dir / fname
        if not local.exists():
            _s3_download(key, str(local), quiet=True)
            downloaded += 1
    log.info("  Downloaded %d new files (%d already local)", downloaded, len(s3_keys) - downloaded)

    available = list(fetch_dir.glob("*.json"))
    log.info("  Total fetch files available: %d", len(available))

    if not available:
        log.error("No fetch files found. Run flood_fetch_only first.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 2: Load TIGER + crosswalk, project to EPSG:5070
    # ---------------------------------------------------------------
    log.info("\n=== LOADING DATA ===")
    tiger_path = Path(args.tiger_dir)
    shp_files = list(tiger_path.glob("*.shp"))
    log.info("Reading TIGER: %s", shp_files[0])
    zcta_geo = gpd.read_file(shp_files[0], engine="pyogrio")
    zcta_geo = zcta_geo[["ZCTA5CE20", "geometry"]].rename(columns={"ZCTA5CE20": "zcta_id"})
    zcta_geo["zcta_id"] = zcta_geo["zcta_id"].astype(str).str.zfill(5)
    zcta_geo = zcta_geo.to_crs("EPSG:4326")
    log.info("  %d ZCTAs loaded", len(zcta_geo))

    xwalk = pd.read_parquet(Path(args.data_dir) / "zcta_county_crosswalk.parquet")
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    zcta_counties = dict(zip(xwalk["zcta_id"], xwalk["county_fips"].astype(str)))
    zcta_geo["county_fips"] = zcta_geo["zcta_id"].map(zcta_counties).fillna("unknown")
    log.info("  %d ZCTA-county assignments", len(zcta_counties))

    # CONUS filter
    state_fips = zcta_geo["county_fips"].str[:2]
    conus_mask = ~state_fips.isin(NON_CONUS_STATE_FIPS) & (zcta_geo["county_fips"] != "unknown")
    before = len(zcta_geo)
    zcta_geo = zcta_geo[conus_mask].reset_index(drop=True)
    log.info("  CONUS filter: %d -> %d ZCTAs", before, len(zcta_geo))

    log.info("  Projecting to EPSG:5070...")
    zcta_proj = zcta_geo.to_crs("EPSG:5070")
    n = len(zcta_proj)

    # Pre-compute ZCTA areas
    zcta_areas = zcta_proj.geometry.area.values

    # ---------------------------------------------------------------
    # Step 3: Validate
    # ---------------------------------------------------------------
    log.info("\n=== VALIDATION ===")
    if not validate(zcta_proj):
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 4: Build manifest + overlay
    # ---------------------------------------------------------------
    log.info("\n=== BUILDING COUNTY MANIFEST ===")
    manifest = []
    for county_fips, idx in zcta_geo.groupby("county_fips").groups.items():
        manifest.append({
            "county_fips": county_fips,
            "idx_list": list(idx),
            "n_zctas": len(list(idx)),
        })
    manifest.sort(key=lambda m: m["n_zctas"])  # small first

    # Check which counties have fetch data
    available_fips = {f.stem for f in fetch_dir.glob("*.json")}
    has_data = [m for m in manifest if m["county_fips"] in available_fips]
    missing = len(manifest) - len(has_data)
    log.info("  %d counties in manifest, %d have fetch data, %d missing",
             len(manifest), len(has_data), missing)
    if missing > 0:
        missing_fips = [m["county_fips"] for m in manifest if m["county_fips"] not in available_fips]
        log.warning("  Missing counties (first 20): %s", missing_fips[:20])

    # Accumulators
    zone_a_area = np.zeros(n, dtype=np.float64)
    zone_x500_area = np.zeros(n, dtype=np.float64)

    log.info("\n=== PHASE 2: OVERLAY (%d counties, sequential) ===", len(has_data))
    counties_with_flood = 0
    total_overlaps = 0
    t_start = time.time()

    for i, entry in enumerate(has_data):
        county_fips = entry["county_fips"]
        t0 = time.time()

        fetch_file = fetch_dir / f"{county_fips}.json"
        with open(fetch_file) as f:
            raw_features = json.load(f)

        result = overlay_county(county_fips, raw_features, zcta_proj, entry["idx_list"])

        n_overlaps = 0
        for pos_idx, area in result["zone_a"].items():
            zone_a_area[pos_idx] += area
            n_overlaps += 1
        for pos_idx, area in result["zone_x500"].items():
            zone_x500_area[pos_idx] += area
            n_overlaps += 1

        if n_overlaps > 0:
            counties_with_flood += 1
        total_overlaps += n_overlaps

        del raw_features, result
        gc.collect()

        elapsed = time.time() - t0
        done = i + 1

        if done % 25 == 0 or done == len(has_data):
            total_elapsed = time.time() - t_start
            rate = done / (total_elapsed / 60) if total_elapsed > 0 else 0
            eta = (len(has_data) - done) / rate if rate > 0 else 0
            log.info("  [%d/%d] %s: %d overlaps, %.1fs (%.1f/min, ETA %.0f min)",
                     done, len(has_data), county_fips, n_overlaps, elapsed, rate, eta)
            sys.stdout.flush()

    log.info("Overlay complete: %d counties with flood, %d ZCTA overlaps",
             counties_with_flood, total_overlaps)

    # ---------------------------------------------------------------
    # Step 5: Compute percentages + Check 7 + output
    # ---------------------------------------------------------------
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

    # Check 7: Distribution validation
    log.info("\n=== CHECK 7: FRACTION DISTRIBUTION ===")
    for col_name, arr in [("zone_a", pct_a), ("zone_x500", pct_x500), ("zone_x", pct_x)]:
        log.info("  %s: min=%.4f max=%.2f mean=%.2f median=%.2f zero=%.1f%%",
                 col_name, np.min(arr), np.max(arr), np.mean(arr), np.median(arr),
                 np.mean(arr == 0) * 100)

    overflow = np.sum((pct_a + pct_x500) > 100.01)
    if overflow > 0:
        log.warning("  %d ZCTAs have zone_a + zone_x500 > 100%%", overflow)

    zone_a_coverage = np.mean(pct_a > 0) * 100
    log.info("  Zone A coverage: %.1f%% of ZCTAs", zone_a_coverage)

    # Summary
    log.info("\n=== SUMMARY ===")
    log.info("ZCTAs:          %d", len(result_df))
    log.info("Counties:       %d total, %d with data, %d missing",
             len(manifest), len(has_data), missing)
    log.info("Zone A >0%%:     %d (%.1f%%)",
             (result_df["flood_pct_zone_a"] > 0).sum(),
             (result_df["flood_pct_zone_a"] > 0).mean() * 100)
    log.info("Zone X500 >0%%:  %d (%.1f%%)",
             (result_df["flood_pct_zone_x500"] > 0).sum(),
             (result_df["flood_pct_zone_x500"] > 0).mean() * 100)
    log.info("SFHA count:     %d", result_df["flood_sfha"].sum())

    # Save + upload
    out_path = output_dir / "flood_zones_zcta.parquet"
    result_df.to_parquet(out_path, index=False)
    log.info("Saved: %s (%d rows)", out_path, len(result_df))
    _s3_upload(str(out_path), S3_OUTPUT_KEY)

    # Provenance
    prov = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "FEMA NFHL ArcGIS REST (pre-fetched from S3)",
        "fetch_files": len(has_data),
        "missing_counties": missing,
        "n_zctas": len(result_df),
        "zone_a_coverage_pct": round(zone_a_coverage, 2),
        "total_elapsed_sec": round(time.time() - t_start, 1),
    }
    prov_path = output_dir / "flood_zones_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2))
    _s3_upload(str(prov_path), S3_PROVENANCE_KEY)

    log.info("\n=== DONE ===")


if __name__ == "__main__":
    main()
