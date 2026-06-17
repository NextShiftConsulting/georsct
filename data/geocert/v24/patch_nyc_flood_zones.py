#!/usr/bin/env python3
"""Patch NYC flood zones in the national parquet.

Downloads DFIRM 360497 (City of New York), extracts S_FLD_HAZ_AR,
overlays against NYC borough ZCTAs, and patches flood_zones_zcta.parquet.

Run locally (not SageMaker) -- small data, single county.

Root cause: FEMA uses DFIRM ID 360497 for all 5 NYC boroughs (a consolidated
community). The overlay code used dfirm_id[:5] -> "36049" (Schuyler County),
missing all NYC ZCTAs. This patch extracts the actual flood zones from the
360497 zip and overlays against NYC borough ZCTAs.
"""

import gc
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from shapely.ops import transform, unary_union
from shapely.validation import make_valid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-yrsn-datasets"
DFIRM_KEY = "rsct_curriculum/series_018/processed/flood_raw/360497.zip"
PARQUET_KEY = "rsct_curriculum/series_018/processed/flood_zones_zcta.parquet"

NYC_BOROUGH_FIPS = {"36005", "36047", "36061", "36081", "36085"}

ZONE_A_PREFIXES = {"A", "AE", "AH", "AO", "AR", "V", "VE"}
ZONE_X500_LABELS = {"X500", "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
                     "B", "AREA OF MODERATE FLOOD HAZARD"}


def get_s3():
    session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "nsc-swarm"))
    return session.client("s3", region_name="us-east-1")


def classify_zone(fld_zone: str, zone_subty: str) -> str:
    """Classify FEMA zone into A, X500, or X."""
    fz = (fld_zone or "").strip().upper()
    zs = (zone_subty or "").strip().upper()
    if fz in ZONE_A_PREFIXES:
        return "A"
    if fz == "X" and zs in ZONE_X500_LABELS:
        return "X500"
    return "X"


def main():
    s3 = get_s3()
    work_dir = Path(tempfile.mkdtemp(prefix="nyc_flood_"))

    # -----------------------------------------------------------------
    # 1. Download and extract DFIRM 360497
    # -----------------------------------------------------------------
    log.info("Downloading DFIRM 360497 from S3...")
    zip_path = work_dir / "360497.zip"
    s3.download_file(BUCKET, DFIRM_KEY, str(zip_path))
    log.info("  %.1f MB", zip_path.stat().st_size / 1e6)

    extract_dir = work_dir / "extract"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    # Find S_FLD_HAZ_AR shapefile
    shp_files = list(extract_dir.rglob("S_FLD_HAZ_AR.shp"))
    if not shp_files:
        log.error("No S_FLD_HAZ_AR.shp found in zip")
        sys.exit(1)
    shp_path = shp_files[0]
    log.info("Found: %s", shp_path)

    # -----------------------------------------------------------------
    # 2. Read flood hazard areas via geopandas/pyogrio
    # -----------------------------------------------------------------
    log.info("Reading flood hazard polygons...")
    flood_gdf = gpd.read_file(str(shp_path), engine="pyogrio")
    src_crs = flood_gdf.crs
    log.info("  %d features, CRS=%s", len(flood_gdf), src_crs)

    # Classify zones
    flood_gdf["_zone_cls"] = flood_gdf.apply(
        lambda r: classify_zone(
            r.get("FLD_ZONE", ""),
            r.get("ZONE_SUBTY", ""),
        ), axis=1
    )
    log.info("  Zone distribution:")
    for cls, cnt in flood_gdf["_zone_cls"].value_counts().items():
        log.info("    %s: %d", cls, cnt)

    # Fix invalid geometries
    invalid_mask = ~flood_gdf.geometry.is_valid
    if invalid_mask.any():
        log.info("  Repairing %d invalid geometries", invalid_mask.sum())
        flood_gdf.loc[invalid_mask, "geometry"] = flood_gdf.loc[invalid_mask].geometry.apply(make_valid)

    # Drop empty
    flood_gdf = flood_gdf[~flood_gdf.geometry.is_empty].copy()

    # Dissolve by zone class
    log.info("Dissolving zone polygons...")
    zone_unions = {}
    for cls in ("A", "X500"):
        subset = flood_gdf[flood_gdf["_zone_cls"] == cls]
        if len(subset) > 0:
            zone_unions[cls] = unary_union(subset.geometry.values)
            log.info("  %s: %d polygons dissolved", cls, len(subset))
        else:
            zone_unions[cls] = None

    del flood_gdf
    gc.collect()

    # -----------------------------------------------------------------
    # 3. Load TIGER ZCTA boundaries for NYC boroughs
    # -----------------------------------------------------------------
    log.info("Loading ZCTA crosswalk...")
    xwalk_path = work_dir / "xwalk.parquet"
    s3.download_file("swarm-floodrsct-data",
                     "raw/geocertdb2026/zcta_county_crosswalk.parquet",
                     str(xwalk_path))
    xwalk = pd.read_parquet(xwalk_path)
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    xwalk["county_fips"] = xwalk["county_fips"].astype(str)
    nyc_zctas = set(xwalk[xwalk["county_fips"].isin(NYC_BOROUGH_FIPS)]["zcta_id"])
    log.info("  NYC ZCTAs from crosswalk: %d", len(nyc_zctas))

    log.info("Loading ZCTA boundaries (large file -- from floodrsct-data)...")
    zcta_path = work_dir / "zcta5.parquet"
    s3.download_file("swarm-floodrsct-data",
                     "raw/geocertdb2026/zcta5_boundaries.parquet",
                     str(zcta_path))
    zcta_geo = gpd.read_parquet(zcta_path)
    if "ZCTA5CE20" in zcta_geo.columns:
        zcta_geo = zcta_geo.rename(columns={"ZCTA5CE20": "zcta_id"})
    zcta_geo["zcta_id"] = zcta_geo["zcta_id"].astype(str).str.zfill(5)
    zcta_nyc = zcta_geo[zcta_geo["zcta_id"].isin(nyc_zctas)].copy()
    log.info("  NYC ZCTAs with geometry: %d", len(zcta_nyc))
    del zcta_geo
    gc.collect()

    # -----------------------------------------------------------------
    # 4. Reproject to EPSG:5070 and overlay
    # -----------------------------------------------------------------
    log.info("Reprojecting to EPSG:5070...")
    zcta_nyc = zcta_nyc.to_crs("EPSG:5070")

    proj = pyproj.Transformer.from_crs(
        pyproj.CRS(src_crs), pyproj.CRS("EPSG:5070"), always_xy=True
    )
    for cls in zone_unions:
        if zone_unions[cls] is not None:
            zone_unions[cls] = transform(proj.transform, zone_unions[cls])

    log.info("Computing overlay for %d NYC ZCTAs...", len(zcta_nyc))
    results = []
    for i, (_, row) in enumerate(zcta_nyc.iterrows()):
        zcta_id = row["zcta_id"]
        zcta_geom = row.geometry
        zcta_area = zcta_geom.area
        if zcta_area <= 0:
            results.append({"zcta_id": zcta_id, "flood_pct_zone_a": 0.0,
                            "flood_pct_zone_x500": 0.0, "flood_pct_zone_x": 100.0,
                            "flood_sfha": False})
            continue

        pct_a = 0.0
        pct_x500 = 0.0
        for cls in ("A", "X500"):
            if zone_unions[cls] is not None:
                try:
                    inter = zcta_geom.intersection(zone_unions[cls])
                    if not inter.is_empty:
                        frac = inter.area / zcta_area * 100
                        if cls == "A":
                            pct_a = round(frac, 2)
                        else:
                            pct_x500 = round(frac, 2)
                except Exception as e:
                    log.warning("  Overlay error for %s/%s: %s", zcta_id, cls, e)

        pct_x = round(max(0, 100 - pct_a - pct_x500), 2)
        results.append({
            "zcta_id": zcta_id,
            "flood_pct_zone_a": pct_a,
            "flood_pct_zone_x500": pct_x500,
            "flood_pct_zone_x": pct_x,
            "flood_sfha": pct_a > 0,
        })
        if (i + 1) % 25 == 0:
            log.info("  %d/%d ZCTAs processed", i + 1, len(zcta_nyc))

    patch_df = pd.DataFrame(results)
    n_zone_a = (patch_df["flood_pct_zone_a"] > 0).sum()
    log.info("NYC overlay complete:")
    log.info("  Zone A coverage: %d/%d ZCTAs (%.1f%%)",
             n_zone_a, len(patch_df), n_zone_a / len(patch_df) * 100)
    log.info("  Zone A: mean=%.2f%%, max=%.2f%%",
             patch_df["flood_pct_zone_a"].mean(), patch_df["flood_pct_zone_a"].max())
    log.info("  Zone X500: mean=%.2f%%, max=%.2f%%",
             patch_df["flood_pct_zone_x500"].mean(), patch_df["flood_pct_zone_x500"].max())

    # -----------------------------------------------------------------
    # 5. Patch national parquet
    # -----------------------------------------------------------------
    log.info("Patching national flood_zones_zcta.parquet...")
    national_path = work_dir / "flood_zones_zcta.parquet"
    s3.download_file(BUCKET, PARQUET_KEY, str(national_path))
    national = pd.read_parquet(national_path)
    national["zcta_id"] = national["zcta_id"].astype(str).str.zfill(5)

    old_nyc = national["zcta_id"].isin(nyc_zctas)
    log.info("  Replacing %d old NYC rows with %d patched rows",
             old_nyc.sum(), len(patch_df))
    national = pd.concat([national[~old_nyc], patch_df], ignore_index=True)
    national = national.sort_values("zcta_id").reset_index(drop=True)

    # Validate
    new_nyc = national[national["zcta_id"].isin(nyc_zctas)]
    log.info("  Post-patch: %d NYC ZCTAs, %d with Zone A",
             len(new_nyc), (new_nyc["flood_pct_zone_a"] > 0).sum())

    # -----------------------------------------------------------------
    # 6. Save and upload
    # -----------------------------------------------------------------
    out_path = work_dir / "flood_zones_zcta_patched.parquet"
    national.to_parquet(out_path, index=False)

    log.info("Uploading patched parquet...")
    s3.upload_file(str(out_path), BUCKET, PARQUET_KEY)
    log.info("  -> s3://%s/%s", BUCKET, PARQUET_KEY)

    FLOODRSCT_BUCKET = "swarm-floodrsct-data"
    FLOODRSCT_KEY = "raw/geocertdb2026/flood_zones_zcta.parquet"
    s3.upload_file(str(out_path), FLOODRSCT_BUCKET, FLOODRSCT_KEY)
    log.info("  -> s3://%s/%s", FLOODRSCT_BUCKET, FLOODRSCT_KEY)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    log.info("\n=== PATCH SUMMARY ===")
    log.info("  DFIRM: 360497 (City of New York)")
    log.info("  Boroughs: %s", sorted(NYC_BOROUGH_FIPS))
    log.info("  NYC ZCTAs patched: %d", len(patch_df))
    log.info("  Zone A coverage: %d/%d (%.1f%%)",
             n_zone_a, len(patch_df), n_zone_a / len(patch_df) * 100)
    log.info("  Zone A mean: %.2f%%", patch_df["flood_pct_zone_a"].mean())
    log.info("  National ZCTAs total: %d", len(national))

    shutil.rmtree(work_dir, ignore_errors=True)
    log.info("Done.")


if __name__ == "__main__":
    main()
