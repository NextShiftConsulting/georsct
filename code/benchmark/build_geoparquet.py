#!/usr/bin/env python3
"""
build_geoparquet.py -- Build GeoCert v24 GeoParquet for Hugging Face release.

Joins ZCTA boundaries (Census TIGER/Line 2022) with:
  - 27 target labels
  - 5 CDC SVI columns
  - 3 FEMA flood zone columns
  - 6 HIFLD hospital/pharmacy columns
  - 2 drive-time columns
  - 3 evaluation split assignments
  - Coverage flags
  - ACS encoder features

Uncertainty is NOT embedded in the main GeoParquet (Option 2: separate concern).
Two standalone uncertainty parquets are published alongside:
  - cdc_places_ci.parquet  (42 cols: 21 low + 21 high)
  - zcta_acs_margins_of_error.parquet  (31 MOE cols)
These are joined at runtime by consumers that need them (e.g., allocator.py).

Output: geocert.geoparquet (~60 MB)

Usage:
    python build_geoparquet.py --dry-run          # Build locally, don't upload
    python build_geoparquet.py                    # Build + upload to S3
    python build_geoparquet.py --simplify 0.001   # Simplify geometries (smaller file)

Prerequisites:
    pip install geopandas pyogrio pandas boto3 requests
"""

import argparse
import json
import logging
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import geopandas as gpd
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-yrsn-datasets"
FEATURES_KEY = "rsct_curriculum/series_018/processed/zcta_features_labels.parquet"
SPLITS_KEY = "rsct_curriculum/series_018/processed/geocert_splits.parquet"
COUNTY_XWALK_KEY = "rsct_curriculum/series_018/processed/zcta_county_crosswalk.parquet"
SVI_KEY = "rsct_curriculum/series_018/processed/svi_zcta.parquet"
FLOOD_KEY = "rsct_curriculum/series_018/processed/flood_zones_zcta.parquet"
HIFLD_KEY = "rsct_curriculum/series_018/processed/hifld_zcta.parquet"
DRIVE_KEY = "rsct_curriculum/series_018/processed/drive_times_zcta.parquet"
OUTPUT_KEY = "rsct_curriculum/series_018/release/geocert_v24.geoparquet"
PROVENANCE_KEY = "rsct_curriculum/series_018/release/geocert_v24_geoparquet_provenance.json"
REGION = "us-east-1"

TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2022/ZCTA520/tl_2022_us_zcta520.zip"
TIGER_LOCAL = Path("/tmp/tl_2022_us_zcta520.zip")
TIGER_DIR = Path("/tmp/tiger_zcta")

# Target columns to include (all 27)
TARGET_COLUMNS = [
    "target_annual_checkup", "target_arthritis", "target_asthma",
    "target_binge_drinking", "target_bp_medicated", "target_cancer",
    "target_cholesterol_screening", "target_chronic_kidney_disease",
    "target_copd", "target_coronary_heart_disease", "target_dental_visit",
    "target_diabetes", "target_elevation", "target_high_blood_pressure",
    "target_high_cholesterol", "target_home_value", "target_income",
    "target_mental_health_not_good", "target_night_lights", "target_obesity",
    "target_physical_health_not_good", "target_physical_inactivity",
    "target_population_density", "target_sleep_less_7hr", "target_smoking",
    "target_stroke", "target_tree_cover",
]

# ACS feature columns to include (encoder features)
ACS_PREFIX = "acs_"


def download_tiger():
    """Download Census TIGER/Line 2022 ZCTA boundaries."""
    if TIGER_DIR.exists() and any(TIGER_DIR.glob("*.shp")):
        log.info("TIGER shapefile already exists at %s", TIGER_DIR)
        return

    if not TIGER_LOCAL.exists():
        log.info("Downloading TIGER/Line ZCTA boundaries...")
        log.info("  URL: %s", TIGER_URL)
        resp = requests.get(TIGER_URL, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(TIGER_LOCAL, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * 1024 * 1024) < 8192:
                    log.info("  %.1f / %.1f MB", downloaded / 1e6, total / 1e6)
        log.info("  Downloaded %.1f MB", TIGER_LOCAL.stat().st_size / 1e6)

    TIGER_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Extracting shapefile...")
    with zipfile.ZipFile(TIGER_LOCAL, "r") as zf:
        zf.extractall(TIGER_DIR)
    log.info("Extracted to %s", TIGER_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Build locally, don't upload to S3")
    parser.add_argument("--simplify", type=float, default=None,
                        help="Simplify geometries (tolerance in degrees, e.g. 0.001)")
    parser.add_argument("--output", type=str, default="/tmp/geocert.geoparquet",
                        help="Local output path")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    timestamp = datetime.now(timezone.utc).isoformat()

    # -- 1. Download TIGER boundaries --
    download_tiger()

    shp_files = list(TIGER_DIR.glob("*.shp"))
    if not shp_files:
        log.error("No .shp file found in %s", TIGER_DIR)
        sys.exit(1)

    log.info("Reading ZCTA boundaries from %s", shp_files[0])
    boundaries = gpd.read_file(shp_files[0], engine="pyogrio")
    boundaries = boundaries[["ZCTA5CE20", "geometry"]].rename(
        columns={"ZCTA5CE20": "zcta_id"}
    )
    boundaries = boundaries.to_crs("EPSG:4326")
    log.info("  %d ZCTA boundaries loaded (CRS: EPSG:4326)", len(boundaries))

    # -- 2. Simplify geometries if requested --
    if args.simplify:
        log.info("Simplifying geometries (tolerance=%.4f degrees)...", args.simplify)
        boundaries["geometry"] = boundaries.geometry.simplify(
            tolerance=args.simplify, preserve_topology=True
        )

    # -- 3. Load features + labels --
    local_features = Path("/tmp/zcta_features_labels.parquet")
    log.info("Downloading features/labels from S3...")
    s3.download_file(BUCKET, FEATURES_KEY, str(local_features))
    df = pd.read_parquet(local_features)
    df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)
    log.info("  %d ZCTAs, %d columns", len(df), len(df.columns))

    # -- 4. Load splits --
    local_splits = Path("/tmp/geocert_splits.parquet")
    log.info("Downloading splits from S3...")
    s3.download_file(BUCKET, SPLITS_KEY, str(local_splits))
    splits = pd.read_parquet(local_splits)
    splits["zcta_id"] = splits["zcta_id"].astype(str).str.zfill(5)
    log.info("  %d split assignments loaded", len(splits))

    # -- 4b. Load ZCTA-county crosswalk (Census FIPS-based, majority by area) --
    local_xwalk = Path("/tmp/zcta_county_crosswalk.parquet")
    try:
        log.info("Downloading ZCTA-county crosswalk from S3...")
        s3.download_file(BUCKET, COUNTY_XWALK_KEY, str(local_xwalk))
        xwalk = pd.read_parquet(local_xwalk)
        xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
        log.info("  %d ZCTA-county assignments loaded", len(xwalk))

        # Replace county_name and state with Census-authoritative values
        # Add county_fips and state_fips
        xwalk_cols = ["zcta_id", "county_fips", "state_fips"]
        # Keep county_name from crosswalk if it's there, but as xwalk_county_name
        if "county_name" in xwalk.columns:
            xwalk_cols.append("county_name")
        if "state" in xwalk.columns:
            xwalk_cols.append("state")

        # Drop existing county/state columns from df if they'll be replaced
        for col in ["county_name", "state", "county_fips", "state_fips"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = df.merge(xwalk[xwalk_cols], on="zcta_id", how="left")
        log.info("  After crosswalk merge: county_fips coverage = %d / %d",
                 df["county_fips"].notna().sum(), len(df))

        # CONUS filter: crosswalk only contains lower-48 + DC.
        # ZCTAs without a crosswalk match are non-CONUS or unresolved.
        n_before = len(df)
        df = df[df["county_fips"].notna()].reset_index(drop=True)
        log.info("  CONUS filter: %d -> %d ZCTAs (dropped %d without crosswalk match)",
                 n_before, len(df), n_before - len(df))
    except Exception as e:
        log.warning("Could not load crosswalk: %s (using existing columns)", e)

    # -- 5. Select columns --
    acs_cols = sorted(c for c in df.columns if c.startswith(ACS_PREFIX))
    meta_cols = ["zcta_id", "latitude", "longitude", "county_fips", "county_name",
                 "state_fips", "state"]
    keep_cols = meta_cols + acs_cols + TARGET_COLUMNS

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        log.warning("Missing columns (will skip): %s", missing)
        keep_cols = [c for c in keep_cols if c in df.columns]

    tabular = df[keep_cols].copy()

    # -- 6. Merge splits --
    split_cols = ["zcta_id", "split_imputation", "split_extrapolation",
                  "split_superres", "has_cdc_places", "has_income", "has_home_value"]
    tabular = tabular.merge(splits[split_cols], on="zcta_id", how="left")
    log.info("  After split merge: %d rows, %d columns", len(tabular), len(tabular.columns))

    # -- 6b. Merge enrichment layers (SVI, flood, HIFLD, drive times) --
    # NOTE: Uncertainty layers (CDC CIs, ACS MOEs) are separate parquets.
    def try_load_enrichment(key: str, name: str) -> pd.DataFrame:
        local_path = Path(f"/tmp/{Path(key).name}")
        try:
            log.info("Downloading %s from S3...", name)
            s3.download_file(BUCKET, key, str(local_path))
            enrich = pd.read_parquet(local_path)
            enrich["zcta_id"] = enrich["zcta_id"].astype(str).str.zfill(5)
            log.info("  Loaded %s: %d rows, %d columns", name, len(enrich), len(enrich.columns))
            return enrich
        except Exception as e:
            log.warning("  Could not load %s: %s (skipping)", name, e)
            return None

    # CDC SVI
    svi_df = try_load_enrichment(SVI_KEY, "CDC SVI")
    if svi_df is not None:
        svi_cols = [c for c in svi_df.columns if c != "zcta_id"]
        tabular = tabular.merge(svi_df, on="zcta_id", how="left")
        log.info("  After SVI merge: %d columns (+%d)", len(tabular.columns), len(svi_cols))

    # FEMA flood zones
    flood_df = try_load_enrichment(FLOOD_KEY, "FEMA flood zones")
    if flood_df is not None:
        flood_cols = [c for c in flood_df.columns if c != "zcta_id"]
        tabular = tabular.merge(flood_df, on="zcta_id", how="left")
        log.info("  After flood merge: %d columns (+%d)", len(tabular.columns), len(flood_cols))

    # HIFLD hospitals/pharmacies
    hifld_df = try_load_enrichment(HIFLD_KEY, "HIFLD facilities")
    if hifld_df is not None:
        hifld_cols = [c for c in hifld_df.columns if c != "zcta_id"]
        tabular = tabular.merge(hifld_df, on="zcta_id", how="left")
        log.info("  After HIFLD merge: %d columns (+%d)", len(tabular.columns), len(hifld_cols))

    # Drive times
    drive_df = try_load_enrichment(DRIVE_KEY, "drive times")
    if drive_df is not None:
        drive_cols = [c for c in drive_df.columns if c != "zcta_id"]
        tabular = tabular.merge(drive_df, on="zcta_id", how="left")
        log.info("  After drive merge: %d columns (+%d)", len(tabular.columns), len(drive_cols))

    # -- 7. Join with boundaries --
    # Left join on tabular (canonical 31,789) -- boundaries is a superset
    geo = boundaries.merge(tabular, on="zcta_id", how="inner")
    log.info("")
    log.info("=== JOIN RESULT ===")
    log.info("  Boundaries:  %d ZCTAs", len(boundaries))
    log.info("  Tabular:     %d ZCTAs", len(tabular))
    log.info("  Joined:      %d ZCTAs", len(geo))
    log.info("  Dropped (no boundary): %d", len(tabular) - len(geo))

    unmatched = set(tabular["zcta_id"]) - set(boundaries["zcta_id"])
    if unmatched:
        log.warning("  %d canonical ZCTAs have no TIGER boundary: %s...",
                    len(unmatched), sorted(unmatched)[:5])

    # -- 8. Column ordering --
    # metadata -> splits -> coverage -> ACS features -> SVI -> flood -> HIFLD ->
    # drive times -> targets -> geometry
    svi_col_names = sorted(c for c in tabular.columns if c.startswith("svi_"))
    flood_col_names = sorted(c for c in tabular.columns if c.startswith("flood_"))
    hifld_col_names = sorted(c for c in tabular.columns if c.startswith("hifld_"))
    drive_col_names = sorted(c for c in tabular.columns if c.startswith("drive_"))

    ordered = (
        ["zcta_id", "county_fips", "county_name", "state_fips", "state",
         "latitude", "longitude"]
        + ["split_imputation", "split_extrapolation", "split_superres"]
        + ["has_cdc_places", "has_income", "has_home_value"]
        + acs_cols
        + svi_col_names
        + flood_col_names
        + hifld_col_names
        + drive_col_names
        + TARGET_COLUMNS
        + ["geometry"]
    )
    ordered = [c for c in ordered if c in geo.columns]
    geo = geo[ordered]

    # -- 9. Save --
    output = Path(args.output)
    log.info("")
    log.info("Writing GeoParquet to %s", output)
    geo.to_parquet(output, index=False)
    size_mb = output.stat().st_size / (1024 * 1024)
    log.info("  Size: %.1f MB", size_mb)
    log.info("  Rows: %d, Columns: %d", len(geo), len(geo.columns))

    # -- 10. Upload --
    if not args.dry_run:
        log.info("Uploading to s3://%s/%s", BUCKET, OUTPUT_KEY)
        s3.upload_file(str(output), BUCKET, OUTPUT_KEY)

        provenance = {
            "operation": "build_geoparquet",
            "timestamp": timestamp,
            "tiger_source": TIGER_URL,
            "features_source": f"s3://{BUCKET}/{FEATURES_KEY}",
            "splits_source": f"s3://{BUCKET}/{SPLITS_KEY}",
            "output": f"s3://{BUCKET}/{OUTPUT_KEY}",
            "crs": "EPSG:4326",
            "simplify_tolerance": args.simplify,
            "n_zctas": len(geo),
            "n_columns": len(geo.columns),
            "n_unmatched_boundaries": len(unmatched) if unmatched else 0,
            "file_size_mb": round(size_mb, 1),
            "columns": list(geo.columns),
        }
        s3.put_object(
            Bucket=BUCKET, Key=PROVENANCE_KEY,
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved: s3://%s/%s", BUCKET, PROVENANCE_KEY)
    else:
        log.info("[DRY RUN] Skipping S3 upload.")

    log.info("")
    log.info("=== COLUMN SUMMARY ===")
    log.info("  Metadata:    zcta_id, county_fips, county_name, state_fips, state, latitude, longitude")
    log.info("  Splits:      split_imputation, split_extrapolation, split_superres")
    log.info("  Coverage:    has_cdc_places, has_income, has_home_value")
    log.info("  Features:    %d ACS columns", len(acs_cols))
    log.info("  SVI:         %d columns", len(svi_col_names))
    log.info("  Flood:       %d columns", len(flood_col_names))
    log.info("  HIFLD:       %d columns", len(hifld_col_names))
    log.info("  Drive times: %d columns", len(drive_col_names))
    log.info("  Targets:     %d task columns", len([c for c in TARGET_COLUMNS if c in geo.columns]))
    log.info("  Geometry:    ZCTA boundary polygons (EPSG:4326)")
    log.info("  TOTAL:       %d columns", len(geo.columns))
    log.info("")
    log.info("  Uncertainty (separate parquets, not embedded):")
    log.info("    cdc_places_ci.parquet             — 42 cols (21 low + 21 high)")
    log.info("    zcta_acs_margins_of_error.parquet — 31 MOE cols")
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()
