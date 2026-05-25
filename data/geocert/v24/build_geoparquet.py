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
    python build_geoparquet.py --dry-run                          # Build locally, don't upload
    python build_geoparquet.py --simplify 0.001                   # Build + upload to S3 + HF
    python build_geoparquet.py --simplify 0.001 --version v23.0.2 # Also tag HF revision

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
from swarm_auth import get_aws_credentials
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
LAGS_KEY = "rsct_curriculum/series_018/processed/zcta_features_labels_with_lags.parquet"
FEATURES_KEY = "rsct_curriculum/series_018/processed/zcta_features_labels.parquet"
SPLITS_KEY = "rsct_curriculum/series_018/processed/geocert_splits.parquet"
COUNTY_XWALK_KEY = "rsct_curriculum/series_018/processed/zcta_county_crosswalk.parquet"
SVI_KEY = "rsct_curriculum/series_018/processed/svi_zcta.parquet"
FLOOD_KEY = "rsct_curriculum/series_018/processed/flood_zones_zcta.parquet"
HIFLD_KEY = "rsct_curriculum/series_018/processed/hifld_zcta.parquet"
DRIVE_KEY = "rsct_curriculum/series_018/processed/drive_times_zcta.parquet"
CDC_CI_KEY = "rsct_curriculum/series_018/processed/cdc_places_ci.parquet"
ACS_MOE_KEY = "rsct_curriculum/series_018/processed/zcta_acs_margins_of_error.parquet"
OUTPUT_KEY = "rsct_curriculum/series_018/release/georsct_simplified_001.geoparquet"
TABLE_KEY = "rsct_curriculum/series_018/release/georsct_table.parquet"
CDC_CI_RELEASE_KEY = "rsct_curriculum/series_018/release/cdc_places_ci.parquet"
ACS_MOE_RELEASE_KEY = "rsct_curriculum/series_018/release/zcta_acs_margins_of_error.parquet"
PROVENANCE_KEY = "rsct_curriculum/series_018/release/geocert_v24_geoparquet_provenance.json"
REGION = "us-east-1"
CROISSANT_PATH = Path(__file__).parent.parent.parent.parent / "evidence" / "specifications" / "croissant.json"
# Fallback: when deployed to SageMaker, croissant.json is alongside code
if not CROISSANT_PATH.exists():
    CROISSANT_PATH = Path(__file__).parent / "croissant.json"

# 14 curated spatial-lag columns (S018X experiment: unique signal beyond SVI/HIFLD)
LAG_COLUMNS = [
    "lag_acs_total_pop", "lag_acs_median_age", "lag_acs_median_home_value",
    "lag_acs_median_rent", "lag_acs_median_year_built", "lag_acs_gini_index",
    "lag_acs_mean_commute_min", "lag_acs_pct_bachelors", "lag_acs_pct_vacant",
    "lag_acs_pct_food_stamps", "lag_acs_pct_no_insurance", "lag_acs_pct_veterans",
    "lag_acs_pct_female", "lag_acs_pct_wfh",
]

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
    parser.add_argument("--output", type=str, default="/tmp/georsct_simplified_001.geoparquet",
                        help="Local output path for geoparquet")
    parser.add_argument("--version", type=str, default=None,
                        help="Version tag for HuggingFace (e.g. v23.0.2). Tags the HF commit.")
    parser.add_argument("--hf-repo", type=str, default="rudymartin/georsct",
                        help="HuggingFace dataset repo")
    args = parser.parse_args()

    # On SageMaker, IAM role provides credentials (no profile needed)
    try:
        _aws = get_aws_credentials()
        boto3.client("sts", region_name=REGION, **_aws).get_caller_identity()
    except Exception:
        session = boto3.Session(region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION, **_aws)
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
    # Align column names with Croissant manifest
    if "acs_median_commute_min" in df.columns and "acs_mean_commute_min" not in df.columns:
        df = df.rename(columns={"acs_median_commute_min": "acs_mean_commute_min"})
    if "split_extrap" in df.columns and "split_extrapolation" not in df.columns:
        df = df.rename(columns={"split_extrap": "split_extrapolation"})
    if "drive_min_to_county_seat" in df.columns and "drive_min_to_county_centroid" not in df.columns:
        df = df.rename(columns={"drive_min_to_county_seat": "drive_min_to_county_centroid"})
    log.info("  %d ZCTAs, %d columns", len(df), len(df.columns))

    # -- 3b. Refresh HIFLD columns from rebuilt hifld_zcta.parquet --
    local_hifld = Path("/tmp/hifld_zcta.parquet")
    try:
        log.info("Downloading fresh HIFLD features from S3...")
        s3.download_file(BUCKET, HIFLD_KEY, str(local_hifld))
        hifld_df = pd.read_parquet(local_hifld)
        hifld_df["zcta_id"] = hifld_df["zcta_id"].astype(str).str.zfill(5)
        hifld_cols = [c for c in hifld_df.columns if c.startswith("hifld_")]
        # Drop stale HIFLD columns from features, replace with fresh
        stale = [c for c in df.columns if c.startswith("hifld_")]
        if stale:
            df = df.drop(columns=stale)
            log.info("  Dropped %d stale HIFLD columns from features", len(stale))
        df = df.merge(hifld_df[["zcta_id"] + hifld_cols], on="zcta_id", how="left")
        log.info("  Merged %d fresh HIFLD columns (%d non-null)",
                 len(hifld_cols), int(df[hifld_cols[0]].notna().sum()))
    except Exception as e:
        log.warning("Could not refresh HIFLD: %s (using features file columns)", e)

    # -- 3c. Refresh drive-time columns from rebuilt drive_times_zcta.parquet --
    local_drive = Path("/tmp/drive_times_zcta.parquet")
    try:
        log.info("Downloading fresh drive-time features from S3...")
        s3.download_file(BUCKET, DRIVE_KEY, str(local_drive))
        drive_df = pd.read_parquet(local_drive)
        for col in ("zcta_id", "zcta"):
            if col in drive_df.columns:
                drive_df = drive_df.rename(columns={col: "zcta_id"})
                break
        drive_df["zcta_id"] = drive_df["zcta_id"].astype(str).str.zfill(5)
        # Rename if needed
        if "drive_min_to_county_seat" in drive_df.columns and "drive_min_to_county_centroid" not in drive_df.columns:
            drive_df = drive_df.rename(columns={"drive_min_to_county_seat": "drive_min_to_county_centroid"})
        drive_cols = [c for c in drive_df.columns if c.startswith("drive_")]
        stale = [c for c in df.columns if c.startswith("drive_")]
        if stale:
            df = df.drop(columns=stale)
            log.info("  Dropped %d stale drive columns from features", len(stale))
        df = df.merge(drive_df[["zcta_id"] + drive_cols], on="zcta_id", how="left")
        log.info("  Merged %d fresh drive columns (%d non-null)",
                 len(drive_cols), int(df[drive_cols[0]].notna().sum()))
    except Exception as e:
        log.warning("Could not refresh drive times: %s (using features file columns)", e)

    # -- 4. Load ZCTA-county crosswalk (Census FIPS-based, majority by area) --
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

    # -- 4b. Join 14 spatial-lag columns from _with_lags parquet --
    local_lags = Path("/tmp/zcta_features_labels_with_lags.parquet")
    try:
        log.info("Downloading spatial-lag features from S3...")
        s3.download_file(BUCKET, LAGS_KEY, str(local_lags))
        lags_df = pd.read_parquet(local_lags)
        lags_df["zcta_id"] = lags_df["zcta_id"].astype(str).str.zfill(5)

        # Select only the 14 curated lag columns + join key
        available_lags = [c for c in LAG_COLUMNS if c in lags_df.columns]
        missing_lags = [c for c in LAG_COLUMNS if c not in lags_df.columns]
        if missing_lags:
            log.warning("Missing lag columns in source: %s", missing_lags)

        # Handle column name mismatch: source has lag_acs_median_commute_min,
        # Croissant expects lag_acs_mean_commute_min
        rename_map = {}
        if "lag_acs_median_commute_min" in lags_df.columns and \
           "lag_acs_mean_commute_min" not in lags_df.columns:
            rename_map["lag_acs_median_commute_min"] = "lag_acs_mean_commute_min"
            if "lag_acs_mean_commute_min" in missing_lags:
                available_lags.append("lag_acs_median_commute_min")
                missing_lags.remove("lag_acs_mean_commute_min")

        lag_merge = lags_df[["zcta_id"] + available_lags].copy()
        if rename_map:
            lag_merge = lag_merge.rename(columns=rename_map)

        df = df.merge(lag_merge, on="zcta_id", how="left")
        log.info("  Joined %d lag columns (%d ZCTAs matched)",
                 len(available_lags), df[available_lags[0] if available_lags
                 else "zcta_id"].notna().sum())
        if missing_lags:
            log.error("MISSING LAG COLUMNS (Croissant violation): %s", missing_lags)
    except Exception as e:
        log.error("Could not load spatial lags: %s", e)
        log.error("The 14 lag_acs_* columns will be MISSING -- Croissant violation!")

    # -- 4c. Generate coverage flags --
    cdc_target_cols = [c for c in df.columns if c.startswith("target_") and c not in
                       ("target_elevation", "target_night_lights", "target_tree_cover",
                        "target_home_value", "target_income", "target_population_density")]
    if cdc_target_cols:
        df["has_cdc_places"] = df[cdc_target_cols[0]].notna()
    else:
        df["has_cdc_places"] = False
    df["has_income"] = df["target_income"].notna() if "target_income" in df.columns else False
    df["has_home_value"] = df["target_home_value"].notna() if "target_home_value" in df.columns else False
    # has_cdc_ci: whether this ZCTA has confidence interval data in sidecar
    try:
        local_ci = Path("/tmp/cdc_places_ci.parquet")
        if not local_ci.exists():
            s3.download_file(BUCKET, CDC_CI_KEY, str(local_ci))
        ci_df = pd.read_parquet(local_ci)
        ci_id_col = "zcta_id" if "zcta_id" in ci_df.columns else "zcta"
        ci_zctas = set(ci_df[ci_id_col].astype(str).str.zfill(5))
        df["has_cdc_ci"] = df["zcta_id"].isin(ci_zctas)
    except Exception:
        df["has_cdc_ci"] = df["has_cdc_places"]
    log.info("  Coverage flags: has_cdc_places=%d, has_income=%d, has_home_value=%d, has_cdc_ci=%d",
             int(df["has_cdc_places"].sum()), int(df["has_income"].sum()),
             int(df["has_home_value"].sum()), int(df["has_cdc_ci"].sum()))

    # -- 5. Select columns --
    acs_cols = sorted(c for c in df.columns if c.startswith(ACS_PREFIX))
    lag_col_names = [c for c in LAG_COLUMNS if c in df.columns]
    svi_col_names = sorted(c for c in df.columns if c.startswith("svi_"))
    flood_col_names = sorted(c for c in df.columns if c.startswith("flood_"))
    hifld_col_names = sorted(c for c in df.columns if c.startswith("hifld_"))
    drive_col_names = sorted(c for c in df.columns if c.startswith("drive_"))
    split_col_names = sorted(c for c in df.columns if c.startswith("split_"))
    coverage_col_names = sorted(c for c in df.columns if c.startswith("has_"))

    meta_cols = ["zcta_id", "latitude", "longitude"]
    # Add county/state cols if present (from crosswalk merge above)
    for c in ["county_fips", "county_name", "state_fips", "state"]:
        if c in df.columns:
            meta_cols.append(c)
    # Add population if present
    if "population" in df.columns:
        meta_cols.append("population")

    target_cols = [c for c in TARGET_COLUMNS if c in df.columns]
    missing_targets = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing_targets:
        log.warning("Missing target columns (will skip): %s", missing_targets)

    keep_cols = (meta_cols + split_col_names + coverage_col_names
                 + acs_cols + lag_col_names + svi_col_names + flood_col_names
                 + hifld_col_names + drive_col_names + target_cols)
    # Deduplicate while preserving order
    seen = set()
    keep_cols = [c for c in keep_cols if c in df.columns and not (c in seen or seen.add(c))]

    tabular = df[keep_cols].copy()
    log.info("  Selected %d columns for output", len(tabular.columns))

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
    # metadata -> splits -> coverage -> ACS features -> spatial lags ->
    # SVI -> flood -> HIFLD -> drive times -> targets -> geometry
    ordered = (
        meta_cols
        + split_col_names
        + coverage_col_names
        + acs_cols
        + lag_col_names
        + svi_col_names
        + flood_col_names
        + hifld_col_names
        + drive_col_names
        + target_cols
        + ["geometry"]
    )
    ordered = [c for c in ordered if c in geo.columns]
    geo = geo[ordered]

    # -- 9. Save GeoParquet (with geometry) --
    output = Path(args.output)
    log.info("")
    log.info("Writing GeoParquet to %s", output)
    geo.to_parquet(output, index=False)
    size_mb = output.stat().st_size / (1024 * 1024)
    log.info("  Size: %.1f MB", size_mb)
    log.info("  Rows: %d, Columns: %d", len(geo), len(geo.columns))

    # -- 9b. Save table parquet (no geometry) --
    table_output = output.parent / "georsct_table.parquet"
    table_df = geo.drop(columns=["geometry"])
    log.info("Writing table parquet to %s", table_output)
    table_df.to_parquet(table_output, index=False)
    table_size_mb = table_output.stat().st_size / (1024 * 1024)
    log.info("  Size: %.1f MB", table_size_mb)

    # -- 9c. Download uncertainty sidecars from S3 --
    sidecar_files = {}
    for key, name in [(CDC_CI_KEY, "cdc_places_ci.parquet"),
                      (ACS_MOE_KEY, "zcta_acs_margins_of_error.parquet")]:
        local_path = output.parent / name
        try:
            log.info("Downloading uncertainty sidecar: %s", name)
            s3.download_file(BUCKET, key, str(local_path))
            sc_df = pd.read_parquet(local_path)
            sc_size = local_path.stat().st_size / (1024 * 1024)
            log.info("  %s: %d rows, %d columns, %.1f MB",
                     name, len(sc_df), len(sc_df.columns), sc_size)
            sidecar_files[name] = local_path
        except Exception as e:
            log.warning("  Could not download %s: %s (skipping)", name, e)

    # -- 9d. Croissant pre-flight validation (GATE) --
    if CROISSANT_PATH.exists():
        log.info("")
        log.info("=== CROISSANT PRE-FLIGHT VALIDATION ===")
        from validate_croissant import extract_manifest_columns, validate_parquet_against_manifest
        import json as _json
        with open(CROISSANT_PATH) as _f:
            _croissant = _json.load(_f)
        _manifest = extract_manifest_columns(_croissant)
        _fields = _manifest.get("georsct-main", [])
        _violations = validate_parquet_against_manifest(str(output), _fields)
        _critical = [v for v in _violations if v["severity"] == "CRITICAL"]
        _warnings = [v for v in _violations if v["severity"] == "WARNING"]
        for v in _violations:
            if v["severity"] == "CRITICAL":
                log.error("[CRITICAL] %s", v["message"])
            elif v["severity"] == "WARNING":
                log.warning("[WARNING] %s", v["message"])
            else:
                log.info("[INFO] %s", v["message"])
        if _critical:
            log.error("CROISSANT VALIDATION FAILED: %d critical violations", len(_critical))
            log.error("Upload BLOCKED. Fix data before publishing.")
            sys.exit(1)
        else:
            log.info("CROISSANT VALIDATION PASSED (%d warnings)", len(_warnings))
    else:
        log.warning("Croissant spec not found at %s -- skipping validation", CROISSANT_PATH)

    # -- 10. Upload --
    if not args.dry_run:
        log.info("Uploading geoparquet to s3://%s/%s", BUCKET, OUTPUT_KEY)
        s3.upload_file(str(output), BUCKET, OUTPUT_KEY)
        log.info("Uploading table parquet to s3://%s/%s", BUCKET, TABLE_KEY)
        s3.upload_file(str(table_output), BUCKET, TABLE_KEY)

        # Upload uncertainty sidecars to release prefix
        for name, release_key in [("cdc_places_ci.parquet", CDC_CI_RELEASE_KEY),
                                  ("zcta_acs_margins_of_error.parquet", ACS_MOE_RELEASE_KEY)]:
            if name in sidecar_files:
                log.info("Uploading %s to s3://%s/%s", name, BUCKET, release_key)
                s3.upload_file(str(sidecar_files[name]), BUCKET, release_key)

        provenance = {
            "operation": "build_geoparquet",
            "timestamp": timestamp,
            "tiger_source": TIGER_URL,
            "features_source": f"s3://{BUCKET}/{FEATURES_KEY}",
            "output_geoparquet": f"s3://{BUCKET}/{OUTPUT_KEY}",
            "output_table": f"s3://{BUCKET}/{TABLE_KEY}",
            "output_cdc_ci": f"s3://{BUCKET}/{CDC_CI_RELEASE_KEY}",
            "output_acs_moe": f"s3://{BUCKET}/{ACS_MOE_RELEASE_KEY}",
            "crs": "EPSG:4326",
            "simplify_tolerance": args.simplify,
            "n_zctas": len(geo),
            "n_columns": len(geo.columns),
            "n_unmatched_boundaries": len(unmatched) if unmatched else 0,
            "geoparquet_size_mb": round(size_mb, 1),
            "table_size_mb": round(table_size_mb, 1),
            "columns": list(geo.columns),
        }
        s3.put_object(
            Bucket=BUCKET, Key=PROVENANCE_KEY,
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved: s3://%s/%s", BUCKET, PROVENANCE_KEY)

        # -- 10b. Upload to HuggingFace + tag --
        try:
            from huggingface_hub import HfApi
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not hf_token:
                log.warning("HF_TOKEN not set -- skipping HuggingFace upload")
                raise RuntimeError("No HF token")
            api = HfApi(token=hf_token)
            hf_repo = args.hf_repo
            hf_files = [
                (str(output), "georsct_simplified_001.geoparquet"),
                (str(table_output), "georsct_table.parquet"),
            ]
            for name, local_path in sidecar_files.items():
                hf_files.append((str(local_path), name))

            log.info("Uploading %d files to HuggingFace %s...", len(hf_files), hf_repo)
            for local_path, repo_path in hf_files:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=hf_repo,
                    repo_type="dataset",
                    commit_message=f"update: {repo_path}",
                )
                log.info("  Uploaded %s", repo_path)

            if args.version:
                api.create_tag(
                    hf_repo,
                    repo_type="dataset",
                    tag=args.version,
                    tag_message=f"{args.version}: {len(geo)} ZCTAs, {len(geo.columns)} cols",
                )
                log.info("  Tagged HF as %s", args.version)
                log.info("  Pin with: load_dataset('%s', revision='%s')",
                         hf_repo, args.version)
        except ImportError:
            log.warning("huggingface_hub not installed -- skipping HF upload")
        except Exception as e:
            log.warning("HF upload failed: %s (S3 upload succeeded)", e)
    else:
        log.info("[DRY RUN] Skipping S3 and HF upload.")

    log.info("")
    log.info("=== COLUMN SUMMARY ===")
    log.info("  Metadata:    %d columns", len(meta_cols))
    log.info("  Splits:      %d columns", len(split_col_names))
    log.info("  Coverage:    %d columns", len(coverage_col_names))
    log.info("  Features:    %d ACS columns", len(acs_cols))
    log.info("  Spatial lag: %d columns", len(lag_col_names))
    log.info("  SVI:         %d columns", len(svi_col_names))
    log.info("  Flood:       %d columns", len(flood_col_names))
    log.info("  HIFLD:       %d columns", len(hifld_col_names))
    log.info("  Drive times: %d columns", len(drive_col_names))
    log.info("  Targets:     %d task columns", len(target_cols))
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
