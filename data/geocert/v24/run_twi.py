#!/usr/bin/env python3
"""
run_twi.py -- SageMaker run script: TWI and watershed features per ZCTA.

Downloads EPA StreamCat / USGS ScienceBase CONUS-wide parquet files for
Topographic Wetness Index (TWI) and basin characteristics (slope), joins
to the COMID-to-ZCTA spatial crosswalk (area-weighted), aggregates to ZCTA.

Instance: ml.m5.xlarge (4 vCPU, 16 GB RAM)
Runtime:  ~15-20 min (download ~100 MB + vectorized join on 3M rows)
Cost:     ~$0.08

--- DATA SOURCE HISTORY ---

The old EPA StreamCat FTP (gaftp.epa.gov) went offline in 2025. Data moved to
USGS ScienceBase. Two URL patterns exist on ScienceBase — only one is public:

  PUBLIC  (used here): catalog/file/get/{item_id}?f=__disk__{hash}
    Returned by the ScienceBase JSON API as "downloadUri" in the files[] array.
    These are the ZIP downloads. No auth required.

  AUTH-GATED (do not use): sciencebase.usgs.gov/manager/download/{hash}
    The parquet/CSV direct-download URLs visible in the browser. Return 0 bytes
    without a USGS session cookie. Not useful from a container.

Strategy: query the JSON API at runtime to get the current downloadUri for the
named ZIP file, then download and extract. Stable item IDs survive file re-issues.

ScienceBase items:
  TWI:   https://www.sciencebase.gov/catalog/item/56f97be4e4b0a6037df06b70
  Slope: https://www.sciencebase.gov/catalog/item/57976a0ce4b021cadec97890

--- FILE FORMAT NOTE ---

StreamCat distributes data as ZIPs containing .txt files (not .csv), but the
content is standard comma-delimited with a header row. pandas.read_csv handles
.txt files identically to .csv. download_zip_csv accepts both extensions.

--- AGGREGATION APPROACH ---

Area-weighted mean from COMID to ZCTA using the comid_zcta_crosswalk.parquet
(built by sagemaker_comid_zcta.py). Each COMID contributes proportionally to
its area_fraction within the ZCTA. Fully vectorized groupby — no Python loops
over ZCTAs. county_crosswalk used only for zero-filling: ensures all ~33K ZCTAs
appear in the output even if they have no NHDPlus catchment overlap.

--- INPUTS ---

  /opt/ml/processing/input/data/   -- comid_zcta_crosswalk.parquet
  /opt/ml/processing/input/county/ -- zcta_county_crosswalk.parquet
  (separate LocalPaths required — SageMaker rejects duplicate LocalPaths)

Output:
  s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/twi_features_zcta.parquet
"""

import io
import json
import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
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

BUCKET = "swarm-yrsn-datasets"
PREFIX = "rsct_curriculum/series_018/processed"
OUTPUT_KEY = f"{PREFIX}/twi_features_zcta.parquet"
PROVENANCE_KEY = f"{PREFIX}/twi_features_zcta_provenance.json"

# USGS ScienceBase item IDs (stable references even if file hashes change)
SB_TWI_ITEM   = "56f97be4e4b0a6037df06b70"   # Topographic Wetness Index CONUS
SB_SLOPE_ITEM = "57976a0ce4b021cadec97890"   # Basin characteristics (slope, etc.)


def _s3():
    return boto3.client("s3")


def get_sciencebase_zip_url(item_id: str, zip_name: str) -> str:
    """Query ScienceBase JSON API and return the public ZIP download URL."""
    url = f"https://www.sciencebase.gov/catalog/item/{item_id}?format=json"
    log.info("  Querying ScienceBase: %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    files = resp.json().get("files", [])
    for f in files:
        if f.get("name", "") == zip_name:
            uri = f.get("downloadUri", "")
            log.info("  Found: %s -> %s", zip_name, uri[:80])
            return uri
    raise RuntimeError(
        f"{zip_name} not found in ScienceBase item {item_id}. "
        f"Files: {[f.get('name') for f in files]}"
    )


def download_zip_csv(url: str, label: str) -> pd.DataFrame:
    """Download a ZIP from URL, extract first CSV or TXT tabular file, return as DataFrame."""
    log.info("Downloading %s...", label)
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    data = b"".join(resp.iter_content(chunk_size=1024 * 1024))
    log.info("  Downloaded %.1f MB for %s", len(data) / 1e6, label)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # StreamCat uses .txt extension for CSV-formatted data
        tabular = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not tabular:
            raise RuntimeError(f"No tabular file found in ZIP for {label}: {zf.namelist()}")
        csv_name = tabular[0]
        log.info("  Extracting: %s", csv_name)
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, dtype={"COMID": str})
    log.info("  Shape: %s | Columns: %s", df.shape, list(df.columns[:10]))
    return df


def normalize_comid(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize COMID column to uppercase string."""
    for col in df.columns:
        if col.upper() == "COMID":
            df = df.rename(columns={col: "COMID"})
            break
    df["COMID"] = df["COMID"].astype(str).str.strip()
    return df


def aggregate_comid_to_zcta(
    twi_df: pd.DataFrame,
    slope_df: pd.DataFrame,
    xwalk: pd.DataFrame,
) -> pd.DataFrame:
    """Area-weighted aggregation from COMID to ZCTA (vectorized)."""
    log.info("Merging TWI + slope on COMID...")
    features = twi_df.merge(slope_df, on="COMID", how="outer")
    log.info("  Feature table: %d COMIDs", len(features))

    merged = xwalk.merge(features, on="COMID", how="left")
    log.info("  Joined to crosswalk: %d rows", len(merged))

    numeric_cols = [c for c in features.columns if c != "COMID"]
    result_parts = {}

    for col in numeric_cols:
        valid = merged[["zcta_id", col, "area_fraction"]].dropna(subset=[col])
        valid = valid.copy()
        valid["_wv"] = valid[col] * valid["area_fraction"]
        g = valid.groupby("zcta_id").agg(_wsum=("_wv", "sum"), _afsum=("area_fraction", "sum"))
        result_parts[col] = (g["_wsum"] / g["_afsum"]).rename(col)

    result = pd.concat(result_parts.values(), axis=1).reset_index()
    log.info("  Aggregated to %d ZCTAs", len(result))
    return result


def select_and_rename(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract TWI and slope columns from a merged feature table.
    ScienceBase _cat parquets use column patterns like CAT_WetIndx, CAT_Slope, etc.
    Returns (twi_df, slope_df) each with COMID + relevant columns renamed.
    """
    cols = list(df.columns)
    twi_cols  = [c for c in cols if "wetindx" in c.lower() or "twi" in c.lower()]
    slp_cols  = [c for c in cols if "slope" in c.lower() or "slp" in c.lower()]

    renames_twi  = {c: f"twi_{c.lower().replace('cat_','').replace('ws','ws_')}" for c in twi_cols}
    renames_slp  = {c: f"slope_{c.lower().replace('cat_','').replace('ws','ws_')}" for c in slp_cols}

    twi_df  = df[["COMID"] + twi_cols].rename(columns=renames_twi)  if twi_cols  else df[["COMID"]]
    slp_df  = df[["COMID"] + slp_cols].rename(columns=renames_slp)  if slp_cols  else df[["COMID"]]

    log.info("  TWI columns: %s", list(twi_df.columns))
    log.info("  Slope columns: %s", list(slp_df.columns))
    return twi_df, slp_df


def derive_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived flood-relevant features from raw TWI and slope columns."""
    # Find the primary TWI and slope columns (catchment-level preferred)
    twi_col   = next((c for c in df.columns if "wetindx" in c.lower()), None)
    slope_col = next((c for c in df.columns if "slope" in c.lower()), None)

    if twi_col:
        df["twi_mean"] = df[twi_col]
        df["twi_pct_high_proxy"] = (df[twi_col] / 15.0).clip(0, 1)

    if slope_col:
        df["slope_mean_pct"] = df[slope_col]

    if twi_col and slope_col:
        df["terrain_flood_index"] = (
            df[twi_col] * (1.0 - (df[slope_col] / 100.0).clip(0, 1))
        ).clip(0, None)

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="/opt/ml/processing/input/data")
    parser.add_argument("--county-dir", default="/opt/ml/processing/input/county")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    data_dir   = Path(args.data_dir)
    county_dir = Path(args.county_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve ScienceBase ZIP URLs at runtime ---
    log.info("=== RESOLVING SCIENCEBASE DOWNLOAD URLS ===")
    twi_url   = get_sciencebase_zip_url(SB_TWI_ITEM,   "TWI_CONUS.zip")
    slope_url = get_sciencebase_zip_url(SB_SLOPE_ITEM, "BASIN_CHAR_CAT_CONUS.zip")

    # --- Download ---
    log.info("=== DOWNLOADING STREAMCAT DATA ===")
    twi_raw   = download_zip_csv(twi_url,   "TWI_CONUS.zip")
    slope_raw = download_zip_csv(slope_url, "BASIN_CHAR_CAT_CONUS.zip")

    twi_raw   = normalize_comid(twi_raw)
    slope_raw = normalize_comid(slope_raw)

    twi_df, _ = select_and_rename(twi_raw)
    _, slp_df = select_and_rename(slope_raw)

    # --- Load COMID-ZCTA crosswalk ---
    log.info("=== LOADING COMID-ZCTA CROSSWALK ===")
    xwalk_path = data_dir / "comid_zcta_crosswalk.parquet"
    if not xwalk_path.exists():
        log.error("comid_zcta_crosswalk.parquet not found at %s", xwalk_path)
        sys.exit(1)
    xwalk = pd.read_parquet(xwalk_path)
    xwalk["COMID"]   = xwalk["COMID"].astype(str)
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    log.info("Crosswalk: %d pairs | %d COMIDs | %d ZCTAs",
             len(xwalk), xwalk["COMID"].nunique(), xwalk["zcta_id"].nunique())

    # --- Aggregate ---
    log.info("=== AGGREGATING TO ZCTA ===")
    result = aggregate_comid_to_zcta(twi_df, slp_df, xwalk)
    result = derive_composite(result)

    # --- Zero-fill all ZCTAs ---
    county_path = county_dir / "zcta_county_crosswalk.parquet"
    if county_path.exists():
        all_zctas = pd.read_parquet(county_path)[["zcta_id"]].drop_duplicates()
        all_zctas["zcta_id"] = all_zctas["zcta_id"].astype(str).str.zfill(5)
        result = all_zctas.merge(result, on="zcta_id", how="left")
        log.info("Zero-filled to %d ZCTAs", len(result))

    # --- Summary ---
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:   %d | Columns: %d", len(result), len(result.columns))
    for col in ("twi_mean", "terrain_flood_index", "slope_mean_pct"):
        if col in result.columns:
            v = result[col].dropna()
            log.info("  %-28s mean=%.2f  min=%.2f  max=%.2f  n=%d/%d",
                     col, v.mean(), v.min(), v.max(), len(v), len(result))

    # --- Save and upload ---
    out_path = output_dir / "twi_features_zcta.parquet"
    result.to_parquet(out_path, index=False)
    log.info("Saved: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    s3 = _s3()
    s3.upload_file(str(out_path), BUCKET, OUTPUT_KEY)
    log.info("  -> s3://%s/%s", BUCKET, OUTPUT_KEY)

    provenance = {
        "operation": "build_twi_features",
        "timestamp": timestamp,
        "source_twi":   f"https://www.sciencebase.gov/catalog/item/{SB_TWI_ITEM}",
        "source_slope": f"https://www.sciencebase.gov/catalog/item/{SB_SLOPE_ITEM}",
        "approximation_method": "comid_area_weighted",
        "n_zctas": len(result),
        "n_comids_in_crosswalk": int(xwalk["COMID"].nunique()),
        "columns": [c for c in result.columns if c != "zcta_id"],
    }
    s3.put_object(Bucket=BUCKET, Key=PROVENANCE_KEY,
                  Body=json.dumps(provenance, indent=2),
                  ContentType="application/json")
    log.info("Done.")


if __name__ == "__main__":
    main()
