#!/usr/bin/env python3
"""
run_twi.py -- SageMaker run script: TWI and watershed features per ZCTA.

Downloads EPA StreamCat pre-computed watershed attributes (WetIndx, Slope)
for all 18 CONUS HUC2 regions, joins to the COMID-to-ZCTA spatial crosswalk
(area-weighted), and aggregates to ZCTA level.

Instance: ml.m5.xlarge (4 vCPU, 16 GB RAM)
Runtime:  ~15-25 min (18 StreamCat downloads + vectorized join)
Cost:     ~$0.10

Inputs (SageMaker mounts):
  /opt/ml/processing/input/data/comid_zcta_crosswalk.parquet
  /opt/ml/processing/input/data/zcta_county_crosswalk.parquet

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

STREAMCAT_BASE = (
    "https://gaftp.epa.gov/epadatacommons/ORD/"
    "NHDPlusLandscapeAttributes/StreamCat/HydroRegions/"
)
CONUS_HUC2 = [f"{i:02d}" for i in range(1, 19)]

STREAMCAT_COLS = {
    "WetIndx": {
        "CAT_WetIndx": "twi_cat_mean",
        "WsWetIndx":   "twi_ws_mean",
    },
    "Slope": {
        "CAT_Slope": "slope_cat_mean_pct",
        "WsSlope":   "slope_ws_mean_pct",
    },
}


def _s3():
    return boto3.client("s3")


def fetch_streamcat_region(metric: str, huc2: str) -> pd.DataFrame:
    fname = f"{metric}_{huc2}.zip"
    url = STREAMCAT_BASE + fname
    log.info("  Fetching %s", fname)
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("  SKIP %s: %s", fname, exc)
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = f"{metric}_{huc2}.csv"
            if csv_name not in zf.namelist():
                csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, dtype={"COMID": str})
    except Exception as exc:
        log.warning("  Parse error %s: %s", fname, exc)
        return pd.DataFrame()

    df["COMID"] = df["COMID"].astype(str).str.strip()
    return df


def fetch_streamcat_national(metric: str, regions: list[str]) -> pd.DataFrame:
    log.info("Fetching StreamCat/%s for %d regions...", metric, len(regions))
    col_map = STREAMCAT_COLS.get(metric, {})
    parts = []
    for huc2 in regions:
        part = fetch_streamcat_region(metric, huc2)
        if part.empty:
            continue
        keep = ["COMID"] + [c for c in col_map if c in part.columns]
        if len(keep) < 2:
            log.warning("  No expected columns in %s_%s", metric, huc2)
            continue
        parts.append(part[keep].rename(columns=col_map))

    if not parts:
        log.error("No data for metric %s", metric)
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    log.info("  %d catchments loaded for %s", len(result), metric)
    return result


def aggregate_comid_to_zcta(
    twi_df: pd.DataFrame, slope_df: pd.DataFrame, xwalk: pd.DataFrame
) -> pd.DataFrame:
    """Area-weighted aggregation from COMID to ZCTA (vectorized)."""
    log.info("Joining StreamCat to %d COMID-ZCTA pairs...", len(xwalk))

    features = twi_df.merge(slope_df, on="COMID", how="outer")
    merged = xwalk.merge(features, on="COMID", how="left")

    numeric_cols = [c for c in features.columns if c != "COMID"]
    result_parts = {}

    for col in numeric_cols:
        valid = merged[["zcta_id", col, "area_fraction"]].dropna(subset=[col])
        weighted = valid.copy()
        weighted["_w_val"] = weighted[col] * weighted["area_fraction"]
        g = weighted.groupby("zcta_id").agg(
            _w_sum=("_w_val", "sum"),
            _af_sum=("area_fraction", "sum"),
        )
        result_parts[col] = (g["_w_sum"] / g["_af_sum"]).rename(col)

    if not result_parts:
        return pd.DataFrame(columns=["zcta_id"])

    result = pd.concat(result_parts.values(), axis=1).reset_index()
    log.info("  Aggregated to %d ZCTAs", len(result))
    return result


def derive_composite(df: pd.DataFrame) -> pd.DataFrame:
    if "twi_cat_mean" in df.columns:
        df["twi_mean"] = df["twi_cat_mean"]
        df["twi_pct_high_proxy"] = (df["twi_cat_mean"] / 15.0).clip(0, 1)
    if "slope_cat_mean_pct" in df.columns:
        df["slope_mean_pct"] = df["slope_cat_mean_pct"]
    if "twi_cat_mean" in df.columns and "slope_cat_mean_pct" in df.columns:
        df["terrain_flood_index"] = (
            df["twi_cat_mean"] * (1.0 - (df["slope_cat_mean_pct"] / 100.0).clip(0, 1))
        ).clip(0, None)
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/opt/ml/processing/input/data")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    regions = args.regions or CONUS_HUC2
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Fetch StreamCat ---
    log.info("=== FETCHING STREAMCAT ===")
    twi_df = fetch_streamcat_national("WetIndx", regions)
    slope_df = fetch_streamcat_national("Slope", regions)

    if twi_df.empty and slope_df.empty:
        log.error("No StreamCat data retrieved")
        sys.exit(1)

    # --- Load COMID-ZCTA crosswalk ---
    log.info("=== LOADING COMID-ZCTA CROSSWALK ===")
    xwalk_path = data_dir / "comid_zcta_crosswalk.parquet"
    if not xwalk_path.exists():
        log.error("comid_zcta_crosswalk.parquet not found at %s", xwalk_path)
        sys.exit(1)

    xwalk = pd.read_parquet(xwalk_path)
    xwalk["COMID"] = xwalk["COMID"].astype(str)
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    log.info("Crosswalk: %d COMID-ZCTA pairs, %d COMIDs, %d ZCTAs",
             len(xwalk), xwalk["COMID"].nunique(), xwalk["zcta_id"].nunique())

    # --- Aggregate ---
    log.info("=== AGGREGATING TO ZCTA ===")
    result = aggregate_comid_to_zcta(twi_df, slope_df, xwalk)
    result = derive_composite(result)

    # --- Zero-fill all ZCTAs from county crosswalk ---
    county_xwalk_path = data_dir / "zcta_county_crosswalk.parquet"
    if county_xwalk_path.exists():
        all_zctas = pd.read_parquet(county_xwalk_path)[["zcta_id"]].drop_duplicates()
        all_zctas["zcta_id"] = all_zctas["zcta_id"].astype(str).str.zfill(5)
        result = all_zctas.merge(result, on="zcta_id", how="left")
        log.info("Zero-filled to %d ZCTAs (from county crosswalk)", len(result))

    # --- Summary ---
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:   %d", len(result))
    log.info("Columns: %s", [c for c in result.columns if c != "zcta_id"])
    for col in ("twi_mean", "terrain_flood_index", "slope_mean_pct"):
        if col in result.columns:
            v = result[col].dropna()
            log.info("  %s: mean=%.2f min=%.2f max=%.2f coverage=%d/%d",
                     col, v.mean(), v.min(), v.max(), len(v), len(result))

    # --- Save and upload ---
    out_path = output_dir / "twi_features_zcta.parquet"
    result.to_parquet(out_path, index=False)
    log.info("Saved: %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)

    s3 = _s3()
    s3.upload_file(str(out_path), BUCKET, OUTPUT_KEY)
    log.info("  -> s3://%s/%s", BUCKET, OUTPUT_KEY)

    provenance = {
        "operation": "build_twi_features",
        "timestamp": timestamp,
        "source": STREAMCAT_BASE,
        "metrics": list(STREAMCAT_COLS.keys()),
        "huc2_regions": regions,
        "approximation_method": "comid_area_weighted",
        "n_zctas": len(result),
        "n_comids_in_crosswalk": int(xwalk["COMID"].nunique()),
    }
    s3.put_object(Bucket=BUCKET, Key=PROVENANCE_KEY,
                  Body=json.dumps(provenance, indent=2),
                  ContentType="application/json")
    log.info("Done.")


if __name__ == "__main__":
    main()
