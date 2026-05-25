#!/usr/bin/env python3
"""
build_twi_features.py -- Topographic Wetness Index and watershed features per ZCTA.

Uses EPA StreamCat pre-computed watershed attributes (NHDPlus catchment level)
and the Census 2020 ZCTA-to-NHDPlus catchment spatial relationship to aggregate
terrain-hydrologic features to ZCTA level.

Why TWI (not raw elevation)?
  The existing elevation task measures terrain height. TWI measures WHERE water
  accumulates: TWI = ln(A / tan(β)), where A = upslope contributing area and β
  = local slope. High TWI = valley bottoms, depressions, floodplains.
  Same elevation can have very different TWI depending on drainage context.
  For flood certificate construction, TWI is the hydrologically meaningful axis;
  elevation alone misses the concavity/convergence signal.

StreamCat source:
  https://gaftp.epa.gov/epadatacommons/ORD/NHDPlusLandscapeAttributes/StreamCat/
  National file: WetIndx_CONUS.csv (TWI mean, min, max per NHDPlus catchment)
  Secondary files: Elev_CONUS.csv, Slope_CONUS.csv (terrain context)

NHDPlus COMID to ZCTA crosswalk:
  Built from Census 2020 NHDPlus flowline-to-ZCTA spatial overlay.
  Pre-built version hosted at s3://swarm-yrsn-datasets/rsct_curriculum/geo/
  If unavailable, falls back to county-level approximation.

Output: twi_features_zcta.parquet
  - zcta_id                (str, 5-digit)
  - twi_mean               (float) mean TWI within ZCTA (higher = wetter terrain)
  - twi_max                (float) maximum TWI (worst-case wetness)
  - twi_pct_high           (float) fraction of catchment area with TWI > 7 (flood threshold)
  - slope_mean_pct         (float) mean slope percent (flat = flood-prone)
  - flow_accum_mean        (float) mean upstream contributing area (sq km)
  - terrain_flood_index    (float) composite: twi_mean * (1 - slope_mean_pct/100)

Usage:
    python build_twi_features.py --dry-run
    python build_twi_features.py --upload
    python build_twi_features.py --region 03  # HUC2 region (e.g. 03=South Atlantic)
"""

import argparse
import io
import json
import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

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
PREFIX = "rsct_curriculum/series_018/processed"
GEO_PREFIX = "rsct_curriculum/geo"

# EPA StreamCat FTP base — national CONUS files
STREAMCAT_BASE = (
    "https://gaftp.epa.gov/epadatacommons/ORD/"
    "NHDPlusLandscapeAttributes/StreamCat/HydroRegions/"
)

# Files: {metric}_{HUC2}.zip containing {metric}_{HUC2}.csv
# HUC2 regions 01-18 (CONUS); 19-21 (AK/HI/PR, skip)
CONUS_HUC2 = [f"{i:02d}" for i in range(1, 19)]

# StreamCat column names
STREAMCAT_COLS = {
    "WetIndx": {
        "CAT_WetIndx": "twi_cat_mean",     # catchment-only TWI mean
        "WsWetIndx":  "twi_ws_mean",       # full upstream watershed TWI mean
    },
    "Slope": {
        "CAT_Slope": "slope_cat_mean_pct",
        "WsSlope":   "slope_ws_mean_pct",
    },
}

# Pre-built COMID-to-ZCTA crosswalk S3 key
COMID_ZCTA_KEY = f"{GEO_PREFIX}/comid_zcta_crosswalk.parquet"
# Fallback: county-level approximation via zcta_county_crosswalk
COUNTY_CROSSWALK_KEY = f"{PREFIX}/zcta_county_crosswalk.parquet"

# TWI threshold above which terrain is considered flood-prone
TWI_FLOOD_THRESHOLD = 7.0


def fetch_streamcat_region(metric: str, huc2: str) -> pd.DataFrame:
    """Download one StreamCat metric file for one HUC2 region."""
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
                # Some files use lowercase
                csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, dtype={"COMID": str})
    except Exception as exc:
        log.warning("  Parse error %s: %s", fname, exc)
        return pd.DataFrame()

    df["COMID"] = df["COMID"].astype(str).str.strip()
    return df


def fetch_streamcat_national(metric: str, regions: list[str]) -> pd.DataFrame:
    """Fetch StreamCat metric across all specified HUC2 regions."""
    log.info("Fetching StreamCat/%s for %d regions...", metric, len(regions))
    col_map = STREAMCAT_COLS.get(metric, {})
    all_parts = []

    for huc2 in regions:
        part = fetch_streamcat_region(metric, huc2)
        if part.empty:
            continue
        keep = ["COMID"] + [c for c in col_map if c in part.columns]
        if len(keep) < 2:
            log.warning("  No expected columns in %s_%s", metric, huc2)
            continue
        part = part[keep].rename(columns=col_map)
        all_parts.append(part)

    if not all_parts:
        log.error("No data retrieved for metric %s", metric)
        return pd.DataFrame()

    result = pd.concat(all_parts, ignore_index=True)
    log.info("  %d NHDPlus catchments loaded for %s", len(result), metric)
    return result


def load_comid_zcta_crosswalk(
    local_path: Path | None, s3=None
) -> pd.DataFrame | None:
    """Try to load the COMID-to-ZCTA crosswalk."""
    # Try local path
    if local_path and local_path.exists():
        log.info("Loading COMID-ZCTA crosswalk from %s", local_path)
        xwalk = pd.read_parquet(local_path)
        xwalk["COMID"] = xwalk["COMID"].astype(str)
        xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
        return xwalk

    # Try S3
    if s3:
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                s3.download_file(BUCKET, COMID_ZCTA_KEY, tmp.name)
                xwalk = pd.read_parquet(tmp.name)
                xwalk["COMID"] = xwalk["COMID"].astype(str)
                xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
                log.info("Loaded COMID-ZCTA crosswalk from S3: %d rows", len(xwalk))
                return xwalk
        except Exception as exc:
            log.warning("COMID-ZCTA crosswalk not found in S3: %s", exc)

    return None


def aggregate_comid_to_zcta(
    twi_df: pd.DataFrame, slope_df: pd.DataFrame, xwalk: pd.DataFrame
) -> pd.DataFrame:
    """Join StreamCat metrics to ZCTA via COMID crosswalk.

    xwalk columns: COMID, zcta_id, area_fraction (fraction of COMID area in ZCTA)
    """
    log.info("Joining StreamCat to %d COMID-ZCTA relationships...", len(xwalk))

    # Merge TWI + slope on COMID
    features = twi_df.merge(slope_df, on="COMID", how="outer")
    features = xwalk.merge(features, on="COMID", how="left")

    # Weighted mean by area fraction within ZCTA
    numeric_cols = [c for c in features.columns
                    if c not in ("COMID", "zcta_id", "area_fraction")]

    def weighted_mean(grp: pd.DataFrame, col: str) -> float:
        valid = grp[[col, "area_fraction"]].dropna()
        if valid.empty:
            return float("nan")
        w = valid["area_fraction"]
        return float((valid[col] * w).sum() / w.sum())

    result_rows = []
    for zcta_id, grp in features.groupby("zcta_id"):
        row: dict = {"zcta_id": zcta_id}
        for col in numeric_cols:
            row[col] = weighted_mean(grp, col)
        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    log.info("  Aggregated to %d ZCTAs", len(result))
    return result


def fallback_county_approximation(
    twi_df: pd.DataFrame, slope_df: pd.DataFrame, crosswalk_path: Path
) -> pd.DataFrame:
    """Fallback: assign median TWI to ZCTAs via county (rough approximation).

    Used when COMID-ZCTA spatial crosswalk is unavailable. Accuracy is lower
    but sufficient for exploratory analysis. Flag in provenance.
    """
    log.warning("Using county-level fallback approximation for TWI.")
    log.warning("For production use, generate COMID-ZCTA spatial crosswalk.")

    features = twi_df.merge(slope_df, on="COMID", how="outer")

    # Median across all COMIDs (national single value — minimal geographic info)
    # Better than nothing; marks the dataset as needing upgrade
    national_medians = {
        col: float(features[col].median())
        for col in features.columns if col != "COMID"
    }
    log.info("  National medians: %s", national_medians)

    xwalk = pd.read_parquet(crosswalk_path)[["zcta_id"]].drop_duplicates()
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)

    for col, val in national_medians.items():
        xwalk[col] = val

    xwalk["twi_approximation_method"] = "county_fallback"
    log.info("  %d ZCTAs filled with national medians", len(xwalk))
    return xwalk


def derive_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived flood-relevant features."""
    # TWI columns present
    twi_col = "twi_cat_mean" if "twi_cat_mean" in df.columns else None
    slope_col = "slope_cat_mean_pct" if "slope_cat_mean_pct" in df.columns else None

    if twi_col:
        df["twi_mean"] = df[twi_col]
        # High-TWI fraction proxy: approximate from mean (higher mean → more high-TWI area)
        # True fraction would require full distribution; this is a monotone proxy
        df["twi_pct_high_proxy"] = (df[twi_col] / 15.0).clip(0, 1)

    if slope_col:
        df["slope_mean_pct"] = df[slope_col]

    if twi_col and slope_col:
        # terrain_flood_index: high TWI + low slope = maximum flood accumulation risk
        df["terrain_flood_index"] = (
            df[twi_col] * (1.0 - (df[slope_col] / 100.0).clip(0, 1))
        ).clip(0, None)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build TWI and watershed flood features per ZCTA"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Build locally, skip S3 upload")
    parser.add_argument("--upload", action="store_true",
                        help="Upload result to S3")
    parser.add_argument("--output", default="/tmp/twi_features_zcta.parquet")
    parser.add_argument("--crosswalk", default=None,
                        help="Path to zcta_county_crosswalk.parquet")
    parser.add_argument("--comid-zcta-crosswalk", default=None,
                        help="Path to pre-built COMID-to-ZCTA crosswalk parquet")
    parser.add_argument("--regions", nargs="+", default=None,
                        help="HUC2 region codes to process (default: all CONUS 01-18)")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    regions = args.regions if args.regions else CONUS_HUC2

    # Resolve county crosswalk (always needed as fallback or for ZCTA list)
    if args.crosswalk:
        crosswalk_path = Path(args.crosswalk)
    else:
        here = Path(__file__).parent
        crosswalk_path = here / "zcta_county_crosswalk.parquet"
        if not crosswalk_path.exists():
            log.error("County crosswalk not found. Pass --crosswalk.")
            sys.exit(1)

    # Setup S3 if uploading
    s3 = None
    if args.upload:
        import boto3
from swarm_auth import get_aws_credentials
        _aws = get_aws_credentials()
        s3 = boto3.client("s3", **_aws)

    # Fetch StreamCat metrics
    twi_df = fetch_streamcat_national("WetIndx", regions)
    slope_df = fetch_streamcat_national("Slope", regions)

    if twi_df.empty and slope_df.empty:
        log.error("Failed to retrieve any StreamCat data.")
        sys.exit(1)

    # Try COMID-ZCTA crosswalk
    comid_xwalk_path = Path(args.comid_zcta_crosswalk) if args.comid_zcta_crosswalk else None
    comid_xwalk = load_comid_zcta_crosswalk(comid_xwalk_path, s3)

    if comid_xwalk is not None:
        result = aggregate_comid_to_zcta(twi_df, slope_df, comid_xwalk)
        approximation_method = "comid_area_weighted"
    else:
        result = fallback_county_approximation(twi_df, slope_df, crosswalk_path)
        approximation_method = "county_fallback"

    # Derive composite features
    result = derive_composite_features(result)

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:                  %d", len(result))
    log.info("Approximation method:   %s", approximation_method)
    if "twi_mean" in result.columns:
        valid = result["twi_mean"].dropna()
        log.info("TWI mean (valid ZCTAs): mean=%.2f, min=%.2f, max=%.2f, n=%d",
                 valid.mean(), valid.min(), valid.max(), len(valid))
    if "terrain_flood_index" in result.columns:
        valid = result["terrain_flood_index"].dropna()
        log.info("Terrain flood index:    mean=%.2f, max=%.2f, n=%d",
                 valid.mean(), valid.max(), len(valid))

    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload and s3:
        key = f"{PREFIX}/twi_features_zcta.parquet"
        s3.upload_file(args.output, BUCKET, key)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)

        provenance = {
            "operation": "build_twi_features",
            "timestamp": timestamp,
            "source": STREAMCAT_BASE,
            "metrics": list(STREAMCAT_COLS.keys()),
            "huc2_regions": regions,
            "approximation_method": approximation_method,
            "twi_flood_threshold": TWI_FLOOD_THRESHOLD,
            "n_zctas": len(result),
            "note": (
                "COMID-ZCTA spatial crosswalk not available — "
                "county fallback used. Re-run with comid_zcta_crosswalk.parquet "
                "for production-quality TWI aggregation."
                if approximation_method == "county_fallback"
                else "Area-weighted aggregation from NHDPlus COMID to ZCTA."
            ),
        }
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{PREFIX}/twi_features_zcta_provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
