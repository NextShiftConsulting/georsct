#!/usr/bin/env python3
"""
fetch_jrc_deltares.py -- SageMaker job: fetch JRC + Deltares shared features.

Extracts two shared feature layers from Microsoft Planetary Computer STAC API
for all ~794 ZCTA centroids in the geocertdb2026 universe:

  1. JRC Global Surface Water (1984-2020): water occurrence stats
  2. Deltares Global Flood Maps: modeled depth at RP 10/50/100

Both are scenario-independent (keyed by ZCTA, not by event), so they live
in the shared/ prefix and are built once for all scenarios.

Source:
  JRC: https://planetarycomputer.microsoft.com/dataset/jrc-gsw
  Deltares: https://planetarycomputer.microsoft.com/dataset/deltares-floods

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_jrc_water_occurrence_pct.parquet
  s3://swarm-floodrsct-data/processed/shared/zcta_deltares_depth.parquet

Usage:
    python fetch_jrc_deltares.py
    python fetch_jrc_deltares.py --jrc-only
    python fetch_jrc_deltares.py --deltares-only
"""

import argparse
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from swarm_auth import get_aws_credentials

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
CENTROID_KEY = "raw/geocertdb2026/zcta_features_labels.parquet"
JRC_CACHE_KEY = "processed/shared/zcta_jrc_water_occurrence_pct.parquet"
DELTARES_CACHE_KEY = "processed/shared/zcta_deltares_depth.parquet"


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_read(s3, key: str):
    """Read parquet from S3; return None if missing."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        log.warning("S3 key not found: %s", key)
        return None
    except Exception as e:
        log.warning("Could not read %s: %s", key, e)
        return None


def s3_write_parquet(s3, df: pd.DataFrame, key: str) -> None:
    """Write DataFrame as parquet to S3."""
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    log.info("Uploaded %d rows x %d cols to s3://%s/%s",
             len(df), len(df.columns), BUCKET, key)


# ---------------------------------------------------------------------------
# Centroid loading
# ---------------------------------------------------------------------------

def load_centroids(s3) -> pd.DataFrame:
    """Load ZCTA centroids from geocertdb2026."""
    static = s3_read(s3, CENTROID_KEY)
    if static is None:
        raise RuntimeError(f"geocertdb2026 not found at s3://{BUCKET}/{CENTROID_KEY}")

    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next(
        (c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()),
        None,
    )
    if not all([zcta_col, lat_col, lon_col]):
        raise RuntimeError(
            f"Missing zcta/lat/lon in geocertdb2026 cols: {list(static.columns)}"
        )

    centroids = (
        static[[zcta_col, lat_col, lon_col]]
        .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
        .dropna(subset=["lat", "lon"])
    )
    centroids["zcta_id"] = centroids["zcta_id"].astype(str)
    log.info("Loaded %d ZCTA centroids", len(centroids))
    return centroids


# ---------------------------------------------------------------------------
# JRC extraction
# ---------------------------------------------------------------------------

def fetch_jrc(s3, centroids: pd.DataFrame) -> pd.DataFrame:
    """Fetch JRC Global Surface Water occurrence for all centroids."""
    log.info("--- JRC Global Surface Water ---")
    t0 = time.time()

    from floodcaster.stac import jrc_centroid_occurrence
    result = jrc_centroid_occurrence(centroids, id_col="zcta_id")

    elapsed = time.time() - t0
    coverage = result["jrc_occurrence_mean"].notna().mean() * 100
    log.info("JRC extraction: %d ZCTAs, %.1f%% coverage, %.0f sec",
             len(result), coverage, elapsed)

    s3_write_parquet(s3, result, JRC_CACHE_KEY)
    return result


# ---------------------------------------------------------------------------
# Deltares extraction
# ---------------------------------------------------------------------------

def fetch_deltares(s3, centroids: pd.DataFrame) -> pd.DataFrame:
    """Fetch Deltares Global Flood Maps depth for all centroids."""
    log.info("--- Deltares Global Flood Maps ---")
    t0 = time.time()

    from floodcaster.batch import deltares_centroid_depth
    result = deltares_centroid_depth(
        centroids, id_col="zcta_id", return_periods=[10, 50, 100],
    )

    elapsed = time.time() - t0
    coverage = result["deltares_depth_ft_rp100"].notna().mean() * 100
    log.info("Deltares extraction: %d ZCTAs, %.1f%% coverage, %.0f sec",
             len(result), coverage, elapsed)

    s3_write_parquet(s3, result, DELTARES_CACHE_KEY)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch JRC + Deltares shared features")
    parser.add_argument("--jrc-only", action="store_true", help="Fetch JRC only")
    parser.add_argument("--deltares-only", action="store_true", help="Fetch Deltares only")
    args = parser.parse_args()

    do_jrc = not args.deltares_only
    do_deltares = not args.jrc_only

    log.info("fetch_jrc_deltares.py starting (jrc=%s, deltares=%s)", do_jrc, do_deltares)

    aws = get_aws_credentials()
    s3 = boto3.client("s3", **aws)

    centroids = load_centroids(s3)

    if do_jrc:
        jrc_df = fetch_jrc(s3, centroids)
        log.info("JRC columns: %s", list(jrc_df.columns))

    if do_deltares:
        deltares_df = fetch_deltares(s3, centroids)
        log.info("Deltares columns: %s", list(deltares_df.columns))

    log.info("fetch_jrc_deltares.py complete")


if __name__ == "__main__":
    main()
