#!/usr/bin/env python3
"""
build_hawaii_reference.py -- Extract Hawaii ZCTA reference data.

Filters national geocertdb2026 tables to Hawaii ZCTAs (968xx) and computes
centroids from zcta5_boundaries geometry. Uploads to a separate S3 prefix
to keep Hawaii projections isolated from CONUS data.

Outputs (s3://swarm-floodrsct-data/raw/hawaii/):
  hawaii_zcta_centroids.parquet   -- zcta_id, latitude, longitude, land_area_m2
  hawaii_nfip_claims.parquet      -- NFIP claims filtered to Hawaii ZCTAs
  hawaii_zcta_adjacency.parquet   -- adjacency edges filtered to Hawaii ZCTAs
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
GEOCERT_PREFIX = "raw/geocertdb2026"
OUTPUT_PREFIX = "raw/hawaii"
REGION = "us-east-1"

OUTPUT_DIR = Path(os.environ.get("LOCAL_OUTPUT_DIR", "outputs/hawaii_reference"))


def s3_client():
    return boto3.client("s3", region_name=REGION)


def s3_load_parquet(s3, key: str) -> pd.DataFrame:
    logger.info("Loading s3://%s/%s", BUCKET, key)
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def is_hawaii_zcta(zcta: str) -> bool:
    return str(zcta).startswith("968")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = s3_client()

    # -----------------------------------------------------------------
    # 1. Extract Hawaii ZCTA centroids from zcta5_boundaries
    # -----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Building Hawaii ZCTA reference data")
    logger.info("=" * 60)

    boundaries = s3_load_parquet(s3, f"{GEOCERT_PREFIX}/zcta5_boundaries.parquet")
    logger.info("National boundaries: %d ZCTAs", len(boundaries))

    hi_bounds = boundaries[boundaries["zcta_id"].astype(str).apply(is_hawaii_zcta)].copy()
    logger.info("Hawaii ZCTAs: %d", len(hi_bounds))

    if len(hi_bounds) == 0:
        logger.error("No Hawaii ZCTAs found in zcta5_boundaries")
        sys.exit(1)

    # Compute centroids from WKB geometry
    try:
        import geopandas as gpd
        from shapely import wkb

        hi_bounds["geom"] = hi_bounds["geometry"].apply(
            lambda g: wkb.loads(g) if isinstance(g, bytes) else g
        )
        gdf = gpd.GeoDataFrame(hi_bounds, geometry="geom", crs="EPSG:4326")
        centroids = gdf.geometry.centroid
        hi_centroids = pd.DataFrame({
            "zcta_id": hi_bounds["zcta_id"].values,
            "latitude": centroids.y.values,
            "longitude": centroids.x.values,
            "land_area_m2": hi_bounds["land_area_m2"].values,
        })
    except ImportError:
        # Fallback: use bounding box centroid from WKB
        logger.warning("geopandas not available; using WKB bounding box centroid fallback")
        from shapely import wkb
        lats, lons = [], []
        for geom_bytes in hi_bounds["geometry"].values:
            geom = wkb.loads(geom_bytes) if isinstance(geom_bytes, bytes) else geom_bytes
            c = geom.centroid
            lats.append(c.y)
            lons.append(c.x)
        hi_centroids = pd.DataFrame({
            "zcta_id": hi_bounds["zcta_id"].values,
            "latitude": lats,
            "longitude": lons,
            "land_area_m2": hi_bounds["land_area_m2"].values,
        })

    logger.info("Centroids computed:")
    logger.info("  Lat range: %.4f to %.4f", hi_centroids["latitude"].min(), hi_centroids["latitude"].max())
    logger.info("  Lon range: %.4f to %.4f", hi_centroids["longitude"].min(), hi_centroids["longitude"].max())

    centroid_path = OUTPUT_DIR / "hawaii_zcta_centroids.parquet"
    hi_centroids.to_parquet(centroid_path, index=False)
    logger.info("Saved: %s (%d rows)", centroid_path, len(hi_centroids))

    # -----------------------------------------------------------------
    # 2. Filter NFIP claims to Hawaii
    # -----------------------------------------------------------------
    nfip = s3_load_parquet(s3, f"{GEOCERT_PREFIX}/nfip_claims_zcta.parquet")
    hi_nfip = nfip[nfip["zcta_id"].astype(str).apply(is_hawaii_zcta)].copy()
    logger.info("Hawaii NFIP claims: %d ZCTAs (%d with claims)",
                len(hi_nfip), hi_nfip["nfip_has_claims"].sum())

    nfip_path = OUTPUT_DIR / "hawaii_nfip_claims.parquet"
    hi_nfip.to_parquet(nfip_path, index=False)
    logger.info("Saved: %s", nfip_path)

    # -----------------------------------------------------------------
    # 3. Filter adjacency to Hawaii
    # -----------------------------------------------------------------
    adj = s3_load_parquet(s3, f"{GEOCERT_PREFIX}/zcta_adjacency.parquet")
    hi_adj = adj[
        adj["zcta_id_1"].astype(str).apply(is_hawaii_zcta) &
        adj["zcta_id_2"].astype(str).apply(is_hawaii_zcta)
    ].copy()
    logger.info("Hawaii adjacency: %d edges", len(hi_adj))

    adj_path = OUTPUT_DIR / "hawaii_zcta_adjacency.parquet"
    hi_adj.to_parquet(adj_path, index=False)
    logger.info("Saved: %s", adj_path)

    # -----------------------------------------------------------------
    # 4. Upload to S3
    # -----------------------------------------------------------------
    for fpath in [centroid_path, nfip_path, adj_path]:
        s3_key = f"{OUTPUT_PREFIX}/{fpath.name}"
        logger.info("Uploading %s -> s3://%s/%s", fpath.name, BUCKET, s3_key)
        s3.upload_file(str(fpath), BUCKET, s3_key)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Hawaii reference data complete")
    logger.info("  ZCTAs:     %d", len(hi_centroids))
    logger.info("  NFIP rows: %d (%d with claims)", len(hi_nfip), hi_nfip["nfip_has_claims"].sum())
    logger.info("  Adj edges: %d", len(hi_adj))
    logger.info("  S3 prefix: s3://%s/%s/", BUCKET, OUTPUT_PREFIX)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
