#!/usr/bin/env python3
"""
stage_zcta_geometry.py -- Stage ZCTA polygon boundaries + adjacency to S3.

Downloads Census TIGER/Line ZCTA5 (2020 vintage) national shapefile,
converts to GeoParquet, builds Queen's contiguity adjacency edge list,
and uploads both to S3.

Outputs:
  s3://swarm-floodrsct-data/raw/geocertdb2026/zcta5_boundaries.parquet
  s3://swarm-floodrsct-data/raw/geocertdb2026/zcta_adjacency.parquet

Usage:
    python stage_zcta_geometry.py --upload
    python stage_zcta_geometry.py --dry-run
"""

import argparse
import hashlib
import io
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# Census TIGER/Line ZCTA5 2020 vintage (used for 2020-era analysis)
TIGER_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/"
    "tl_2023_us_zcta520.zip"
)

BOUNDARIES_KEY = "raw/geocertdb2026/zcta5_boundaries.parquet"
ADJACENCY_KEY = "raw/geocertdb2026/zcta_adjacency.parquet"


def download_tiger_zcta(tmp_dir: str) -> gpd.GeoDataFrame:
    """Download and read the national ZCTA5 shapefile."""
    zip_path = Path(tmp_dir) / "zcta5.zip"
    log.info("Downloading ZCTA5 shapefile from Census TIGER (%s)", TIGER_URL)

    resp = requests.get(TIGER_URL, stream=True, timeout=600)
    resp.raise_for_status()
    total = 0
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)
    log.info("Downloaded %.1f MB", total / 1e6)

    # Extract and read
    extract_dir = Path(tmp_dir) / "zcta5"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    shp_files = list(extract_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No .shp in TIGER zip")

    gdf = gpd.read_file(shp_files[0])
    log.info("Read %d ZCTA5 polygons, CRS=%s", len(gdf), gdf.crs)

    # Standardize column names
    zcta_col = next((c for c in gdf.columns if "ZCTA5" in c), None)
    if zcta_col and zcta_col != "zcta_id":
        gdf = gdf.rename(columns={zcta_col: "zcta_id"})

    # Keep only id + geometry (boundaries file stays lean)
    keep = ["zcta_id", "geometry"]
    extra = [c for c in ["ALAND20", "AWATER20"] if c in gdf.columns]
    if extra:
        gdf = gdf.rename(columns={"ALAND20": "land_area_m2", "AWATER20": "water_area_m2"})
        keep.extend(["land_area_m2", "water_area_m2"])

    gdf = gdf[[c for c in keep if c in gdf.columns]]
    gdf = gdf.to_crs("EPSG:4326")
    log.info("Standardized: %d polygons in EPSG:4326", len(gdf))
    return gdf


def build_adjacency(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Build Queen's contiguity adjacency edge list via spatial touches.

    Two ZCTAs are queen-contiguous if their geometries touch or share
    any boundary point. Uses STRtree spatial index for performance.
    """
    log.info("Building Queen's contiguity adjacency for %d polygons...", len(gdf))

    # Use projected CRS for reliable spatial predicates
    proj = gdf.to_crs("EPSG:5070")  # Albers Equal Area for CONUS
    sindex = proj.sindex

    edges = []
    for idx, row in proj.iterrows():
        candidates = list(sindex.intersection(row.geometry.bounds))
        for cand_idx in candidates:
            if cand_idx <= idx:
                continue  # avoid duplicates + self
            if row.geometry.touches(proj.geometry.iloc[cand_idx]) or \
               row.geometry.intersects(proj.geometry.iloc[cand_idx]):
                edges.append({
                    "zcta_id_1": gdf.iloc[idx]["zcta_id"],
                    "zcta_id_2": gdf.iloc[cand_idx]["zcta_id"],
                })

    adj_df = pd.DataFrame(edges)
    n_zctas_with_neighbors = len(
        set(adj_df["zcta_id_1"].tolist() + adj_df["zcta_id_2"].tolist())
    )
    mean_degree = len(adj_df) * 2 / max(len(gdf), 1)
    log.info(
        "Adjacency: %d edges, %d ZCTAs with neighbors, mean degree %.1f",
        len(adj_df), n_zctas_with_neighbors, mean_degree,
    )
    return adj_df


def upload_parquet(df, s3, key: str, is_geo: bool = False) -> str:
    """Write parquet to /tmp and upload to S3. Returns SHA-256."""
    local = f"/tmp/{Path(key).name}"
    if is_geo and isinstance(df, gpd.GeoDataFrame):
        df.to_parquet(local, index=False)
    else:
        df.to_parquet(local, index=False)
    size = Path(local).stat().st_size

    with open(local, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    s3.upload_file(local, BUCKET, key)
    log.info("Uploaded s3://%s/%s (%.1f MB, sha256=%s)", BUCKET, key, size / 1e6, sha[:16])
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage ZCTA5 boundaries + adjacency to S3"
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-adjacency", action="store_true",
                        help="Skip adjacency build (slow on full national set)")
    args = parser.parse_args()

    s3 = get_s3_client()

    # Check if already staged
    for key in [BOUNDARIES_KEY, ADJACENCY_KEY]:
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
            log.info("Already exists: s3://%s/%s", BUCKET, key)
            if not args.dry_run:
                continue
        except Exception:
            pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        gdf = download_tiger_zcta(tmp_dir)

        if args.dry_run:
            log.info("[DRY RUN] Would upload %d ZCTA boundaries to %s", len(gdf), BOUNDARIES_KEY)
            return

        if args.upload:
            upload_parquet(gdf, s3, BOUNDARIES_KEY, is_geo=True)

            if not args.skip_adjacency:
                adj = build_adjacency(gdf)
                upload_parquet(adj, s3, ADJACENCY_KEY)
            else:
                log.info("Skipping adjacency build (--skip-adjacency)")
        else:
            local_bound = "/tmp/zcta5_boundaries.parquet"
            gdf.to_parquet(local_bound, index=False)
            log.info("Saved locally: %s", local_bound)

            if not args.skip_adjacency:
                adj = build_adjacency(gdf)
                local_adj = "/tmp/zcta_adjacency.parquet"
                adj.to_parquet(local_adj, index=False)
                log.info("Saved locally: %s", local_adj)

    log.info("stage_zcta_geometry complete")


if __name__ == "__main__":
    main()
