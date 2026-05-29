#!/usr/bin/env python3
"""
fetch_nyc_sewersheds.py -- SageMaker job: download NYC DEP sewer-shed polygons.

Attempts to download NYC sewer drainage area boundaries from:
  1. NYC Open Data Socrata (dataset nbry-ynxa if available)
  2. NYC DEP GIS portal (direct shapefile download)

These polygons are the spatial unit for the NYC FloodRSCT experiments —
each sewer-shed is one observation row in the (sewershed, event) table.

Output:
  s3://swarm-floodrsct-data/raw/nyc_sewersheds/nyc_sewersheds.gpkg
  s3://swarm-floodrsct-data/raw/nyc_sewersheds/nyc_sewersheds.parquet  (centroid + attributes, no geometry)
"""

import logging
import sys
import tempfile
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
SOCRATA_DOMAIN = "data.cityofnewyork.us"

# NYC DEP sewer-shed dataset candidates (try in order)
SOURCES = [
    {
        "name": "NYC Open Data Socrata (nbry-ynxa)",
        "url": f"https://{SOCRATA_DOMAIN}/resource/nbry-ynxa.json",
        "type": "socrata_json",
    },
    {
        "name": "NYC DEP GeoJSON export (arcgis)",
        "url": (
            "https://maps.nyc.gov/arcgis/rest/services/Environment/DEP_Sewershed/MapServer/0/query"
            "?where=1%3D1&outFields=*&f=geojson&returnGeometry=true"
        ),
        "type": "geojson",
    },
]


def try_socrata(url: str) -> pd.DataFrame | None:
    """Attempt Socrata JSON download; return DataFrame or None."""
    params = {"$limit": 5000, "$offset": 0}
    all_records = []
    while True:
        resp = requests.get(url, params=params, timeout=120)
        if resp.status_code != 200:
            return None
        records = resp.json()
        if not records:
            break
        all_records.extend(records)
        if len(records) < 5000:
            break
        params["$offset"] += 5000
    return pd.DataFrame(all_records) if all_records else None


def try_geojson(url: str) -> tuple[object | None, pd.DataFrame | None]:
    """Attempt GeoJSON download; return (GeoDataFrame, attribute-only DataFrame) or (None, None)."""
    try:
        import geopandas as gpd
        resp = requests.get(url, timeout=300)
        if resp.status_code != 200:
            return None, None
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            f.write(resp.content)
            tmp_path = f.name
        gdf = gpd.read_file(tmp_path)
        log.info("GeoJSON source: %d features, columns: %s", len(gdf), list(gdf.columns[:10]))
        # Attribute-only copy for parquet (geometry-free)
        attr_df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        if "geometry" in gdf.columns:
            centroids = gdf.geometry.centroid
            attr_df["centroid_lon"] = centroids.x
            attr_df["centroid_lat"] = centroids.y
        return gdf, attr_df
    except ImportError:
        log.warning("geopandas not available; skipping GeoJSON path")
        return None, None
    except Exception as e:
        log.warning("GeoJSON download failed: %s", e)
        return None, None


def upload_file(local_path: str, s3_key: str) -> None:
    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)
    s3.upload_file(local_path, BUCKET, s3_key)
    log.info("Uploaded to s3://%s/%s", BUCKET, s3_key)


def main() -> None:
    gdf = None
    attr_df = None

    for source in SOURCES:
        log.info("Trying source: %s", source["name"])
        if source["type"] == "socrata_json":
            df = try_socrata(source["url"])
            if df is not None and not df.empty:
                log.info("Socrata source succeeded: %d records", len(df))
                attr_df = df
                break
        elif source["type"] == "geojson":
            gdf, attr_df = try_geojson(source["url"])
            if attr_df is not None and not attr_df.empty:
                log.info("GeoJSON source succeeded: %d features", len(attr_df))
                break

    if attr_df is None or attr_df.empty:
        log.error(
            "All sewer-shed sources failed. Manual download required from "
            "https://data.cityofnewyork.us or NYC DEP GIS portal."
        )
        sys.exit(1)

    # Upload GeoPackage if geopandas available
    if gdf is not None:
        gpkg_path = "/tmp/nyc_sewersheds.gpkg"
        try:
            gdf.to_file(gpkg_path, driver="GPKG")
            upload_file(gpkg_path, "raw/nyc_sewersheds/nyc_sewersheds.gpkg")
        except Exception as e:
            log.warning("GeoPackage export failed: %s", e)

    # Always upload attribute parquet
    parquet_path = "/tmp/nyc_sewersheds.parquet"
    attr_df.to_parquet(parquet_path, index=False)
    upload_file(parquet_path, "raw/nyc_sewersheds/nyc_sewersheds.parquet")

    log.info("fetch_nyc_sewersheds complete: %d sewer-sheds", len(attr_df))


if __name__ == "__main__":
    main()
