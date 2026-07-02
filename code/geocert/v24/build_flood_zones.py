#!/usr/bin/env python3
"""
build_flood_zones.py -- Compute FEMA flood zone coverage per ZCTA.

Uses the FEMA National Flood Hazard Layer (NFHL) via ArcGIS REST API
(layer 28: Flood Hazard Zones) to compute what percentage of each ZCTA
falls within SFHA (Special Flood Hazard Area) zones.

Strategy: county-level bbox queries with adaptive quadtree splitting for
dense areas, then local spatial overlay in EPSG:5070 for area fractions.

Centroid-only approach was evaluated and rejected: 84% false negative rate
in Houston (2/25 SFHA vs 23/25 with polygon overlay). Flood exposure along
bayous/floodplains covers 1-40% of ZCTAs without touching centroids.

Service: https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer
Layer 28 maxRecordCount=2000 but server 500s on geometry payloads >1000.
Use resultRecordCount=1000 with resultOffset pagination, falling back to
quadtree split if pagination fails.

Output: flood_zones_zcta.parquet
  - zcta_id (str, 5-digit zero-padded)
  - flood_pct_zone_a (float64, 0-100) -- 100-year / 1% annual chance (SFHA)
  - flood_pct_zone_x500 (float64, 0-100) -- 500-year / 0.2% annual chance
  - flood_pct_zone_x (float64, 0-100) -- minimal risk (remainder)
  - flood_sfha (bool) -- True if any Zone A coverage > 0

Usage:
    python build_flood_zones.py --output /tmp/flood_zones_zcta.parquet
    python build_flood_zones.py --upload
    python build_flood_zones.py --max-zctas 200   # test run
"""

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Polygon, box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# FEMA NFHL ArcGIS REST service
# Correct base: /arcgis/ (the old /gis/nfhl/ path returns 404)
NFHL_BASE = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
NFHL_LAYER = 28  # Flood Hazard Zones
NFHL_QUERY_URL = f"{NFHL_BASE}/{NFHL_LAYER}/query"

# Server advertises maxRecordCount=2000 but 500-errors on geometry payloads
# at that size. 1000 is reliable for both fetch and pagination.
PAGE_SIZE = 1000
# Minimum cell size for quadtree splitting (degrees)
MIN_CELL_DEG = 0.05

BUCKET = "swarm-yrsn-datasets"
S3_KEY = "rsct_curriculum/series_018/processed/flood_zones_zcta.parquet"
PROVENANCE_KEY = "rsct_curriculum/series_018/processed/flood_zones_zcta_provenance.json"
XWALK_KEY = "rsct_curriculum/series_018/processed/zcta_county_crosswalk.parquet"


def load_zcta_boundaries(zcta_geo_path: str) -> gpd.GeoDataFrame:
    """Load ZCTA boundaries from shapefile or GeoParquet."""
    path = Path(zcta_geo_path)
    if path.suffix in (".geoparquet", ".parquet"):
        geo = gpd.read_parquet(path)
    elif path.suffix == ".shp":
        geo = gpd.read_file(path, engine="pyogrio")
    else:
        geo = gpd.read_file(path)

    if "ZCTA5CE20" in geo.columns:
        geo = geo.rename(columns={"ZCTA5CE20": "zcta_id"})
    elif "zcta5" in geo.columns:
        geo = geo.rename(columns={"zcta5": "zcta_id"})

    geo["zcta_id"] = geo["zcta_id"].astype(str).str.zfill(5)
    geo = geo[["zcta_id", "geometry"]].copy()
    geo = geo.to_crs("EPSG:4326")
    log.info("Loaded %d ZCTA boundaries", len(geo))
    return geo


def _esri_to_shapely(geometry: dict) -> Polygon:
    """Convert Esri JSON rings to shapely Polygon."""
    rings = geometry.get("rings", [])
    if not rings:
        return None
    try:
        poly = Polygon(rings[0], rings[1:]) if len(rings) > 1 else Polygon(rings[0])
        return poly.buffer(0) if not poly.is_valid else poly
    except Exception:
        return None


def _fetch_page(bbox_str: str, offset: int = 0, max_retries: int = 3):
    """Fetch one page of NFHL features for a bbox. Returns (features, exceeded)."""
    params = {
        "where": "1=1",
        "geometry": bbox_str,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF",
        "returnGeometry": "true",
        "f": "json",
        "resultRecordCount": PAGE_SIZE,
        "resultOffset": offset,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(NFHL_QUERY_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                return None, False  # signal to caller: try splitting
            features = data.get("features", [])
            exceeded = data.get("exceededTransferLimit", False)
            return features, exceeded
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None, False


def fetch_nfhl_for_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> list:
    """Fetch all NFHL features for a bbox with pagination + adaptive splitting.

    First tries pagination (resultOffset). If any page returns an error
    (server 500 on dense geometry), falls back to quadtree splitting.
    """
    bbox_str = f"{xmin},{ymin},{xmax},{ymax}"

    # Try paginated fetch
    all_features = []
    offset = 0
    while True:
        features, exceeded = _fetch_page(bbox_str, offset)

        if features is None:
            # Server error -- fall back to quadtree split
            if (xmax - xmin) > MIN_CELL_DEG and (ymax - ymin) > MIN_CELL_DEG:
                return _fetch_quadtree(xmin, ymin, xmax, ymax)
            return all_features  # can't split further, return what we have

        if not features:
            break

        all_features.extend(features)

        if len(features) < PAGE_SIZE and not exceeded:
            break
        offset += len(features)
        time.sleep(0.15)

    return all_features


def _fetch_quadtree(
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> list:
    """Split bbox into 4 quadrants and fetch each recursively."""
    mx = (xmin + xmax) / 2
    my = (ymin + ymax) / 2
    all_features = []
    for bx in [(xmin, mx), (mx, xmax)]:
        for by in [(ymin, my), (my, ymax)]:
            feats = fetch_nfhl_for_bbox(bx[0], by[0], bx[1], by[1])
            all_features.extend(feats)
            time.sleep(0.1)
    return all_features


def classify_zone(fld_zone: str, zone_subty: str) -> str:
    """Classify a flood zone into A (100yr), X500 (500yr), or X (minimal)."""
    if pd.isna(fld_zone):
        return "X"
    zone = str(fld_zone).strip().upper()
    subty = str(zone_subty).strip().upper() if pd.notna(zone_subty) else ""

    if zone in ("A", "AE", "AH", "AO", "AR", "A99", "VE", "V"):
        return "A"
    if "FLOODWAY" in zone:
        return "A"

    if zone == "X" and ("500" in subty or "0.2" in subty or "SHADED" in subty):
        return "X500"
    if "0.2 PCT" in zone:
        return "X500"

    return "X"


def load_county_crosswalk() -> pd.DataFrame:
    """Load ZCTA-county crosswalk from local file or S3."""
    local = Path(__file__).parent / "zcta_county_crosswalk.parquet"
    if not local.exists():
        local = Path(tempfile.gettempdir()) / "zcta_county_crosswalk.parquet"
    if not local.exists():
        log.info("Downloading crosswalk from S3...")
        import boto3
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.download_file(BUCKET, XWALK_KEY, str(local))
    xwalk = pd.read_parquet(local)
    xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
    return xwalk


def compute_flood_coverage(
    zcta_boundaries: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Compute flood zone area fractions per ZCTA via county-batched NFHL queries.

    Groups ZCTAs by county, queries NFHL for each county bbox with pagination
    and adaptive splitting, then does local spatial overlay in EPSG:5070.
    """
    # Load crosswalk for county grouping
    xwalk = load_county_crosswalk()
    zcta_counties = dict(zip(xwalk["zcta_id"], xwalk["county_fips"]))

    zcta_boundaries = zcta_boundaries.copy()
    zcta_boundaries["county_fips"] = zcta_boundaries["zcta_id"].map(zcta_counties)
    n_missing = zcta_boundaries["county_fips"].isna().sum()
    if n_missing > 0:
        log.warning("  %d ZCTAs have no county -- grouped as 'unknown'", n_missing)
        zcta_boundaries["county_fips"] = zcta_boundaries["county_fips"].fillna("unknown")

    # Project for area calculations
    zcta_proj = zcta_boundaries.to_crs("EPSG:5070")
    zcta_areas = zcta_proj.geometry.area.values

    # Accumulators indexed by position
    n = len(zcta_boundaries)
    zone_a_area = np.zeros(n, dtype=np.float64)
    zone_x500_area = np.zeros(n, dtype=np.float64)

    # Group by county
    counties = zcta_boundaries.groupby("county_fips").groups
    log.info("  %d counties, %d ZCTAs", len(counties), n)

    api_calls = [0]  # mutable for inner counting
    counties_with_flood = 0
    total_features = 0

    for ci, (county_fips, idx_list) in enumerate(sorted(counties.items())):
        if ci % 100 == 0:
            log.info("  County %d/%d (%s) -- %d features so far, %d counties with flood",
                     ci, len(counties), county_fips, total_features, counties_with_flood)

        # Bbox for this county's ZCTAs
        bounds = zcta_boundaries.iloc[idx_list].total_bounds
        pad = 0.01
        raw_features = fetch_nfhl_for_bbox(
            bounds[0] - pad, bounds[1] - pad,
            bounds[2] + pad, bounds[3] + pad,
        )

        if not raw_features:
            time.sleep(0.05)
            continue

        counties_with_flood += 1
        total_features += len(raw_features)

        # Convert Esri JSON to shapely + classify
        zone_a_polys = []
        zone_x500_polys = []
        for f in raw_features:
            zclass = classify_zone(
                f["attributes"].get("FLD_ZONE", ""),
                f["attributes"].get("ZONE_SUBTY", ""),
            )
            if zclass not in ("A", "X500"):
                continue
            poly = _esri_to_shapely(f["geometry"])
            if poly is None:
                continue
            if zclass == "A":
                zone_a_polys.append(poly)
            else:
                zone_x500_polys.append(poly)

        # Overlay per zone class
        for polys, accum in [(zone_a_polys, zone_a_area), (zone_x500_polys, zone_x500_area)]:
            if not polys:
                continue
            try:
                zone_gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_crs("EPSG:5070")
                zone_union = zone_gdf.geometry.union_all()
                for pos_idx in idx_list:
                    zcta_geom = zcta_proj.iloc[pos_idx].geometry
                    intersection = zcta_geom.intersection(zone_union)
                    if not intersection.is_empty:
                        accum[pos_idx] += intersection.area
            except Exception as e:
                log.warning("  Overlay error county %s: %s", county_fips, e)

        time.sleep(0.05)

    log.info("  Done: %d features from %d counties with flood data",
             total_features, counties_with_flood)

    # Convert areas to percentages
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_a = np.where(zcta_areas > 0,
                         np.minimum(zone_a_area / zcta_areas * 100, 100.0), 0.0)
        pct_x500 = np.where(zcta_areas > 0,
                            np.minimum(zone_x500_area / zcta_areas * 100, 100.0), 0.0)
    pct_x = np.maximum(100.0 - pct_a - pct_x500, 0.0)

    return pd.DataFrame({
        "zcta_id": zcta_boundaries["zcta_id"].values,
        "flood_pct_zone_a": np.round(pct_a, 2),
        "flood_pct_zone_x500": np.round(pct_x500, 2),
        "flood_pct_zone_x": np.round(pct_x, 2),
        "flood_sfha": pct_a > 0,
    })


def main():
    parser = argparse.ArgumentParser(description="Build FEMA flood zone coverage")
    parser.add_argument("--output", default="/tmp/flood_zones_zcta.parquet")
    parser.add_argument("--zcta-boundaries", default="/tmp/tiger_zcta",
                        help="Path to ZCTA boundaries (shapefile dir or geoparquet)")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--max-zctas", type=int, default=None,
                        help="Limit number of ZCTAs for testing")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    # Load ZCTA boundaries
    zcta_path = Path(args.zcta_boundaries)
    if zcta_path.is_dir():
        shp_files = list(zcta_path.glob("*.shp"))
        if not shp_files:
            log.error("No .shp file in %s", zcta_path)
            sys.exit(1)
        zcta_geo = load_zcta_boundaries(str(shp_files[0]))
    else:
        zcta_geo = load_zcta_boundaries(str(zcta_path))

    if args.max_zctas:
        zcta_geo = zcta_geo.head(args.max_zctas)
        log.info("Limited to %d ZCTAs for testing", len(zcta_geo))

    # Compute flood coverage
    result = compute_flood_coverage(zcta_geo)

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs:         %d", len(result))
    log.info("Zone A >0%%:    %d (%.1f%%)",
             (result["flood_pct_zone_a"] > 0).sum(),
             (result["flood_pct_zone_a"] > 0).mean() * 100)
    log.info("Zone X500 >0%%: %d (%.1f%%)",
             (result["flood_pct_zone_x500"] > 0).sum(),
             (result["flood_pct_zone_x500"] > 0).mean() * 100)
    log.info("Mean Zone A:   %.2f%%", result["flood_pct_zone_a"].mean())
    log.info("Mean Zone X500: %.2f%%", result["flood_pct_zone_x500"].mean())
    log.info("SFHA count:    %d", result["flood_sfha"].sum())

    # Save
    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.upload_file(args.output, BUCKET, S3_KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, S3_KEY)

        provenance = {
            "operation": "build_flood_zones",
            "timestamp": timestamp,
            "source": f"{NFHL_BASE}/{NFHL_LAYER}",
            "method": "county_batched_polygon_overlay",
            "page_size": PAGE_SIZE,
            "n_zctas": len(result),
            "zone_a_count": int((result["flood_pct_zone_a"] > 0).sum()),
            "zone_x500_count": int((result["flood_pct_zone_x500"] > 0).sum()),
            "sfha_count": int(result["flood_sfha"].sum()),
            "mean_zone_a_pct": round(float(result["flood_pct_zone_a"].mean()), 2),
        }
        s3.put_object(
            Bucket=BUCKET, Key=PROVENANCE_KEY,
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
