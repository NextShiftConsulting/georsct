#!/usr/bin/env python3
"""
build_r1_features.py -- SageMaker job: compute R1 supplemental features.

Reads existing assembled parquet + raw spatial data sources. Produces a
supplement parquet with columns that train_r1_hydrology.py joins onto R0.

R1 adds hydrologic and infrastructure features not in the R0 baseline:
  - nhd_catchment_area_km2: NHDPlus V2 catchment area at ZCTA centroid
  - levee_nearest_km, levee_condition_rating: USACE NLD proximity (NOLA, NYC)
  - sewershed_type: NYC DEP sewershed classification (NYC only)

All features use centroid-based spatial joins. No polygon overlay.

Usage:
    python build_r1_features.py --scenario houston --upload
    python build_r1_features.py --scenario southwest_florida --upload
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client, load_processed_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# Scenario -> NHDPlus VPU mapping
SCENARIO_VPU = {
    "houston": "12",
    "riverside_coachella": "18",
    "southwest_florida": "03S",
    "nyc": "02",
    "new_orleans": "08",
}

# Scenarios with USACE levee data
LEVEE_SCENARIOS = {
    "new_orleans": "raw/usace_levees/new_orleans_levees.parquet",
    "nyc": "raw/usace_levees/nyc_levees.parquet",
}

# NYC sewershed data
SEWERSHED_KEY = "raw/nyc_sewersheds/nyc_sewersheds.gpkg"
SEWERSHED_PARQUET_KEY = "raw/nyc_sewersheds/nyc_sewersheds.parquet"

RESULTS_PREFIX = "processed"


def s3_read_parquet(s3, key: str) -> pd.DataFrame:
    """Read a parquet file from S3. Returns empty DataFrame on failure."""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception as exc:
        log.warning("Cannot read s3://%s/%s: %s", BUCKET, key, exc)
        return pd.DataFrame()


def load_zcta_centroids(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Load ZCTA centroids from geocertdb2026."""
    static = s3_read_parquet(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if static.empty:
        return pd.DataFrame(columns=["zcta_id", "lat", "lon"])

    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns
                    if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        log.error("Cannot find zcta/lat/lon columns in geocertdb2026")
        return pd.DataFrame(columns=["zcta_id", "lat", "lon"])

    df = (static[[zcta_col, lat_col, lon_col]]
          .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
          .query("zcta_id in @zcta_ids")
          .dropna(subset=["lat", "lon"]))
    log.info("Loaded %d ZCTA centroids", len(df))
    return df


def build_nhd_catchment(s3, centroids: pd.DataFrame, vpu: str) -> pd.DataFrame:
    """Spatial join: ZCTA centroids -> NHDPlus catchments -> catchment area."""
    nhd_key = f"raw/nhdplus/catchments/v2/catchments_vpu{vpu}.parquet"
    nhd_df = s3_read_parquet(s3, nhd_key)
    if nhd_df.empty:
        log.warning("No NHDPlus data for VPU %s", vpu)
        return pd.DataFrame({
            "zcta_id": centroids["zcta_id"].values,
            "nhd_catchment_area_km2": np.nan,
        })

    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely import wkb
    except ImportError:
        log.error("geopandas/shapely required for spatial join")
        return pd.DataFrame({
            "zcta_id": centroids["zcta_id"].values,
            "nhd_catchment_area_km2": np.nan,
        })

    # Build centroid GeoDataFrame
    geom = [Point(r["lon"], r["lat"]) for _, r in centroids.iterrows()]
    centroid_gdf = gpd.GeoDataFrame(centroids, geometry=geom, crs="EPSG:4326")

    # Deserialize WKB geometry if needed
    if nhd_df["geometry"].dtype == object and len(nhd_df) > 0:
        sample = nhd_df["geometry"].iloc[0]
        if isinstance(sample, bytes):
            nhd_df = nhd_df.copy()
            nhd_df["geometry"] = nhd_df["geometry"].apply(wkb.loads)
    nhd_gdf = gpd.GeoDataFrame(nhd_df, geometry="geometry", crs="EPSG:4326")

    joined = gpd.sjoin(centroid_gdf, nhd_gdf[["comid", "area_sq_km", "geometry"]],
                       how="left", predicate="within")

    out = joined[["zcta_id", "area_sq_km"]].rename(
        columns={"area_sq_km": "nhd_catchment_area_km2"}
    )
    # Deduplicate (centroid may fall in overlapping catchments)
    out = out.groupby("zcta_id", as_index=False).first()

    matched_pct = out["nhd_catchment_area_km2"].notna().mean() * 100
    log.info("NHDPlus VPU %s: %d ZCTAs, %.1f%% matched", vpu, len(out), matched_pct)
    return out


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def build_levee_features(s3, centroids: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Nearest-levee distance + condition rating for NOLA and NYC."""
    if scenario not in LEVEE_SCENARIOS:
        return pd.DataFrame({
            "zcta_id": centroids["zcta_id"].values,
            "levee_nearest_km": np.nan,
            "levee_condition_rating": np.nan,
        })

    levee_df = s3_read_parquet(s3, LEVEE_SCENARIOS[scenario])
    if levee_df.empty:
        log.warning("No levee data for %s", scenario)
        return pd.DataFrame({
            "zcta_id": centroids["zcta_id"].values,
            "levee_nearest_km": np.nan,
            "levee_condition_rating": np.nan,
        })

    # Levee data has lat/lon or geometry. Try centroid_lat/centroid_lon first.
    # USACE NLD records are levee SYSTEMS, not segments. Use centroid of leveed area.
    # For now, if no explicit lat/lon, try to derive from geometry.
    lat_col = next((c for c in levee_df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in levee_df.columns
                    if "lon" in c.lower() or "lng" in c.lower()), None)

    if not lat_col or not lon_col:
        # Try to get centroid from geometry if it exists
        if "geometry" in levee_df.columns:
            try:
                import geopandas as gpd
                from shapely import wkb
                gdf = levee_df.copy()
                sample = gdf["geometry"].iloc[0]
                if isinstance(sample, bytes):
                    gdf["geometry"] = gdf["geometry"].apply(wkb.loads)
                gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
                levee_df["_levee_lat"] = gdf.geometry.centroid.y
                levee_df["_levee_lon"] = gdf.geometry.centroid.x
                lat_col, lon_col = "_levee_lat", "_levee_lon"
            except Exception as exc:
                log.warning("Cannot extract levee geometry: %s", exc)
                return pd.DataFrame({
                    "zcta_id": centroids["zcta_id"].values,
                    "levee_nearest_km": np.nan,
                    "levee_condition_rating": np.nan,
                })
        else:
            log.warning("No lat/lon or geometry in levee data")
            return pd.DataFrame({
                "zcta_id": centroids["zcta_id"].values,
                "levee_nearest_km": np.nan,
                "levee_condition_rating": np.nan,
            })

    levee_lats = levee_df[lat_col].values.astype(float)
    levee_lons = levee_df[lon_col].values.astype(float)

    # Parse condition_rating to numeric
    cond_col = "condition_rating"
    if cond_col not in levee_df.columns:
        cond_col = next((c for c in levee_df.columns if "condition" in c.lower()), None)
    cond_vals = levee_df[cond_col].values if cond_col else np.full(len(levee_df), np.nan)

    results = []
    for _, row in centroids.iterrows():
        dists = haversine_km(row["lat"], row["lon"], levee_lats, levee_lons)
        idx = np.nanargmin(dists)
        results.append({
            "zcta_id": row["zcta_id"],
            "levee_nearest_km": float(dists[idx]),
            "levee_condition_rating": cond_vals[idx] if cond_col else np.nan,
        })

    out = pd.DataFrame(results)
    log.info("Levee features (%s): %d ZCTAs, median dist %.1f km",
             scenario, len(out), out["levee_nearest_km"].median())
    return out


def build_sewershed_features(s3, centroids: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """NYC sewershed assignment via centroid-in-polygon."""
    if scenario != "nyc":
        return pd.DataFrame({
            "zcta_id": centroids["zcta_id"].values,
            "sewershed_name": None,
        })

    # Try gpkg first (has geometry), fall back to parquet (centroids only)
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        # Download gpkg to /tmp
        local_gpkg = "/tmp/nyc_sewersheds.gpkg"
        s3.download_file(BUCKET, SEWERSHED_KEY, local_gpkg)
        sewer_gdf = gpd.read_file(local_gpkg).to_crs("EPSG:4326")

        geom = [Point(r["lon"], r["lat"]) for _, r in centroids.iterrows()]
        centroid_gdf = gpd.GeoDataFrame(centroids, geometry=geom, crs="EPSG:4326")

        name_col = next((c for c in sewer_gdf.columns
                         if "sewershed" in c.lower() or "wwtp" in c.lower()), None)
        if not name_col:
            name_col = sewer_gdf.columns[0]

        joined = gpd.sjoin(centroid_gdf, sewer_gdf[[name_col, "geometry"]],
                           how="left", predicate="within")
        out = joined[["zcta_id", name_col]].rename(columns={name_col: "sewershed_name"})
        out = out.groupby("zcta_id", as_index=False).first()
        log.info("Sewershed assignment: %d ZCTAs, %.1f%% matched",
                 len(out), out["sewershed_name"].notna().mean() * 100)
        return out

    except Exception as exc:
        log.warning("Sewershed spatial join failed: %s — using centroid fallback", exc)
        # Nearest-centroid assignment from the parquet (no geometry)
        sewer_df = s3_read_parquet(s3, SEWERSHED_PARQUET_KEY)
        if sewer_df.empty or "centroid_lat" not in sewer_df.columns:
            return pd.DataFrame({
                "zcta_id": centroids["zcta_id"].values,
                "sewershed_name": None,
            })

        name_col = next((c for c in sewer_df.columns
                         if "sewershed" in c.lower() or "wwtp" in c.lower()), "Sewershed")
        sewer_lats = sewer_df["centroid_lat"].values
        sewer_lons = sewer_df["centroid_lon"].values
        sewer_names = sewer_df[name_col].values

        results = []
        for _, row in centroids.iterrows():
            dists = haversine_km(row["lat"], row["lon"], sewer_lats, sewer_lons)
            idx = np.nanargmin(dists)
            results.append({
                "zcta_id": row["zcta_id"],
                "sewershed_name": str(sewer_names[idx]),
            })
        return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True,
                        choices=list(SCENARIO_VPU.keys()))
    parser.add_argument("--upload", action="store_true",
                        help="Upload results to S3")
    args = parser.parse_args()

    scenario = args.scenario
    vpu = SCENARIO_VPU[scenario]
    log.info("build_r1_features: scenario=%s, VPU=%s", scenario, vpu)

    s3 = get_s3_client()

    # Load existing parquet for ZCTA IDs
    df = load_processed_parquet(s3, scenario)
    zcta_ids = df["zcta_id"].unique().tolist()
    log.info("Scenario %s: %d unique ZCTAs", scenario, len(zcta_ids))

    # Load centroids
    centroids = load_zcta_centroids(s3, zcta_ids)
    if centroids.empty:
        log.error("No centroids loaded -- cannot compute R1 features")
        sys.exit(1)

    # Build R1 features
    nhd = build_nhd_catchment(s3, centroids, vpu)
    levee = build_levee_features(s3, centroids, scenario)
    sewer = build_sewershed_features(s3, centroids, scenario)

    # Merge all on zcta_id
    r1 = nhd.merge(levee, on="zcta_id", how="outer")
    r1 = r1.merge(sewer, on="zcta_id", how="outer")

    # Summary
    log.info("R1 supplement: %d rows, %d columns", len(r1), len(r1.columns))
    for col in r1.columns:
        if col == "zcta_id":
            continue
        pct = r1[col].notna().mean() * 100
        log.info("  %s: %.1f%% non-null", col, pct)

    if args.upload:
        # Upload to S3
        buf = io.BytesIO()
        r1.to_parquet(buf, index=False)
        buf.seek(0)
        key = f"{RESULTS_PREFIX}/{scenario}/{scenario}_r1_supplement.parquet"
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        log.info("Uploaded s3://%s/%s", BUCKET, key)

        # Write metadata
        meta = {
            "scenario": scenario,
            "vpu": vpu,
            "n_zctas": len(r1),
            "columns": list(r1.columns),
            "coverage": {
                col: float(r1[col].notna().mean())
                for col in r1.columns if col != "zcta_id"
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        meta_key = f"{RESULTS_PREFIX}/{scenario}/{scenario}_r1_meta.json"
        s3.put_object(Bucket=BUCKET, Key=meta_key,
                      Body=json.dumps(meta, indent=2).encode())
        log.info("Uploaded s3://%s/%s", BUCKET, meta_key)
    else:
        local = f"/tmp/{scenario}_r1_supplement.parquet"
        r1.to_parquet(local, index=False)
        log.info("Saved locally: %s", local)


if __name__ == "__main__":
    main()
