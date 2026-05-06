#!/usr/bin/env python3
"""
build_drive_times.py -- Compute drive times from ZCTA centroids to key facilities.

Uses OSRM (Open Source Routing Machine) public API to compute:
  - drive_min_to_nearest_hospital: minutes to nearest hospital
  - drive_min_to_county_seat: minutes to county seat

Requires HIFLD hospital data (from build_hifld_features.py) for hospital locations.

Output: drive_times_zcta.parquet

Usage:
    python build_drive_times.py --output /tmp/drive_times_zcta.parquet
    python build_drive_times.py --upload
"""

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# OSRM public demo API (rate limit: ~1 req/sec)
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"

# County centroids: derived from ZCTA centroids grouped by county_fips
# (population-weighted centroid, since we already have the crosswalk)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def build_county_centroids(zcta_df: pd.DataFrame) -> pd.DataFrame:
    """Build county centroids from ZCTA centroids + crosswalk.

    Uses mean of ZCTA centroids per county_fips as proxy for county seat.
    Requires zcta_df to have county_fips, latitude, longitude columns.
    Falls back to loading the crosswalk parquet if county_fips is missing.
    """
    if "county_fips" not in zcta_df.columns:
        # Look next to the script first, then /tmp
        xwalk_path = Path(__file__).parent / "zcta_county_crosswalk.parquet"
        if not xwalk_path.exists():
            import tempfile
            xwalk_path = Path(tempfile.gettempdir()) / "zcta_county_crosswalk.parquet"
        if not xwalk_path.exists():
            log.error("Need county_fips column or crosswalk at %s", xwalk_path)
            sys.exit(1)
        xwalk = pd.read_parquet(xwalk_path)
        xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
        zcta_df = zcta_df.merge(xwalk[["zcta_id", "county_fips"]], on="zcta_id", how="left")

    county = (
        zcta_df.dropna(subset=["county_fips", "latitude", "longitude"])
        .groupby("county_fips")
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .reset_index()
    )
    log.info("  %d county centroids built from ZCTA means", len(county))
    return county


def osrm_route_duration(
    src_lon: float, src_lat: float,
    dst_lon: float, dst_lat: float,
) -> float:
    """Get driving duration in minutes between two points via OSRM."""
    coords = f"{src_lon},{src_lat};{dst_lon},{dst_lat}"
    url = f"{OSRM_ROUTE_URL}/{coords}"
    params = {"overview": "false", "annotations": "false"}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == "Ok" and data.get("routes"):
            return data["routes"][0]["duration"] / 60.0  # seconds -> minutes
    except (requests.RequestException, KeyError, IndexError):
        pass

    return float("nan")


def osrm_table_durations(
    sources: list,
    destinations: list,
    batch_size: int = 50,
) -> np.ndarray:
    """Compute duration matrix using OSRM table API.

    sources: list of (lon, lat)
    destinations: list of (lon, lat)
    Returns: matrix of durations in minutes (sources x destinations)
    """
    n_src = len(sources)
    n_dst = len(destinations)
    result = np.full((n_src, n_dst), np.nan)

    # OSRM table API accepts up to ~100 coordinates total
    # Process source batches
    for i in range(0, n_src, batch_size):
        src_batch = sources[i:i + batch_size]
        batch_n = len(src_batch)

        # Combine sources + destinations
        all_coords = src_batch + destinations
        coord_str = ";".join(f"{lon},{lat}" for lon, lat in all_coords)

        src_indices = list(range(batch_n))
        dst_indices = list(range(batch_n, batch_n + n_dst))

        url = f"{OSRM_TABLE_URL}/{coord_str}"
        params = {
            "sources": ";".join(str(x) for x in src_indices),
            "destinations": ";".join(str(x) for x in dst_indices),
            "annotations": "duration",
        }

        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") == "Ok":
                durations = data["durations"]
                for j in range(batch_n):
                    for k in range(n_dst):
                        val = durations[j][k]
                        if val is not None:
                            result[i + j, k] = val / 60.0  # seconds -> minutes
        except (requests.RequestException, KeyError, IndexError) as e:
            log.warning("OSRM table batch %d failed: %s", i, e)

        time.sleep(1.0)  # rate limit for public API

    return result


def find_nearest_hospitals(
    zcta_centroids: pd.DataFrame,
    hospitals: pd.DataFrame,
    k: int = 3,
) -> pd.DataFrame:
    """For each ZCTA, find the k nearest hospitals by haversine distance."""
    zcta_lats = zcta_centroids["latitude"].values
    zcta_lons = zcta_centroids["longitude"].values
    hosp_lats = hospitals["latitude"].values
    hosp_lons = hospitals["longitude"].values

    nearest = []
    for i in range(len(zcta_lats)):
        dists = np.array([
            haversine_km(zcta_lats[i], zcta_lons[i], hosp_lats[j], hosp_lons[j])
            for j in range(len(hosp_lats))
        ])
        top_k = np.argsort(dists)[:k]
        nearest.append({
            "zcta_idx": i,
            "nearest_hosp_indices": top_k.tolist(),
            "nearest_hosp_dists_km": dists[top_k].tolist(),
        })

    return nearest


def main():
    parser = argparse.ArgumentParser(description="Build drive-time features")
    parser.add_argument("--output", default="/tmp/drive_times_zcta.parquet")
    parser.add_argument("--zcta-data",
                        default="/tmp/zcta_features_labels.parquet",
                        help="ZCTA data with latitude/longitude")
    parser.add_argument("--hifld-data",
                        default="/tmp/hifld_zcta.parquet",
                        help="HIFLD output (for hospital locations fallback)")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--max-zctas", type=int, default=None,
                        help="Limit ZCTAs for testing")
    parser.add_argument("--use-haversine", action="store_true",
                        help="Use haversine estimate instead of OSRM (faster)")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    # Load ZCTA centroids
    zcta = pd.read_parquet(args.zcta_data)
    zcta["zcta_id"] = zcta["zcta_id"].astype(str).str.zfill(5)

    if "latitude" not in zcta.columns or "longitude" not in zcta.columns:
        log.error("ZCTA data must have latitude/longitude columns")
        sys.exit(1)

    zcta = zcta[["zcta_id", "latitude", "longitude"]].copy()

    if args.max_zctas:
        zcta = zcta.head(args.max_zctas)
        log.info("Limited to %d ZCTAs", len(zcta))

    log.info("Loaded %d ZCTA centroids", len(zcta))

    # Build county centroids from ZCTA data
    zcta_full = pd.read_parquet(args.zcta_data)
    zcta_full["zcta_id"] = zcta_full["zcta_id"].astype(str).str.zfill(5)
    county_seats = build_county_centroids(zcta_full)

    # Load hospital locations from CMS via build_hifld_features
    # Each hospital is resolved to a ZCTA centroid (zip->ZCTA mapping)
    log.info("Loading hospital locations from CMS...")
    from build_hifld_features import fetch_cms_hospitals, resolve_zip_to_zcta
    cms_raw = fetch_cms_hospitals()
    cms_raw = resolve_zip_to_zcta(cms_raw, zcta)
    hospitals_raw = cms_raw.dropna(subset=["_lat", "_lon"]).copy()
    hospitals_raw = hospitals_raw.rename(columns={"_lat": "latitude", "_lon": "longitude"})
    # Deduplicate: one row per unique hospital ZCTA (centroid-level precision)
    hospitals_raw = hospitals_raw.drop_duplicates(subset=["_zcta_id"])
    log.info("  %d unique hospital ZCTAs", len(hospitals_raw))

    # Compute drive times
    if args.use_haversine:
        log.info("Using haversine distance estimate (speed factor 1.4x)...")
        SPEED_FACTOR = 1.4  # road detour factor
        AVG_SPEED_KMH = 50.0  # blended urban/rural

        def vectorized_nearest_min(
            src_lats: np.ndarray, src_lons: np.ndarray,
            dst_lats: np.ndarray, dst_lons: np.ndarray,
        ) -> np.ndarray:
            """Compute drive-time estimate (min) from each src to nearest dst."""
            R = 6371.0
            src_lat_r = np.radians(src_lats)
            src_lon_r = np.radians(src_lons)
            nearest_km = np.full(len(src_lats), np.inf)

            # Iterate over destinations (fewer than sources)
            for j in range(len(dst_lats)):
                dlat = np.radians(dst_lats[j]) - src_lat_r
                dlon = np.radians(dst_lons[j]) - src_lon_r
                a = (np.sin(dlat / 2) ** 2 +
                     np.cos(src_lat_r) * math.cos(math.radians(dst_lats[j])) *
                     np.sin(dlon / 2) ** 2)
                d = R * 2 * np.arcsin(np.sqrt(a))
                nearest_km = np.minimum(nearest_km, d)

            return (nearest_km * SPEED_FACTOR / AVG_SPEED_KMH) * 60

        zcta_lats = zcta["latitude"].values
        zcta_lons = zcta["longitude"].values

        log.info("  Computing nearest hospital drive times...")
        nearest_hosp_min = vectorized_nearest_min(
            zcta_lats, zcta_lons,
            hospitals_raw["latitude"].values, hospitals_raw["longitude"].values,
        )

        log.info("  Computing nearest county seat drive times...")
        nearest_cs_min = vectorized_nearest_min(
            zcta_lats, zcta_lons,
            county_seats["latitude"].values, county_seats["longitude"].values,
        )

    else:
        log.info("Using OSRM routing API...")
        log.info("  This will take ~15-30 minutes for %d ZCTAs", len(zcta))

        nearest_hosp_min = np.full(len(zcta), np.nan)
        nearest_cs_min = np.full(len(zcta), np.nan)

        # For each ZCTA, find nearest hospital and county seat by haversine,
        # then get OSRM driving time to the top-1 nearest
        hosp_lats = hospitals_raw["latitude"].values
        hosp_lons = hospitals_raw["longitude"].values
        cs_lats = county_seats["latitude"].values
        cs_lons = county_seats["longitude"].values

        for i in range(len(zcta)):
            lat = zcta.iloc[i]["latitude"]
            lon = zcta.iloc[i]["longitude"]

            # Find nearest hospital by haversine
            min_hosp_dist = np.inf
            min_hosp_idx = 0
            for j in range(len(hosp_lats)):
                d = haversine_km(lat, lon, hosp_lats[j], hosp_lons[j])
                if d < min_hosp_dist:
                    min_hosp_dist = d
                    min_hosp_idx = j

            # OSRM route to nearest hospital
            dur = osrm_route_duration(
                lon, lat,
                hosp_lons[min_hosp_idx], hosp_lats[min_hosp_idx],
            )
            nearest_hosp_min[i] = dur

            # Find nearest county seat
            min_cs_dist = np.inf
            min_cs_idx = 0
            for j in range(len(cs_lats)):
                d = haversine_km(lat, lon, cs_lats[j], cs_lons[j])
                if d < min_cs_dist:
                    min_cs_dist = d
                    min_cs_idx = j

            dur = osrm_route_duration(
                lon, lat,
                cs_lons[min_cs_idx], cs_lats[min_cs_idx],
            )
            nearest_cs_min[i] = dur

            if i % 100 == 0:
                log.info("  OSRM: %d/%d ZCTAs", i, len(zcta))

            time.sleep(0.5)  # rate limit

    # Build result
    result = pd.DataFrame({
        "zcta_id": zcta["zcta_id"].values,
        "drive_min_to_nearest_hospital": np.round(nearest_hosp_min, 1),
        "drive_min_to_county_seat": np.round(nearest_cs_min, 1),
    })

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs: %d", len(result))
    log.info("Hospital drive time:")
    log.info("  Mean:   %.1f min", result["drive_min_to_nearest_hospital"].mean())
    log.info("  Median: %.1f min", result["drive_min_to_nearest_hospital"].median())
    log.info("  Max:    %.1f min", result["drive_min_to_nearest_hospital"].max())
    log.info("  >60min: %d ZCTAs",
             (result["drive_min_to_nearest_hospital"] > 60).sum())
    log.info("County seat drive time:")
    log.info("  Mean:   %.1f min", result["drive_min_to_county_seat"].mean())
    log.info("  Median: %.1f min", result["drive_min_to_county_seat"].median())

    # Save
    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
        BUCKET = "swarm-yrsn-datasets"
        KEY = "rsct_curriculum/series_018/processed/drive_times_zcta.parquet"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.upload_file(args.output, BUCKET, KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, KEY)

        provenance = {
            "operation": "build_drive_times",
            "timestamp": timestamp,
            "method": "haversine_estimate" if args.use_haversine else "osrm",
            "n_zctas": len(result),
            "n_hospitals": int(len(hospitals_raw)),
            "n_county_seats": int(len(county_seats)),
            "mean_hosp_min": round(float(result["drive_min_to_nearest_hospital"].mean()), 1),
            "mean_county_seat_min": round(float(result["drive_min_to_county_seat"].mean()), 1),
        }
        s3.put_object(
            Bucket=BUCKET,
            Key="rsct_curriculum/series_018/processed/drive_times_zcta_provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
