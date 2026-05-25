#!/usr/bin/env python3
"""
build_hifld_features.py -- Build hospital & pharmacy access features per ZCTA.

Data sources:
  - Hospitals: HIFLD Open Data Hospitals CSV (8,013 hospitals with BEDS, TRAUMA,
    lat/lon; source: hifld-geoplatform.hub.arcgis.com). Local file or S3
    artifact. Provides n_hospitals, n_hospital_beds, nearest_hospital_km,
    nearest_trauma_center_km.
  - Pharmacies: CDC Vaccines.gov provider locations (~41K unique pharmacies
    with lat/lon). Bulk CSV download. Provides n_pharmacies,
    nearest_pharmacy_km.

Hospital assignment: haversine nearest-ZCTA from HIFLD lat/lon coordinates.
Pharmacy assignment: haversine nearest-ZCTA from CDC lat/lon coordinates.

Computes per-ZCTA:
  - hifld_n_hospitals: count of HIFLD hospitals assigned to the ZCTA
  - hifld_n_hospital_beds: sum of BEDS for hospitals in the ZCTA (sentinel
    -999 treated as 0)
  - hifld_nearest_hospital_km: haversine to nearest hospital
  - hifld_n_pharmacies: count of CDC pharmacies in the ZCTA
  - hifld_nearest_pharmacy_km: haversine to nearest pharmacy
  - hifld_nearest_trauma_center_km: haversine to nearest trauma center

Output: hifld_zcta.parquet

Usage:
    python build_hifld_features.py --output /tmp/hifld_zcta.parquet
    python build_hifld_features.py --upload
"""

import argparse
import json
import logging
import math
import sys
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

# HIFLD Hospitals (local CSV or S3 artifact)
# Source: https://hifld-geoplatform.hub.arcgis.com/pages/hifld-open
# Vintage: SOURCEDATE 2012-2022, VAL_DATE 2013-2022 (8,013 hospitals)
BUCKET = "swarm-yrsn-datasets"
HIFLD_HOSPITALS_S3_KEY = "rsct_curriculum/series_018/source/HIFLD_2022_Hospitals.csv"
HIFLD_HOSPITALS_LOCAL = Path(__file__).parent.parent.parent.parent / "V3" / "data" / "HIFLD_2022_Hospitals.csv"

# CDC Vaccines.gov pharmacy locations (bulk CSV)
CDC_PHARMACY_URL = "https://data.cdc.gov/api/views/5jp2-pgaw/rows.csv?accessType=DOWNLOAD"

# S3 output
HIFLD_OUTPUT_KEY = "rsct_curriculum/series_018/processed/hifld_zcta.parquet"
REGION = "us-east-1"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance in km between two (lat, lon) points."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def haversine_km_vectorized(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: float, lon2: float,
) -> np.ndarray:
    """Vectorized haversine: array of points to single point."""
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_hifld_hospitals() -> pd.DataFrame:
    """Load HIFLD Hospitals from local CSV, SageMaker mount, or S3."""
    # Check paths in priority order
    sagemaker_path = Path("/opt/ml/processing/input/hifld/HIFLD_2022_Hospitals.csv")
    candidates = [HIFLD_HOSPITALS_LOCAL, sagemaker_path]
    for p in candidates:
        if p.exists():
            log.info("Loading HIFLD hospitals from %s", p)
            df = pd.read_csv(p)
            break
    else:
        local_path = Path("/tmp/HIFLD_2022_Hospitals.csv")
        if not local_path.exists():
            log.info("Downloading HIFLD hospitals from S3...")
            import boto3
from swarm_auth import get_aws_credentials
            try:
                s3 = boto3.client("s3", region_name=REGION, **_aws)
            except Exception:
                s3 = boto3.client("s3", region_name=REGION)
            s3.download_file(BUCKET, HIFLD_HOSPITALS_S3_KEY, str(local_path))
        df = pd.read_csv(local_path)

    # Filter to OPEN status and CONUS
    df = df[df["STATUS"] == "OPEN"].copy()
    conus_states = {
        "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
        "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
        "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
        "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
        "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    }
    df = df[df["STATE"].isin(conus_states)].copy()

    # Clean BEDS: sentinel -999 -> 0
    df["BEDS"] = df["BEDS"].replace(-999, 0).clip(lower=0)

    # Standardize columns for assignment
    df["_lat"] = df["LATITUDE"].astype(float)
    df["_lon"] = df["LONGITUDE"].astype(float)
    df["_zip5"] = df["ZIP"].astype(str).str[:5].str.zfill(5)

    log.info("HIFLD hospitals loaded: %d (CONUS, OPEN)", len(df))
    log.info("  BEDS: median=%d, max=%d, zero=%d",
             df["BEDS"].median(), df["BEDS"].max(), (df["BEDS"] == 0).sum())
    return df


def fetch_cdc_pharmacies() -> pd.DataFrame:
    """Fetch pharmacy locations from CDC Vaccines.gov bulk CSV."""
    local_path = Path("/tmp/cdc_pharmacies.csv")
    if not local_path.exists():
        log.info("Downloading CDC pharmacy data (~50 MB)...")
        resp = requests.get(CDC_PHARMACY_URL, stream=True, timeout=120)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        log.info("  Downloaded %.1f MB", local_path.stat().st_size / 1e6)

    df = pd.read_csv(local_path, dtype=str)
    log.info("CDC pharmacy rows (raw): %d", len(df))

    # Deduplicate on provider_location_guid (multiple rows per vaccine type)
    if "provider_location_guid" in df.columns:
        df = df.drop_duplicates(subset=["provider_location_guid"])
    log.info("CDC pharmacies (unique): %d", len(df))

    # Parse coordinates
    df["_lat"] = pd.to_numeric(df.get("latitude", pd.Series(dtype=float)),
                               errors="coerce")
    df["_lon"] = pd.to_numeric(df.get("longitude", pd.Series(dtype=float)),
                               errors="coerce")

    # Filter to CONUS with valid coordinates
    df = df.dropna(subset=["_lat", "_lon"])
    conus_states = {
        "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
        "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
        "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
        "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
        "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    }
    if "loc_admin_state" in df.columns:
        df = df[df["loc_admin_state"].isin(conus_states)].copy()

    log.info("CDC pharmacies (CONUS, geocoded): %d", len(df))
    return df


def load_zcta_centroids(zcta_path: str) -> pd.DataFrame:
    """Load ZCTA centroids from features/labels parquet or GeoParquet."""
    path = Path(zcta_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        import geopandas as gpd
        geo = gpd.read_file(path)
        df = pd.DataFrame({
            "zcta_id": geo["ZCTA5CE20"] if "ZCTA5CE20" in geo.columns else geo["zcta_id"],
            "latitude": geo.geometry.centroid.y,
            "longitude": geo.geometry.centroid.x,
        })

    df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)

    if "latitude" not in df.columns or "longitude" not in df.columns:
        log.error("ZCTA file must have latitude/longitude columns")
        sys.exit(1)

    return df[["zcta_id", "latitude", "longitude"]].copy()


def assign_facilities_to_zctas(
    zcta_centroids: pd.DataFrame,
    facilities: pd.DataFrame,
    facility_type: str,
) -> dict:
    """For each ZCTA, count facilities and find nearest distance.

    Uses a simple approach: assign each facility to its nearest ZCTA centroid
    (for counts), and compute min distance from each ZCTA to any facility.
    """
    fac_lats = facilities["_lat"].values
    fac_lons = facilities["_lon"].values
    valid_mask = ~(np.isnan(fac_lats) | np.isnan(fac_lons))
    fac_lats = fac_lats[valid_mask]
    fac_lons = fac_lons[valid_mask]

    log.info("Computing %s distances for %d ZCTAs x %d facilities...",
             facility_type, len(zcta_centroids), len(fac_lats))

    zcta_lats = zcta_centroids["latitude"].values
    zcta_lons = zcta_centroids["longitude"].values

    # For each facility, find nearest ZCTA (for counting)
    # For each ZCTA, find nearest facility (for distance)
    nearest_dist = np.full(len(zcta_centroids), np.inf)
    facility_count = np.zeros(len(zcta_centroids), dtype=int)

    # Process in chunks to manage memory
    chunk_size = 500
    for i in range(0, len(fac_lats), chunk_size):
        chunk_lats = fac_lats[i:i + chunk_size]
        chunk_lons = fac_lons[i:i + chunk_size]

        for j in range(len(chunk_lats)):
            # Distance from this facility to all ZCTAs
            dists = haversine_km_vectorized(
                zcta_lats, zcta_lons, chunk_lats[j], chunk_lons[j]
            )
            # Update nearest distance
            nearest_dist = np.minimum(nearest_dist, dists)
            # Assign facility to nearest ZCTA
            nearest_zcta_idx = np.argmin(dists)
            facility_count[nearest_zcta_idx] += 1

    return {
        "nearest_dist": nearest_dist,
        "count": facility_count,
    }




def main():
    parser = argparse.ArgumentParser(description="Build hospital & pharmacy access features")
    parser.add_argument("--output", default="/tmp/hifld_zcta.parquet")
    parser.add_argument("--zcta-data",
                        default="/tmp/zcta_features_labels.parquet",
                        help="Path to ZCTA data with latitude/longitude")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    # Load ZCTA centroids
    zcta = load_zcta_centroids(args.zcta_data)
    log.info("Loaded %d ZCTA centroids", len(zcta))

    # ----------------------------------------------------------------
    # HOSPITALS (HIFLD 2020 -- has BEDS, TRAUMA, lat/lon)
    # ----------------------------------------------------------------
    hospitals = load_hifld_hospitals()
    if hospitals.empty:
        log.error("No hospitals loaded")
        sys.exit(1)

    # Identify trauma centers (HIFLD has TRAUMA column: LEVEL I-V or NOT AVAILABLE)
    trauma_hospitals = hospitals[
        hospitals["TRAUMA"].astype(str).str.startswith("LEVEL")
    ].copy()
    log.info("Trauma centers: %d", len(trauma_hospitals))

    # Assign hospitals to nearest ZCTA + compute distances
    hosp_result = assign_facilities_to_zctas(zcta, hospitals, "hospitals")

    # Nearest trauma center distance
    if len(trauma_hospitals) > 0:
        trauma_result = assign_facilities_to_zctas(zcta, trauma_hospitals, "trauma")
    else:
        trauma_result = {"nearest_dist": np.full(len(zcta), np.nan)}

    # Count hospitals and sum beds per ZCTA (assign each hospital to nearest ZCTA)
    zcta_lats = zcta["latitude"].values
    zcta_lons = zcta["longitude"].values
    hosp_zcta_idx = np.zeros(len(hospitals), dtype=int)
    for i, (lat, lon) in enumerate(zip(hospitals["_lat"].values,
                                        hospitals["_lon"].values)):
        dists = haversine_km_vectorized(zcta_lats, zcta_lons, lat, lon)
        hosp_zcta_idx[i] = np.argmin(dists)

    hosp_count = np.zeros(len(zcta), dtype=int)
    hosp_beds = np.zeros(len(zcta), dtype=int)
    beds_array = hospitals["BEDS"].values
    for i, idx in enumerate(hosp_zcta_idx):
        hosp_count[idx] += 1
        hosp_beds[idx] += beds_array[i]

    # ----------------------------------------------------------------
    # PHARMACIES (CDC Vaccines.gov -- has lat/lon)
    # ----------------------------------------------------------------
    pharmacies = fetch_cdc_pharmacies()

    if not pharmacies.empty:
        pharm_result = assign_facilities_to_zctas(zcta, pharmacies, "pharmacies")

        # Count pharmacies per ZCTA
        pharm_count = np.zeros(len(zcta), dtype=int)
        for i, (lat, lon) in enumerate(zip(pharmacies["_lat"].values,
                                            pharmacies["_lon"].values)):
            dists = haversine_km_vectorized(zcta_lats, zcta_lons, lat, lon)
            pharm_count[np.argmin(dists)] += 1

        pharm_nearest = np.round(pharm_result["nearest_dist"], 2)
    else:
        log.warning("No pharmacy data -- columns will be NaN")
        pharm_count = np.zeros(len(zcta), dtype=int)
        pharm_nearest = np.full(len(zcta), np.nan)

    # ----------------------------------------------------------------
    # BUILD RESULT
    # ----------------------------------------------------------------
    result = pd.DataFrame({
        "zcta_id": zcta["zcta_id"].values,
        "hifld_n_hospitals": hosp_count,
        "hifld_n_hospital_beds": hosp_beds,
        "hifld_nearest_hospital_km": np.round(hosp_result["nearest_dist"], 2),
        "hifld_n_pharmacies": pharm_count,
        "hifld_nearest_pharmacy_km": pharm_nearest,
        "hifld_nearest_trauma_center_km": np.round(trauma_result["nearest_dist"], 2),
    })

    # Replace inf with NaN
    for col in result.columns:
        if result[col].dtype == float:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs: %d", len(result))
    log.info("Hospitals (HIFLD 2020):")
    log.info("  Total: %d", len(hospitals))
    log.info("  ZCTAs with >= 1 hospital: %d", (hosp_count > 0).sum())
    log.info("  Total beds: %d", hosp_beds.sum())
    log.info("  Mean nearest hospital: %.1f km",
             result["hifld_nearest_hospital_km"].mean())
    log.info("  Max nearest hospital:  %.1f km",
             result["hifld_nearest_hospital_km"].max())
    log.info("Trauma centers:")
    log.info("  Total: %d", len(trauma_hospitals))
    log.info("  Mean nearest trauma: %.1f km",
             result["hifld_nearest_trauma_center_km"].mean())
    log.info("Pharmacies (CDC Vaccines.gov):")
    log.info("  Total: %d", len(pharmacies))
    log.info("  ZCTAs with >= 1 pharmacy: %d", (pharm_count > 0).sum())
    log.info("  Mean nearest pharmacy: %.1f km",
             result["hifld_nearest_pharmacy_km"].mean())

    # Save
    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
        try:
            s3 = boto3.client("s3", region_name=REGION, **_aws)
        except Exception:
            s3 = boto3.client("s3", region_name=REGION)
        s3.upload_file(args.output, BUCKET, HIFLD_OUTPUT_KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, HIFLD_OUTPUT_KEY)

        provenance = {
            "operation": "build_hifld_features",
            "timestamp": timestamp,
            "hospital_source": "HIFLD Open Data Hospitals (hifld-geoplatform.hub.arcgis.com)",
            "pharmacy_source": "CDC Vaccines.gov (data.cdc.gov/5jp2-pgaw)",
            "hospital_vintage": "SOURCEDATE 2012-2022, VAL_DATE 2013-2022",
            "n_zctas": len(result),
            "n_hospitals": int(len(hospitals)),
            "n_trauma_centers": int(len(trauma_hospitals)),
            "n_pharmacies": int(len(pharmacies)),
            "total_beds": int(hosp_beds.sum()),
        }
        s3.put_object(
            Bucket=BUCKET,
            Key="rsct_curriculum/series_018/processed/hifld_zcta_provenance.json",
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance saved.")

    log.info("Done.")


if __name__ == "__main__":
    main()
