#!/usr/bin/env python3
"""
build_hifld_features.py -- Build hospital & pharmacy access features per ZCTA.

Data source: CMS Hospital Compare API (replaces HIFLD, whose ArcGIS endpoints
moved to opaque hash-based service names in early 2026).

Geocoding strategy: CMS provides zip_code but no lat/lon. We resolve zip→ZCTA
via two paths:

  1. Direct match: zip_code[:5] == zcta_id (covers ~94.5% of CONUS hospitals)
  2. Fallback: nearest ZCTA by numeric ZIP proximity (rough geographic sort)

The proper resolution uses the HUD USPS ZIP-to-ZCTA crosswalk (many-to-many:
multiple delivery ZIPs fold into one ZCTA, and one ZIP can straddle multiple
ZCTAs weighted by RES_RATIO). This requires a HUD API key:

  Register: https://www.huduser.gov/hudapi/public/register?comingfrom=1
  Python client: https://etam4260.github.io/hudpy/build/html/index.html
  Set HUD_API_KEY env var after registration.

When HUD_API_KEY is set, the script uses hudpy to fetch the authoritative
ZIP→ZCTA crosswalk, replacing the numeric fallback with proper weighted
assignment. Each hospital inherits its assigned ZCTA's centroid coordinates
from the existing features parquet. Nearest-hospital distance is then
centroid-to-centroid between the target ZCTA and the nearest ZCTA that
contains a hospital. No external geocoding API is needed.

Computes per-ZCTA:
  - hifld_n_hospitals: count of CMS-registered hospitals in the ZCTA
  - hifld_n_hospital_beds: (unavailable from CMS Compare — set to NaN)
  - hifld_nearest_hospital_km: haversine distance to nearest hospital ZCTA centroid
  - hifld_n_pharmacies: (placeholder — CMS has no pharmacy endpoint)
  - hifld_nearest_pharmacy_km: (placeholder)
  - hifld_nearest_trauma_center_km: nearest hospital with emergency services

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

# CMS Hospital Compare (replaces HIFLD which moved to opaque service names)
CMS_HOSPITAL_URL = "https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0"
CMS_PAGE_SIZE = 500  # API max before it returns empty

# FDA pharmacy data or NPPES for pharmacy locations
# We use zip-code assignment since CMS doesn't have lat/lon


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


def fetch_cms_hospitals() -> pd.DataFrame:
    """Fetch all hospitals from CMS Hospital Compare API with pagination."""
    all_results = []
    offset = 0

    while True:
        log.info("Fetching hospitals offset=%d ...", offset)
        resp = requests.get(
            CMS_HOSPITAL_URL,
            params={"limit": CMS_PAGE_SIZE, "offset": offset},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        if not results:
            break

        all_results.extend(results)
        log.info("  Got %d (total: %d)", len(results), len(all_results))

        if len(results) < CMS_PAGE_SIZE:
            break
        offset += CMS_PAGE_SIZE

    log.info("Total hospitals fetched: %d", len(all_results))
    return pd.DataFrame(all_results)


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


def _fetch_hud_crosswalk(zip_codes: list) -> dict:
    """Fetch ZIP->ZCTA crosswalk from HUD API via hudpy.

    HUD doesn't have a direct ZIP->ZCTA endpoint, so we use ZIP->TRACT
    and then map TRACT->ZCTA using the Census relationship file.

    For hospital assignment, a simpler approach: the HUD ZIP->TRACT crosswalk
    gives us the dominant tract (by RES_RATIO), and tract[:5] approximates
    the county — but we actually just need the ZCTA. Since ZCTAs and ZIPs
    are both 5-digit and largely 1:1, we query HUD for each unmatched ZIP
    to get its dominant tract, then look up which ZCTA contains that tract.

    In practice: hudpy.hud_cw.hud_cw_zip_tract() returns tract mappings.
    We keep a tract->ZCTA lookup from our existing Census relationship file.

    Returns dict mapping zip5 -> zcta_id.
    Returns empty dict if HUD_API_KEY is not set or hudpy not installed.
    """
    import os
    hud_key = os.environ.get("HUD_API_KEY")
    if not hud_key:
        log.info("HUD_API_KEY not set -- skipping HUD crosswalk")
        return {}

    try:
        from hudpy.hud_key import hud_set_key
        from hudpy.hud_cw import hud_cw_zip_tract
        hud_set_key(hud_key)

        log.info("Fetching HUD ZIP->TRACT crosswalk for %d unmatched ZIPs...",
                 len(zip_codes))

        # Load tract->ZCTA from Census relationship file (same one fetch_svi.py uses)
        tract_zcta_url = ("https://www2.census.gov/geo/docs/maps-data/data/"
                          "rel2020/zcta520/tab20_zcta520_tract20_natl.txt")
        log.info("  Downloading Census tract->ZCTA relationship file...")
        tract_zcta = pd.read_csv(tract_zcta_url, sep="|", dtype=str)
        # Build tract -> dominant ZCTA (by AREALAND_PART)
        tract_zcta["AREALAND_PART"] = pd.to_numeric(
            tract_zcta["AREALAND_PART"], errors="coerce"
        ).fillna(0)
        best_zcta = (
            tract_zcta.sort_values("AREALAND_PART", ascending=False)
            .drop_duplicates("GEOID_TRACT_20")
        )
        tract_to_zcta = dict(zip(
            best_zcta["GEOID_TRACT_20"].str.zfill(11),
            best_zcta["GEOID_ZCTA5_20"].str.zfill(5),
        ))
        log.info("  Tract->ZCTA lookup: %d tracts", len(tract_to_zcta))

        # Query HUD for each unmatched ZIP
        result = {}
        for i, zip5 in enumerate(zip_codes):
            try:
                df = hud_cw_zip_tract(zip5)
                if df is not None and not df.empty:
                    # Take tract with highest res_ratio
                    df["res_ratio"] = pd.to_numeric(df["res_ratio"], errors="coerce")
                    best_row = df.sort_values("res_ratio", ascending=False).iloc[0]
                    tract = str(best_row.get("tract", "")).zfill(11)
                    zcta = tract_to_zcta.get(tract)
                    if zcta:
                        result[zip5] = zcta
            except Exception:
                pass  # skip individual failures

            if (i + 1) % 50 == 0:
                log.info("  HUD queries: %d/%d (resolved: %d)",
                         i + 1, len(zip_codes), len(result))

        log.info("  HUD crosswalk resolved: %d / %d ZIPs", len(result), len(zip_codes))
        return result

    except ImportError:
        log.warning("hudpy not installed (pip install hudpy). Using fallback.")
        return {}
    except Exception as e:
        log.warning("HUD crosswalk fetch failed: %s. Using fallback.", e)
        return {}


def resolve_zip_to_zcta(
    hospitals: pd.DataFrame,
    zcta_centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve hospital zip_code to ZCTA, assigning centroid lat/lon.

    Strategy (in priority order):
      1. Direct match: zip_code[:5] == zcta_id (covers ~94.5% of CONUS)
      2. HUD crosswalk: authoritative ZIP->ZCTA via hudpy (if HUD_API_KEY set)
      3. Nearest-ZCTA fallback: numeric ZIP proximity (rough geographic sort)

    The ZIP-to-ZCTA relationship is many-to-many:
      - Many ZIPs -> one ZCTA (PO box / unique-delivery ZIPs fold in)
      - One ZIP -> many ZCTAs (ZIP boundary straddles ZCTA boundary)
    For facility assignment, we take the dominant ZCTA (highest RES_RATIO
    from HUD, or direct match). This is acceptable because we only need
    the ZCTA-level count and distance, not sub-ZCTA precision.

    Returns hospitals with _zcta_id, _lat, _lon columns added.
    """
    zcta_set = set(zcta_centroids["zcta_id"])
    zcta_lookup = zcta_centroids.set_index("zcta_id")

    hospitals = hospitals.copy()
    hospitals["_zip5"] = hospitals["zip_code"].astype(str).str[:5].str.zfill(5)

    # Step 1: direct match
    direct_mask = hospitals["_zip5"].isin(zcta_set)
    hospitals.loc[direct_mask, "_zcta_id"] = hospitals.loc[direct_mask, "_zip5"]

    n_direct = direct_mask.sum()
    log.info("ZIP->ZCTA direct match: %d / %d (%.1f%%)",
             n_direct, len(hospitals), 100 * n_direct / len(hospitals))

    # Step 2: HUD crosswalk for unmatched (if available)
    unmatched_mask = ~direct_mask & hospitals["_zcta_id"].isna()
    if unmatched_mask.any():
        unmatched_zips = hospitals.loc[unmatched_mask, "_zip5"].unique().tolist()
        hud_map = _fetch_hud_crosswalk(unmatched_zips)
        if hud_map:
            for idx in hospitals[unmatched_mask].index:
                zip5 = hospitals.loc[idx, "_zip5"]
                zcta = hud_map.get(zip5)
                if zcta and zcta in zcta_set:
                    hospitals.loc[idx, "_zcta_id"] = zcta

            n_hud = (~hospitals["_zcta_id"].isna()).sum() - n_direct
            log.info("ZIP->ZCTA HUD crosswalk: %d additional matches", n_hud)

    # Step 3: nearest-ZCTA fallback for still-unmatched
    still_unmatched = hospitals[hospitals["_zcta_id"].isna()]
    if len(still_unmatched) > 0:
        zcta_ids = zcta_centroids["zcta_id"].values

        for idx in still_unmatched.index:
            zip5 = int(hospitals.loc[idx, "_zip5"])
            zcta_nums = zcta_ids.astype(int)
            nearest_idx = np.argmin(np.abs(zcta_nums - zip5))
            hospitals.loc[idx, "_zcta_id"] = zcta_ids[nearest_idx]

        log.info("ZIP->ZCTA numeric fallback: %d hospitals", len(still_unmatched))

    # Assign centroid coordinates from resolved ZCTA
    hospitals["_lat"] = hospitals["_zcta_id"].map(zcta_lookup["latitude"]).astype(float)
    hospitals["_lon"] = hospitals["_zcta_id"].map(zcta_lookup["longitude"]).astype(float)

    resolved = hospitals["_lat"].notna().sum()
    log.info("Hospitals with coordinates: %d / %d", resolved, len(hospitals))

    return hospitals


def main():
    parser = argparse.ArgumentParser(description="Build hospital access features")
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

    # -- Fetch hospitals from CMS --
    hospitals = fetch_cms_hospitals()
    if hospitals.empty:
        log.error("No hospitals fetched from CMS API")
        sys.exit(1)

    # Filter to CONUS (exclude territories without ZCTA coverage)
    conus_states = {
        "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
        "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
        "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
        "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
        "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "AK", "HI",
    }
    hospitals = hospitals[hospitals["state"].isin(conus_states)].copy()
    log.info("CONUS hospitals: %d", len(hospitals))

    # Resolve ZIP -> ZCTA -> centroid coordinates
    hospitals = resolve_zip_to_zcta(hospitals, zcta)
    hospitals = hospitals.dropna(subset=["_lat", "_lon"])

    # Identify emergency-capable hospitals (proxy for trauma centers)
    # CMS has emergency_services column (Yes/No)
    emergency_hospitals = hospitals[
        hospitals["emergency_services"].astype(str).str.lower() == "yes"
    ].copy()
    log.info("Emergency-capable hospitals: %d", len(emergency_hospitals))

    # -- Compute per-ZCTA features --

    # Count hospitals per ZCTA (direct group-by on resolved ZCTA)
    hosp_counts = hospitals.groupby("_zcta_id").size().reset_index(name="n")
    hosp_count_map = dict(zip(hosp_counts["_zcta_id"], hosp_counts["n"]))

    # Nearest hospital distance (ZCTA centroid to ZCTA centroid)
    hosp_result = assign_facilities_to_zctas(zcta, hospitals, "hospitals")

    # Nearest emergency hospital distance
    if len(emergency_hospitals) > 0:
        emerg_result = assign_facilities_to_zctas(zcta, emergency_hospitals, "emergency")
    else:
        emerg_result = {"nearest_dist": np.full(len(zcta), np.nan)}

    # Build result
    result = pd.DataFrame({
        "zcta_id": zcta["zcta_id"].values,
        "hifld_n_hospitals": [hosp_count_map.get(z, 0) for z in zcta["zcta_id"]],
        "hifld_n_hospital_beds": np.nan,  # CMS Compare lacks bed counts
        "hifld_nearest_hospital_km": np.round(hosp_result["nearest_dist"], 2),
        "hifld_n_pharmacies": np.nan,  # No pharmacy source available
        "hifld_nearest_pharmacy_km": np.nan,
        "hifld_nearest_trauma_center_km": np.round(emerg_result["nearest_dist"], 2),
    })

    # Replace inf with NaN
    for col in result.columns:
        if result[col].dtype == float:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    # Summary
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("ZCTAs: %d", len(result))
    log.info("Hospitals:")
    log.info("  CMS hospitals assigned: %d", len(hospitals))
    log.info("  ZCTAs with >= 1 hospital: %d",
             (result["hifld_n_hospitals"] > 0).sum())
    log.info("  Mean nearest hospital: %.1f km",
             result["hifld_nearest_hospital_km"].mean())
    log.info("  Max nearest hospital:  %.1f km",
             result["hifld_nearest_hospital_km"].max())
    log.info("Emergency (trauma proxy):")
    log.info("  Emergency hospitals: %d", len(emergency_hospitals))
    log.info("  Mean nearest emergency: %.1f km",
             result["hifld_nearest_trauma_center_km"].mean())
    log.info("  Max nearest emergency:  %.1f km",
             result["hifld_nearest_trauma_center_km"].max())

    # Save
    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
        BUCKET = "swarm-yrsn-datasets"
        KEY = "rsct_curriculum/series_018/processed/hifld_zcta.parquet"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.upload_file(args.output, BUCKET, KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, KEY)

        provenance = {
            "operation": "build_hifld_features",
            "timestamp": timestamp,
            "hospital_source": CMS_HOSPITAL_URL,
            "geocoding": "zip5->zcta direct match + nearest-zcta fallback",
            "n_zctas": len(result),
            "n_hospitals": int(len(hospitals)),
            "n_emergency": int(len(emergency_hospitals)),
            "direct_match_pct": round(100 * hospitals["_zip5"].isin(
                set(zcta["zcta_id"])).sum() / len(hospitals), 1),
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
