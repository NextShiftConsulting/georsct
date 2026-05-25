#!/usr/bin/env python3
"""
build_drive_times.py -- Compute drive times from ZCTA centroids to key facilities.

Uses OSRM (Open Source Routing Machine) table API for batched road-network routing:
  - drive_min_to_nearest_hospital: OSRM drive time to nearest HIFLD hospital (min)
  - drive_min_to_county_centroid: OSRM drive time to county centroid (min)

Hospital locations come from HIFLD 2022 Open Data via load_hifld_hospitals().
County centroids are the mean of ZCTA centroids per county_fips.

Strategy:
  1. Haversine pre-filter finds nearest destination per ZCTA (vectorized, seconds)
  2. OSRM table API batches 25 source-destination pairs per call (~1,300 calls total)
  3. Rate-limited at 1 req/sec -> ~22 min for 31K ZCTAs

A --use-haversine fallback estimates drive time without API calls.

Output: drive_times_zcta.parquet

Usage:
    python build_drive_times.py --zcta-data /path/to/zcta_features_labels.parquet --output /tmp/drive_times_zcta.parquet
    python build_drive_times.py --upload
    python build_drive_times.py --use-haversine   # fast estimate, no API calls
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

# Ensure stdout works on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# OSRM public demo API
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving"

# S3 output
BUCKET = "swarm-yrsn-datasets"
DRIVE_OUTPUT_KEY = "rsct_curriculum/series_018/processed/drive_times_zcta.parquet"
REGION = "us-east-1"


def vectorized_haversine_to_point(
    src_lats: np.ndarray, src_lons: np.ndarray,
    dst_lat: float, dst_lon: float,
) -> np.ndarray:
    """Haversine distance (km) from array of sources to a single destination."""
    R = 6371.0
    src_lat_r = np.radians(src_lats)
    src_lon_r = np.radians(src_lons)
    dst_lat_r = math.radians(dst_lat)
    dst_lon_r = math.radians(dst_lon)
    dlat = dst_lat_r - src_lat_r
    dlon = dst_lon_r - src_lon_r
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(src_lat_r) * math.cos(dst_lat_r) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def find_nearest_indices(
    zcta_lats: np.ndarray, zcta_lons: np.ndarray,
    fac_lats: np.ndarray, fac_lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each ZCTA, find nearest facility index and haversine distance (km)."""
    n_zcta = len(zcta_lats)
    nearest_idx = np.zeros(n_zcta, dtype=int)
    nearest_km = np.full(n_zcta, np.inf)

    for j in range(len(fac_lats)):
        d = vectorized_haversine_to_point(zcta_lats, zcta_lons, fac_lats[j], fac_lons[j])
        closer = d < nearest_km
        nearest_km[closer] = d[closer]
        nearest_idx[closer] = j

    return nearest_idx, nearest_km


def osrm_table_paired(
    src_coords: list[tuple[float, float]],
    dst_coords: list[tuple[float, float]],
    batch_size: int = 25,
    rate_limit: float = 1.0,
) -> np.ndarray:
    """Get drive times for paired source-destination using OSRM table API.

    Each src_coords[i] is routed to dst_coords[i] (1:1 pairing).
    Batches multiple pairs per API call to reduce total calls.

    Args:
        src_coords: list of (lon, lat) for sources
        dst_coords: list of (lon, lat) for destinations
        batch_size: pairs per API call (25 pairs = 50 coordinates)
        rate_limit: seconds between API calls

    Returns:
        Array of drive times in minutes (NaN on failure)
    """
    n = len(src_coords)
    result = np.full(n, np.nan)

    total_batches = (n + batch_size - 1) // batch_size
    log.info("  OSRM table API: %d pairs in %d batches of %d",
             n, total_batches, batch_size)

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_n = batch_end - batch_start

        # Build coordinate string: sources first, then destinations
        # Each source i maps to destination i
        all_coords = []
        for i in range(batch_start, batch_end):
            all_coords.append(src_coords[i])
        for i in range(batch_start, batch_end):
            all_coords.append(dst_coords[i])

        coord_str = ";".join(f"{lon},{lat}" for lon, lat in all_coords)

        src_indices = list(range(batch_n))
        dst_indices = list(range(batch_n, 2 * batch_n))

        url = f"{OSRM_TABLE_URL}/{coord_str}"
        params = {
            "sources": ";".join(str(x) for x in src_indices),
            "destinations": ";".join(str(x) for x in dst_indices),
            "annotations": "duration",
        }

        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=60)
                if resp.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()

                if data.get("code") == "Ok":
                    durations = data["durations"]
                    for j in range(batch_n):
                        val = durations[j][j]  # paired: source j -> dest j
                        if val is not None:
                            result[batch_start + j] = val / 60.0
                break
            except (requests.RequestException, KeyError, IndexError) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                log.warning("OSRM batch %d failed after 3 attempts: %s",
                            batch_start, e)

        # Progress logging
        done = batch_end
        if done % (batch_size * 20) == 0 or done == n:
            nan_count = np.isnan(result[:done]).sum()
            log.info("  OSRM: %d/%d (%.0f%%, %d NaN)",
                     done, n, 100 * done / n, nan_count)

        time.sleep(rate_limit)

    return result


def build_county_centroids(zcta_df: pd.DataFrame) -> pd.DataFrame:
    """Build county centroids from ZCTA centroids grouped by county_fips."""
    if "county_fips" not in zcta_df.columns:
        xwalk_candidates = [
            Path("/opt/ml/processing/input/data/zcta_county_crosswalk.parquet"),
            Path(__file__).parent / "zcta_county_crosswalk.parquet",
        ]
        for xp in xwalk_candidates:
            if xp.exists():
                xwalk = pd.read_parquet(xp)
                xwalk["zcta_id"] = xwalk["zcta_id"].astype(str).str.zfill(5)
                zcta_df = zcta_df.merge(xwalk[["zcta_id", "county_fips"]], on="zcta_id", how="left")
                break
        else:
            log.error("Need county_fips column or crosswalk parquet")
            sys.exit(1)

    county = (
        zcta_df.dropna(subset=["county_fips", "latitude", "longitude"])
        .groupby("county_fips")
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .reset_index()
    )
    log.info("  %d county centroids built from ZCTA means", len(county))
    return county


def main():
    parser = argparse.ArgumentParser(description="Build drive-time features")
    parser.add_argument("--output", default="/tmp/drive_times_zcta.parquet")
    parser.add_argument("--zcta-data",
                        default="/tmp/zcta_features_labels.parquet",
                        help="ZCTA data with latitude/longitude")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--max-zctas", type=int, default=None,
                        help="Limit ZCTAs for testing")
    parser.add_argument("--use-haversine", action="store_true",
                        help="Use haversine estimate instead of OSRM (faster)")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="OSRM table API batch size (pairs per call)")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    # ---- Load ZCTA centroids ----
    log.info("Loading ZCTA centroids from %s", args.zcta_data)
    zcta_full = pd.read_parquet(args.zcta_data)
    zcta_full["zcta_id"] = zcta_full["zcta_id"].astype(str).str.zfill(5)

    if "latitude" not in zcta_full.columns or "longitude" not in zcta_full.columns:
        log.error("ZCTA data must have latitude/longitude columns")
        sys.exit(1)

    zcta = zcta_full[["zcta_id", "latitude", "longitude"]].copy()
    if args.max_zctas:
        zcta = zcta.head(args.max_zctas)
        log.info("Limited to %d ZCTAs for testing", len(zcta))
    log.info("  %d ZCTA centroids loaded", len(zcta))

    # ---- Load HIFLD hospitals ----
    log.info("Loading HIFLD 2022 hospitals...")
    from build_hifld_features import load_hifld_hospitals
    hospitals = load_hifld_hospitals()
    hosp_lats = hospitals["_lat"].values
    hosp_lons = hospitals["_lon"].values
    log.info("  %d OPEN CONUS hospitals", len(hospitals))

    # ---- Build county centroids ----
    log.info("Building county centroids...")
    county_centroids = build_county_centroids(zcta_full)
    cc_lats = county_centroids["latitude"].values
    cc_lons = county_centroids["longitude"].values

    zcta_lats = zcta["latitude"].values
    zcta_lons = zcta["longitude"].values
    n_zcta = len(zcta)

    # ---- Find nearest by haversine (vectorized pre-filter) ----
    log.info("Finding nearest hospital per ZCTA (haversine pre-filter)...")
    nearest_hosp_idx, nearest_hosp_km = find_nearest_indices(
        zcta_lats, zcta_lons, hosp_lats, hosp_lons,
    )
    log.info("  Haversine nearest hospital: median=%.1f km, max=%.1f km",
             np.median(nearest_hosp_km), np.max(nearest_hosp_km))

    log.info("Finding nearest county centroid per ZCTA...")
    nearest_cc_idx, nearest_cc_km = find_nearest_indices(
        zcta_lats, zcta_lons, cc_lats, cc_lons,
    )

    if args.use_haversine:
        # ---- Haversine estimate (no API calls) ----
        SPEED_FACTOR = 1.4
        AVG_SPEED_KMH = 50.0
        log.info("Using haversine estimate (detour=%.1fx, speed=%.0f km/h)...",
                 SPEED_FACTOR, AVG_SPEED_KMH)

        nearest_hosp_min = (nearest_hosp_km * SPEED_FACTOR / AVG_SPEED_KMH) * 60
        nearest_cc_min = (nearest_cc_km * SPEED_FACTOR / AVG_SPEED_KMH) * 60
        method = "haversine_estimate"

    else:
        # ---- OSRM table API (batched, with S3 checkpointing) ----
        import boto3
from swarm_auth import get_aws_credentials
        try:
            _s3 = boto3.client("s3", region_name=REGION, **_aws)
        except Exception:
            _s3 = boto3.client("s3", region_name=REGION)

        CHECKPOINT_KEY = "rsct_code/geocert_v24/inputs/drive_times_checkpoint.npz"
        checkpoint_path = Path("/tmp/drive_times_checkpoint.npz")

        # Try to resume from checkpoint
        nearest_hosp_min = np.full(n_zcta, np.nan)
        nearest_cc_min = np.full(n_zcta, np.nan)
        hosp_done = False
        cc_done = False

        try:
            _s3.download_file(BUCKET, CHECKPOINT_KEY, str(checkpoint_path))
            ckpt = np.load(checkpoint_path)
            if "hosp_min" in ckpt and len(ckpt["hosp_min"]) == n_zcta:
                nearest_hosp_min = ckpt["hosp_min"]
                hosp_filled = np.count_nonzero(~np.isnan(nearest_hosp_min))
                if hosp_filled > n_zcta * 0.95:
                    hosp_done = True
                    log.info("Checkpoint: hospital routing COMPLETE (%d/%d filled)",
                             hosp_filled, n_zcta)
            if "cc_min" in ckpt and len(ckpt["cc_min"]) == n_zcta:
                nearest_cc_min = ckpt["cc_min"]
                cc_filled = np.count_nonzero(~np.isnan(nearest_cc_min))
                if cc_filled > n_zcta * 0.95:
                    cc_done = True
                    log.info("Checkpoint: county routing COMPLETE (%d/%d filled)",
                             cc_filled, n_zcta)
        except Exception:
            log.info("No checkpoint found, starting fresh")

        def save_checkpoint():
            np.savez(checkpoint_path,
                     hosp_min=nearest_hosp_min, cc_min=nearest_cc_min)
            _s3.upload_file(str(checkpoint_path), BUCKET, CHECKPOINT_KEY)

        est_calls = (n_zcta + args.batch_size - 1) // args.batch_size
        log.info("OSRM table API routing: %d ZCTAs, batch=%d, ~%d calls per pass",
                 n_zcta, args.batch_size, est_calls)

        if not hosp_done:
            hosp_src = [(zcta_lons[i], zcta_lats[i]) for i in range(n_zcta)]
            hosp_dst = [(hosp_lons[nearest_hosp_idx[i]], hosp_lats[nearest_hosp_idx[i]])
                        for i in range(n_zcta)]
            log.info("Routing to nearest hospitals...")
            nearest_hosp_min = osrm_table_paired(
                hosp_src, hosp_dst, batch_size=args.batch_size,
            )
            log.info("Checkpointing hospital results...")
            save_checkpoint()
        else:
            log.info("Skipping hospital routing (checkpoint complete)")

        if not cc_done:
            cc_src = [(zcta_lons[i], zcta_lats[i]) for i in range(n_zcta)]
            cc_dst = [(cc_lons[nearest_cc_idx[i]], cc_lats[nearest_cc_idx[i]])
                      for i in range(n_zcta)]
            log.info("Routing to nearest county centroids...")
            nearest_cc_min = osrm_table_paired(
                cc_src, cc_dst, batch_size=args.batch_size,
            )
            log.info("Checkpointing county results...")
            save_checkpoint()
        else:
            log.info("Skipping county routing (checkpoint complete)")

        method = "osrm_table"

    # ---- Build result ----
    result = pd.DataFrame({
        "zcta_id": zcta["zcta_id"].values,
        "drive_min_to_nearest_hospital": np.round(nearest_hosp_min, 1),
        "drive_min_to_county_centroid": np.round(nearest_cc_min, 1),
    })

    # ---- Summary ----
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("Method: %s", method)
    log.info("ZCTAs: %d", len(result))
    for col in ["drive_min_to_nearest_hospital", "drive_min_to_county_centroid"]:
        vals = result[col].dropna()
        log.info("%s:", col)
        log.info("  Mean:   %.1f min", vals.mean())
        log.info("  Median: %.1f min", vals.median())
        log.info("  Max:    %.1f min", vals.max())
        log.info("  NaN:    %d", result[col].isna().sum())
        if col == "drive_min_to_nearest_hospital":
            log.info("  >60min: %d ZCTAs", (vals > 60).sum())

    # ---- Save ----
    result.to_parquet(args.output, index=False)
    log.info("Saved: %s (%.1f KB)", args.output,
             Path(args.output).stat().st_size / 1024)

    if args.upload:
        import boto3
        try:
            s3 = boto3.client("s3", region_name=REGION, **_aws)
        except Exception:
            s3 = boto3.client("s3", region_name=REGION)

        s3.upload_file(args.output, BUCKET, DRIVE_OUTPUT_KEY)
        log.info("Uploaded to s3://%s/%s", BUCKET, DRIVE_OUTPUT_KEY)

        provenance = {
            "operation": "build_drive_times",
            "timestamp": timestamp,
            "method": method,
            "hospital_source": "HIFLD 2022 Open Data (hifld-geoplatform.hub.arcgis.com)",
            "routing_source": "OSRM public table API (router.project-osrm.org), OpenStreetMap data",
            "n_zctas": len(result),
            "n_hospitals": int(len(hospitals)),
            "n_county_centroids": int(len(county_centroids)),
            "mean_hosp_min": round(float(result["drive_min_to_nearest_hospital"].mean()), 1),
            "mean_county_min": round(float(result["drive_min_to_county_centroid"].mean()), 1),
            "nan_hosp": int(result["drive_min_to_nearest_hospital"].isna().sum()),
            "nan_county": int(result["drive_min_to_county_centroid"].isna().sum()),
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
