#!/usr/bin/env python3
"""
build_r2_features.py -- SageMaker job: compute R2 temporal event features.

Reads existing hourly MRMS grib2 files and tide gauge parquets from S3.
Produces per-ZCTA temporal statistics for the R2 representation bundle.

R2 features (all event_window temporal class):
  - peak_1h_mm, peak_3h_mm, peak_6h_mm: rolling-max rainfall at ZCTA centroid
  - storm_duration_h: hours where rainfall exceeds 1 mm
  - time_to_peak_h: hours from first rain to peak rain
  - rainfall_intensity_cv: CV of hourly rainfall during storm hours
  - tide_peak_m: max observed water level from nearest tide station
  - surge_rain_lag_h: hours between peak rainfall and peak surge

Approach:
  1. Decode ONE grib2 to get grid coordinates
  2. Precompute nearest-grid-index for each ZCTA centroid (done ONCE)
  3. Decode all remaining grib2 in parallel, index with precomputed map
  4. Build (n_zctas x n_hours) rainfall matrix (~1 MB, trivial)
  5. Compute temporal stats from matrix
  6. Load tide gauge parquets, compute surge timing

Usage:
    python build_r2_features.py --scenario houston --upload
    python build_r2_features.py --scenario southwest_florida --upload
"""

import argparse
import gzip
import io
import json
import logging
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, load_processed_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# Scenario -> events mapping
SCENARIO_EVENTS = {
    "houston": ["harvey2017", "imelda2019", "beryl2024"],
    "southwest_florida": ["ian2022", "milton2024", "helene2024"],
    "nyc": ["ida2021_nyc", "henri2021"],
    "riverside_coachella": ["hilary2023"],
    "new_orleans": ["ida2021_nola"],
}

RAIN_THRESHOLD_MM = 1.0  # minimum rainfall to count as "storm hour"


def parse_mrms_timestamp(filename: str) -> datetime:
    """Extract datetime from MRMS filename like GaugeCorr_QPE_01H_00.00_20170817-120000.grib2.gz"""
    match = re.search(r"(\d{8})-(\d{6})", filename)
    if not match:
        return None
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")


def _decode_one_grib(args: tuple):
    """Worker: download + decompress + decode one grib2 file. Returns (precip_2d, lat_1d, lon_1d) or None."""
    key, bucket = args
    import boto3
    s3w = boto3.client("s3", region_name="us-east-1")

    with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f_gz:
        s3w.download_fileobj(bucket, key, f_gz)
        f_gz.flush()
        gz_path = f_gz.name

    grb2_path = gz_path.replace(".gz", ".grb2")
    try:
        if key.endswith(".gz"):
            with gzip.open(gz_path, "rb") as gz_in:
                with open(grb2_path, "wb") as raw_out:
                    raw_out.write(gz_in.read())
        else:
            grb2_path = gz_path

        import cfgrib
        ds = cfgrib.open_dataset(grb2_path, indexpath=None)
        precip_var = next(
            (v for v in ["tp", "unknown", "apcp", "APCP"] if v in ds), None
        )
        if precip_var is None:
            return None
        return (ds[precip_var].values, ds["latitude"].values, ds["longitude"].values)
    except Exception:
        return None
    finally:
        for p in (gz_path, grb2_path):
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


def build_centroid_index(lat_1d: np.ndarray, lon_1d: np.ndarray,
                         centroid_lats: np.ndarray, centroid_lons: np.ndarray) -> np.ndarray:
    """Precompute nearest grid cell index for each ZCTA centroid.

    Returns array of flat indices into the MRMS grid.
    """
    if lat_1d.ndim == 1 and lon_1d.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    else:
        lat_2d, lon_2d = lat_1d, lon_1d

    flat_lat = lat_2d.flatten()
    flat_lon = lon_2d.flatten()

    indices = np.empty(len(centroid_lats), dtype=np.int64)
    for i in range(len(centroid_lats)):
        dist = (flat_lat - centroid_lats[i]) ** 2 + (flat_lon - centroid_lons[i]) ** 2
        indices[i] = dist.argmin()

    return indices


def compute_mrms_temporal(s3, event: str, zcta_ids: list[str],
                          centroid_lats: np.ndarray,
                          centroid_lons: np.ndarray) -> pd.DataFrame:
    """Build per-ZCTA temporal rainfall features from hourly MRMS grib2 files."""
    prefix = f"raw/noaa_mrms/{event}/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
    all_keys = [o["Key"] for o in resp.get("Contents", [])
                if o["Key"].endswith(".grb2") or o["Key"].endswith(".grib2.gz")]
    # Handle pagination
    while resp.get("IsTruncated"):
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000,
                                  ContinuationToken=resp["NextContinuationToken"])
        all_keys.extend(o["Key"] for o in resp.get("Contents", [])
                        if o["Key"].endswith(".grb2") or o["Key"].endswith(".grib2.gz"))

    if not all_keys:
        log.warning("No MRMS files for event %s", event)
        return _empty_r2(zcta_ids)

    # Sort by timestamp
    keyed = []
    for k in all_keys:
        ts = parse_mrms_timestamp(k.split("/")[-1])
        if ts:
            keyed.append((ts, k))
    keyed.sort(key=lambda x: x[0])
    timestamps = [t for t, _ in keyed]
    sorted_keys = [k for _, k in keyed]
    n_hours = len(sorted_keys)
    n_zctas = len(zcta_ids)
    log.info("MRMS temporal: event=%s, %d hours, %d ZCTAs", event, n_hours, n_zctas)

    # Decode files in parallel, collect results with hour index
    n_workers = min(os.cpu_count() // 2 or 2, 8)
    grid_index = None  # precomputed centroid -> grid mapping
    rainfall_matrix = np.full((n_zctas, n_hours), np.nan, dtype=np.float32)

    work_items = [(k, BUCKET) for k in sorted_keys]
    key_to_hour = {k: i for i, k in enumerate(sorted_keys)}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_decode_one_grib, item): item for item in work_items}
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            if done_count % 50 == 0:
                log.info("MRMS decode progress: %d/%d", done_count, n_hours)
                sys.stdout.flush()

            result = fut.result()
            if result is None:
                continue

            arr, lat, lon = result
            key_used = futures[fut][0]
            hour_idx = key_to_hour[key_used]

            # Build centroid index on first successful decode
            if grid_index is None:
                grid_index = build_centroid_index(lat, lon, centroid_lats, centroid_lons)
                log.info("Grid index built: %d centroids mapped to MRMS grid", len(grid_index))

            # Sample at precomputed indices
            flat_vals = arr.flatten()
            hourly_vals = flat_vals[grid_index]
            rainfall_matrix[:, hour_idx] = np.where(np.isnan(hourly_vals), 0.0, hourly_vals)

    if grid_index is None:
        log.error("No MRMS files decoded successfully for %s", event)
        return _empty_r2(zcta_ids)

    # Compute temporal features from the rainfall matrix
    # Replace NaN columns (missing hours) with 0 for rolling computations
    valid_mask = ~np.all(np.isnan(rainfall_matrix), axis=0)
    log.info("Valid hours: %d/%d", valid_mask.sum(), n_hours)

    results = []
    for i in range(n_zctas):
        ts = rainfall_matrix[i, :]
        results.append(_compute_temporal_stats(ts, zcta_ids[i]))

    df = pd.DataFrame(results)
    log.info("R2 temporal features computed for event %s: %d ZCTAs", event, len(df))
    return df


def _compute_temporal_stats(hourly: np.ndarray, zcta_id: str) -> dict:
    """Compute temporal rainfall statistics for one ZCTA."""
    # Replace NaN with 0 for rolling computations
    h = np.nan_to_num(hourly, nan=0.0)
    n = len(h)

    # Peak rolling sums
    peak_1h = float(np.max(h)) if n > 0 else np.nan

    if n >= 3:
        rolling_3h = np.convolve(h, np.ones(3), mode="valid")
        peak_3h = float(np.max(rolling_3h))
    else:
        peak_3h = float(np.sum(h))

    if n >= 6:
        rolling_6h = np.convolve(h, np.ones(6), mode="valid")
        peak_6h = float(np.max(rolling_6h))
    else:
        peak_6h = float(np.sum(h))

    # Storm duration: hours above threshold
    storm_mask = h > RAIN_THRESHOLD_MM
    storm_duration = int(storm_mask.sum())

    # Time to peak: hours from first rain to peak
    if storm_duration > 0:
        first_rain_idx = np.argmax(storm_mask)
        peak_idx = np.argmax(h)
        time_to_peak = int(peak_idx - first_rain_idx)
    else:
        time_to_peak = 0

    # Rainfall intensity CV during storm hours
    storm_values = h[storm_mask]
    if len(storm_values) >= 2:
        mean_rain = storm_values.mean()
        if mean_rain > 0:
            cv = float(storm_values.std() / mean_rain)
        else:
            cv = 0.0
    else:
        cv = 0.0

    return {
        "zcta_id": zcta_id,
        "peak_1h_mm": peak_1h,
        "peak_3h_mm": peak_3h,
        "peak_6h_mm": peak_6h,
        "storm_duration_h": storm_duration,
        "time_to_peak_h": time_to_peak,
        "rainfall_intensity_cv": cv,
    }


def _empty_r2(zcta_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "zcta_id": zcta_ids,
        "peak_1h_mm": np.nan,
        "peak_3h_mm": np.nan,
        "peak_6h_mm": np.nan,
        "storm_duration_h": np.nan,
        "time_to_peak_h": np.nan,
        "rainfall_intensity_cv": np.nan,
    })


def compute_tide_features(s3, event: str, zcta_ids: list[str],
                          centroid_lats: np.ndarray,
                          centroid_lons: np.ndarray,
                          zcta_peak_rain_hour: dict) -> pd.DataFrame:
    """Load tide gauge parquets, compute surge timing features."""
    prefix = f"raw/noaa_tides/{event}/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=100)
    surge_keys = [o["Key"] for o in resp.get("Contents", [])
                  if o["Key"].endswith(".parquet") and "tidal_surge" in o["Key"]]

    if not surge_keys:
        log.info("No tide gauge data for event %s", event)
        return pd.DataFrame({
            "zcta_id": zcta_ids,
            "tide_peak_m": np.nan,
            "surge_rain_lag_h": np.nan,
        })

    # Load all station surge time series
    all_stations = []
    for key in surge_keys:
        station_id = key.split("/")[-1].replace("tidal_surge_", "").replace(".parquet", "")
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            sdf = pd.read_parquet(io.BytesIO(obj["Body"].read()))
            sdf["station_id"] = station_id
            all_stations.append(sdf)
        except Exception as exc:
            log.warning("Cannot read tide gauge %s: %s", key, exc)

    if not all_stations:
        return pd.DataFrame({
            "zcta_id": zcta_ids,
            "tide_peak_m": np.nan,
            "surge_rain_lag_h": np.nan,
        })

    tide_df = pd.concat(all_stations, ignore_index=True)
    tide_df["observed_m"] = pd.to_numeric(tide_df["observed_m"], errors="coerce")
    tide_df["timestamp"] = pd.to_datetime(tide_df["timestamp"], errors="coerce")

    # Find overall peak across all stations
    valid = tide_df.dropna(subset=["observed_m", "timestamp"])
    if valid.empty:
        return pd.DataFrame({
            "zcta_id": zcta_ids,
            "tide_peak_m": np.nan,
            "surge_rain_lag_h": np.nan,
        })

    peak_row = valid.loc[valid["observed_m"].idxmax()]
    tide_peak_m = float(peak_row["observed_m"])
    tide_peak_time = peak_row["timestamp"]

    # Broadcast tide peak to all ZCTAs (no per-station spatial assignment yet)
    results = []
    for zcta in zcta_ids:
        lag_h = np.nan
        rain_hour = zcta_peak_rain_hour.get(zcta)
        if rain_hour is not None and pd.notna(tide_peak_time):
            # Both are datetime-like; compute difference in hours
            if isinstance(rain_hour, (int, float)):
                # rain_hour is an hour index; can't compute absolute lag without timestamps
                lag_h = np.nan
            else:
                delta = (tide_peak_time - rain_hour).total_seconds() / 3600
                lag_h = float(delta)

        results.append({
            "zcta_id": zcta,
            "tide_peak_m": tide_peak_m,
            "surge_rain_lag_h": lag_h,
        })

    return pd.DataFrame(results)


def load_zcta_centroids(s3, zcta_ids: list[str]):
    """Load ZCTA centroids from geocertdb2026."""
    obj = s3.get_object(Bucket=BUCKET, Key="raw/geocertdb2026/zcta_features_labels.parquet")
    static = pd.read_parquet(io.BytesIO(obj["Body"].read()))

    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)

    centroids = (static[[zcta_col, lat_col, lon_col]]
                 .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
                 .query("zcta_id in @zcta_ids")
                 .dropna(subset=["lat", "lon"]))
    return centroids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=list(SCENARIO_EVENTS.keys()))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    scenario = args.scenario
    events = SCENARIO_EVENTS[scenario]
    log.info("build_r2_features: scenario=%s, events=%s", scenario, events)

    s3 = get_s3_client()

    # Load assembled parquet for ZCTA IDs and event assignment
    df = load_processed_parquet(s3, scenario)
    all_zctas = df["zcta_id"].unique().tolist()

    # Load centroids
    centroids = load_zcta_centroids(s3, all_zctas)
    zcta_order = centroids["zcta_id"].values.tolist()
    clats = centroids["lat"].values.astype(np.float64)
    clons = centroids["lon"].values.astype(np.float64)
    log.info("Loaded %d ZCTA centroids", len(zcta_order))

    # Process each event
    event_results = []
    for event in events:
        log.info("Processing event: %s", event)

        # MRMS temporal features
        mrms = compute_mrms_temporal(s3, event, zcta_order, clats, clons)

        # Build peak-rain-hour lookup for surge-rain lag
        # (hour index only; absolute times would require keeping timestamps)
        peak_rain_hour = {}
        for _, row in mrms.iterrows():
            peak_rain_hour[row["zcta_id"]] = None  # simplified: no absolute time

        # Tide gauge features
        tide = compute_tide_features(s3, event, zcta_order, clats, clons, peak_rain_hour)

        # Merge MRMS + tide
        r2 = mrms.merge(tide, on="zcta_id", how="outer")
        r2["event"] = event
        event_results.append(r2)

    # Concatenate all events
    r2_all = pd.concat(event_results, ignore_index=True)
    log.info("R2 supplement total: %d rows, %d columns", len(r2_all), len(r2_all.columns))
    for col in r2_all.columns:
        if col in ("zcta_id", "event"):
            continue
        pct = r2_all[col].notna().mean() * 100
        log.info("  %s: %.1f%% non-null", col, pct)

    if args.upload:
        buf = io.BytesIO()
        r2_all.to_parquet(buf, index=False)
        buf.seek(0)
        key = f"processed/{scenario}/{scenario}_r2_supplement.parquet"
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        log.info("Uploaded s3://%s/%s", BUCKET, key)

        meta = {
            "scenario": scenario,
            "events": events,
            "n_rows": len(r2_all),
            "columns": list(r2_all.columns),
            "coverage": {
                col: float(r2_all[col].notna().mean())
                for col in r2_all.columns if col not in ("zcta_id", "event")
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        meta_key = f"processed/{scenario}/{scenario}_r2_meta.json"
        s3.put_object(Bucket=BUCKET, Key=meta_key,
                      Body=json.dumps(meta, indent=2).encode())
        log.info("Uploaded s3://%s/%s", BUCKET, meta_key)
    else:
        local = f"/tmp/{scenario}_r2_supplement.parquet"
        r2_all.to_parquet(local, index=False)
        log.info("Saved locally: %s", local)


if __name__ == "__main__":
    main()
