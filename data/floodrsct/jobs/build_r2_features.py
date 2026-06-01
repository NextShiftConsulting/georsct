#!/usr/bin/env python3
"""
build_r2_features.py -- SageMaker job: compute R2 temporal event features.

Reads existing hourly MRMS grib2 files and tide gauge parquets from S3.
Produces per-ZCTA temporal statistics for the R2 representation bundle.

R2 features (all event_window temporal class):
  MRMS observed rainfall:
  - peak_1h_mm, peak_3h_mm, peak_6h_mm: rolling-max rainfall at ZCTA centroid
  - storm_duration_h: hours where rainfall exceeds 1 mm
  - time_to_peak_h: hours from first rain to peak rain
  - rainfall_intensity_cv: CV of hourly rainfall during storm hours
  HRRR QPF (forecast rainfall):
  - hrrr_qpf_total_mm: cumulative HRRR forecast precip at ZCTA centroid
  - hrrr_qpf_peak_mm: max single-init HRRR QPF
  - hrrr_forecast_bias_mm: QPF total minus MRMS observed total
  Tide/surge:
  - tide_peak_m: max observed water level from nearest tide station
  - surge_rain_lag_h: hours between peak rainfall and peak surge
  Storm dynamics:
  - storm_approach_speed_kph: TC forward speed at closest approach (HURDAT2)

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
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, load_processed_parquet

CODE_DIR = Path("/opt/ml/processing/input/code")

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
    "riverside_coachella": ["hilary2023", "ar_flood_2023"],
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
                          centroid_lons: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Build per-ZCTA temporal rainfall features from hourly MRMS grib2 files.

    Returns:
        (df, peak_rain_hours): DataFrame of temporal stats and dict mapping
        zcta_id -> datetime of peak rainfall hour (for surge-rain lag).
    """
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
        return _empty_r2(zcta_ids), {}

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
        return _empty_r2(zcta_ids), {}

    # Compute temporal features from the rainfall matrix
    # Replace NaN columns (missing hours) with 0 for rolling computations
    valid_mask = ~np.all(np.isnan(rainfall_matrix), axis=0)
    log.info("Valid hours: %d/%d", valid_mask.sum(), n_hours)

    results = []
    for i in range(n_zctas):
        ts = rainfall_matrix[i, :]
        results.append(_compute_temporal_stats(ts, zcta_ids[i]))

    df = pd.DataFrame(results)

    # Build peak-rain-hour lookup: zcta_id -> datetime of peak rainfall
    peak_rain_hours = {}
    for i, zcta in enumerate(zcta_ids):
        h = np.nan_to_num(rainfall_matrix[i, :], nan=0.0)
        if np.any(h > RAIN_THRESHOLD_MM):
            peak_idx = int(np.argmax(h))
            if peak_idx < len(timestamps):
                peak_rain_hours[zcta] = timestamps[peak_idx]

    log.info("R2 temporal features computed for event %s: %d ZCTAs, %d with peak rain time",
             event, len(df), len(peak_rain_hours))
    return df, peak_rain_hours


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


def parse_hrrr_timestamp(filename: str) -> datetime:
    """Extract datetime from HRRR filename like hrrr.20170817.t00z.wrfsfcf01.grib2"""
    match = re.search(r"hrrr\.(\d{8})\.t(\d{2})z", filename)
    if not match:
        return None
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H")


def _decode_one_hrrr(args: tuple):
    """Worker: download + decode one HRRR grib2 file.

    HRRR grib2 files are ~100 MB with hundreds of variables. We filter to
    APCP (accumulated precipitation) only via cfgrib backend_kwargs.
    Returns (precip_2d, lat_2d, lon_2d) or None.
    """
    key, bucket = args
    import boto3
    s3w = boto3.client("s3", region_name="us-east-1")

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        s3w.download_fileobj(bucket, key, f)
        f.flush()
        tmp_path = f.name

    try:
        import cfgrib
        # Filter to APCP only — avoids indexing all ~100+ messages
        ds = cfgrib.open_dataset(
            tmp_path, indexpath=None,
            backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
        )
        precip_var = next(
            (v for v in ["tp", "unknown", "apcp", "APCP"] if v in ds), None
        )
        if precip_var is None:
            # Fallback: try without filter
            ds = cfgrib.open_dataset(tmp_path, indexpath=None)
            precip_var = next(
                (v for v in ["tp", "unknown", "apcp", "APCP"] if v in ds), None
            )
            if precip_var is None:
                return None
        return (ds[precip_var].values, ds["latitude"].values, ds["longitude"].values)
    except Exception:
        return None
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


def compute_hrrr_temporal(s3, event: str, zcta_ids: list[str],
                          centroid_lats: np.ndarray,
                          centroid_lons: np.ndarray,
                          mrms_total: dict | None = None) -> pd.DataFrame:
    """Build per-ZCTA HRRR QPF features from 6-hourly grib2 files.

    Args:
        s3: boto3 S3 client.
        event: Event name (e.g. 'harvey2017').
        zcta_ids: List of ZCTA IDs.
        centroid_lats, centroid_lons: ZCTA centroid coordinates.
        mrms_total: Optional dict of zcta_id -> MRMS total_rainfall_mm
            for computing forecast bias. If None, bias column is NaN.

    Returns:
        DataFrame with columns: zcta_id, hrrr_qpf_total_mm,
        hrrr_qpf_peak_mm, hrrr_forecast_bias_mm.
    """
    prefix = f"raw/noaa_hrrr/{event}/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
    all_keys = [o["Key"] for o in resp.get("Contents", [])
                if o["Key"].endswith(".grib2")]
    while resp.get("IsTruncated"):
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000,
                                  ContinuationToken=resp["NextContinuationToken"])
        all_keys.extend(o["Key"] for o in resp.get("Contents", [])
                        if o["Key"].endswith(".grib2"))

    if not all_keys:
        log.warning("No HRRR files for event %s", event)
        return _empty_hrrr(zcta_ids)

    # Sort by timestamp
    keyed = []
    for k in all_keys:
        ts = parse_hrrr_timestamp(k.split("/")[-1])
        if ts:
            keyed.append((ts, k))
    keyed.sort(key=lambda x: x[0])
    sorted_keys = [k for _, k in keyed]
    n_inits = len(sorted_keys)
    n_zctas = len(zcta_ids)
    log.info("HRRR QPF: event=%s, %d init times, %d ZCTAs", event, n_inits, n_zctas)

    # Decode in parallel — HRRR files are large (~100 MB) so limit workers
    n_workers = min(os.cpu_count() // 2 or 2, 4)
    grid_index = None
    qpf_matrix = np.full((n_zctas, n_inits), np.nan, dtype=np.float32)

    work_items = [(k, BUCKET) for k in sorted_keys]
    key_to_idx = {k: i for i, k in enumerate(sorted_keys)}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_decode_one_hrrr, item): item for item in work_items}
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            if done_count % 10 == 0:
                log.info("HRRR decode progress: %d/%d", done_count, n_inits)
                sys.stdout.flush()

            result = fut.result()
            if result is None:
                continue

            arr, lat, lon = result
            key_used = futures[fut][0]
            init_idx = key_to_idx[key_used]

            if grid_index is None:
                grid_index = build_centroid_index(lat, lon, centroid_lats, centroid_lons)
                log.info("HRRR grid index built: %d centroids mapped", len(grid_index))

            flat_vals = arr.flatten()
            qpf_vals = flat_vals[grid_index]
            qpf_matrix[:, init_idx] = np.where(np.isnan(qpf_vals), 0.0, qpf_vals)

    if grid_index is None:
        log.error("No HRRR files decoded successfully for %s", event)
        return _empty_hrrr(zcta_ids)

    valid_inits = (~np.all(np.isnan(qpf_matrix), axis=0)).sum()
    log.info("HRRR valid init times: %d/%d", valid_inits, n_inits)

    # Compute per-ZCTA QPF stats
    results = []
    for i in range(n_zctas):
        h = np.nan_to_num(qpf_matrix[i, :], nan=0.0)
        total = float(np.sum(h))
        peak = float(np.max(h)) if len(h) > 0 else np.nan

        bias = np.nan
        if mrms_total and zcta_ids[i] in mrms_total:
            obs = mrms_total[zcta_ids[i]]
            if not np.isnan(obs):
                bias = total - obs

        results.append({
            "zcta_id": zcta_ids[i],
            "hrrr_qpf_total_mm": total,
            "hrrr_qpf_peak_mm": peak,
            "hrrr_forecast_bias_mm": bias,
        })

    df = pd.DataFrame(results)
    log.info("HRRR QPF features computed for event %s: %d ZCTAs", event, len(df))
    return df


def _empty_hrrr(zcta_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "zcta_id": zcta_ids,
        "hrrr_qpf_total_mm": np.nan,
        "hrrr_qpf_peak_mm": np.nan,
        "hrrr_forecast_bias_mm": np.nan,
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


def compute_storm_approach_speed(
    s3, event: str, storm_id: str,
    peak_window: tuple[str, str],
    zcta_ids: list[str],
    centroid_lats: np.ndarray, centroid_lons: np.ndarray,
) -> pd.DataFrame:
    """Compute TC forward speed (km/h) at closest approach to scenario centroid.

    Uses consecutive HURDAT2 6-hourly fixes. Forward speed = haversine distance
    between fixes / time delta. Returns the speed at the fix closest to the
    mean scenario centroid during the peak window.

    Returns DataFrame: zcta_id, storm_approach_speed_kph (same value for all ZCTAs
    in a scenario — this is a storm-level, not ZCTA-level, feature).
    """
    empty = pd.DataFrame({"zcta_id": zcta_ids, "storm_approach_speed_kph": np.nan})

    if not storm_id:
        return empty

    obj = s3.get_object(Bucket=BUCKET, Key="raw/hurdat2/storm_tracks.parquet")
    hurdat = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    if hurdat.empty:
        return empty

    # Filter to this storm + peak window
    ts_col = hurdat["timestamp"]
    if ts_col.dt.tz is None:
        ts_col = ts_col.dt.tz_localize("UTC")
    mask = (
        (hurdat["storm_id"] == storm_id) &
        (ts_col >= pd.Timestamp(peak_window[0], tz="UTC")) &
        (ts_col <= pd.Timestamp(peak_window[1], tz="UTC"))
    )
    track = hurdat[mask].sort_values("timestamp").reset_index(drop=True)
    if len(track) < 2:
        log.warning("storm_approach_speed: < 2 fixes for %s in peak window", storm_id)
        return empty

    # Scenario centroid (mean of all ZCTA centroids)
    sc_lat = float(np.nanmean(centroid_lats))
    sc_lon = float(np.nanmean(centroid_lons))

    # Find the fix closest to scenario centroid
    R = 6371.0
    best_idx = 0
    best_dist = np.inf
    for i, row in track.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue
        dlat = np.radians(row["lat"] - sc_lat)
        dlon = np.radians(row["lon"] - sc_lon)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(sc_lat)) * np.cos(np.radians(row["lat"])) *
             np.sin(dlon / 2) ** 2)
        d = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        if d < best_dist:
            best_dist = d
            best_idx = i

    # Compute forward speed between closest fix and adjacent fix
    # Use the segment ending at closest fix (or starting, if closest is first)
    if best_idx == 0:
        idx_a, idx_b = 0, 1
    else:
        idx_a, idx_b = best_idx - 1, best_idx

    fix_a = track.loc[idx_a]
    fix_b = track.loc[idx_b]
    if pd.isna(fix_a["lat"]) or pd.isna(fix_b["lat"]):
        return empty

    # Haversine between consecutive fixes
    dlat = np.radians(fix_b["lat"] - fix_a["lat"])
    dlon = np.radians(fix_b["lon"] - fix_a["lon"])
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(fix_a["lat"])) * np.cos(np.radians(fix_b["lat"])) *
         np.sin(dlon / 2) ** 2)
    dist_km = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Time delta in hours
    t_a = pd.Timestamp(fix_a["timestamp"])
    t_b = pd.Timestamp(fix_b["timestamp"])
    if t_a.tz is None:
        t_a = t_a.tz_localize("UTC")
    if t_b.tz is None:
        t_b = t_b.tz_localize("UTC")
    dt_hours = (t_b - t_a).total_seconds() / 3600
    if dt_hours <= 0:
        return empty

    speed_kph = float(dist_km / dt_hours)
    log.info("storm_approach_speed: %s closest at %.0f km, speed=%.1f kph",
             storm_id, best_dist, speed_kph)

    return pd.DataFrame({
        "zcta_id": zcta_ids,
        "storm_approach_speed_kph": speed_kph,
    })


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

    # Load scenario config for storm IDs
    cfg_path = CODE_DIR / f"{scenario}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg_events = cfg.get("events", {})

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

        # MRMS temporal features + peak rain timestamps for surge-rain lag
        mrms, peak_rain_hour = compute_mrms_temporal(s3, event, zcta_order, clats, clons)

        # HRRR QPF features (needs MRMS totals for forecast bias)
        mrms_total_lookup = dict(zip(mrms["zcta_id"], mrms["peak_1h_mm"]))
        # Use existing total_rainfall_mm from assembled parquet if available
        event_rows = df[df.get("event", pd.Series()) == event] if "event" in df.columns else pd.DataFrame()
        if not event_rows.empty and "total_rainfall_mm" in event_rows.columns:
            mrms_total_lookup = dict(zip(event_rows["zcta_id"], event_rows["total_rainfall_mm"]))
        hrrr = compute_hrrr_temporal(s3, event, zcta_order, clats, clons, mrms_total_lookup)

        # Tide gauge features
        tide = compute_tide_features(s3, event, zcta_order, clats, clons, peak_rain_hour)

        # Storm approach speed from HURDAT2
        # Map S3 event key back to config event name (e.g., ida2021_nyc -> ida2021)
        cfg_event_key = next(
            (k for k in cfg_events if event.startswith(k)), None
        )
        if cfg_event_key and cfg_events[cfg_event_key].get("nhc_storm_id"):
            ev_cfg = cfg_events[cfg_event_key]
            storm_speed = compute_storm_approach_speed(
                s3, event, ev_cfg["nhc_storm_id"],
                tuple(ev_cfg["peak_window"]),
                zcta_order, clats, clons,
            )
        else:
            storm_speed = pd.DataFrame({
                "zcta_id": zcta_order, "storm_approach_speed_kph": np.nan,
            })

        # Merge MRMS + HRRR + tide + storm speed
        r2 = mrms.merge(hrrr, on="zcta_id", how="outer")
        r2 = r2.merge(tide, on="zcta_id", how="outer")
        r2 = r2.merge(storm_speed, on="zcta_id", how="outer")
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
