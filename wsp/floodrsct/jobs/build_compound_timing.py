"""
build_compound_timing.py -- Compound flooding temporal features for R2.

Derives event-timing features from MRMS rainfall, NOAA tides, and HURDAT2
storm tracks. These capture the temporal dynamics of compound flooding:
  - Rainfall intensity, duration, peak timing
  - Surge duration, peak timing
  - Rain-surge overlap window
  - Lag between peak rain and peak tide
  - Compound flooding indicator

Design inspired by QGIS Rain to Flood Analysis plugin and Temporal Controller
patterns. Event timing is a layer, not a footnote.

Usage:
    python build_compound_timing.py --scenario houston --upload
    python build_compound_timing.py --all-scenarios --upload
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client

MODELABLE = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]


# ---------------------------------------------------------------------------
# Rainfall features from MRMS event summaries
# ---------------------------------------------------------------------------

def compute_rainfall_features(mrms_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ZCTA rainfall timing features from MRMS time series.

    Expected columns: zcta_id, event, timestamp, rainfall_mm_hr

    Returns DataFrame with columns:
        rain_peak_intensity_mm_hr: Maximum hourly rainfall rate
        rain_total_mm: Cumulative rainfall over event
        rain_duration_hr: Hours with rainfall > 1 mm/hr
        rain_peak_hour: Hour offset of peak rainfall from event start
        rain_p90_intensity: 90th percentile hourly intensity
    """
    if mrms_df.empty:
        return pd.DataFrame()

    mrms_df = mrms_df.copy()
    mrms_df["timestamp"] = pd.to_datetime(mrms_df["timestamp"])

    results = []
    for (zcta, event), group in mrms_df.groupby(["zcta_id", "event"]):
        group = group.sort_values("timestamp")
        rain = group["rainfall_mm_hr"].values
        ts = group["timestamp"]

        event_start = ts.min()
        peak_idx = np.argmax(rain)
        peak_time = ts.iloc[peak_idx]

        results.append({
            "zcta_id": zcta,
            "event": event,
            "rain_peak_intensity_mm_hr": float(np.max(rain)),
            "rain_total_mm": float(np.sum(rain)),
            "rain_duration_hr": float(np.sum(rain > 1.0)),
            "rain_peak_hour": float((peak_time - event_start).total_seconds() / 3600),
            "rain_p90_intensity": float(np.percentile(rain[rain > 0], 90)) if (rain > 0).any() else 0.0,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Surge features from tide gauge data
# ---------------------------------------------------------------------------

def compute_surge_features(tides_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ZCTA surge timing features from tide observations.

    Expected columns: zcta_id, event, timestamp, observed_m, predicted_m

    Returns DataFrame with columns:
        surge_max_m: Maximum observed - predicted water level
        surge_duration_hr: Hours where surge > 0.3m above predicted
        surge_peak_hour: Hour offset of peak surge from event start
        tide_range_m: Tidal range (max - min predicted) during event
    """
    if tides_df.empty:
        return pd.DataFrame()

    tides_df = tides_df.copy()
    tides_df["timestamp"] = pd.to_datetime(tides_df["timestamp"])
    tides_df["surge_m"] = tides_df["observed_m"] - tides_df["predicted_m"]

    results = []
    for (zcta, event), group in tides_df.groupby(["zcta_id", "event"]):
        group = group.sort_values("timestamp")
        surge = group["surge_m"].values
        predicted = group["predicted_m"].values
        ts = group["timestamp"]

        event_start = ts.min()
        peak_idx = np.argmax(surge)
        peak_time = ts.iloc[peak_idx]

        results.append({
            "zcta_id": zcta,
            "event": event,
            "surge_max_m": float(np.max(surge)),
            "surge_duration_hr": float(np.sum(surge > 0.3)),
            "surge_peak_hour": float((peak_time - event_start).total_seconds() / 3600),
            "tide_range_m": float(np.ptp(predicted)),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Compound flooding features
# ---------------------------------------------------------------------------

def compute_compound_features(
    rain_df: pd.DataFrame, surge_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge rainfall and surge features and compute compound indicators.

    Returns DataFrame with additional columns:
        rain_surge_lag_hr: Abs difference between rain peak and surge peak
        rain_surge_overlap_hr: Estimated hours of simultaneous rain + surge
        compound_score: Combined intensity metric (rain * surge interaction)
    """
    if rain_df.empty or surge_df.empty:
        return rain_df if not rain_df.empty else surge_df

    merged = rain_df.merge(surge_df, on=["zcta_id", "event"], how="outer")

    # Lag: absolute time difference between peak rain and peak surge
    merged["rain_surge_lag_hr"] = np.abs(
        merged["rain_peak_hour"].fillna(0) - merged["surge_peak_hour"].fillna(0)
    )

    # Overlap: estimate concurrent hours (rain duration intersected with surge)
    # Simplified: min(rain_duration, surge_duration) scaled by temporal proximity
    rain_dur = merged["rain_duration_hr"].fillna(0)
    surge_dur = merged["surge_duration_hr"].fillna(0)
    lag = merged["rain_surge_lag_hr"]
    overlap = np.clip(np.minimum(rain_dur, surge_dur) - lag, 0, None)
    merged["rain_surge_overlap_hr"] = overlap

    # Compound score: interaction term (both high = compound event)
    rain_norm = merged["rain_total_mm"].fillna(0) / max(merged["rain_total_mm"].max(), 1)
    surge_norm = merged["surge_max_m"].fillna(0) / max(merged["surge_max_m"].max(), 0.01)
    merged["compound_score"] = rain_norm * surge_norm

    return merged


# ---------------------------------------------------------------------------
# Storm proximity features from HURDAT2
# ---------------------------------------------------------------------------

def compute_storm_features(
    storm_df: pd.DataFrame, zcta_centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Compute storm timing features from HURDAT2 tracks.

    Expected storm_df columns: storm_id, event, timestamp, lat, lon,
        max_wind_kt, status
    Expected zcta_centroids columns: zcta_id, event, latitude, longitude

    Returns DataFrame with columns:
        storm_closest_km: Minimum distance to storm track
        storm_closest_hour: Hour of closest approach from event start
        storm_max_wind_kt: Maximum sustained wind at closest approach
        storm_duration_hr: Total storm duration in study area
    """
    if storm_df.empty or zcta_centroids.empty:
        return pd.DataFrame()

    storm_df = storm_df.copy()
    storm_df["timestamp"] = pd.to_datetime(storm_df["timestamp"])

    results = []
    for event in storm_df["event"].unique():
        event_storms = storm_df[storm_df["event"] == event]
        event_zctas = zcta_centroids[zcta_centroids["event"] == event]

        if event_storms.empty or event_zctas.empty:
            continue

        event_start = event_storms["timestamp"].min()

        for _, zcta in event_zctas.iterrows():
            zcta_lat, zcta_lon = zcta["latitude"], zcta["longitude"]

            # Approximate distance in km (Haversine-like)
            dlat = (event_storms["lat"].values - zcta_lat) * 110.57
            dlon = (event_storms["lon"].values - zcta_lon) * 111.32 * np.cos(np.radians(zcta_lat))
            dists = np.sqrt(dlat**2 + dlon**2)

            closest_idx = np.argmin(dists)
            closest_time = event_storms["timestamp"].iloc[closest_idx]

            results.append({
                "zcta_id": zcta["zcta_id"],
                "event": event,
                "storm_closest_km": float(dists[closest_idx]),
                "storm_closest_hour": float((closest_time - event_start).total_seconds() / 3600),
                "storm_max_wind_kt": float(event_storms.iloc[closest_idx]["max_wind_kt"]),
                "storm_duration_hr": float(
                    (event_storms["timestamp"].max() - event_storms["timestamp"].min()).total_seconds() / 3600
                ),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# S3 data loading
# ---------------------------------------------------------------------------

def _load_parquet(s3, key: str) -> pd.DataFrame:
    """Load a parquet from S3, return empty DataFrame if not found."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        log.info("Loaded %d rows from %s", len(df), key)
        return df
    except s3.exceptions.ClientError:
        log.warning("Not found: %s", key)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_r2_timing(s3, scenario: str, upload: bool = False) -> dict:
    """Build R2 compound timing features for one scenario."""
    # Load MRMS event rainfall summaries
    mrms_df = _load_parquet(s3, f"processed/{scenario}/{scenario}_mrms_event.parquet")
    rain_features = compute_rainfall_features(mrms_df)
    log.info("Rainfall features: %d rows", len(rain_features))

    # Load tide observations
    tides_df = _load_parquet(s3, f"processed/{scenario}/{scenario}_tides_event.parquet")
    surge_features = compute_surge_features(tides_df)
    log.info("Surge features: %d rows", len(surge_features))

    # Compound features
    compound = compute_compound_features(rain_features, surge_features)
    log.info("Compound features: %d rows, %d columns", len(compound), len(compound.columns))

    # Storm proximity (if HURDAT2 available)
    storm_df = _load_parquet(s3, "processed/hurdat2/storm_tracks_events.parquet")
    if not storm_df.empty:
        centroids = _load_parquet(s3, f"processed/{scenario}/{scenario}_zcta_centroids.parquet")
        storm_features = compute_storm_features(storm_df, centroids)
        if not storm_features.empty:
            compound = compound.merge(storm_features, on=["zcta_id", "event"], how="left")
            log.info("Added storm features: %d columns total", len(compound.columns))

    summary = {
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(compound),
        "n_columns": len(compound.columns),
        "columns": list(compound.columns),
        "status": "COMPLETE" if not compound.empty else "NO_DATA",
    }

    if upload and not compound.empty:
        buf = io.BytesIO()
        compound.to_parquet(buf, compression="zstd", index=False)
        key = f"processed/{scenario}/{scenario}_r2_compound_timing.parquet"
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        log.info("Uploaded %s (%d rows)", key, len(compound))
        summary["s3_key"] = f"s3://{BUCKET}/{key}"

        json_key = f"processed/{scenario}/{scenario}_r2_compound_timing_meta.json"
        s3.put_object(
            Bucket=BUCKET, Key=json_key,
            Body=json.dumps(summary, indent=2).encode(),
            ContentType="application/json",
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build R2 compound timing features")
    parser.add_argument("--scenario", choices=MODELABLE)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    if not args.scenario and not args.all_scenarios:
        parser.error("specify --scenario or --all-scenarios")

    s3 = get_s3_client()
    scenarios = MODELABLE if args.all_scenarios else [args.scenario]

    for scenario in scenarios:
        log.info("=== Compound timing: %s ===", scenario)
        result = build_r2_timing(s3, scenario, args.upload)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
