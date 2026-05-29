#!/usr/bin/env python3
"""
build_event_dataset.py -- SageMaker job: assemble per-scenario (unit, event) tables.

Joins all raw pulls into a single analysis-ready parquet per scenario.
Output schema has four layers:

  Layer 1 inputs:  raw hydro + impact measurements per (unit, event)
  Layer 2 static:  geocertdb2026 ZCTA features (ACS, SVI, flood zones, TWI)
  Layer 2 reserved: certificate slots (null; filled by experiment scripts)
  Observability:   data quality / sensor coverage flags per (unit, event)

Layer 3 (retrieval grounding) and Layer 4 (rationale) are NOT populated here.
They are added by experiment-time scripts on top of this table.

Outputs:
  s3://swarm-floodrsct-data/processed/houston/houston_event_features.parquet
  s3://swarm-floodrsct-data/processed/new_orleans/no_event_features.parquet
  s3://swarm-floodrsct-data/processed/nyc/nyc_event_features.parquet
  s3://swarm-floodrsct-data/processed/riverside_coachella/rc_event_features.parquet
  s3://swarm-floodrsct-data/processed/southwest_florida/swfl_event_features.parquet

Usage:
    python build_event_dataset.py --scenario houston
    python build_event_dataset.py --scenario new_orleans
    python build_event_dataset.py --scenario nyc
    python build_event_dataset.py --scenario riverside_coachella
    python build_event_dataset.py --scenario southwest_florida
"""

import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
CONFIG_DIR = Path("/opt/ml/processing/input/config")

# MRMS requires cfgrib + geopandas for spatial aggregation
# If not available, rainfall_total_mm will be NaN (flagged in obs_mrms_coverage_pct)
try:
    import cfgrib
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    log.warning("cfgrib or geopandas not available; MRMS spatial aggregation disabled")


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

CERTIFICATE_SLOTS = {
    # Layer 1 — upstream model output (filled by experiment or surrogate job)
    "pred_risk_score": float,
    "pred_model_id": str,
    # Layer 2 — RSCT certificate fields (filled by experiment scripts)
    "cert_r": float,
    "cert_s": float,
    "cert_n": float,
    "cert_kappa": float,
    "cert_action": str,         # Trust / Review / Escalate / Suppress
    "cert_geometry_flags": str, # JSON string
    # Layer 3 — retrieval grounding (filled by evidence registry scripts)
    "retrieval_grounding_score": float,
    "retrieval_tier1_count": float,
    "retrieval_tier2_count": float,
    "retrieval_tier3_count": float,
    # Layer 4 — rationale (filled by rationale generation scripts)
    "rationale_text": str,
    "rationale_acceptability": float,
}

OBSERVABILITY_COLS = [
    "obs_gauge_count",         # USGS gauges within/near unit
    "obs_gauge_distance_km",   # distance to nearest active gauge
    "obs_mrms_coverage_pct",   # fraction of event-window hours present in S3
    "obs_has_hwm",             # USGS STN marks exist for this unit
    "obs_has_311",             # 311 reports exist during event window
    "obs_nfip_event_claims",   # event-specific DR-filtered claim count
    "obs_feature_modal_frac",  # fraction flood-modal vs proxy features (alpha input)
    "obs_missing_sensor_flag", # True if any primary sensor source missing
]


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_read(s3, key: str) -> Optional[pd.DataFrame]:
    """Read a parquet from swarm-floodrsct-data; return None if missing."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        log.warning("S3 key not found: %s", key)
        return None
    except Exception as e:
        log.warning("Could not read %s: %s", key, e)
        return None


def s3_upload(df: pd.DataFrame, key: str, s3) -> None:
    local = f"/tmp/{Path(key).name}"
    df.to_parquet(local, index=False)
    s3.upload_file(local, BUCKET, key)
    log.info("Uploaded %d rows x %d cols to s3://%s/%s", len(df), len(df.columns), BUCKET, key)


# ---------------------------------------------------------------------------
# Geocertdb2026 static features
# ---------------------------------------------------------------------------

def load_geocert_static(s3, scenario: str, zcta_ids: list[str]) -> pd.DataFrame:
    """Load static ZCTA features from geocertdb2026 and filter to scenario ZCTAs."""
    key = f"raw/geocertdb2026/scenarios/{scenario}/zcta_features_labels.parquet"
    df = s3_read(s3, key)
    if df is None:
        # Fallback: load national and filter
        log.warning("Scenario subset not found; loading national and filtering")
        df = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if df is None:
        log.error("geocertdb2026 not available — run copy_geocertdb2026 job first")
        return pd.DataFrame()
    zcta_col = next((c for c in df.columns if "zcta" in c.lower()), None)
    if zcta_col:
        df = df[df[zcta_col].isin(zcta_ids)].copy()
        df = df.rename(columns={zcta_col: "zcta_id"}) if zcta_col != "zcta_id" else df
    log.info("Geocert static: %d ZCTAs loaded", len(df))
    return df


# ---------------------------------------------------------------------------
# NWIS gauge features
# ---------------------------------------------------------------------------

def aggregate_nwis(s3, scenario: str, event: str,
                   zcta_ids: list[str], cfg: dict) -> pd.DataFrame:
    """
    Load NWIS timeseries, compute peak stage and flow per event,
    assign each gauge to its nearest ZCTA using lat/lon from crosswalk.
    Returns DataFrame indexed by zcta_id with peak_stage_ft, peak_flow_cfs.
    """
    key = f"raw/usgs_nwis/{scenario}_{event}.parquet"
    nwis = s3_read(s3, key)
    if nwis is None or nwis.empty:
        log.warning("No NWIS data for %s / %s", scenario, event)
        return _empty_nwis(zcta_ids)

    # Peak values per site
    peaks = (
        nwis.groupby("site_no")
        .agg(peak_stage_ft=("stage_ft", "max"), peak_flow_cfs=("flow_cfs", "max"))
        .reset_index()
    )

    # Load site metadata from NWIS crosswalk if available
    # Fallback: use anchor site lat/lon from config or assign to zcta centroid
    site_lat_lon = _load_site_coords(s3, scenario)
    if site_lat_lon is not None:
        peaks = peaks.merge(site_lat_lon, on="site_no", how="left")
        peaks = _assign_nearest_zcta(peaks, zcta_ids, s3)
    else:
        # Cannot spatial-assign without coords — flag as missing
        log.warning("Site coordinates unavailable; NWIS gauge-to-ZCTA assignment skipped")
        return _empty_nwis(zcta_ids)

    # Aggregate to ZCTA (some ZCTAs may have >1 gauge; take max)
    zcta_peaks = (
        peaks.groupby("zcta_id")
        .agg(peak_stage_ft=("peak_stage_ft", "max"),
             peak_flow_cfs=("peak_flow_cfs", "max"),
             obs_gauge_count=("site_no", "count"),
             obs_gauge_distance_km=("dist_km", "min"))
        .reset_index()
    )
    # Fill ZCTAs with no nearby gauge
    all_zctas = pd.DataFrame({"zcta_id": zcta_ids})
    result = all_zctas.merge(zcta_peaks, on="zcta_id", how="left")
    result["obs_gauge_count"] = result["obs_gauge_count"].fillna(0).astype(int)
    return result


def _load_site_coords(s3, scenario: str) -> Optional[pd.DataFrame]:
    """Try to load site coordinate table from S3."""
    key = f"raw/usgs_nwis/{scenario}_site_coords.parquet"
    return s3_read(s3, key)


def _assign_nearest_zcta(df: pd.DataFrame, zcta_ids: list[str],
                          s3) -> pd.DataFrame:
    """Add zcta_id and dist_km to each site row using haversine to ZCTA centroids."""
    centroids_key = "raw/geocertdb2026/zcta_features_labels.parquet"
    geo = s3_read(s3, centroids_key)
    if geo is None or "latitude" not in geo.columns:
        df["zcta_id"] = None
        df["dist_km"] = np.nan
        return df

    geo = geo[geo["zcta_id"].isin(zcta_ids)][["zcta_id", "latitude", "longitude"]].dropna()
    assignments = []
    for _, site in df.iterrows():
        if pd.isna(site.get("latitude")) or pd.isna(site.get("longitude")):
            assignments.append((None, np.nan))
            continue
        dlat = np.radians(geo["latitude"].values - site["latitude"])
        dlon = np.radians(geo["longitude"].values - site["longitude"])
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(site["latitude"])) *
             np.cos(np.radians(geo["latitude"].values)) *
             np.sin(dlon / 2) ** 2)
        dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        idx = dist.argmin()
        assignments.append((geo.iloc[idx]["zcta_id"], dist[idx]))
    df["zcta_id"] = [a[0] for a in assignments]
    df["dist_km"] = [a[1] for a in assignments]
    return df


def _empty_nwis(zcta_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "zcta_id": zcta_ids,
        "peak_stage_ft": np.nan,
        "peak_flow_cfs": np.nan,
        "obs_gauge_count": 0,
        "obs_gauge_distance_km": np.nan,
    })


# ---------------------------------------------------------------------------
# MRMS rainfall
# ---------------------------------------------------------------------------

def aggregate_mrms_rainfall(s3, event: str, zcta_ids: list[str]) -> pd.DataFrame:
    """
    List all grib2 files for the event, compute coverage %, then aggregate
    to ZCTA rainfall totals using spatial overlay (requires cfgrib + geopandas).

    Falls back to NaN + low coverage flag if libraries are unavailable.
    """
    prefix = f"raw/noaa_mrms/{event}/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    files = [o["Key"] for o in resp.get("Contents", [])
             if o["Key"].endswith(".grb2") or o["Key"].endswith(".grib2.gz")]
    log.info("MRMS: %d grib2 files for event %s", len(files), event)

    empty = pd.DataFrame({
        "zcta_id": zcta_ids,
        "rainfall_total_mm": np.nan,
        "obs_mrms_coverage_pct": 0.0,
    })

    if not files:
        log.warning("No MRMS files for event %s", event)
        return empty

    if not HAS_GEO:
        log.warning("cfgrib/geopandas not available; MRMS aggregation skipped")
        empty["obs_mrms_coverage_pct"] = 0.0
        return empty

    # Spatial aggregation — download files, read with cfgrib, overlay on ZCTA polygons
    try:
        return _mrms_spatial_aggregate(s3, files, zcta_ids, event)
    except Exception as e:
        log.error("MRMS spatial aggregation failed: %s", e)
        empty["obs_mrms_coverage_pct"] = 0.0
        return empty


def _mrms_spatial_aggregate(s3, file_keys: list[str], zcta_ids: list[str],
                             event: str) -> pd.DataFrame:
    """Download grib2 files, read precipitation values, overlay on ZCTA polygons."""
    import gzip
    import tempfile
    import geopandas as gpd
    import cfgrib

    # Download a sample to get grid coordinates, then accumulate
    hourly_arrays = []
    lat_arr = lon_arr = None

    for key in file_keys:
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f_gz:
            s3.download_fileobj(BUCKET, key, f_gz)
            f_gz.flush()
            gz_path = f_gz.name

        # Decompress .grib2.gz -> .grb2 for cfgrib/eccodes
        grb2_path = gz_path.replace(".gz", ".grb2")
        try:
            if key.endswith(".gz"):
                with gzip.open(gz_path, "rb") as gz_in:
                    with open(grb2_path, "wb") as raw_out:
                        raw_out.write(gz_in.read())
            else:
                grb2_path = gz_path  # already uncompressed

            ds = cfgrib.open_dataset(grb2_path, indexpath=None)
            precip_var = next(
                (v for v in ["tp", "unknown", "apcp", "APCP"] if v in ds),
                None,
            )
            if precip_var is None:
                continue
            arr = ds[precip_var].values  # 2D grid
            if lat_arr is None:
                lat_arr = ds["latitude"].values
                lon_arr = ds["longitude"].values
            hourly_arrays.append(arr)
        except Exception as e:
            log.debug("Could not read %s: %s", key, e)
        finally:
            for p in (gz_path, grb2_path):
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass

    if not hourly_arrays:
        return pd.DataFrame({"zcta_id": zcta_ids, "rainfall_total_mm": np.nan,
                             "obs_mrms_coverage_pct": 0.0})

    total_mm = np.nansum(hourly_arrays, axis=0)
    coverage = len(hourly_arrays) / max(len(file_keys), 1)

    # Flatten grid to points and assign to ZCTAs by nearest centroid
    flat_lat = lat_arr.flatten()
    flat_lon = lon_arr.flatten()
    flat_val = total_mm.flatten()

    # Load ZCTA centroids from geocertdb2026
    geo = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if geo is None or "latitude" not in geo.columns:
        return pd.DataFrame({"zcta_id": zcta_ids, "rainfall_total_mm": np.nan,
                             "obs_mrms_coverage_pct": coverage})

    geo = geo[geo["zcta_id"].isin(zcta_ids)][["zcta_id", "latitude", "longitude"]].dropna()

    # Bilinear-approximate: for each ZCTA centroid, find nearest grid point
    zcta_rainfall = []
    for _, row in geo.iterrows():
        dlat = flat_lat - row["latitude"]
        dlon = flat_lon - row["longitude"]
        dist = dlat ** 2 + dlon ** 2
        nearest = dist.argmin()
        zcta_rainfall.append({"zcta_id": row["zcta_id"],
                               "rainfall_total_mm": float(flat_val[nearest])})

    df = pd.DataFrame(zcta_rainfall)
    df["obs_mrms_coverage_pct"] = coverage
    # Fill ZCTAs with no MRMS match
    all_zctas = pd.DataFrame({"zcta_id": zcta_ids})
    return all_zctas.merge(df, on="zcta_id", how="left")


# ---------------------------------------------------------------------------
# STN high-water marks
# ---------------------------------------------------------------------------

def aggregate_hwm(s3, event: str, zcta_ids: list[str]) -> pd.DataFrame:
    # Check both locations: pre-uploaded S3 parquets and fetch_surge_hwm output
    hwm = s3_read(s3, f"raw/usgs_stn/{event}_hwm.parquet")
    if hwm is None or hwm.empty:
        hwm = s3_read(s3, f"raw/surge_estimates/{event}/hwm_{event}.parquet")
    empty = pd.DataFrame({"zcta_id": zcta_ids, "hwm_count": 0,
                           "hwm_max_elev_ft": np.nan, "obs_has_hwm": False})
    if hwm is None or hwm.empty:
        return empty

    # Assign HWMs to ZCTAs by nearest centroid (same haversine approach)
    geo = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if geo is None or "latitude" not in geo.columns:
        return empty

    geo = geo[geo["zcta_id"].isin(zcta_ids)][["zcta_id", "latitude", "longitude"]].dropna()
    hwm = hwm.dropna(subset=["latitude", "longitude"])

    assigned = []
    for _, mark in hwm.iterrows():
        dlat = np.radians(geo["latitude"].values - mark["latitude"])
        dlon = np.radians(geo["longitude"].values - mark["longitude"])
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(mark["latitude"])) *
             np.cos(np.radians(geo["latitude"].values)) *
             np.sin(dlon / 2) ** 2)
        dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        idx = dist.argmin()
        # Only assign to nearest ZCTA if within 5 km
        if dist[idx] <= 5.0:
            assigned.append({"zcta_id": geo.iloc[idx]["zcta_id"],
                             "elev_ft": mark.get("elev_ft", np.nan)})

    if not assigned:
        return empty

    adf = pd.DataFrame(assigned)
    agg = adf.groupby("zcta_id").agg(
        hwm_count=("elev_ft", "count"),
        hwm_max_elev_ft=("elev_ft", "max"),
    ).reset_index()
    agg["obs_has_hwm"] = True

    all_zctas = pd.DataFrame({"zcta_id": zcta_ids})
    result = all_zctas.merge(agg, on="zcta_id", how="left")
    result["hwm_count"] = result["hwm_count"].fillna(0).astype(int)
    result["obs_has_hwm"] = result["obs_has_hwm"].fillna(False)
    return result


# ---------------------------------------------------------------------------
# 311 reports (Houston / NYC)
# ---------------------------------------------------------------------------

def aggregate_311(s3, source: str, event: str, zcta_ids: list[str]) -> pd.DataFrame:
    """source: 'houston' or 'nyc'"""
    if source == "houston":
        key = f"raw/houston_311/{event}_311.parquet"
    else:
        key = f"raw/nyc_311/{event}_flooding_311.parquet"

    reports = s3_read(s3, key)
    empty = pd.DataFrame({"zcta_id": zcta_ids, "complaints_311_count": 0,
                           "obs_has_311": False})
    if reports is None or reports.empty:
        return empty

    if "zcta_id" not in reports.columns:
        return empty

    agg = (reports.groupby("zcta_id").size()
           .reset_index(name="complaints_311_count"))
    agg["obs_has_311"] = True

    all_zctas = pd.DataFrame({"zcta_id": zcta_ids})
    result = all_zctas.merge(agg, on="zcta_id", how="left")
    result["complaints_311_count"] = result["complaints_311_count"].fillna(0).astype(int)
    result["obs_has_311"] = result["obs_has_311"].fillna(False)
    return result


# ---------------------------------------------------------------------------
# Event-specific NFIP claims (DR-filtered)
# ---------------------------------------------------------------------------

def load_nfip_event_claims(s3, dr_number: int, zcta_ids: list[str]) -> pd.DataFrame:
    key = f"raw/openfema/nfip_claims_dr{dr_number}.parquet"
    claims = s3_read(s3, key)
    empty = pd.DataFrame({"zcta_id": zcta_ids,
                           "nfip_event_claim_count": 0,
                           "nfip_event_total_loss": 0.0,
                           "obs_nfip_event_claims": 0})
    if claims is None or claims.empty or "zcta_id" not in claims.columns:
        return empty

    claims["zcta_id"] = claims["zcta_id"].astype(str).str.zfill(5)
    loss_col = "amountPaidOnBuildingClaim"
    if loss_col not in claims.columns:
        log.warning("NFIP: %s not found in claims columns %s; total_loss will be 0",
                    loss_col, list(claims.columns))
        agg = claims.groupby("zcta_id").agg(
            nfip_event_claim_count=("zcta_id", "count"),
        ).reset_index()
        agg["nfip_event_total_loss"] = 0.0
    else:
        agg = claims.groupby("zcta_id").agg(
            nfip_event_claim_count=("zcta_id", "count"),
            nfip_event_total_loss=(loss_col, "sum"),
        ).reset_index()
    agg["obs_nfip_event_claims"] = agg["nfip_event_claim_count"]

    all_zctas = pd.DataFrame({"zcta_id": zcta_ids})
    result = all_zctas.merge(agg, on="zcta_id", how="left")
    for col in ["nfip_event_claim_count", "obs_nfip_event_claims"]:
        result[col] = result[col].fillna(0).astype(int)
    result["nfip_event_total_loss"] = result["nfip_event_total_loss"].fillna(0.0)
    return result


# ---------------------------------------------------------------------------
# HURDAT2 storm proximity
# ---------------------------------------------------------------------------

def compute_storm_proximity(s3, storm_id: str,
                             peak_window: tuple[str, str],
                             zcta_ids: list[str]) -> pd.DataFrame:
    """Compute distance from each ZCTA to storm track within peak window."""
    hurdat = s3_read(s3, "raw/hurdat2/storm_tracks.parquet")
    empty = pd.DataFrame({"zcta_id": zcta_ids, "storm_min_dist_km": np.nan,
                           "storm_landfall_category": np.nan})
    if hurdat is None or hurdat.empty:
        return empty

    # Localize timestamp if tz-naive (fetcher writes naive UTC datetimes)
    ts_col = hurdat["timestamp"]
    if ts_col.dt.tz is None:
        ts_col = ts_col.dt.tz_localize("UTC")
    track = hurdat[
        (hurdat["storm_id"] == storm_id) &
        (ts_col >= pd.Timestamp(peak_window[0], tz="UTC")) &
        (ts_col <= pd.Timestamp(peak_window[1], tz="UTC"))
    ]
    if track.empty:
        return empty

    geo = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if geo is None or "latitude" not in geo.columns:
        return empty

    geo = geo[geo["zcta_id"].isin(zcta_ids)][["zcta_id", "latitude", "longitude"]].dropna()
    # Derive Saffir-Simpson category from max_wind_kt (fetcher writes status + max_wind_kt, not category)
    def _saffir_simpson(kt):
        if pd.isna(kt):
            return np.nan
        if kt >= 137:
            return 5
        if kt >= 113:
            return 4
        if kt >= 96:
            return 3
        if kt >= 83:
            return 2
        if kt >= 64:
            return 1
        return 0  # tropical storm / depression
    if "max_wind_kt" in track.columns:
        landfall_cat = track["max_wind_kt"].apply(_saffir_simpson).max()
    else:
        landfall_cat = np.nan

    rows = []
    for _, zcta_row in geo.iterrows():
        min_dist = np.inf
        for _, fix in track.iterrows():
            if pd.isna(fix["lat"]) or pd.isna(fix["lon"]):
                continue
            dlat = np.radians(fix["lat"] - zcta_row["latitude"])
            dlon = np.radians(fix["lon"] - zcta_row["longitude"])
            a = (np.sin(dlat / 2) ** 2 +
                 np.cos(np.radians(zcta_row["latitude"])) *
                 np.cos(np.radians(fix["lat"])) *
                 np.sin(dlon / 2) ** 2)
            dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
            min_dist = min(min_dist, dist)
        rows.append({"zcta_id": zcta_row["zcta_id"], "storm_min_dist_km": min_dist})

    df = pd.DataFrame(rows)
    df["storm_landfall_category"] = landfall_cat
    all_zctas = pd.DataFrame({"zcta_id": zcta_ids})
    return all_zctas.merge(df, on="zcta_id", how="left")


# ---------------------------------------------------------------------------
# NOAA tides / SLOSH (New Orleans + SW Florida)
# ---------------------------------------------------------------------------

def aggregate_tides(s3, prefix_pattern: str, event: str,
                    zcta_ids: list[str]) -> pd.DataFrame:
    """
    Load all tide parquets matching prefix_pattern, compute peak surge per
    station, then average to ZCTA by nearest station.
    """
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix_pattern)
    keys = [o["Key"] for o in resp.get("Contents", [])
            if event in o["Key"] and o["Key"].endswith(".parquet")]
    empty = pd.DataFrame({"zcta_id": zcta_ids, "max_surge_m": np.nan,
                           "max_water_level_m": np.nan})
    if not keys:
        log.warning("No tide files matching %s / %s", prefix_pattern, event)
        return empty

    frames = []
    for key in keys:
        df = s3_read(s3, key)
        if df is not None and "observed_m" in df.columns:
            station_id = df["station_id"].iloc[0] if "station_id" in df.columns else key
            peak_wl = df["observed_m"].max()
            peak_surge = df["surge_m"].max() if "surge_m" in df.columns else np.nan
            frames.append({"station_id": station_id,
                           "max_water_level_m": peak_wl,
                           "max_surge_m": peak_surge})

    if not frames:
        return empty

    # Simple approach: assign all ZCTAs the max surge across all stations
    # A future version should do spatial interpolation
    peaks = pd.DataFrame(frames)
    max_surge = peaks["max_surge_m"].max()
    max_wl = peaks["max_water_level_m"].max()
    result = pd.DataFrame({"zcta_id": zcta_ids})
    result["max_surge_m"] = max_surge
    result["max_water_level_m"] = max_wl
    return result


# ---------------------------------------------------------------------------
# Observability composite
# ---------------------------------------------------------------------------

def compute_observability_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive obs_feature_modal_frac and obs_missing_sensor_flag."""
    # Modal fraction: ratio of flood-primary features to total non-null features
    flood_modal_cols = ["peak_stage_ft", "peak_flow_cfs", "rainfall_total_mm",
                        "hwm_count", "flood_pct_zone_a", "twi_twi", "max_surge_m"]
    proxy_cols = ["acs_pct_below_poverty", "acs_median_hh_income", "svi_overall",
                  "acs_total_pop", "acs_pct_renter_occupied"]
    all_feature_cols = flood_modal_cols + proxy_cols

    present_flood = sum(1 for c in flood_modal_cols if c in df.columns and df[c].notna().any())
    present_all = sum(1 for c in all_feature_cols if c in df.columns and df[c].notna().any())
    modal_frac = present_flood / max(present_all, 1)
    df["obs_feature_modal_frac"] = modal_frac

    # Missing sensor flag: True if both peak_stage_ft and rainfall_total_mm are NaN
    df["obs_missing_sensor_flag"] = (
        df.get("peak_stage_ft", pd.Series([np.nan] * len(df))).isna() &
        df.get("rainfall_total_mm", pd.Series([np.nan] * len(df))).isna()
    )
    return df


# ---------------------------------------------------------------------------
# Scenario-specific assemblers
# ---------------------------------------------------------------------------

def build_houston(s3, cfg: dict) -> pd.DataFrame:
    """Assemble Houston (zcta, event) table."""
    xwalk = s3_read(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
    harris_zctas = xwalk[xwalk["county_fips"] == "48201"]["zcta_id"].tolist()
    log.info("Houston: %d Harris County ZCTAs", len(harris_zctas))

    static = load_geocert_static(s3, "houston", harris_zctas)

    event_map = {
        "harvey2017": {"dr": 4332, "storm_id": "AL092017",
                       "peak_window": ("2017-08-25", "2017-09-02")},
        "imelda2019": {"dr": 4466, "storm_id": "AL132019",
                       "peak_window": ("2019-09-17", "2019-09-21")},
        "beryl2024":  {"dr": 4781, "storm_id": "AL022024",
                       "peak_window": ("2024-07-08", "2024-07-12")},
    }

    # Feature contract derived fields (computed once, shared across events)
    impervious = build_impervious_features(s3, harris_zctas)
    catchments = build_catchment_features(s3, harris_zctas, vpu="12")
    levees     = build_levee_features(s3, harris_zctas, "houston")
    # drainage_capacity_status: operational — not available from public archive
    drainage_op = _operational_unknown(
        pd.DataFrame({"zcta_id": harris_zctas}),
        "drainage_capacity_status",
        note="HCFCD operational pump/gate telemetry; no public historical archive.",
    )

    rows = []
    for event_name, ev in event_map.items():
        log.info("Houston event: %s", event_name)
        nwis  = aggregate_nwis(s3, "houston", event_name, harris_zctas, cfg)
        mrms  = aggregate_mrms_rainfall(s3, event_name, harris_zctas)
        hwm   = aggregate_hwm(s3, event_name, harris_zctas)
        s311  = aggregate_311(s3, "houston", event_name, harris_zctas)
        nfip  = load_nfip_event_claims(s3, ev["dr"], harris_zctas)
        storm = compute_storm_proximity(s3, ev["storm_id"], ev["peak_window"], harris_zctas)

        base = pd.DataFrame({"zcta_id": harris_zctas, "event": event_name,
                              "scenario": "houston"})
        for part in [nwis, mrms, hwm, s311, nfip, storm,
                     impervious, catchments, levees, drainage_op]:
            base = base.merge(part, on="zcta_id", how="left")
        if not static.empty:
            base = base.merge(static, on="zcta_id", how="left")
        base = compute_observability_flags(base)
        rows.append(base)

    df = pd.concat(rows, ignore_index=True)
    _add_certificate_slots(df)
    return df


def build_new_orleans(s3, cfg: dict) -> pd.DataFrame:
    """Assemble New Orleans (zcta, event) table."""
    xwalk = s3_read(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
    no_zctas = xwalk[xwalk["county_fips"] == "22071"]["zcta_id"].tolist()
    log.info("New Orleans: %d Orleans Parish ZCTAs", len(no_zctas))

    static = load_geocert_static(s3, "new_orleans", no_zctas)

    # Feature contract derived fields
    levee_feats = build_levee_features(s3, no_zctas, "new_orleans")
    elevation   = build_elevation_features(s3, no_zctas, "new_orleans")
    # pump_station_status: operational (Ida 2021 partially hand-coded)
    pump_evidence = _load_local_evidence("/opt/ml/processing/input/evidence/no_pump_stations_ida2021.csv")
    # pump_station_status for non-Ida events: operational_unknown
    pump_op = _operational_unknown(
        pd.DataFrame({"zcta_id": no_zctas}),
        "pump_station_status",
        note="S&WB pump telemetry not publicly archived; Ida 2021 partially available via evidence CSV.",
    )

    event_map = {
        "ida2021": {"dr": 4611, "storm_id": "AL092021",
                    "peak_window": ("2021-08-29", "2021-09-01")},
    }

    rows = []
    for event_name, ev in event_map.items():
        log.info("New Orleans event: %s", event_name)
        nwis   = aggregate_nwis(s3, "new_orleans", event_name, no_zctas, cfg)
        tides  = aggregate_tides(s3, "raw/noaa_tides/", event_name, no_zctas)
        nfip   = load_nfip_event_claims(s3, ev["dr"], no_zctas)
        storm  = compute_storm_proximity(s3, ev["storm_id"], ev["peak_window"], no_zctas)

        base = pd.DataFrame({"zcta_id": no_zctas, "event": event_name,
                              "scenario": "new_orleans"})
        for part in [nwis, tides, nfip, storm, levee_feats, elevation, pump_op]:
            base = base.merge(part, on="zcta_id", how="left")
        if not static.empty:
            base = base.merge(static, on="zcta_id", how="left")
        # Attach pump evidence for Ida 2021 (hand-coded override for pump_station_status)
        if event_name == "ida2021" and pump_evidence is not None and "pump_available" in pump_evidence.columns:
            ida_pump = pump_evidence[["district_id", "pump_available"]].rename(
                columns={"pump_available": "pump_available_ida"}
            )
            base = base.merge(ida_pump, left_on="zcta_id", right_on="district_id", how="left")
            # Override operational_unknown where hand-coded data exists
            has_data = base["pump_available_ida"].notna()
            base.loc[has_data, "pump_station_status"] = base.loc[has_data, "pump_available_ida"].map(
                {True: "operational", False: "degraded", 1: "operational", 0: "degraded"}
            ).fillna("unknown")
            base.loc[has_data, "_fs_pump_station_status"] = "present"
        base = compute_observability_flags(base)
        rows.append(base)

    df = pd.concat(rows, ignore_index=True)
    _add_certificate_slots(df)
    return df


def build_nyc(s3, cfg: dict) -> pd.DataFrame:
    """Assemble NYC (zcta, event) table."""
    xwalk = s3_read(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
    nyc_fips = ["36061", "36047", "36081", "36005", "36085"]
    nyc_zctas = xwalk[xwalk["county_fips"].isin(nyc_fips)]["zcta_id"].tolist()
    log.info("NYC: %d ZCTAs", len(nyc_zctas))

    static = load_geocert_static(s3, "nyc", nyc_zctas)

    # Feature contract derived fields
    impervious  = build_impervious_features(s3, nyc_zctas)
    sewer_feats = build_sewershed_features(s3, nyc_zctas)
    subway_feats = build_subway_features(s3, nyc_zctas)
    subway_evidence = _load_local_evidence("/opt/ml/processing/input/evidence/nyc_subway_flooding_ida2021.csv")

    event_map = {
        "ida2021":   {"dr": 4615, "storm_id": "AL092021",
                      "peak_window": ("2021-09-01", "2021-09-02")},
        "henri2021": {"dr": None, "storm_id": "AL082021",
                      "peak_window": ("2021-08-21", "2021-08-22")},
    }

    rows = []
    for event_name, ev in event_map.items():
        log.info("NYC event: %s", event_name)
        nwis  = aggregate_nwis(s3, "nyc", event_name, nyc_zctas, cfg)
        mrms  = aggregate_mrms_rainfall(s3, f"ida2021_nyc" if "ida" in event_name else event_name,
                                        nyc_zctas)
        s311  = aggregate_311(s3, "nyc", event_name, nyc_zctas)
        storm = compute_storm_proximity(s3, ev["storm_id"], ev["peak_window"], nyc_zctas)
        nfip  = (load_nfip_event_claims(s3, ev["dr"], nyc_zctas)
                 if ev["dr"] else pd.DataFrame({"zcta_id": nyc_zctas}))

        base = pd.DataFrame({"zcta_id": nyc_zctas, "event": event_name,
                              "scenario": "nyc"})
        for part in [nwis, mrms, s311, nfip, storm,
                     impervious, sewer_feats, subway_feats]:
            base = base.merge(part, on="zcta_id", how="left")
        if not static.empty:
            base = base.merge(static, on="zcta_id", how="left")
        # Subway flooding evidence (hand-coded Ida 2021 overlay)
        if event_name == "ida2021" and subway_evidence is not None and "flooding_observed" in subway_evidence.columns:
            flooded = subway_evidence[subway_evidence["flooding_observed"] == True]
            # TODO: spatial join flooded stations to ZCTAs when station coords available
            base["subway_flooded_count_nearby"] = 0  # placeholder until spatial join
        base = compute_observability_flags(base)
        rows.append(base)

    df = pd.concat(rows, ignore_index=True)
    _add_certificate_slots(df)
    return df


def build_riverside_coachella(s3, cfg: dict) -> pd.DataFrame:
    """Assemble Riverside-Coachella (zcta, event) variogram-input table."""
    xwalk = s3_read(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
    rc_fips = ["06065", "06025"]
    rc_zctas = xwalk[xwalk["county_fips"].isin(rc_fips)]["zcta_id"].tolist()
    log.info("Riverside-Coachella: %d ZCTAs", len(rc_zctas))

    static = load_geocert_static(s3, "riverside_coachella", rc_zctas)

    # Feature contract derived fields
    burn_scars  = build_burn_scar_features(s3, rc_zctas)
    catchments  = build_catchment_features(s3, rc_zctas, vpu="18")
    # road_access_status: operational — not available from public archive
    road_op = _operational_unknown(
        pd.DataFrame({"zcta_id": rc_zctas}),
        "road_access_status",
        note="CALTRANS real-time road closure feeds; no public historical archive at ZCTA granularity.",
    )

    event_map = {
        "hilary2023":    {"dr": 4699, "storm_id": "EP082023",
                          "peak_window": ("2023-08-20", "2023-08-21")},
        "ar_flood_2023": {"dr": None, "storm_id": None,
                          "peak_window": ("2023-03-10", "2023-03-15")},
    }

    rows = []
    for event_name, ev in event_map.items():
        log.info("Riverside-Coachella event: %s", event_name)
        nwis  = aggregate_nwis(s3, "riverside_coachella", event_name, rc_zctas, cfg)
        mrms  = aggregate_mrms_rainfall(s3, event_name, rc_zctas)
        nfip  = (load_nfip_event_claims(s3, ev["dr"], rc_zctas)
                 if ev["dr"] else pd.DataFrame({"zcta_id": rc_zctas}))
        storm = (compute_storm_proximity(s3, ev["storm_id"], ev["peak_window"], rc_zctas)
                 if ev["storm_id"] else pd.DataFrame({"zcta_id": rc_zctas}))

        base = pd.DataFrame({"zcta_id": rc_zctas, "event": event_name,
                              "scenario": "riverside_coachella"})
        for part in [nwis, mrms, nfip, storm, burn_scars, catchments, road_op]:
            base = base.merge(part, on="zcta_id", how="left")
        if not static.empty:
            base = base.merge(static, on="zcta_id", how="left")
        base = compute_observability_flags(base)
        rows.append(base)

    df = pd.concat(rows, ignore_index=True)
    _add_certificate_slots(df)
    return df


def build_southwest_florida(s3, cfg: dict) -> pd.DataFrame:
    """Assemble SW Florida (zcta, event) variogram-input table."""
    xwalk = s3_read(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
    swfl_fips = ["12021", "12071", "12115", "12081", "12057", "12103"]
    swfl_zctas = xwalk[xwalk["county_fips"].isin(swfl_fips)]["zcta_id"].tolist()
    log.info("SW Florida: %d ZCTAs", len(swfl_zctas))

    static = load_geocert_static(s3, "southwest_florida", swfl_zctas)

    # Feature contract derived fields
    elevation       = build_elevation_features(s3, swfl_zctas, "southwest_florida")
    coastal_dist    = build_coastal_distance_features(s3, swfl_zctas)
    levee_feats     = build_levee_features(s3, swfl_zctas, "southwest_florida")
    # evacuation_route_status: operational — not available from public archive
    evac_op = _operational_unknown(
        pd.DataFrame({"zcta_id": swfl_zctas}),
        "evacuation_route_status",
        note="FL county OES evacuation route status not publicly archived historically. "
             "Static evacuation zones exist (FL DEM) but route operational status is real-time only.",
    )

    event_map = {
        "ian2022":    {"dr": 4673, "storm_id": "AL092022",
                       "peak_window": ("2022-09-28", "2022-09-29")},
        "helene2024": {"dr": 4828, "storm_id": "AL092024",
                       "peak_window": ("2024-09-26", "2024-09-27")},
        "milton2024": {"dr": 4834, "storm_id": "AL142024",
                       "peak_window": ("2024-10-09", "2024-10-10")},
    }

    rows = []
    for event_name, ev in event_map.items():
        log.info("SW Florida event: %s", event_name)
        nwis  = aggregate_nwis(s3, "southwest_florida", event_name, swfl_zctas, cfg)
        mrms  = aggregate_mrms_rainfall(s3, event_name, swfl_zctas)
        tides = aggregate_tides(s3, "raw/noaa_tides/", event_name, swfl_zctas)
        nfip  = load_nfip_event_claims(s3, ev["dr"], swfl_zctas)
        storm = compute_storm_proximity(s3, ev["storm_id"], ev["peak_window"], swfl_zctas)

        slosh = build_slosh_features(s3, event_name, swfl_zctas)

        base = pd.DataFrame({"zcta_id": swfl_zctas, "event": event_name,
                              "scenario": "southwest_florida"})
        for part in [nwis, mrms, tides, nfip, storm,
                     slosh, elevation, coastal_dist, levee_feats, evac_op]:
            base = base.merge(part, on="zcta_id", how="left")
        if not static.empty:
            base = base.merge(static, on="zcta_id", how="left")
        base = compute_observability_flags(base)
        rows.append(base)

    df = pd.concat(rows, ignore_index=True)
    _add_certificate_slots(df)
    return df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_certificate_slots(df: pd.DataFrame) -> None:
    """Add null certificate + observability columns reserved for experiment scripts."""
    for col, dtype in CERTIFICATE_SLOTS.items():
        if col not in df.columns:
            df[col] = None if dtype == str else np.nan
    for col in OBSERVABILITY_COLS:
        if col not in df.columns:
            df[col] = np.nan


def _load_local_evidence(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        log.warning("Evidence file not found: %s", path)
        return None
    df = pd.read_csv(p, comment="#")
    if df.empty or df.columns.tolist() == ["district_id"]:
        log.warning("Evidence file is a template stub (no data rows): %s", path)
        return None
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Feature contract build functions
# Each function reads one raw dataset from S3 and returns a per-ZCTA DataFrame.
# All functions are idempotent and return NaN / operational_unknown columns
# with field_status when source data is unavailable.
# ---------------------------------------------------------------------------

# field_status sentinel values — match FEATURE_CONTRACT.yaml
_FS_MISSING = "missing_source_data"
_FS_OPERATIONAL = "operational_status_unavailable"
_OPERATIONAL_NOTE = (
    "Historical public archive does not provide real-time operational status for this field."
)


def _operational_unknown(df: pd.DataFrame, col: str, note: str = _OPERATIONAL_NOTE) -> pd.DataFrame:
    """Set col to 'unknown' with a field_status column."""
    df[col] = "unknown"
    df[f"_fs_{col}"] = _FS_OPERATIONAL
    df[f"_fs_{col}_reason"] = note
    return df


def build_impervious_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Derive impervious_pct per ZCTA from NLCD 2021 GeoTIFFs.

    Requires: raw/nlcd/impervious/v2021/*.tif + ZCTA centroid lat/lon from geocertdb2026.
    Returns DataFrame with columns: zcta_id, impervious_pct, _fs_impervious_pct.
    """
    empty = pd.DataFrame({"zcta_id": zcta_ids, "impervious_pct": np.nan,
                          "_fs_impervious_pct": _FS_MISSING})

    if not HAS_GEO:
        log.warning("build_impervious_features: geopandas/rasterio not available; returning NaN")
        return empty

    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import mapping, box
    except ImportError:
        log.warning("build_impervious_features: rasterio not available; returning NaN")
        return empty

    # Find NLCD raster: national .img file or per-state TIFs
    nlcd_key = None
    for prefix in ["raw/nlcd/impervious_2021/", "raw/nlcd/impervious/v2021/"]:
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        for o in resp.get("Contents", []):
            k = o["Key"]
            if k.endswith((".img", ".tif", ".tiff")):
                nlcd_key = k
                break
        if nlcd_key:
            break
    if not nlcd_key:
        log.warning("build_impervious_features: no NLCD raster found in S3; returning NaN")
        return empty

    log.info("build_impervious_features: using %s", nlcd_key)

    # Load ZCTA centroids from geocertdb2026
    static_key = "raw/geocertdb2026/zcta_features_labels.parquet"
    static = s3_read(s3, static_key)
    if static is None:
        return empty
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        log.warning("build_impervious_features: missing zcta/lat/lon in geocertdb2026; returning NaN")
        return empty

    static = static[[zcta_col, lat_col, lon_col]].rename(
        columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"}
    )
    centroids = static[static["zcta_id"].isin(zcta_ids)].dropna(subset=["lat", "lon"])

    # Download national raster (26 GB .img) -- stream to /tmp
    local_raster = f"/tmp/nlcd_{Path(nlcd_key).stem}{Path(nlcd_key).suffix}"
    log.info("build_impervious_features: downloading %s (this may take a few minutes)", nlcd_key)
    s3.download_file(BUCKET, nlcd_key, local_raster)

    # Use windowed reads per ZCTA centroid (avoids loading full 26 GB into RAM)
    # NLCD is in EPSG:5070 (Albers Equal Area, meters) -- must reproject centroid
    results = []
    with rasterio.open(local_raster) as src:
        nodata = src.nodata if src.nodata is not None else 127
        raster_crs = src.crs
        # Build transformer: WGS84 lat/lon -> raster native CRS
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        is_projected = raster_crs.is_projected  # True for EPSG:5070
        for _, row in centroids.iterrows():
            # Reproject centroid to raster CRS, then build 500m buffer
            x, y = transformer.transform(row["lon"], row["lat"])
            if is_projected:
                delta = 500  # 500 meters in native CRS units
            else:
                delta = 0.005  # fallback ~500m in degrees if raster is geographic
            geom = [mapping(box(x - delta, y - delta, x + delta, y + delta))]
            try:
                masked, _ = rio_mask(src, geom, crop=True, nodata=nodata)
                vals = masked[0].flatten()
                valid = vals[(vals != nodata) & (vals >= 0) & (vals <= 100)]
                imp_pct = float(np.mean(valid)) if len(valid) > 0 else np.nan
            except Exception:
                imp_pct = np.nan
            results.append({"zcta_id": row["zcta_id"], "impervious_pct": imp_pct})

    # Clean up local file
    try:
        Path(local_raster).unlink()
    except OSError:
        pass

    if not results:
        return empty

    out = pd.DataFrame(results).groupby("zcta_id", as_index=False)["impervious_pct"].mean()
    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(out, on="zcta_id", how="left")
    out["_fs_impervious_pct"] = np.where(out["impervious_pct"].notna(), "present", _FS_MISSING)
    log.info("build_impervious_features: %d ZCTAs, %.1f%% with data",
             len(out), (out["impervious_pct"].notna().mean() * 100))
    return out


def build_elevation_features(s3, zcta_ids: list[str], region: str) -> pd.DataFrame:
    """Derive elevation_m_msl per ZCTA from USGS 3DEP GeoTIFFs.

    Requires: raw/dem/3dep/v1/{region}/*.tif
    Returns DataFrame with columns: zcta_id, elevation_m_msl, _fs_elevation_m_msl.
    """
    empty = pd.DataFrame({"zcta_id": zcta_ids, "elevation_m_msl": np.nan,
                          "_fs_elevation_m_msl": _FS_MISSING})

    if not HAS_GEO:
        log.warning("build_elevation_features: geopandas/rasterio not available; returning NaN")
        return empty

    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import mapping, box
    except ImportError:
        return empty

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"raw/dem/3dep/v1/{region}/")
    tif_keys = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".tif")]
    if not tif_keys:
        log.warning("build_elevation_features: no 3DEP TIFs for region=%s; returning NaN", region)
        return empty

    static = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if static is None:
        return empty
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        return empty

    centroids = (static[[zcta_col, lat_col, lon_col]]
                 .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
                 .query("zcta_id in @zcta_ids")
                 .dropna(subset=["lat", "lon"]))

    elev_results: dict[str, list[float]] = {}
    for tif_key in tif_keys:
        local_tif = f"/tmp/dem_{Path(tif_key).stem}.tif"
        s3.download_file(BUCKET, tif_key, local_tif)
        with rasterio.open(local_tif) as src:
            nodata = src.nodata if src.nodata is not None else -9999
            raster_crs = src.crs
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            is_projected = raster_crs.is_projected
            for _, row in centroids.iterrows():
                x, y = transformer.transform(row["lon"], row["lat"])
                if is_projected:
                    delta = 1000  # 1km in native CRS meters
                else:
                    delta = 0.01  # ~1km in degrees for geographic CRS
                geom = [mapping(box(x - delta, y - delta, x + delta, y + delta))]
                try:
                    masked, _ = rio_mask(src, geom, crop=True, nodata=nodata)
                    vals = masked[0].flatten()
                    valid = vals[vals != nodata]
                    if len(valid) > 0:
                        # NAD83 DEM is in feet for NED; convert to meters
                        elev_m = float(np.mean(valid)) * 0.3048
                        elev_results.setdefault(row["zcta_id"], []).append(elev_m)
                except Exception:
                    pass

    rows = [{"zcta_id": z, "elevation_m_msl": np.mean(v)} for z, v in elev_results.items()]
    if not rows:
        return empty

    out = pd.DataFrame(rows)
    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(out, on="zcta_id", how="left")
    out["_fs_elevation_m_msl"] = np.where(out["elevation_m_msl"].notna(), "present", _FS_MISSING)
    log.info("build_elevation_features: %d ZCTAs, %.1f%% with data (region=%s)",
             len(out), out["elevation_m_msl"].notna().mean() * 100, region)
    return out


def build_burn_scar_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Derive burn_scar_overlap per ZCTA from MTBS perimeter GeoParquet.

    Requires: raw/mtbs/perimeters/v2023/burn_perims_ca_2015_2023.parquet
    Returns DataFrame with columns: zcta_id, burn_scar_overlap, _fs_burn_scar_overlap.
    """
    empty = pd.DataFrame({"zcta_id": zcta_ids, "burn_scar_overlap": False,
                          "_fs_burn_scar_overlap": _FS_MISSING})

    if not HAS_GEO:
        return empty

    try:
        import geopandas as gpd
    except ImportError:
        return empty

    mtbs_key = "raw/mtbs/perimeters/v2023/burn_perims_ca_2015_2023.parquet"
    mtbs_df = s3_read(s3, mtbs_key)
    if mtbs_df is None or mtbs_df.empty:
        log.warning("build_burn_scar_features: no MTBS data found; returning False for all ZCTAs")
        return empty

    # Load ZCTA polygons (from geocertdb2026 or TIGER)
    # Fallback: use centroid-based point-in-polygon
    static = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if static is None:
        return empty
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        return empty

    centroids = (static[[zcta_col, lat_col, lon_col]]
                 .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
                 .query("zcta_id in @zcta_ids")
                 .dropna(subset=["lat", "lon"]))

    # Build burn scar GeoDataFrame (union of all perimeters)
    from shapely.geometry import Point
    burn_gdf = gpd.GeoDataFrame(mtbs_df, geometry="geometry", crs="EPSG:4326") if "geometry" in mtbs_df.columns \
        else gpd.GeoDataFrame(mtbs_df, crs="EPSG:4326")
    if burn_gdf.empty:
        return empty

    burn_union = burn_gdf.unary_union  # single multipolygon for quick containment check

    overlaps = []
    for _, row in centroids.iterrows():
        pt = Point(row["lon"], row["lat"])
        overlaps.append({"zcta_id": row["zcta_id"], "burn_scar_overlap": burn_union.contains(pt)})

    out = pd.DataFrame(overlaps)
    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(out, on="zcta_id", how="left")
    out["burn_scar_overlap"] = out["burn_scar_overlap"].fillna(False)
    out["_fs_burn_scar_overlap"] = "present"
    pct_burned = out["burn_scar_overlap"].mean() * 100
    log.info("build_burn_scar_features: %d ZCTAs, %.1f%% in burn scar", len(out), pct_burned)
    return out


def build_catchment_features(s3, zcta_ids: list[str], vpu: str) -> pd.DataFrame:
    """Derive upstream_catchment_km2, wash_segment_id / bayou_segment_id from NHDPlus V2.

    Requires: raw/nhdplus/catchments/v2/catchments_vpu{vpu}.parquet
    Returns DataFrame: zcta_id, upstream_catchment_km2, wash_segment_id (or bayou_segment_id).
    """
    empty = pd.DataFrame({
        "zcta_id": zcta_ids,
        "upstream_catchment_km2": np.nan,
        "wash_segment_id": None,
        "_fs_upstream_catchment_km2": _FS_MISSING,
    })

    if not HAS_GEO:
        return empty

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        return empty

    nhd_key = f"raw/nhdplus/catchments/v2/catchments_vpu{vpu}.parquet"
    nhd_df = s3_read(s3, nhd_key)
    if nhd_df is None or nhd_df.empty:
        log.warning("build_catchment_features: no NHDPlus data for VPU %s; returning NaN", vpu)
        return empty

    static = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if static is None:
        return empty
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        return empty

    centroids = (static[[zcta_col, lat_col, lon_col]]
                 .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
                 .query("zcta_id in @zcta_ids")
                 .dropna(subset=["lat", "lon"]))
    centroid_gdf = gpd.GeoDataFrame(
        centroids,
        geometry=[Point(r["lon"], r["lat"]) for _, r in centroids.iterrows()],
        crs="EPSG:4326",
    )

    nhd_gdf = gpd.GeoDataFrame(nhd_df, geometry="geometry", crs="EPSG:4326")
    joined = gpd.sjoin(centroid_gdf, nhd_gdf[["comid", "area_sq_km", "geometry"]],
                       how="left", predicate="within")

    seg_col = "wash_segment_id" if vpu == "18" else "bayou_segment_id"
    out = joined[["zcta_id", "comid", "area_sq_km"]].rename(
        columns={"comid": seg_col, "area_sq_km": "upstream_catchment_km2"}
    )
    out[seg_col] = out[seg_col].astype(str).replace("nan", None)
    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(out, on="zcta_id", how="left")
    out["_fs_upstream_catchment_km2"] = np.where(
        out["upstream_catchment_km2"].notna(), "present", _FS_MISSING
    )
    log.info("build_catchment_features: %d ZCTAs, %.1f%% matched (VPU %s)",
             len(out), out["upstream_catchment_km2"].notna().mean() * 100, vpu)
    return out


def build_slosh_features(s3, event: str, zcta_ids: list[str]) -> pd.DataFrame:
    """Derive slosh_max_surge_m and slosh_category per ZCTA from NHC SLOSH grids.

    Requires: raw/noaa_slosh/{event}/*.zip or *.nc (SLOSH MOM grid)
    Returns DataFrame: zcta_id, slosh_max_surge_m, slosh_category, _fs_slosh_max_surge_m.
    """
    empty = pd.DataFrame({
        "zcta_id": zcta_ids,
        "slosh_max_surge_m": np.nan,
        "slosh_category": None,
        "_fs_slosh_max_surge_m": _FS_MISSING,
    })

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"raw/noaa_slosh/{event}/")
    slosh_keys = [o["Key"] for o in resp.get("Contents", [])
                  if not o["Key"].endswith("MANUAL_DOWNLOAD_REQUIRED.txt")]
    if not slosh_keys:
        log.warning("build_slosh_features: no SLOSH data for event=%s; "
                    "check for MANUAL_DOWNLOAD_REQUIRED.txt", event)
        return empty

    # SLOSH MOM grids are ASCII or NetCDF; attempt NetCDF first
    nc_keys = [k for k in slosh_keys if k.endswith(".nc") or k.endswith(".grb2")]
    if not nc_keys:
        log.warning("build_slosh_features: SLOSH keys found but no .nc/.grb2 for event=%s; "
                    "returning NaN (manual inspection required)", event)
        return empty

    if not HAS_GEO:
        log.warning("build_slosh_features: geopandas not available; returning NaN")
        return empty

    try:
        import xarray as xr
        from shapely.geometry import Point
        import geopandas as gpd
    except ImportError:
        log.warning("build_slosh_features: xarray not available; returning NaN")
        return empty

    static = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if static is None:
        return empty
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        return empty

    centroids = (static[[zcta_col, lat_col, lon_col]]
                 .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
                 .query("zcta_id in @zcta_ids")
                 .dropna(subset=["lat", "lon"]))

    surge_vals: dict[str, float] = {}
    for nc_key in nc_keys[:3]:  # limit to first 3 to avoid memory issues
        local = f"/tmp/slosh_{Path(nc_key).stem}.nc"
        s3.download_file(BUCKET, nc_key, local)
        try:
            ds = xr.open_dataset(local, engine="netcdf4")
            # SLOSH MOM variable names vary; try common ones
            surge_var = next(
                (v for v in ds.data_vars
                 if any(k in v.lower() for k in ["surge", "mom", "meow", "water"])),
                None,
            )
            if surge_var is None:
                continue
            surge_da = ds[surge_var]
            lat_da = ds.get("lat") or ds.get("latitude") or ds.coords.get("lat")
            lon_da = ds.get("lon") or ds.get("longitude") or ds.coords.get("lon")
            if lat_da is None or lon_da is None:
                continue
            lat_arr = lat_da.values.flatten()
            lon_arr = lon_da.values.flatten()
            surge_arr = surge_da.values.flatten()

            for _, row in centroids.iterrows():
                # Nearest-neighbor lookup
                dists = np.sqrt((lat_arr - row["lat"])**2 + (lon_arr - row["lon"])**2)
                idx = np.argmin(dists)
                val = float(surge_arr[idx]) if not np.isnan(surge_arr[idx]) else np.nan
                if row["zcta_id"] not in surge_vals or val > surge_vals.get(row["zcta_id"], 0):
                    surge_vals[row["zcta_id"]] = val
        except Exception as exc:
            log.warning("SLOSH read error for %s: %s", nc_key, exc)

    if not surge_vals:
        return empty

    def surge_to_category(m: float) -> Optional[str]:
        if np.isnan(m):
            return None
        if m < 1.5:
            return "1"
        elif m < 2.4:
            return "2"
        elif m < 3.7:
            return "3"
        elif m < 5.5:
            return "4"
        return "5"

    rows = [{"zcta_id": z, "slosh_max_surge_m": v,
             "slosh_category": surge_to_category(v)} for z, v in surge_vals.items()]
    out = pd.DataFrame(rows)
    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(out, on="zcta_id", how="left")
    out["_fs_slosh_max_surge_m"] = np.where(out["slosh_max_surge_m"].notna(), "present", _FS_MISSING)
    log.info("build_slosh_features: %d ZCTAs, %.1f%% with surge data (event=%s)",
             len(out), out["slosh_max_surge_m"].notna().mean() * 100, event)
    return out


def build_sewershed_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Assign sewer_shed_id to each NYC ZCTA via spatial join with NYC DEP sewersheds.

    Requires: raw/nyc_sewersheds/nyc_sewersheds.parquet (attribute table from .gpkg)
    Returns DataFrame: zcta_id, sewer_shed_id, _fs_sewer_shed_id.
    """
    empty = pd.DataFrame({"zcta_id": zcta_ids, "sewer_shed_id": None,
                           "_fs_sewer_shed_id": _FS_MISSING})

    if not HAS_GEO:
        return empty

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        return empty

    sewer_key = "raw/nyc_sewersheds/nyc_sewersheds.parquet"
    sewer_df = s3_read(s3, sewer_key)
    if sewer_df is None or sewer_df.empty:
        log.warning("build_sewershed_features: sewershed parquet not found; returning null")
        return empty

    # Find the ID column
    id_col = next((c for c in sewer_df.columns
                   if any(k in c.lower() for k in ["shed_id", "drainage_id", "district", "id"])),
                  sewer_df.columns[0])
    geom_col = "geometry" if "geometry" in sewer_df.columns else None
    if geom_col is None:
        log.warning("build_sewershed_features: no geometry column in sewershed parquet; returning null")
        return empty

    sewer_gdf = gpd.GeoDataFrame(sewer_df, geometry=geom_col, crs="EPSG:4326")

    # Load ZCTA centroids
    static = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if static is None:
        return empty
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        return empty

    centroids = (static[[zcta_col, lat_col, lon_col]]
                 .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
                 .query("zcta_id in @zcta_ids")
                 .dropna(subset=["lat", "lon"]))
    centroid_gdf = gpd.GeoDataFrame(
        centroids,
        geometry=[Point(r["lon"], r["lat"]) for _, r in centroids.iterrows()],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(centroid_gdf, sewer_gdf[[id_col, geom_col]],
                       how="left", predicate="within")
    joined = joined.rename(columns={id_col: "sewer_shed_id"})
    out = joined[["zcta_id", "sewer_shed_id"]].drop_duplicates("zcta_id")
    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(out, on="zcta_id", how="left")
    out["_fs_sewer_shed_id"] = np.where(out["sewer_shed_id"].notna(), "present", _FS_MISSING)
    log.info("build_sewershed_features: %d ZCTAs, %.1f%% matched",
             len(out), out["sewer_shed_id"].notna().mean() * 100)
    return out


def build_subway_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Count subway stations per ZCTA and compute nearest-station distance.

    Requires: raw/mta/subway_stations/v1/subway_stations.parquet
    Returns DataFrame: zcta_id, subway_station_count, nearest_subway_distance_m.
    """
    empty = pd.DataFrame({
        "zcta_id": zcta_ids,
        "subway_station_count": 0,
        "nearest_subway_distance_m": np.nan,
        "_fs_subway_station_count": _FS_MISSING,
    })

    stations_key = "raw/mta/subway_stations/v1/subway_stations.parquet"
    stations = s3_read(s3, stations_key)
    if stations is None or stations.empty:
        log.warning("build_subway_features: MTA station data not found; returning 0/NaN")
        return empty

    if "latitude" not in stations.columns or "longitude" not in stations.columns:
        return empty

    stations = stations.dropna(subset=["latitude", "longitude"])

    # Load ZCTA centroids
    static = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if static is None:
        return empty
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        return empty

    centroids = (static[[zcta_col, lat_col, lon_col]]
                 .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
                 .query("zcta_id in @zcta_ids")
                 .dropna(subset=["lat", "lon"]))

    R_KM = 6371.0

    def haversine_km(lat1, lon1, lat2, lon2):
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
        return R_KM * 2 * np.arcsin(np.sqrt(a))

    # ZCTA bounding radius ~1.5 km for NYC ZCTAs
    THRESHOLD_KM = 1.5
    rows = []
    for _, row in centroids.iterrows():
        dists = haversine_km(
            row["lat"], row["lon"],
            stations["latitude"].values, stations["longitude"].values,
        )
        within = (dists <= THRESHOLD_KM).sum()
        nearest_m = float(dists.min() * 1000) if len(dists) > 0 else np.nan
        rows.append({
            "zcta_id": row["zcta_id"],
            "subway_station_count": int(within),
            "nearest_subway_distance_m": nearest_m,
        })

    out = pd.DataFrame(rows)
    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(out, on="zcta_id", how="left")
    out["subway_station_count"] = out["subway_station_count"].fillna(0).astype(int)
    out["_fs_subway_station_count"] = "present"
    log.info("build_subway_features: %d ZCTAs, mean=%.1f stations/ZCTA",
             len(out), out["subway_station_count"].mean())
    return out


def build_levee_features(s3, zcta_ids: list[str], scenario: str) -> pd.DataFrame:
    """Assign levee_condition_rating and canal_proximity_m per ZCTA.

    Requires: raw/usace_levees/{scenario}_levees.parquet
    Returns DataFrame: zcta_id, levee_condition_rating, _fs_levee_condition_rating.
    """
    empty = pd.DataFrame({"zcta_id": zcta_ids, "levee_condition_rating": np.nan,
                          "canal_proximity_m": np.nan,
                          "_fs_levee_condition_rating": _FS_MISSING})

    levee_key = f"raw/usace_levees/{scenario}_levees.parquet"
    levees = s3_read(s3, levee_key)
    if levees is None or levees.empty:
        log.warning("build_levee_features: no levee data for scenario=%s", scenario)
        return empty

    rating_col = next((c for c in levees.columns if "condition" in c.lower() or "rating" in c.lower()), None)
    lat_col = next((c for c in levees.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in levees.columns if "lon" in c.lower() or "lng" in c.lower()), None)

    # Load ZCTA centroids
    static = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if static is None:
        return empty
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    slat = next((c for c in static.columns if "lat" in c.lower()), None)
    slon = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, slat, slon]):
        return empty

    centroids = (static[[zcta_col, slat, slon]]
                 .rename(columns={zcta_col: "zcta_id", slat: "lat", slon: "lon"})
                 .query("zcta_id in @zcta_ids")
                 .dropna(subset=["lat", "lon"]))

    R_KM = 6371.0
    rows = []
    for _, crow in centroids.iterrows():
        if lat_col and lon_col:
            lats = levees[lat_col].values
            lons = levees[lon_col].values
            dlat = np.radians(lats - crow["lat"])
            dlon = np.radians(lons - crow["lon"])
            a = (np.sin(dlat / 2)**2
                 + np.cos(np.radians(crow["lat"])) * np.cos(np.radians(lats)) * np.sin(dlon / 2)**2)
            dists_km = R_KM * 2 * np.arcsin(np.sqrt(a))
            nearest_idx = int(np.argmin(dists_km))
            nearest_dist_m = float(dists_km[nearest_idx] * 1000)
            rating = float(levees.iloc[nearest_idx][rating_col]) if rating_col else np.nan
        else:
            nearest_dist_m = np.nan
            rating = float(levees[rating_col].median()) if rating_col else np.nan

        rows.append({
            "zcta_id": crow["zcta_id"],
            "levee_condition_rating": rating,
            "canal_proximity_m": nearest_dist_m,
        })

    out = pd.DataFrame(rows)
    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(out, on="zcta_id", how="left")
    out["_fs_levee_condition_rating"] = np.where(
        out["levee_condition_rating"].notna(), "present", _FS_MISSING
    )
    log.info("build_levee_features: %d ZCTAs for scenario=%s", len(out), scenario)
    return out


def build_coastal_distance_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Compute coastal_distance_m from ZCTA centroid to nearest TIGER coastline point.

    Approximation: uses precomputed bounding box of FL/Gulf coast as a set of
    lat/lon reference points. Replace with full TIGER coastline when available.

    Returns DataFrame: zcta_id, coastal_distance_m, _fs_coastal_distance_m.
    """
    empty = pd.DataFrame({"zcta_id": zcta_ids, "coastal_distance_m": np.nan,
                          "_fs_coastal_distance_m": _FS_MISSING})

    # Load ZCTA centroids
    static = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if static is None:
        return empty
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        return empty

    centroids = (static[[zcta_col, lat_col, lon_col]]
                 .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
                 .query("zcta_id in @zcta_ids")
                 .dropna(subset=["lat", "lon"]))

    # SW Florida Gulf Coast reference points (0.1-degree grid)
    # These cover the Lee/Charlotte/Sarasota coastline adequately
    coast_lats = np.arange(25.5, 29.1, 0.1)
    coast_lons_gulf = np.full_like(coast_lats, -82.0)  # approximate Gulf shore longitude
    coast_lats_all = np.concatenate([coast_lats, coast_lats])
    coast_lons_all = np.concatenate([coast_lons_gulf, coast_lons_gulf - 0.5])

    R_KM = 6371.0
    rows = []
    for _, row in centroids.iterrows():
        dlat = np.radians(coast_lats_all - row["lat"])
        dlon = np.radians(coast_lons_all - row["lon"])
        a = (np.sin(dlat / 2)**2
             + np.cos(np.radians(row["lat"])) * np.cos(np.radians(coast_lats_all)) * np.sin(dlon / 2)**2)
        dists_km = R_KM * 2 * np.arcsin(np.sqrt(a))
        rows.append({"zcta_id": row["zcta_id"], "coastal_distance_m": float(dists_km.min() * 1000)})

    out = pd.DataFrame(rows)
    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(out, on="zcta_id", how="left")
    out["_fs_coastal_distance_m"] = np.where(
        out["coastal_distance_m"].notna(), "present", _FS_MISSING
    )
    log.info("build_coastal_distance_features: %d ZCTAs, median=%.1f km",
             len(out), out["coastal_distance_m"].median() / 1000)
    return out


# ---------------------------------------------------------------------------
# Updated scenario assemblers — wire in feature contract build functions
# ---------------------------------------------------------------------------

BUILDERS = {
    "houston":             build_houston,
    "new_orleans":         build_new_orleans,
    "nyc":                 build_nyc,
    "riverside_coachella": build_riverside_coachella,
    "southwest_florida":   build_southwest_florida,
}

OUTPUT_KEYS = {
    "houston":             "processed/houston/houston_event_features.parquet",
    "new_orleans":         "processed/new_orleans/no_event_features.parquet",
    "nyc":                 "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida":   "processed/southwest_florida/swfl_event_features.parquet",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=list(BUILDERS.keys()))
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name="us-east-1")

    cfg_path = CONFIG_DIR / f"{args.scenario}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    builder = BUILDERS[args.scenario]
    df = builder(s3, cfg)

    if df.empty:
        log.error("No data assembled for scenario %s", args.scenario)
        sys.exit(1)

    log.info("Schema summary for %s:", args.scenario)
    log.info("  Rows: %d, Columns: %d", len(df), len(df.columns))
    log.info("  Events: %s", df["event"].unique().tolist() if "event" in df.columns else "?")
    log.info("  Certificate slots null: %s",
             {c: df[c].isna().sum() for c in CERTIFICATE_SLOTS if c in df.columns})

    s3_upload(df, OUTPUT_KEYS[args.scenario], s3)
    log.info("build_event_dataset complete: %s", args.scenario)


if __name__ == "__main__":
    main()
