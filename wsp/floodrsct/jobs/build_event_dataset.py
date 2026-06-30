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
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import boto3
from swarm_auth import get_aws_credentials
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"


# ---------------------------------------------------------------------------
# Subprocess isolation for STAC/rasterio calls
# ---------------------------------------------------------------------------
# pip-installed rasterio bundles GDAL linked against curl 8.x, but the
# PyTorch SageMaker base image ships system curl 7.x.  Any rasterio call
# that opens a remote URL via VSICURL causes a SIGSEGV from ABI mismatch.
# We run all floodcaster STAC extraction in a child process so a segfault
# kills the child -- not the main job.  The caller gets NaN gracefully.

def _run_in_subprocess(func_module: str, func_name: str,
                       args_df: pd.DataFrame, id_col: str,
                       extra_kwargs: dict | None = None,
                       timeout: int = 600) -> pd.DataFrame | None:
    """Run a floodcaster extraction function in a child process.

    Returns the result DataFrame, or None if the child crashes (segfault).
    """
    import multiprocessing as mp
    import tempfile

    extra_kwargs = extra_kwargs or {}

    def _worker(in_path, out_path, mod, fn, kw):
        import importlib
        df = pd.read_parquet(in_path)
        m = importlib.import_module(mod)
        f = getattr(m, fn)
        result = f(df, id_col=id_col, **kw)
        result.to_parquet(out_path, index=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.parquet")
        out_path = os.path.join(tmpdir, "output.parquet")
        args_df.to_parquet(in_path, index=False)

        proc = mp.Process(
            target=_worker,
            args=(in_path, out_path, func_module, func_name, extra_kwargs),
        )
        proc.start()
        proc.join(timeout=timeout)

        if proc.exitcode != 0:
            code = proc.exitcode if proc.exitcode is not None else -1
            log.warning(
                "_run_in_subprocess[%s.%s]: child exited with code %d "
                "(segfault=%s); returning None",
                func_module, func_name, code, code == -11 or code == 139,
            )
            if proc.is_alive():
                proc.kill()
            return None

        return pd.read_parquet(out_path)


def _list_s3_keys(s3, prefix: str) -> list[dict]:
    """Paginated list_objects_v2. Returns all Contents dicts under prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    contents = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        contents.extend(page.get("Contents", []))
    return contents


# Launcher uploads configs alongside code into the same S3 prefix,
# so they land in /opt/ml/processing/input/code/ on the container.
CONFIG_DIR = Path("/opt/ml/processing/input/code")

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
    if isinstance(df, gpd.GeoDataFrame):
        df.to_parquet(local, index=False)
        log.info("Writing GeoParquet (CRS=%s)", df.crs)
    else:
        df.to_parquet(local, index=False)
    s3.upload_file(local, BUCKET, key)
    log.info("Uploaded %d rows x %d cols to s3://%s/%s", len(df), len(df.columns), BUCKET, key)


def _attach_geometry(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert assembled DataFrame to GeoDataFrame with Point geometry from centroids.

    Uses existing latitude/longitude columns. Returns GeoDataFrame with
    EPSG:4326 CRS and RFC-compliant GeoParquet metadata when written.
    (DOE Amendment v1.2 Change 12)
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        log.warning("No lat/lon columns -- returning plain DataFrame (no geometry)")
        return df
    geometry = [
        Point(lon, lat) if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(df["latitude"], df["longitude"])
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    log.info("Attached Point geometry to %d rows (CRS=EPSG:4326)", len(gdf))
    return gdf


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
# Spatial W-Matrix Features (DOE Amendment v1.2 Change 13)
# ---------------------------------------------------------------------------

_ADJ_CACHE: Optional[pd.DataFrame] = None


def _load_adjacency(s3) -> Optional[pd.DataFrame]:
    """Load ZCTA Queen's contiguity adjacency edge list from S3 (cached)."""
    global _ADJ_CACHE
    if _ADJ_CACHE is not None:
        return _ADJ_CACHE
    for key in [
        "raw/geocertdb2026/zcta_adjacency.parquet",
        "raw/geocert/zcta_adjacency.parquet",
    ]:
        adj = s3_read(s3, key)
        if adj is not None:
            log.info("Loaded adjacency from %s: %d edges", key, len(adj))
            _ADJ_CACHE = adj
            return adj
    raise FileNotFoundError(
        "zcta_adjacency.parquet not found on S3. W-matrix features require "
        "adjacency data. Upload adjacency first or run enrich_adjacency."
    )


def _build_w_neighbors(adj_df: pd.DataFrame, zcta_ids: list[str]) -> dict[str, list[str]]:
    """Build queen-contiguity neighbor dict from adjacency edge list."""
    # Determine column names (zcta_id_1/zcta_id_2 or zcta_from/zcta_to)
    cols = adj_df.columns.tolist()
    if "zcta_id_1" in cols and "zcta_id_2" in cols:
        c1, c2 = "zcta_id_1", "zcta_id_2"
    elif "zcta_from" in cols and "zcta_to" in cols:
        c1, c2 = "zcta_from", "zcta_to"
    else:
        c1, c2 = cols[0], cols[1]

    zcta_set = set(zcta_ids)
    # Filter to scenario ZCTAs on both sides
    mask = adj_df[c1].isin(zcta_set) & adj_df[c2].isin(zcta_set)
    sub = adj_df[mask]

    neighbors: dict[str, list[str]] = {z: [] for z in zcta_ids}
    for _, row in sub.iterrows():
        a, b = str(row[c1]), str(row[c2])
        if a in neighbors:
            neighbors[a].append(b)
        if b in neighbors:
            neighbors[b].append(a)

    # Deduplicate
    for z in neighbors:
        neighbors[z] = list(set(neighbors[z]))

    n_with = sum(1 for v in neighbors.values() if v)
    log.info("W-matrix: %d/%d ZCTAs have neighbors", n_with, len(zcta_ids))
    return neighbors


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def compute_w_matrix_features(
    s3, zcta_ids: list[str], static_df: pd.DataFrame,
    event_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute spatial W-matrix features from Queen's contiguity adjacency.

    Returns DataFrame with columns:
      zcta_degree, zcta_mean_neighbor_dist_km,
      wlag_flood_zone_pct, wlag_population_density, wlag_median_income,
      wlag_impervious_pct, wlag_rainfall_mm (if event_df provided),
      wlag_nfip_claims (if event_df provided)

    (DOE Amendment v1.2 Change 13)
    """
    adj_df = _load_adjacency(s3)

    neighbors = _build_w_neighbors(adj_df, zcta_ids)

    # Build centroid lookup from static features
    centroids = {}
    if "latitude" in static_df.columns and "longitude" in static_df.columns:
        for _, row in static_df.iterrows():
            zid = str(row.get("zcta_id", ""))
            if zid and pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
                centroids[zid] = (row["latitude"], row["longitude"])

    # Build value lookup for spatial lag source columns
    static_lookup = {}
    for _, row in static_df.iterrows():
        zid = str(row.get("zcta_id", ""))
        if zid:
            static_lookup[zid] = row

    event_lookup = {}
    if event_df is not None:
        for _, row in event_df.iterrows():
            zid = str(row.get("zcta_id", ""))
            if zid:
                event_lookup[zid] = row

    # Spatial lag source columns (static -- from geocertdb2026 static features)
    STATIC_LAG_MAP = {
        "wlag_flood_zone_pct": "flood_pct_zone_a",
        "wlag_population_density": "population",
        "wlag_median_income": "acs_median_hh_income",
    }
    # Spatial lag source columns (event -- from assembled event base)
    EVENT_LAG_MAP = {
        "wlag_impervious_pct": "impervious_pct",
        "wlag_cropland_pct": "cropland_pct",
        "wlag_rainfall_mm": "rainfall_total_mm",
        "wlag_nfip_claims": "nfip_event_claim_count",
    }

    rows = []
    for zcta in zcta_ids:
        nbrs = neighbors.get(zcta, [])
        degree = len(nbrs)

        # Mean neighbor distance
        mean_dist = np.nan
        if degree > 0 and zcta in centroids:
            lat0, lon0 = centroids[zcta]
            dists = []
            for nb in nbrs:
                if nb in centroids:
                    dists.append(_haversine_km(lat0, lon0, *centroids[nb]))
            if dists:
                mean_dist = float(np.mean(dists))

        rec = {"zcta_id": zcta, "zcta_degree": degree,
               "zcta_mean_neighbor_dist_km": mean_dist}

        # Row-standardized spatial lag of static features
        for out_col, src_col in STATIC_LAG_MAP.items():
            rec[out_col] = _compute_lag(nbrs, static_lookup, src_col)

        # Row-standardized spatial lag of event features
        for out_col, src_col in EVENT_LAG_MAP.items():
            rec[out_col] = _compute_lag(nbrs, event_lookup, src_col)

        rows.append(rec)

    return pd.DataFrame(rows)


def _compute_lag(
    neighbors: list[str], lookup: dict, col: str,
) -> float:
    """Row-standardized spatial lag: mean of col across neighbors."""
    if not neighbors or not lookup:
        return np.nan
    vals = []
    for nb in neighbors:
        row = lookup.get(nb)
        if row is not None:
            v = row.get(col)
            if v is not None and pd.notna(v):
                vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan


def _empty_w_features(zcta_ids: list[str]) -> pd.DataFrame:
    """Return empty W-matrix feature DataFrame when adjacency is unavailable."""
    return pd.DataFrame({
        "zcta_id": zcta_ids,
        "zcta_degree": 0,
        "zcta_mean_neighbor_dist_km": np.nan,
        "wlag_flood_zone_pct": np.nan,
        "wlag_population_density": np.nan,
        "wlag_median_income": np.nan,
        "wlag_impervious_pct": np.nan,
        "wlag_rainfall_mm": np.nan,
        "wlag_nfip_claims": np.nan,
    })


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
    site_lat_lon = _load_site_coords(s3, scenario)
    if site_lat_lon is not None:
        peaks = peaks.merge(site_lat_lon, on="site_no", how="left")
        # Determine max assignment radius from regime config (default 30 km)
        max_radius = float(cfg.get("regime", {}).get(
            "spatial_autocorrelation_range_km", 30))
        assigned = _assign_nearest_zcta(peaks, zcta_ids, s3,
                                        max_radius_km=max_radius)
    else:
        log.warning("Site coordinates unavailable; NWIS gauge-to-ZCTA assignment skipped")
        return _empty_nwis(zcta_ids)

    # assigned is already ZCTA-level (one row per ZCTA within radius)
    if "zcta_id" not in assigned.columns or assigned.empty:
        return _empty_nwis(zcta_ids)

    zcta_peaks = (
        assigned.groupby("zcta_id")
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
                          s3, max_radius_km: float = 30.0) -> pd.DataFrame:
    """Assign each ZCTA to the nearest gauge within max_radius_km.

    Previous logic mapped gauge→ZCTA (one gauge, one ZCTA). That left most
    ZCTAs unassigned when gauge density is low. This version inverts the
    assignment: for each ZCTA centroid, find the nearest gauge within radius
    and propagate its peak values. Multiple ZCTAs can share the same gauge.
    """
    centroids_key = "raw/geocertdb2026/zcta_features_labels.parquet"
    geo = s3_read(s3, centroids_key)
    if geo is None or "latitude" not in geo.columns:
        df["zcta_id"] = None
        df["dist_km"] = np.nan
        return df

    geo = geo[geo["zcta_id"].isin(zcta_ids)][["zcta_id", "latitude", "longitude"]].dropna()

    # Filter gauges with valid coordinates
    valid_sites = df.dropna(subset=["latitude", "longitude"]).copy()
    if valid_sites.empty:
        df["zcta_id"] = None
        df["dist_km"] = np.nan
        return df

    site_lats = valid_sites["latitude"].values
    site_lons = valid_sites["longitude"].values

    # For each ZCTA, find nearest gauge
    results = []
    for _, zcta_row in geo.iterrows():
        dlat = np.radians(site_lats - zcta_row["latitude"])
        dlon = np.radians(site_lons - zcta_row["longitude"])
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(zcta_row["latitude"])) *
             np.cos(np.radians(site_lats)) *
             np.sin(dlon / 2) ** 2)
        dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        idx = dist.argmin()
        min_dist = dist[idx]
        if min_dist <= max_radius_km:
            row = valid_sites.iloc[idx].copy()
            results.append({
                "zcta_id": zcta_row["zcta_id"],
                "dist_km": min_dist,
                "site_no": row["site_no"],
                "peak_stage_ft": row.get("peak_stage_ft", np.nan),
                "peak_flow_cfs": row.get("peak_flow_cfs", np.nan),
            })

    if not results:
        df["zcta_id"] = None
        df["dist_km"] = np.nan
        return df

    log.info("NWIS ZCTA assignment: %d/%d ZCTAs within %.0f km of a gauge",
             len(results), len(geo), max_radius_km)
    return pd.DataFrame(results)


def _empty_nwis(zcta_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "zcta_id": zcta_ids,
        "peak_stage_ft": np.nan,
        "peak_flow_cfs": np.nan,
        "obs_gauge_count": 0,
        "obs_gauge_distance_km": np.nan,
    })


# ---------------------------------------------------------------------------
# Merge helper (dedup guard)
# ---------------------------------------------------------------------------

def _safe_merge_parts(base: pd.DataFrame, parts: list, zcta_col: str = "zcta_id") -> pd.DataFrame:
    """Merge feature DataFrames onto base with row-duplication detection."""
    expected = len(base)
    for part in parts:
        base = base.merge(part, on=zcta_col, how="left")
        if len(base) != expected:
            log.error("Row duplication after merging %s: got %d rows, expected %d",
                      [c for c in part.columns if c != zcta_col], len(base), expected)
            base = base.drop_duplicates(subset=[zcta_col, "event"], keep="first")
    return base


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
    all_objs = _list_s3_keys(s3, prefix)
    files = [o["Key"] for o in all_objs
             if o["Key"].endswith(".grb2") or o["Key"].endswith(".grib2.gz")]
    log.info("MRMS: %d grib2 files for event %s", len(files), event)

    empty = pd.DataFrame({
        "zcta_id": zcta_ids,
        "rainfall_total_mm": np.nan,
        "obs_mrms_coverage_pct": 0.0,
    })

    if not files:
        log.warning("No MRMS files for event %s; trying GPM IMERG fallback", event)
        imerg_result = _aggregate_gpm_imerg(s3, event, zcta_ids)
        if imerg_result is not None:
            return imerg_result
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


def _aggregate_gpm_imerg(s3, event: str, zcta_ids: list[str]) -> Optional[pd.DataFrame]:
    """Fallback rainfall aggregation using GPM IMERG daily NetCDF4 files.

    Used for pre-MRMS events (Sandy 2012) where MRMS data doesn't exist.
    GPM IMERG Final Run V07B has 0.1° resolution, global, back to 2000.
    """
    prefix = f"raw/gpm_imerg/daily/{event}/"
    all_objs = _list_s3_keys(s3, prefix)
    nc_files = [o["Key"] for o in all_objs if o["Key"].endswith(".nc4")]
    if not nc_files:
        log.warning("No GPM IMERG files for event %s", event)
        return None

    log.info("GPM IMERG fallback: %d nc4 files for event %s", len(nc_files), event)

    try:
        import netCDF4
    except ImportError:
        log.warning("netCDF4 not available; GPM IMERG fallback skipped")
        return None

    # Load ZCTA centroids for point sampling
    centroids_key = "raw/geocertdb2026/zcta_features_labels.parquet"
    geo = s3_read(s3, centroids_key)
    if geo is None:
        return None
    geo = geo[geo["zcta_id"].isin(zcta_ids)][["zcta_id", "latitude", "longitude"]].dropna()

    # Accumulate daily precipitation across all days
    accum = np.zeros(len(geo))
    valid_days = 0

    for key in nc_files:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as tmp:
                tmp.write(obj["Body"].read())
                tmp_path = tmp.name

            ds = netCDF4.Dataset(tmp_path, "r")
            # GPM IMERG V07B: precipitation var = 'precipitation', dims = [time, lon, lat]
            # Units: mm/day for daily product
            precip = ds.variables["precipitation"][0, :, :]  # (lon, lat)
            lats = ds.variables["lat"][:]
            lons = ds.variables["lon"][:]
            ds.close()
            os.unlink(tmp_path)

            # Sample at ZCTA centroids using nearest-neighbor
            for i, (_, row) in enumerate(geo.iterrows()):
                lat_idx = np.argmin(np.abs(lats - row["latitude"]))
                lon_idx = np.argmin(np.abs(lons - row["longitude"]))
                val = float(precip[lon_idx, lat_idx])
                if val >= 0:  # GPM uses negative for missing
                    accum[i] += val
            valid_days += 1
        except Exception as e:
            log.warning("GPM IMERG file %s failed: %s", key, e)
            continue

    if valid_days == 0:
        return None

    log.info("GPM IMERG: accumulated %d days, %d ZCTAs, min=%.1f max=%.1f mean=%.1f mm",
             valid_days, len(geo), accum.min(), accum.max(), accum.mean())

    result = pd.DataFrame({
        "zcta_id": geo["zcta_id"].values,
        "rainfall_total_mm": accum,
        "obs_mrms_coverage_pct": 0.0,  # Not MRMS; flag as 0 coverage
        "obs_precip_source": "gpm_imerg",
    })
    # Fill ZCTAs without centroid match
    all_zctas = pd.DataFrame({"zcta_id": zcta_ids})
    return all_zctas.merge(result, on="zcta_id", how="left")


def _process_one_grib(args: tuple) -> tuple:
    """Worker: download + decompress + read one grib2 file.

    Returns (arr, lat, lon) on success, or a string error message on failure.
    Never returns None silently — callers can distinguish success tuples from
    error strings to aggregate diagnostics.
    """
    import gzip
    import tempfile
    key, bucket = args
    import boto3
    from swarm_auth import get_aws_credentials
    s3w = boto3.client("s3", region_name="us-east-1", **get_aws_credentials())

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
        available_vars = list(ds.data_vars)
        precip_var = next(
            (v for v in ["tp", "unknown", "apcp", "APCP"] if v in ds),
            None,
        )
        if precip_var is None:
            return f"NO_PRECIP_VAR:vars={available_vars}"
        arr = ds[precip_var].values
        # MRMS uses negative sentinels (-3 = no data, -1 = range-folded, -2 = below threshold).
        # Clamp to zero so sentinels don't corrupt the hourly accumulation sum.
        arr = np.where(arr < 0, 0.0, arr)
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        return (arr, lat, lon)
    except Exception as exc:
        return f"EXCEPTION:{type(exc).__name__}:{str(exc)[:120]}"
    finally:
        for p in (gz_path, grb2_path):
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


def _mrms_spatial_aggregate(s3, file_keys: list[str], zcta_ids: list[str],
                             event: str) -> pd.DataFrame:
    """Download grib2 files, read precipitation values, overlay on ZCTA polygons.

    Uses ProcessPoolExecutor to parallelize gzip decompression + cfgrib reads
    across all available CPU cores.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Each CONUS grib2 decompresses to ~100 MB. Cap at half the vCPUs so
    # the main process + accumulator have headroom. On ml.m5.4xlarge (16
    # vCPU, 64 GB): 8 workers x 100 MB = 0.8 GB concurrent -- well within
    # memory. On ml.m5.2xlarge (8 vCPU, 32 GB): 4 workers x 100 MB = 0.4 GB.
    n_workers = min(os.cpu_count() // 2, max(1, os.cpu_count() or 4))
    log.info("MRMS parallel decode: %d files, %d workers", len(file_keys), n_workers)

    # Accumulate running sum instead of storing all 168 grids (~16 GB)
    total_mm = None
    lat_arr = lon_arr = None
    n_valid = 0

    work_items = [(key, BUCKET) for key in file_keys]
    error_counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one_grib, item): item for item in work_items}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 25 == 0:
                log.info("MRMS decode progress: %d/%d (valid=%d)", done, len(file_keys), n_valid)
            result = fut.result()
            if isinstance(result, str):
                # Error message from worker — aggregate for summary
                tag = result.split(":")[0]
                error_counts[tag] = error_counts.get(tag, 0) + 1
                if error_counts[tag] <= 3:
                    log.warning("MRMS worker error [%d/%d]: %s", done, len(file_keys), result[:200])
                continue
            arr, lat, lon = result
            if total_mm is None:
                total_mm = np.where(np.isnan(arr), 0.0, arr)
            else:
                total_mm += np.where(np.isnan(arr), 0.0, arr)
            n_valid += 1
            if lat_arr is None:
                lat_arr = lat
                lon_arr = lon

    n_errors = sum(error_counts.values())
    log.info("MRMS decode complete: %d/%d valid, %d errors %s",
             n_valid, len(file_keys), n_errors, dict(error_counts) if error_counts else "")
    if n_valid > 0 and total_mm is not None:
        log.info("MRMS accumulation stats: min=%.2f max=%.2f mean=%.2f",
                 float(np.nanmin(total_mm)), float(np.nanmax(total_mm)), float(np.nanmean(total_mm)))

    if total_mm is None:
        log.error("MRMS: ALL %d files failed decode. No rainfall data for event %s. "
                  "Error breakdown: %s", len(file_keys), event, dict(error_counts))
        return pd.DataFrame({"zcta_id": zcta_ids, "rainfall_total_mm": np.nan,
                             "obs_mrms_coverage_pct": 0.0})

    coverage = n_valid / max(len(file_keys), 1)

    # cfgrib returns 1D coordinate arrays; meshgrid to match 2D precip grid
    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon_arr, lat_arr)
    else:
        lat_2d, lon_2d = lat_arr, lon_arr

    # MRMS GRIB2 uses 0-360 longitude (e.g., Houston = 264.6). ZCTA centroids
    # use -180/180 (Houston = -95.4). Convert grid to -180/180 so the nearest-
    # centroid lookup finds the correct pixel.
    if lon_2d.max() > 180:
        lon_2d = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
        log.info("MRMS lon converted from 0-360 to -180/180 (range: %.1f to %.1f)",
                 lon_2d.min(), lon_2d.max())

    flat_lat = lat_2d.flatten()
    flat_lon = lon_2d.flatten()
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

    # Sanity check: rainfall should not be all zeros when the grid has signal
    r = df["rainfall_total_mm"]
    log.info("MRMS ZCTA assignment: %d ZCTAs, min=%.2f max=%.2f mean=%.2f zeros=%d/%d",
             len(df), r.min(), r.max(), r.mean(), (r == 0).sum(), len(r))

    # Fill ZCTAs with no MRMS match
    all_zctas = pd.DataFrame({"zcta_id": zcta_ids})
    return all_zctas.merge(df, on="zcta_id", how="left")


# ---------------------------------------------------------------------------
# STN high-water marks
# ---------------------------------------------------------------------------

def aggregate_hwm(s3, event: str, zcta_ids: list[str]) -> pd.DataFrame:
    # Check fetch_surge_hwm output first (state-filtered, verified),
    # then fall back to legacy pre-uploaded parquets.
    hwm = s3_read(s3, f"raw/surge_estimates/{event}/hwm_{event}.parquet")
    if hwm is None or hwm.empty:
        hwm = s3_read(s3, f"raw/usgs_stn/{event}_hwm.parquet")
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
# 311 reports (Houston / NYC / New Orleans)
# ---------------------------------------------------------------------------

def _assign_zcta_by_proximity(
    s3, reports: pd.DataFrame, zcta_ids: list[str],
) -> pd.DataFrame:
    """Assign ZCTA to geocoded 311 records by nearest centroid (< 5 km)."""
    geo = s3_read(s3, "raw/geocertdb2026/zcta_features_labels.parquet")
    if geo is None or "latitude" not in geo.columns:
        return reports
    geo = geo[geo["zcta_id"].isin(zcta_ids)][["zcta_id", "latitude", "longitude"]].dropna()
    if geo.empty:
        return reports

    valid = reports.dropna(subset=["latitude", "longitude"]).copy()
    if valid.empty:
        return reports

    assigned_zctas = []
    geo_lat = np.radians(geo["latitude"].values)
    geo_lon = np.radians(geo["longitude"].values)
    for _, row in valid.iterrows():
        rlat = np.radians(row["latitude"])
        rlon = np.radians(row["longitude"])
        dlat = geo_lat - rlat
        dlon = geo_lon - rlon
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(rlat) * np.cos(geo_lat) * np.sin(dlon / 2) ** 2)
        dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
        idx = dist.argmin()
        if dist[idx] <= 5.0:
            assigned_zctas.append(geo.iloc[idx]["zcta_id"])
        else:
            assigned_zctas.append(None)

    valid["zcta_id"] = assigned_zctas
    valid = valid.dropna(subset=["zcta_id"])
    return valid


def aggregate_311(s3, source: str, event: str, zcta_ids: list[str]) -> pd.DataFrame:
    """source: 'houston', 'nyc', or 'new_orleans'"""
    if source == "houston":
        key = f"raw/houston_311/{event}_311.parquet"
    elif source == "new_orleans":
        key = f"raw/nola_311/{event}_flooding_311.parquet"
    else:
        key = f"raw/nyc_311/{event}_flooding_311.parquet"

    reports = s3_read(s3, key)
    empty = pd.DataFrame({"zcta_id": zcta_ids, "flood_311_count": 0,
                           "obs_has_311": False})
    if reports is None or reports.empty:
        return empty

    # NOLA current dataset (2jgv-pqrq) has lat/lon but no ZIP column.
    # Assign to nearest ZCTA by centroid distance if zcta_id is missing.
    if "zcta_id" not in reports.columns and source == "new_orleans":
        reports = _assign_zcta_by_proximity(s3, reports, zcta_ids)

    if "zcta_id" not in reports.columns:
        return empty

    agg = (reports.groupby("zcta_id").size()
           .reset_index(name="flood_311_count"))
    agg["obs_has_311"] = True

    all_zctas = pd.DataFrame({"zcta_id": zcta_ids})
    result = all_zctas.merge(agg, on="zcta_id", how="left")
    result["flood_311_count"] = result["flood_311_count"].fillna(0).astype(int)
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
    all_objs = _list_s3_keys(s3, prefix_pattern)
    keys = [o["Key"] for o in all_objs
            if event in o["Key"] and o["Key"].endswith(".parquet")]
    empty = pd.DataFrame({"zcta_id": zcta_ids, "max_surge_m": np.nan,
                           "max_water_level_m": np.nan})
    if not keys:
        log.warning("No tide files matching %s / %s", prefix_pattern, event)
        return empty

    frames = []
    for key in keys:
        df = s3_read(s3, key)
        if df is None:
            continue
        # Handle both column naming conventions:
        # fetch_noaa_tides.py writes "observed_m", fetch_noaa_tides_swfl.py writes "water_level_m"
        wl_col = next((c for c in ["observed_m", "water_level_m"] if c in df.columns), None)
        if wl_col is not None:
            station_id = df["station_id"].iloc[0] if "station_id" in df.columns else key
            peak_wl = float(df[wl_col].max())
            # surge_m = observed - predicted. If predictions missing, fall back
            # to observed water level as surge proxy (conservative: includes tide)
            peak_surge = np.nan
            if "surge_m" in df.columns and df["surge_m"].notna().any():
                peak_surge = float(df["surge_m"].max())
            elif peak_wl > 0:
                # Fallback: use observed water level as surge proxy
                # This overestimates by tidal amplitude (~0.3m Gulf, ~1.5m Atlantic)
                # but is far better than NaN for hazard characterization
                peak_surge = peak_wl
                log.info("  Station %s: surge_m unavailable, using observed WL=%.2fm as proxy",
                         station_id, peak_wl)
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
    houston_fips = ["48201", "48157", "48339", "48167", "48039", "48071"]
    harris_zctas = xwalk[xwalk["county_fips"].isin(houston_fips)]["zcta_id"].tolist()
    log.info("Houston: %d metro ZCTAs (6 counties)", len(harris_zctas))

    static = load_geocert_static(s3, "houston", harris_zctas)

    event_map = {
        "harvey2017": {"dr": 4332, "storm_id": "AL092017",
                       "peak_window": ("2017-08-25", "2017-09-02")},
        "imelda2019": {"dr": 4466, "storm_id": "AL112019",
                       "peak_window": ("2019-09-17", "2019-09-21")},
        "beryl2024":  {"dr": 4781, "storm_id": "AL022024",
                       "peak_window": ("2024-07-08", "2024-07-12")},
    }

    # Feature contract derived fields (computed once, shared across events)
    impervious = build_impervious_features(s3, harris_zctas)
    cropland   = build_cropland_features(s3, harris_zctas)
    jrc_water  = build_jrc_water_features(s3, harris_zctas)
    catchments = build_catchment_features(s3, harris_zctas, vpu="12")
    levees     = build_levee_features(s3, harris_zctas, "houston")
    elevation  = build_elevation_features(s3, harris_zctas, "houston")
    deltares   = build_deltares_depth_features(s3, harris_zctas)
    hydrology  = build_hydrology_features(s3, harris_zctas, "houston")
    buildings  = build_building_features(s3, harris_zctas)
    depressions = build_depression_features(s3, harris_zctas)
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
        tides = aggregate_tides(s3, "raw/noaa_tides/", event_name, harris_zctas)
        s311  = aggregate_311(s3, "houston", event_name, harris_zctas)
        nfip  = load_nfip_event_claims(s3, ev["dr"], harris_zctas)
        storm = compute_storm_proximity(s3, ev["storm_id"], ev["peak_window"], harris_zctas)
        sar_ev = build_sentinel1_event_features(
            s3, harris_zctas,
            {"peak_window": ev["peak_window"], "s3_event_key": ev.get("s3_event_key", event_name), "scenario": "houston"},
            jrc_water,
        )

        base = pd.DataFrame({"zcta_id": harris_zctas, "event": event_name,
                              "scenario": "houston"})
        base = _safe_merge_parts(base, [nwis, mrms, hwm, tides, s311, nfip, storm,
                                        sar_ev, impervious, cropland, jrc_water, catchments,
                                        levees, elevation, drainage_op, deltares, hydrology,
                                        buildings, depressions])
        if not static.empty:
            base = _safe_merge_parts(base, [static])
        w_feats = compute_w_matrix_features(s3, harris_zctas, static, base)
        base = _safe_merge_parts(base, [w_feats])
        base = compute_observability_flags(base)
        rows.append(base)

    df = pd.concat(rows, ignore_index=True)
    _add_certificate_slots(df)
    return _attach_geometry(df)


def build_new_orleans(s3, cfg: dict) -> pd.DataFrame:
    """Assemble New Orleans (zcta, event) table."""
    xwalk = s3_read(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
    # 5-parish Greater New Orleans metro: Orleans, Jefferson, Plaquemines,
    # St. Bernard, St. Tammany (~66 ZCTAs, up from 20 Orleans-only)
    metro_fips = ["22051", "22071", "22075", "22087", "22103"]
    no_zctas = xwalk[xwalk["county_fips"].isin(metro_fips)]["zcta_id"].tolist()
    log.info("New Orleans metro: %d ZCTAs across %d parishes", len(no_zctas), len(metro_fips))

    static = load_geocert_static(s3, "new_orleans", no_zctas)

    # Feature contract derived fields
    impervious   = build_impervious_features(s3, no_zctas)
    cropland     = build_cropland_features(s3, no_zctas)
    jrc_water    = build_jrc_water_features(s3, no_zctas)
    levee_feats  = build_levee_features(s3, no_zctas, "new_orleans")
    elevation    = build_elevation_features(s3, no_zctas, "new_orleans")
    coastal_dist = build_coastal_distance_features(s3, no_zctas)
    deltares     = build_deltares_depth_features(s3, no_zctas)
    hydrology    = build_hydrology_features(s3, no_zctas, "new_orleans")
    buildings    = build_building_features(s3, no_zctas)
    depressions  = build_depression_features(s3, no_zctas)
    # pump_station_status: operational (Ida 2021 partially hand-coded)
    pump_evidence = _load_local_evidence("/opt/ml/processing/input/evidence/no_pump_stations_ida2021.csv")
    # pump_station_status for non-Ida events: operational_unknown
    pump_op = _operational_unknown(
        pd.DataFrame({"zcta_id": no_zctas}),
        "pump_station_status",
        note="S&WB pump telemetry not publicly archived; Ida 2021 partially available via evidence CSV.",
    )

    event_map = {
        "katrina2005": {"dr": 1603, "storm_id": "AL122005",
                        "peak_window": ("2005-08-29", "2005-08-30"),
                        "s3_event_key": "katrina2005_nola"},
        "isaac2012":   {"dr": 4080, "storm_id": "AL092012",
                        "peak_window": ("2012-08-28", "2012-08-30"),
                        "s3_event_key": "isaac2012_nola"},
        "barry2019":   {"dr": 4458, "storm_id": "AL022019",
                        "peak_window": ("2019-07-12", "2019-07-14"),
                        "s3_event_key": "barry2019_nola"},
        "ida2021":     {"dr": 4611, "storm_id": "AL092021",
                        "peak_window": ("2021-08-29", "2021-09-01"),
                        "s3_event_key": "ida2021_nola"},
    }

    rows = []
    for event_name, ev in event_map.items():
        s3_key = ev.get("s3_event_key", event_name)
        log.info("New Orleans event: %s (S3 key: %s)", event_name, s3_key)
        nwis   = aggregate_nwis(s3, "new_orleans", event_name, no_zctas, cfg)
        mrms   = aggregate_mrms_rainfall(s3, s3_key, no_zctas)
        tides  = aggregate_tides(s3, "raw/noaa_tides/", s3_key, no_zctas)
        hwm    = aggregate_hwm(s3, s3_key, no_zctas)
        s311   = aggregate_311(s3, "new_orleans", event_name, no_zctas)
        nfip   = load_nfip_event_claims(s3, ev["dr"], no_zctas)
        storm  = compute_storm_proximity(s3, ev["storm_id"], ev["peak_window"], no_zctas)
        slosh  = build_slosh_features(s3, s3_key, no_zctas)
        sar_ev = build_sentinel1_event_features(
            s3, no_zctas,
            {"peak_window": ev["peak_window"], "s3_event_key": ev.get("s3_event_key", event_name), "scenario": "new_orleans"},
            jrc_water,
        )

        base = pd.DataFrame({"zcta_id": no_zctas, "event": event_name,
                              "scenario": "new_orleans"})
        base = _safe_merge_parts(base, [nwis, mrms, tides, hwm, s311, nfip, storm,
                                        slosh, sar_ev, impervious, cropland, jrc_water,
                                        levee_feats, elevation, coastal_dist,
                                        pump_op, deltares, hydrology,
                                        buildings, depressions])
        if not static.empty:
            base = _safe_merge_parts(base, [static])
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
        w_feats = compute_w_matrix_features(s3, no_zctas, static, base)
        base = _safe_merge_parts(base, [w_feats])
        base = compute_observability_flags(base)
        rows.append(base)

    df = pd.concat(rows, ignore_index=True)
    _add_certificate_slots(df)
    return _attach_geometry(df)


def build_nyc(s3, cfg: dict) -> pd.DataFrame:
    """Assemble NYC (zcta, event) table."""
    xwalk = s3_read(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
    nyc_fips = ["36061", "36047", "36081", "36005", "36085"]
    nyc_zctas = xwalk[xwalk["county_fips"].isin(nyc_fips)]["zcta_id"].tolist()
    log.info("NYC: %d ZCTAs", len(nyc_zctas))

    static = load_geocert_static(s3, "nyc", nyc_zctas)

    # Feature contract derived fields
    impervious   = build_impervious_features(s3, nyc_zctas)
    cropland     = build_cropland_features(s3, nyc_zctas)
    jrc_water    = build_jrc_water_features(s3, nyc_zctas)
    elevation    = build_elevation_features(s3, nyc_zctas, "nyc")
    sewer_feats  = build_sewershed_features(s3, nyc_zctas)
    subway_feats = build_subway_features(s3, nyc_zctas)
    deltares     = build_deltares_depth_features(s3, nyc_zctas)
    hydrology    = build_hydrology_features(s3, nyc_zctas, "nyc")
    buildings    = build_building_features(s3, nyc_zctas)
    depressions  = build_depression_features(s3, nyc_zctas)
    subway_evidence = _load_local_evidence("/opt/ml/processing/input/evidence/nyc_subway_flooding_ida2021.csv")

    event_map = {
        "sandy2012":     {"dr": 4085, "storm_id": "AL182012",
                          "peak_window": ("2012-10-29", "2012-10-30")},
        "ida2021":       {"dr": 4615, "storm_id": "AL092021",
                          "peak_window": ("2021-09-01", "2021-09-02"),
                          "s3_event_key": "ida2021_nyc"},
        "henri2021":     {"dr": None, "storm_id": "AL082021",
                          "peak_window": ("2021-08-21", "2021-08-22")},
        "nyc_flood_2023": {"dr": 4755, "storm_id": None,
                           "peak_window": ("2023-09-29", "2023-09-30")},
    }

    rows = []
    for event_name, ev in event_map.items():
        s3_key = ev.get("s3_event_key", event_name)
        log.info("NYC event: %s (S3 key: %s)", event_name, s3_key)
        nwis  = aggregate_nwis(s3, "nyc", event_name, nyc_zctas, cfg)
        mrms  = aggregate_mrms_rainfall(s3, s3_key, nyc_zctas)
        tides = aggregate_tides(s3, "raw/noaa_tides/", s3_key, nyc_zctas)
        hwm   = aggregate_hwm(s3, s3_key, nyc_zctas)
        s311  = aggregate_311(s3, "nyc", event_name, nyc_zctas)
        storm = compute_storm_proximity(s3, ev["storm_id"], ev["peak_window"], nyc_zctas)
        nfip  = (load_nfip_event_claims(s3, ev["dr"], nyc_zctas)
                 if ev["dr"] else pd.DataFrame({"zcta_id": nyc_zctas}))
        sar_ev = build_sentinel1_event_features(
            s3, nyc_zctas,
            {"peak_window": ev["peak_window"], "s3_event_key": ev.get("s3_event_key", event_name), "scenario": "nyc"},
            jrc_water,
        )

        base = pd.DataFrame({"zcta_id": nyc_zctas, "event": event_name,
                              "scenario": "nyc"})
        base = _safe_merge_parts(base, [nwis, mrms, tides, hwm, s311, nfip, storm,
                                        sar_ev, impervious, cropland, jrc_water,
                                        elevation, sewer_feats, subway_feats,
                                        deltares, hydrology,
                                        buildings, depressions])
        if not static.empty:
            base = _safe_merge_parts(base, [static])
        # Subway flooding evidence (hand-coded Ida 2021 overlay)
        if event_name == "ida2021" and subway_evidence is not None and "flooding_observed" in subway_evidence.columns:
            flooded = subway_evidence[subway_evidence["flooding_observed"] == True]
            # TODO: spatial join flooded stations to ZCTAs when station coords available
            base["subway_flooded_count_nearby"] = 0  # placeholder until spatial join
        w_feats = compute_w_matrix_features(s3, nyc_zctas, static, base)
        base = _safe_merge_parts(base, [w_feats])
        base = compute_observability_flags(base)
        rows.append(base)

    df = pd.concat(rows, ignore_index=True)
    _add_certificate_slots(df)
    return _attach_geometry(df)


def build_riverside_coachella(s3, cfg: dict) -> pd.DataFrame:
    """Assemble Riverside-Coachella (zcta, event) variogram-input table."""
    xwalk = s3_read(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
    rc_fips = ["06065", "06025"]
    rc_zctas = xwalk[xwalk["county_fips"].isin(rc_fips)]["zcta_id"].tolist()
    log.info("Riverside-Coachella: %d ZCTAs", len(rc_zctas))

    static = load_geocert_static(s3, "riverside_coachella", rc_zctas)

    # Feature contract derived fields
    impervious  = build_impervious_features(s3, rc_zctas)
    cropland    = build_cropland_features(s3, rc_zctas)
    jrc_water   = build_jrc_water_features(s3, rc_zctas)
    burn_scars  = build_burn_scar_features(s3, rc_zctas)
    catchments  = build_catchment_features(s3, rc_zctas, vpu="18")
    elevation   = build_elevation_features(s3, rc_zctas, "socal")
    deltares    = build_deltares_depth_features(s3, rc_zctas)
    hydrology   = build_hydrology_features(s3, rc_zctas, "riverside_coachella")
    buildings   = build_building_features(s3, rc_zctas)
    depressions = build_depression_features(s3, rc_zctas)
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
        hwm   = aggregate_hwm(s3, event_name, rc_zctas)
        nfip  = (load_nfip_event_claims(s3, ev["dr"], rc_zctas)
                 if ev["dr"] else pd.DataFrame({"zcta_id": rc_zctas}))
        storm = (compute_storm_proximity(s3, ev["storm_id"], ev["peak_window"], rc_zctas)
                 if ev["storm_id"] else pd.DataFrame({"zcta_id": rc_zctas}))
        sar_ev = build_sentinel1_event_features(
            s3, rc_zctas,
            {"peak_window": ev["peak_window"], "s3_event_key": ev.get("s3_event_key", event_name), "scenario": "riverside_coachella"},
            jrc_water,
        )

        base = pd.DataFrame({"zcta_id": rc_zctas, "event": event_name,
                              "scenario": "riverside_coachella"})
        base = _safe_merge_parts(base, [nwis, mrms, hwm, nfip, storm,
                                        sar_ev, impervious, cropland, jrc_water,
                                        burn_scars, catchments, elevation,
                                        road_op, deltares, hydrology,
                                        buildings, depressions])
        if not static.empty:
            base = _safe_merge_parts(base, [static])
        w_feats = compute_w_matrix_features(s3, rc_zctas, static, base)
        base = _safe_merge_parts(base, [w_feats])
        base = compute_observability_flags(base)
        rows.append(base)

    df = pd.concat(rows, ignore_index=True)
    _add_certificate_slots(df)
    return _attach_geometry(df)


def build_southwest_florida(s3, cfg: dict) -> pd.DataFrame:
    """Assemble SW Florida (zcta, event) variogram-input table."""
    xwalk = s3_read(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
    swfl_fips = ["12021", "12071", "12015", "12115", "12081", "12057", "12103"]
    swfl_zctas = xwalk[xwalk["county_fips"].isin(swfl_fips)]["zcta_id"].tolist()
    log.info("SW Florida: %d ZCTAs", len(swfl_zctas))

    static = load_geocert_static(s3, "southwest_florida", swfl_zctas)

    # Feature contract derived fields
    impervious      = build_impervious_features(s3, swfl_zctas)
    cropland        = build_cropland_features(s3, swfl_zctas)
    jrc_water       = build_jrc_water_features(s3, swfl_zctas)
    elevation       = build_elevation_features(s3, swfl_zctas, "southwest_florida")
    coastal_dist    = build_coastal_distance_features(s3, swfl_zctas)
    levee_feats     = build_levee_features(s3, swfl_zctas, "southwest_florida")
    deltares        = build_deltares_depth_features(s3, swfl_zctas)
    hydrology       = build_hydrology_features(s3, swfl_zctas, "southwest_florida")
    buildings       = build_building_features(s3, swfl_zctas)
    depressions     = build_depression_features(s3, swfl_zctas)
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
        hwm   = aggregate_hwm(s3, event_name, swfl_zctas)
        nfip  = load_nfip_event_claims(s3, ev["dr"], swfl_zctas)
        storm = compute_storm_proximity(s3, ev["storm_id"], ev["peak_window"], swfl_zctas)

        slosh = build_slosh_features(s3, event_name, swfl_zctas)
        sar_ev = build_sentinel1_event_features(
            s3, swfl_zctas,
            {"peak_window": ev["peak_window"], "s3_event_key": ev.get("s3_event_key", event_name), "scenario": "southwest_florida"},
            jrc_water,
        )

        base = pd.DataFrame({"zcta_id": swfl_zctas, "event": event_name,
                              "scenario": "southwest_florida"})
        base = _safe_merge_parts(base, [nwis, mrms, tides, hwm, nfip, storm,
                                        slosh, sar_ev, impervious, cropland, jrc_water,
                                        elevation, coastal_dist, levee_feats,
                                        evac_op, deltares, hydrology,
                                        buildings, depressions])
        if not static.empty:
            base = _safe_merge_parts(base, [static])
        w_feats = compute_w_matrix_features(s3, swfl_zctas, static, base)
        base = _safe_merge_parts(base, [w_feats])
        base = compute_observability_flags(base)
        rows.append(base)

    df = pd.concat(rows, ignore_index=True)
    _add_certificate_slots(df)
    return _attach_geometry(df)


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


_JRC_WATER_CACHE_KEY = "processed/shared/zcta_jrc_water_occurrence_pct.parquet"

_IMPERVIOUS_CACHE_KEY = "processed/shared/zcta_impervious_pct.parquet"

_DELTARES_DEPTH_CACHE_KEY = "processed/shared/zcta_deltares_depth.parquet"
_HYDROLOGY_CACHE_KEY = "processed/shared/zcta_hydrology.parquet"
_BUILDINGS_CACHE_KEY = "processed/shared/zcta_buildings.parquet"
_DEPRESSIONS_CACHE_KEY = "processed/shared/zcta_depressions.parquet"


def build_impervious_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Derive impervious_pct per ZCTA from NLCD 2021 GeoTIFFs.

    Cache-first: checks for pre-computed parquet at
    s3://{BUCKET}/{_IMPERVIOUS_CACHE_KEY} before doing raster extraction.
    The cache is ZCTA-level and scenario-independent — compute once, share
    across all scenario builds.

    Requires: raw/nlcd/impervious/v2021/*.tif + ZCTA centroid lat/lon from geocertdb2026.
    Returns DataFrame with columns: zcta_id, impervious_pct, _fs_impervious_pct.
    """
    empty = pd.DataFrame({"zcta_id": zcta_ids, "impervious_pct": np.nan,
                          "_fs_impervious_pct": _FS_MISSING})

    # --- Cache lookup: skip raster extraction if pre-computed values exist ---
    try:
        cached = s3_read(s3, _IMPERVIOUS_CACHE_KEY)
        if cached is not None and "zcta_id" in cached.columns:
            cached["zcta_id"] = cached["zcta_id"].astype(str)
            out = pd.DataFrame({"zcta_id": zcta_ids}).merge(
                cached[["zcta_id", "impervious_pct"]], on="zcta_id", how="left",
            )
            hit_rate = out["impervious_pct"].notna().mean() * 100
            if hit_rate > 50:
                out["_fs_impervious_pct"] = np.where(
                    out["impervious_pct"].notna(), "present", _FS_MISSING,
                )
                log.info(
                    "build_impervious_features: cache hit (%s), "
                    "%d/%d ZCTAs matched (%.0f%%)",
                    _IMPERVIOUS_CACHE_KEY, out["impervious_pct"].notna().sum(),
                    len(zcta_ids), hit_rate,
                )
                return out
            else:
                log.info(
                    "build_impervious_features: cache exists but only %.0f%% "
                    "ZCTAs matched — falling through to raster extraction",
                    hit_rate,
                )
    except Exception as e:
        log.info("build_impervious_features: no cache (%s), will extract from raster", e)

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

    # Find NLCD raster: prefer pre-converted .tif over raw .img.
    # The .tif is a lossless conversion (gdal_translate -co COMPRESS=LZW)
    # that pip-installed rasterio can read without HFA driver.
    nlcd_key = None
    _nlcd_img_fallback = None
    for prefix in ["raw/nlcd/impervious_2021/", "raw/nlcd/impervious/v2021/"]:
        all_objs = _list_s3_keys(s3, prefix)
        for o in all_objs:
            k = o["Key"]
            if k.endswith((".tif", ".tiff")):
                nlcd_key = k
                break
            elif k.endswith(".img") and _nlcd_img_fallback is None:
                _nlcd_img_fallback = k
        if nlcd_key:
            break
    if not nlcd_key:
        nlcd_key = _nlcd_img_fallback
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

    # .img (Erdas Imagine HFA) is not readable by pip-installed rasterio and
    # causes segfault via python GDAL bindings. Convert to GeoTIFF using
    # gdal_translate (CLI), which reads HFA reliably. The output .tif is
    # readable by pip-installed rasterio without any driver issues.
    _use_gdal_fallback = False
    if local_raster.endswith(".img"):
        import subprocess
        tif_path = local_raster.replace(".img", ".tif")
        log.info("build_impervious_features: converting .img -> .tif via gdal_translate")
        result = subprocess.run(
            ["gdal_translate", "-of", "GTiff", "-co", "COMPRESS=LZW",
             local_raster, tif_path],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode == 0 and Path(tif_path).exists():
            log.info("build_impervious_features: conversion OK (%.1f GB)",
                     Path(tif_path).stat().st_size / 1e9)
            # Remove original .img to free disk space
            Path(local_raster).unlink(missing_ok=True)
            local_raster = tif_path
        else:
            log.error("build_impervious_features: gdal_translate failed: %s",
                      result.stderr[:500])
            return empty

    # Use windowed reads per ZCTA centroid (avoids loading full 26 GB into RAM)
    # NLCD is in EPSG:5070 (Albers Equal Area, meters) -- must reproject centroid
    results = []

    if _use_gdal_fallback:
        # osgeo.gdal can read .img (HFA) even when pip-installed rasterio cannot
        from osgeo import gdal, osr
        from pyproj import Transformer
        gdal.UseExceptions()
        ds = gdal.Open(local_raster)
        if ds is None:
            log.error("build_impervious_features: gdal.Open also failed; returning NaN")
            return empty
        gt = ds.GetGeoTransform()  # (x_origin, pixel_w, 0, y_origin, 0, pixel_h)
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue() if band.GetNoDataValue() is not None else 127
        # Get raster SRS for reprojection
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(ds.GetProjection())
        epsg = raster_srs.GetAuthorityCode(None) or "5070"
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        delta = 500  # meters in EPSG:5070
        for _, row in centroids.iterrows():
            x, y = transformer.transform(row["lon"], row["lat"])
            # Convert geo coords to pixel coords for the 1km buffer window
            col_min = int((x - delta - gt[0]) / gt[1])
            row_min = int((y - delta - gt[3]) / gt[5])
            col_max = int((x + delta - gt[0]) / gt[1])
            row_max = int((y + delta - gt[3]) / gt[5])
            # Clamp to raster extent
            col_min, col_max = max(0, min(col_min, col_max)), min(ds.RasterXSize, max(col_min, col_max))
            row_min, row_max = max(0, min(row_min, row_max)), min(ds.RasterYSize, max(row_min, row_max))
            w = col_max - col_min
            h = row_max - row_min
            if w <= 0 or h <= 0:
                results.append({"zcta_id": row["zcta_id"], "impervious_pct": np.nan})
                continue
            try:
                arr = band.ReadAsArray(col_min, row_min, w, h)
                valid = arr[(arr != nodata) & (arr >= 0) & (arr <= 100)]
                imp_pct = float(np.mean(valid)) if len(valid) > 0 else np.nan
            except Exception:
                imp_pct = np.nan
            results.append({"zcta_id": row["zcta_id"], "impervious_pct": imp_pct})
        ds = None  # close
    else:
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

    extracted = pd.DataFrame(results).groupby("zcta_id", as_index=False)["impervious_pct"].mean()

    # --- Cache write: save ALL extracted ZCTAs (merge with any existing cache) ---
    try:
        existing = s3_read(s3, _IMPERVIOUS_CACHE_KEY)
        if existing is not None and "zcta_id" in existing.columns:
            existing["zcta_id"] = existing["zcta_id"].astype(str)
            # Merge: new extractions overwrite stale values
            combined = pd.concat([existing, extracted], ignore_index=True)
            combined = combined.drop_duplicates(subset="zcta_id", keep="last")
        else:
            combined = extracted
        buf = BytesIO()
        combined.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=_IMPERVIOUS_CACHE_KEY, Body=buf.getvalue())
        log.info("build_impervious_features: cached %d ZCTAs to %s",
                 len(combined), _IMPERVIOUS_CACHE_KEY)
    except Exception as e:
        log.warning("build_impervious_features: cache write failed: %s", e)

    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(extracted, on="zcta_id", how="left")
    out["_fs_impervious_pct"] = np.where(out["impervious_pct"].notna(), "present", _FS_MISSING)
    log.info("build_impervious_features: %d ZCTAs, %.1f%% with data",
             len(out), (out["impervious_pct"].notna().mean() * 100))
    return out


_CROPLAND_CACHE_KEY = "processed/shared/zcta_cropland_pct.parquet"

# NLCD 2021 land cover classes for cropland
_CROPLAND_CLASSES = frozenset({81, 82})  # 81=Pasture/Hay, 82=Cultivated Crops


def build_cropland_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Derive cropland_pct per ZCTA from NLCD 2021 Land Cover raster.

    Categorical raster: each 30m pixel is one of 16 land cover classes.
    cropland_pct = percentage of valid pixels within 500m of ZCTA centroid
    that are class 81 (Pasture/Hay) or 82 (Cultivated Crops).

    Cache-first: checks for pre-computed parquet at
    s3://{BUCKET}/{_CROPLAND_CACHE_KEY} before raster extraction.

    Returns DataFrame with columns: zcta_id, cropland_pct, _fs_cropland_pct.
    """
    empty = pd.DataFrame({"zcta_id": zcta_ids, "cropland_pct": np.nan,
                          "_fs_cropland_pct": _FS_MISSING})

    # --- Cache lookup ---
    try:
        cached = s3_read(s3, _CROPLAND_CACHE_KEY)
        if cached is not None and "zcta_id" in cached.columns:
            cached["zcta_id"] = cached["zcta_id"].astype(str)
            out = pd.DataFrame({"zcta_id": zcta_ids}).merge(
                cached[["zcta_id", "cropland_pct"]], on="zcta_id", how="left",
            )
            hit_rate = out["cropland_pct"].notna().mean() * 100
            if hit_rate > 50:
                out["_fs_cropland_pct"] = np.where(
                    out["cropland_pct"].notna(), "present", _FS_MISSING,
                )
                log.info(
                    "build_cropland_features: cache hit (%s), "
                    "%d/%d ZCTAs matched (%.0f%%)",
                    _CROPLAND_CACHE_KEY, out["cropland_pct"].notna().sum(),
                    len(zcta_ids), hit_rate,
                )
                return out
    except Exception as e:
        log.info("build_cropland_features: no cache (%s), will extract from raster", e)

    if not HAS_GEO:
        log.warning("build_cropland_features: geopandas/rasterio not available; returning NaN")
        return empty

    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import mapping, box
    except ImportError:
        log.warning("build_cropland_features: rasterio not available; returning NaN")
        return empty

    # Find NLCD land cover raster: prefer .tif over .img
    nlcd_key = None
    _nlcd_img_fallback = None
    for prefix in ["raw/nlcd/land_cover_2021/", "raw/nlcd/landcover/v2021/"]:
        all_objs = _list_s3_keys(s3, prefix)
        for o in all_objs:
            k = o["Key"]
            if k.endswith((".tif", ".tiff")):
                nlcd_key = k
                break
            elif k.endswith(".img") and _nlcd_img_fallback is None:
                _nlcd_img_fallback = k
        if nlcd_key:
            break
    if not nlcd_key:
        nlcd_key = _nlcd_img_fallback
    if not nlcd_key:
        log.warning("build_cropland_features: no NLCD land cover raster found in S3; returning NaN")
        return empty

    log.info("build_cropland_features: using %s", nlcd_key)

    # Load ZCTA centroids
    static_key = "raw/geocertdb2026/zcta_features_labels.parquet"
    static = s3_read(s3, static_key)
    if static is None:
        return empty
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        log.warning("build_cropland_features: missing zcta/lat/lon in geocertdb2026; returning NaN")
        return empty

    static = static[[zcta_col, lat_col, lon_col]].rename(
        columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"}
    )
    centroids = static[static["zcta_id"].isin(zcta_ids)].dropna(subset=["lat", "lon"])

    # Download raster to /tmp
    local_raster = f"/tmp/nlcd_lc_{Path(nlcd_key).stem}{Path(nlcd_key).suffix}"
    log.info("build_cropland_features: downloading %s", nlcd_key)
    s3.download_file(BUCKET, nlcd_key, local_raster)

    # Convert .img -> .tif if needed
    if local_raster.endswith(".img"):
        import subprocess as _sp
        tif_path = local_raster.replace(".img", ".tif")
        log.info("build_cropland_features: converting .img -> .tif via gdal_translate")
        result = _sp.run(
            ["gdal_translate", "-of", "GTiff", "-co", "COMPRESS=LZW",
             local_raster, tif_path],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode == 0 and Path(tif_path).exists():
            Path(local_raster).unlink(missing_ok=True)
            local_raster = tif_path
        else:
            log.error("build_cropland_features: gdal_translate failed: %s",
                      result.stderr[:500])
            return empty

    # Extract cropland percentage per ZCTA centroid (500m buffer)
    results = []
    with rasterio.open(local_raster) as src:
        nodata = src.nodata if src.nodata is not None else 0
        raster_crs = src.crs
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        is_projected = raster_crs.is_projected
        for _, row in centroids.iterrows():
            x, y = transformer.transform(row["lon"], row["lat"])
            delta = 500 if is_projected else 0.005
            geom = [mapping(box(x - delta, y - delta, x + delta, y + delta))]
            try:
                masked, _ = rio_mask(src, geom, crop=True, nodata=nodata)
                vals = masked[0].flatten()
                # Exclude nodata and unclassified (0)
                valid = vals[(vals != nodata) & (vals > 0)]
                if len(valid) > 0:
                    crop_count = np.isin(valid, list(_CROPLAND_CLASSES)).sum()
                    crop_pct = float(crop_count / len(valid) * 100)
                else:
                    crop_pct = np.nan
            except Exception:
                crop_pct = np.nan
            results.append({"zcta_id": row["zcta_id"], "cropland_pct": crop_pct})

    # Cleanup local file
    try:
        Path(local_raster).unlink()
    except OSError:
        pass

    if not results:
        return empty

    extracted = pd.DataFrame(results).groupby("zcta_id", as_index=False)["cropland_pct"].mean()

    # --- Cache write ---
    try:
        existing = s3_read(s3, _CROPLAND_CACHE_KEY)
        if existing is not None and "zcta_id" in existing.columns:
            existing["zcta_id"] = existing["zcta_id"].astype(str)
            combined = pd.concat([existing, extracted], ignore_index=True)
            combined = combined.drop_duplicates(subset="zcta_id", keep="last")
        else:
            combined = extracted
        buf = BytesIO()
        combined.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=_CROPLAND_CACHE_KEY, Body=buf.getvalue())
        log.info("build_cropland_features: cached %d ZCTAs to %s",
                 len(combined), _CROPLAND_CACHE_KEY)
    except Exception as e:
        log.warning("build_cropland_features: cache write failed: %s", e)

    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(extracted, on="zcta_id", how="left")
    out["_fs_cropland_pct"] = np.where(out["cropland_pct"].notna(), "present", _FS_MISSING)
    log.info("build_cropland_features: %d ZCTAs, %.1f%% with data",
             len(out), (out["cropland_pct"].notna().mean() * 100))
    return out


def build_jrc_water_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Derive JRC water occurrence stats per ZCTA from Planetary Computer.

    Uses floodcaster.stac.jrc_centroid_occurrence to fetch JRC Global Surface
    Water (1984-2020) from Microsoft Planetary Computer STAC API. Extracts
    mean/max occurrence and pct_ever_wet within ~1 km of each ZCTA centroid.

    Cache-first: checks for pre-computed parquet at
    s3://{BUCKET}/{_JRC_WATER_CACHE_KEY} before STAC extraction.

    Returns DataFrame with columns:
        zcta_id, jrc_occurrence_mean, jrc_occurrence_max, jrc_pct_ever_wet,
        _fs_jrc_occurrence_mean.
    """
    empty = pd.DataFrame({
        "zcta_id": zcta_ids,
        "jrc_occurrence_mean": np.nan,
        "jrc_occurrence_max": np.nan,
        "jrc_pct_ever_wet": np.nan,
        "_fs_jrc_occurrence_mean": _FS_MISSING,
    })

    # --- Cache lookup ---
    try:
        cached = s3_read(s3, _JRC_WATER_CACHE_KEY)
        if cached is not None and "zcta_id" in cached.columns:
            cached["zcta_id"] = cached["zcta_id"].astype(str)
            out = pd.DataFrame({"zcta_id": zcta_ids}).merge(
                cached[["zcta_id", "jrc_occurrence_mean", "jrc_occurrence_max", "jrc_pct_ever_wet"]],
                on="zcta_id", how="left",
            )
            hit_rate = out["jrc_occurrence_mean"].notna().mean()
            if hit_rate > 0.5:
                out["_fs_jrc_occurrence_mean"] = np.where(
                    out["jrc_occurrence_mean"].notna(), "present", _FS_MISSING,
                )
                log.info("build_jrc_water_features: cache hit (%.0f%% coverage)", hit_rate * 100)
                return out
    except Exception as e:
        log.info("build_jrc_water_features: no cache (%s), extracting from STAC", e)

    # --- Load ZCTA centroids ---
    static_key = "raw/geocertdb2026/zcta_features_labels.parquet"
    static = s3_read(s3, static_key)
    if static is None:
        log.warning("build_jrc_water_features: geocertdb2026 not available; returning NaN")
        return empty

    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        log.warning("build_jrc_water_features: missing zcta/lat/lon columns; returning NaN")
        return empty

    static = static[[zcta_col, lat_col, lon_col]].rename(
        columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"},
    )
    centroids = static[static["zcta_id"].isin(zcta_ids)].dropna(subset=["lat", "lon"])

    if centroids.empty:
        log.warning("build_jrc_water_features: no centroids matched; returning NaN")
        return empty

    # --- Extract from Planetary Computer via floodcaster ---
    try:
        extracted = _run_in_subprocess(
            "floodcaster.stac", "jrc_centroid_occurrence",
            centroids, id_col="zcta_id", timeout=600,
        )
        if extracted is None:
            log.warning("build_jrc_water_features: STAC extraction crashed (segfault); returning NaN")
            return empty
    except Exception as e:
        log.warning("build_jrc_water_features: STAC extraction failed: %s", e)
        return empty

    # --- Cache write ---
    try:
        existing = s3_read(s3, _JRC_WATER_CACHE_KEY)
        if existing is not None and "zcta_id" in existing.columns:
            existing["zcta_id"] = existing["zcta_id"].astype(str)
            combined = pd.concat([existing, extracted], ignore_index=True)
            combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")
        else:
            combined = extracted
        buf = BytesIO()
        combined.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=_JRC_WATER_CACHE_KEY, Body=buf.getvalue())
        log.info("build_jrc_water_features: cached %d ZCTAs to %s",
                 len(combined), _JRC_WATER_CACHE_KEY)
    except Exception as e:
        log.warning("build_jrc_water_features: cache write failed: %s", e)

    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(extracted, on="zcta_id", how="left")
    out["_fs_jrc_occurrence_mean"] = np.where(
        out["jrc_occurrence_mean"].notna(), "present", _FS_MISSING,
    )
    log.info("build_jrc_water_features: %d ZCTAs, %.1f%% with data",
             len(out), (out["jrc_occurrence_mean"].notna().mean() * 100))
    return out


def build_deltares_depth_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Derive Deltares modeled flood depth per ZCTA from Planetary Computer.

    Uses floodcaster.batch.deltares_centroid_depth to fetch Deltares Global Flood
    Maps for return periods 10, 50, 100 from Microsoft Planetary Computer STAC API.
    Extracts mean depth, max depth, and inundation percentage within ~1 km of each
    ZCTA centroid.

    Cache-first: checks for pre-computed parquet at
    s3://{BUCKET}/{_DELTARES_DEPTH_CACHE_KEY} before STAC extraction.

    Returns DataFrame with columns:
        zcta_id, deltares_depth_ft_rp10, deltares_max_depth_ft_rp10,
        deltares_inundation_pct_rp10, ...(same for rp50, rp100),
        _fs_deltares_depth_ft_rp100.
    """
    rps = [10, 50, 100]
    data_cols = []
    for rp in rps:
        data_cols.extend([
            f"deltares_depth_ft_rp{rp}",
            f"deltares_max_depth_ft_rp{rp}",
            f"deltares_inundation_pct_rp{rp}",
        ])

    empty = pd.DataFrame({"zcta_id": zcta_ids})
    for col in data_cols:
        empty[col] = np.nan
    empty["_fs_deltares_depth_ft_rp100"] = _FS_MISSING

    # --- Cache lookup ---
    try:
        cached = s3_read(s3, _DELTARES_DEPTH_CACHE_KEY)
        if cached is not None and "zcta_id" in cached.columns:
            cached["zcta_id"] = cached["zcta_id"].astype(str)
            out = pd.DataFrame({"zcta_id": zcta_ids}).merge(
                cached[[c for c in cached.columns if c in ["zcta_id"] + data_cols]],
                on="zcta_id", how="left",
            )
            hit_rate = out["deltares_depth_ft_rp100"].notna().mean()
            if hit_rate > 0.5:
                out["_fs_deltares_depth_ft_rp100"] = np.where(
                    out["deltares_depth_ft_rp100"].notna(), "present", _FS_MISSING,
                )
                log.info("build_deltares_depth_features: cache hit (%.0f%% coverage)", hit_rate * 100)
                return out
    except Exception as e:
        log.info("build_deltares_depth_features: no cache (%s), extracting from STAC", e)

    # --- Load ZCTA centroids ---
    static_key = "raw/geocertdb2026/zcta_features_labels.parquet"
    static = s3_read(s3, static_key)
    if static is None:
        log.warning("build_deltares_depth_features: geocertdb2026 not available; returning NaN")
        return empty

    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        log.warning("build_deltares_depth_features: missing zcta/lat/lon columns; returning NaN")
        return empty

    static = static[[zcta_col, lat_col, lon_col]].rename(
        columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"},
    )
    centroids = static[static["zcta_id"].isin(zcta_ids)].dropna(subset=["lat", "lon"])

    if centroids.empty:
        log.warning("build_deltares_depth_features: no centroids matched; returning NaN")
        return empty

    # --- Extract from Planetary Computer via floodcaster ---
    try:
        extracted = _run_in_subprocess(
            "floodcaster.batch", "deltares_centroid_depth",
            centroids, id_col="zcta_id",
            extra_kwargs={"return_periods": rps}, timeout=600,
        )
        if extracted is None:
            log.warning("build_deltares_depth_features: STAC extraction crashed (segfault); returning NaN")
            return empty
    except Exception as e:
        log.warning("build_deltares_depth_features: STAC extraction failed: %s", e)
        return empty

    # --- Cache write ---
    try:
        existing = s3_read(s3, _DELTARES_DEPTH_CACHE_KEY)
        if existing is not None and "zcta_id" in existing.columns:
            existing["zcta_id"] = existing["zcta_id"].astype(str)
            combined = pd.concat([existing, extracted], ignore_index=True)
            combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")
        else:
            combined = extracted
        buf = BytesIO()
        combined.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=_DELTARES_DEPTH_CACHE_KEY, Body=buf.getvalue())
        log.info("build_deltares_depth_features: cached %d ZCTAs to %s",
                 len(combined), _DELTARES_DEPTH_CACHE_KEY)
    except Exception as e:
        log.warning("build_deltares_depth_features: cache write failed: %s", e)

    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(extracted, on="zcta_id", how="left")
    out["_fs_deltares_depth_ft_rp100"] = np.where(
        out["deltares_depth_ft_rp100"].notna(), "present", _FS_MISSING,
    )
    log.info("build_deltares_depth_features: %d ZCTAs, %.1f%% with data",
             len(out), (out["deltares_depth_ft_rp100"].notna().mean() * 100))
    return out


def build_hydrology_features(s3, zcta_ids: list[str], scenario: str = "shared") -> pd.DataFrame:
    """Derive DEM-based hydrology stats per ZCTA from Planetary Computer.

    Uses floodcaster.batch.hydrology_centroid_stats to fetch Copernicus DEM
    and compute HAND, TWI, GFI, and SPI within ~1 km of each ZCTA centroid.

    Cache-first: checks for scenario-specific pre-computed parquet at
    s3://{BUCKET}/processed/shared/zcta_hydrology_{scenario}.parquet,
    then falls back to the global cache at {_HYDROLOGY_CACHE_KEY}.

    Returns DataFrame with columns:
        zcta_id, hand_mean_m, twi_mean, gfi_mean, spi_mean,
        _fs_hand_mean_m.
    """
    data_cols = ["hand_mean_m", "twi_mean", "gfi_mean", "spi_mean"]

    empty = pd.DataFrame({"zcta_id": zcta_ids})
    for col in data_cols:
        empty[col] = np.nan
    empty["_fs_hand_mean_m"] = _FS_MISSING

    # --- Cache lookup (scenario-specific first, then global fallback) ---
    scenario_cache_key = f"processed/shared/zcta_hydrology_{scenario}.parquet"
    for cache_key in [scenario_cache_key, _HYDROLOGY_CACHE_KEY]:
        try:
            cached = s3_read(s3, cache_key)
            if cached is not None and "zcta_id" in cached.columns:
                cached["zcta_id"] = cached["zcta_id"].astype(str)
                out = pd.DataFrame({"zcta_id": zcta_ids}).merge(
                    cached[[c for c in cached.columns if c in ["zcta_id"] + data_cols]],
                    on="zcta_id", how="left",
                )
                hit_rate = out["hand_mean_m"].notna().mean()
                if hit_rate > 0.5:
                    out["_fs_hand_mean_m"] = np.where(
                        out["hand_mean_m"].notna(), "present", _FS_MISSING,
                    )
                    log.info("build_hydrology_features: cache hit at %s (%.0f%% coverage)", cache_key, hit_rate * 100)
                    return out
        except Exception as e:
            log.info("build_hydrology_features: cache %s not available (%s)", cache_key, e)

    # --- Load ZCTA centroids ---
    static_key = "raw/geocertdb2026/zcta_features_labels.parquet"
    static = s3_read(s3, static_key)
    if static is None:
        log.warning("build_hydrology_features: geocertdb2026 not available; returning NaN")
        return empty

    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        log.warning("build_hydrology_features: missing zcta/lat/lon columns; returning NaN")
        return empty

    static = static[[zcta_col, lat_col, lon_col]].rename(
        columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"},
    )
    centroids = static[static["zcta_id"].isin(zcta_ids)].dropna(subset=["lat", "lon"])

    if centroids.empty:
        log.warning("build_hydrology_features: no centroids matched; returning NaN")
        return empty

    # --- Extract from Planetary Computer via floodcaster ---
    # Runs in subprocess to survive GDAL/curl segfault (see module docstring).
    try:
        extracted = _run_in_subprocess(
            "floodcaster.batch", "hydrology_centroid_stats",
            centroids, id_col="zcta_id", timeout=600,
        )
        if extracted is None:
            log.warning("build_hydrology_features: STAC extraction crashed (segfault); returning NaN")
            return empty
    except Exception as e:
        log.warning("build_hydrology_features: extraction failed: %s", e)
        return empty

    # --- Cache write ---
    try:
        existing = s3_read(s3, _HYDROLOGY_CACHE_KEY)
        if existing is not None and "zcta_id" in existing.columns:
            existing["zcta_id"] = existing["zcta_id"].astype(str)
            combined = pd.concat([existing, extracted], ignore_index=True)
            combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")
        else:
            combined = extracted
        buf = BytesIO()
        combined.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=_HYDROLOGY_CACHE_KEY, Body=buf.getvalue())
        log.info("build_hydrology_features: cached %d ZCTAs to %s",
                 len(combined), _HYDROLOGY_CACHE_KEY)
    except Exception as e:
        log.warning("build_hydrology_features: cache write failed: %s", e)

    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(extracted, on="zcta_id", how="left")
    out["_fs_hand_mean_m"] = np.where(
        out["hand_mean_m"].notna(), "present", _FS_MISSING,
    )
    log.info("build_hydrology_features: %d ZCTAs, %.1f%% with data",
             len(out), (out["hand_mean_m"].notna().mean() * 100))
    return out


def build_sentinel1_event_features(
    s3, zcta_ids: list[str], event_cfg: dict, jrc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive Sentinel-1 SAR inundation per ZCTA for a specific event.

    Uses floodcaster.batch.sentinel1_centroid_inundation to search for S1
    overpasses within the event's peak_window and compute water fraction.
    Derives sar_water_pct_anomaly by subtracting JRC baseline.

    Args:
        s3: boto3 S3 client.
        zcta_ids: List of ZCTA IDs.
        event_cfg: Dict with keys: "peak_window" (tuple of 2 date strings),
                   "s3_event_key" or event name for cache key,
                   "scenario" for cache path.
        jrc_df: DataFrame with zcta_id + jrc_pct_ever_wet from build_jrc_water_features.

    Returns DataFrame with columns:
        zcta_id, sar_water_pct, sar_water_pct_anomaly,
        sar_acquisition_lag_days, _fs_sar_water_pct.
    """
    data_cols = ["sar_water_pct", "sar_water_pct_anomaly", "sar_acquisition_lag_days"]

    empty = pd.DataFrame({"zcta_id": zcta_ids})
    for col in data_cols:
        empty[col] = np.nan
    empty["_fs_sar_water_pct"] = _FS_MISSING

    event_key = event_cfg.get("s3_event_key", event_cfg.get("event_name", "unknown"))
    scenario = event_cfg.get("scenario", "shared")
    cache_key = f"processed/{scenario}/zcta_sar_{event_key}.parquet"

    peak = event_cfg.get("peak_window")
    if not peak or len(peak) < 2:
        log.warning("build_sentinel1_event_features: no peak_window in event_cfg")
        return empty

    # --- Cache lookup ---
    try:
        cached = s3_read(s3, cache_key)
        if cached is not None and "zcta_id" in cached.columns:
            cached["zcta_id"] = cached["zcta_id"].astype(str)
            out = pd.DataFrame({"zcta_id": zcta_ids}).merge(
                cached[[c for c in cached.columns if c in ["zcta_id"] + data_cols]],
                on="zcta_id", how="left",
            )
            hit_rate = out["sar_water_pct"].notna().mean()
            if hit_rate > 0.5:
                out["_fs_sar_water_pct"] = np.where(
                    out["sar_water_pct"].notna(), "present", _FS_MISSING,
                )
                log.info("build_sentinel1_event_features[%s]: cache hit (%.0f%%)", event_key, hit_rate * 100)
                return out
    except Exception as e:
        log.info("build_sentinel1_event_features[%s]: no cache (%s)", event_key, e)

    # --- Load ZCTA centroids ---
    static_key = "raw/geocertdb2026/zcta_features_labels.parquet"
    static = s3_read(s3, static_key)
    if static is None:
        log.warning("build_sentinel1_event_features: geocertdb2026 not available; returning NaN")
        return empty

    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    if not all([zcta_col, lat_col, lon_col]):
        log.warning("build_sentinel1_event_features: missing zcta/lat/lon columns; returning NaN")
        return empty

    static = static[[zcta_col, lat_col, lon_col]].rename(
        columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"},
    )
    centroids = static[static["zcta_id"].isin(zcta_ids)].dropna(subset=["lat", "lon"])

    if centroids.empty:
        log.warning("build_sentinel1_event_features: no centroids matched; returning NaN")
        return empty

    # --- Extract from Planetary Computer via floodcaster ---
    try:
        extracted = _run_in_subprocess(
            "floodcaster.batch", "sentinel1_centroid_inundation",
            centroids, id_col="zcta_id",
            extra_kwargs={"event_start": peak[0], "event_end": peak[1]},
            timeout=600,
        )
        if extracted is None:
            log.warning("build_sentinel1_event_features[%s]: STAC extraction crashed (segfault); returning NaN", event_key)
            return empty
    except Exception as e:
        log.warning("build_sentinel1_event_features[%s]: extraction failed: %s", event_key, e)
        return empty

    # --- Derive anomaly: sar_water_pct - jrc_pct_ever_wet ---
    if "jrc_pct_ever_wet" in jrc_df.columns:
        extracted = extracted.merge(
            jrc_df[["zcta_id", "jrc_pct_ever_wet"]], on="zcta_id", how="left",
        )
        extracted["sar_water_pct_anomaly"] = (
            extracted["sar_water_pct"] - extracted["jrc_pct_ever_wet"]
        )
        extracted = extracted.drop(columns=["jrc_pct_ever_wet"])
    else:
        extracted["sar_water_pct_anomaly"] = np.nan

    # Drop sar_acquisition_dt before cache (not a feature column)
    if "sar_acquisition_dt" in extracted.columns:
        extracted = extracted.drop(columns=["sar_acquisition_dt"])

    # --- Cache write ---
    try:
        existing = s3_read(s3, cache_key)
        if existing is not None and "zcta_id" in existing.columns:
            existing["zcta_id"] = existing["zcta_id"].astype(str)
            combined = pd.concat([existing, extracted], ignore_index=True)
            combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")
        else:
            combined = extracted
        buf = BytesIO()
        combined.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=cache_key, Body=buf.getvalue())
        log.info("build_sentinel1_event_features[%s]: cached %d ZCTAs", event_key, len(combined))
    except Exception as e:
        log.warning("build_sentinel1_event_features[%s]: cache write failed: %s", event_key, e)

    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(
        extracted[["zcta_id"] + data_cols], on="zcta_id", how="left",
    )
    out["_fs_sar_water_pct"] = np.where(
        out["sar_water_pct"].notna(), "present", _FS_MISSING,
    )
    log.info("build_sentinel1_event_features[%s]: %d ZCTAs, %.1f%% with data",
             event_key, len(out), (out["sar_water_pct"].notna().mean() * 100))
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

    all_objs = _list_s3_keys(s3, f"raw/dem/3dep/v1/{region}/")
    tif_keys = [o["Key"] for o in all_objs if o["Key"].endswith(".tif")]
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

    # Geometry may be stored as WKB bytes in parquet -- deserialize if needed
    if nhd_df["geometry"].dtype == object and isinstance(nhd_df["geometry"].iloc[0], bytes):
        from shapely import wkb
        nhd_df = nhd_df.copy()
        nhd_df["geometry"] = nhd_df["geometry"].apply(wkb.loads)
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


def _slosh_surge_to_category(m: float) -> Optional[str]:
    """Convert surge depth (m) to Saffir-Simpson category string."""
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


# Saffir-Simpson category at landfall for each SW Florida event.
# Used to select the correct MOM GeoTIFF layer.
_SWFL_EVENT_CATEGORY = {
    "ian2022": 4,
    "helene2024": 4,
    "milton2024": 3,
}

# MOM national GeoTIFF key template (Cat 1-5)
_MOM_KEY_TEMPLATE = "raw/noaa_slosh/mom_national/us_Category{cat}_MOM_Inundation_HIGH.tif"


def build_slosh_features(s3, event: str, zcta_ids: list[str]) -> pd.DataFrame:
    """Derive slosh_max_surge_m and slosh_category per ZCTA from NHC SLOSH MOM.

    Primary path: national MOM GeoTIFF at raw/noaa_slosh/mom_national/.
    The event determines which SS category layer to sample (e.g. Ian=Cat4).
    MOM is basin-specific and invariant -- same grid regardless of storm.

    Fallback: legacy per-event NetCDF at raw/noaa_slosh/{event}/ (deprecated).

    Returns DataFrame: zcta_id, slosh_max_surge_m, slosh_category, _fs_slosh_max_surge_m.
    """
    empty = pd.DataFrame({
        "zcta_id": zcta_ids,
        "slosh_max_surge_m": np.nan,
        "slosh_category": None,
        "_fs_slosh_max_surge_m": _FS_MISSING,
    })

    # Resolve storm category for MOM lookup
    cat = _SWFL_EVENT_CATEGORY.get(event)
    if cat is None:
        log.warning("build_slosh_features: no SS category mapping for event=%s; "
                    "returning NaN", event)
        return empty

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

    # --- Primary path: national MOM GeoTIFF via rasterio ---
    mom_key = _MOM_KEY_TEMPLATE.format(cat=cat)
    result = _build_slosh_from_mom_geotiff(s3, mom_key, cat, centroids, zcta_ids, event)
    if result is not None:
        return result

    # --- Fallback: legacy per-event NetCDF (deprecated) ---
    log.info("build_slosh_features: MOM GeoTIFF path failed; trying legacy per-event path")
    result = _build_slosh_from_legacy_nc(s3, event, centroids, zcta_ids)
    if result is not None:
        return result

    return empty


def _build_slosh_from_mom_geotiff(
    s3, mom_key: str, cat: int, centroids: pd.DataFrame,
    zcta_ids: list[str], event: str,
) -> Optional[pd.DataFrame]:
    """Sample national MOM GeoTIFF at ZCTA centroids. Returns DataFrame or None.

    Resource assumptions:
        - GeoTIFF is 318,457 x 223,758 px (uint8). Uncompressed = 66.4 GB.
          MUST NOT call src.read(1). Use src.sample() for point queries.
        - On-disk file is ~1.2 GB (LZW compressed). Downloaded to /tmp.
        - src.sample() reads only the disk blocks containing the queried
          pixels via GDAL's virtual filesystem -- memory cost is ~1 MB for
          coordinate arrays + GDAL block cache (default 5% of RAM).

    Value semantics (from NHC XML metadata):
        - 0:       no inundation (land above surge)
        - 1-98:    surge depth in feet above ground (NAVD88 datum)
        - 99:      levee-protected area -- NOT a valid depth; treated as NaN
        - nodata:  outside SLOSH basin coverage (open ocean, far inland)

    Unit conversion:
        - Raw values are in feet. Converted to meters via FT_TO_M = 0.3048.
        - Output column: slosh_max_surge_m (meters above ground).

    Basin coverage (SW Florida events):
        - ian2022:     Cat 4, basin FTM (Fort Myers)
        - helene2024:  Cat 4, basin APF (Apalachicola-Fort Myers)
        - milton2024:  Cat 3, basin TBW (Tampa Bay)
        National MOM covers all basins; category selects the GeoTIFF layer.
    """
    try:
        import rasterio
    except ImportError:
        log.warning("build_slosh_features: rasterio not available; cannot read MOM GeoTIFF")
        return None

    local = f"/tmp/slosh_mom_cat{cat}.tif"
    try:
        s3.download_file(BUCKET, mom_key, local)
    except Exception as exc:
        log.warning("build_slosh_features: failed to download %s: %s", mom_key, exc)
        return None

    surge_vals: dict[str, float] = {}
    try:
        with rasterio.open(local) as src:
            log.info("build_slosh_features: opened MOM Cat %d (%s, %dx%d, crs=%s)",
                     cat, mom_key, src.width, src.height, src.crs)
            nodata = src.nodata

            # Use src.sample() to read only the needed pixels -- the full
            # raster is 318K x 224K (66 GB) and cannot fit in memory.
            coords = list(zip(centroids["lon"].values, centroids["lat"].values))
            zcta_list = centroids["zcta_id"].values

            for (zcta_id, sample_val) in zip(zcta_list, src.sample(coords)):
                val = float(sample_val[0])
                if nodata is not None and val == nodata:
                    val = np.nan
                elif val >= 99:
                    # NHC metadata: value 99 = levee-protected areas;
                    # not valid surge depth.
                    val = np.nan
                elif val <= 0:
                    val = np.nan
                surge_vals[zcta_id] = val

            log.info("build_slosh_features: sampled %d centroids from MOM Cat %d",
                     len(coords), cat)
    except Exception as exc:
        log.warning("build_slosh_features: rasterio read error: %s", exc)
        return None
    finally:
        if Path(local).exists():
            os.unlink(local)

    # MOM values are in feet; convert to meters
    FT_TO_M = 0.3048
    rows = []
    for z in zcta_ids:
        val_ft = surge_vals.get(z, np.nan)
        val_m = val_ft * FT_TO_M if not np.isnan(val_ft) else np.nan
        rows.append({
            "zcta_id": z,
            "slosh_max_surge_m": val_m,
            "slosh_category": str(cat) if not np.isnan(val_m) else None,
        })
    out = pd.DataFrame(rows)
    out["_fs_slosh_max_surge_m"] = np.where(
        out["slosh_max_surge_m"].notna(), "present", _FS_MISSING)
    pct = out["slosh_max_surge_m"].notna().mean() * 100
    log.info("build_slosh_features: %d ZCTAs, %.1f%% with MOM surge (event=%s, cat=%d)",
             len(out), pct, event, cat)
    return out


def _build_slosh_from_legacy_nc(
    s3, event: str, centroids: pd.DataFrame, zcta_ids: list[str],
) -> Optional[pd.DataFrame]:
    """Legacy fallback: read per-event SLOSH NetCDF files. Returns DataFrame or None."""
    all_objs = _list_s3_keys(s3, f"raw/noaa_slosh/{event}/")
    slosh_keys = [o["Key"] for o in all_objs
                  if not o["Key"].endswith("MANUAL_DOWNLOAD_REQUIRED.txt")]
    if not slosh_keys:
        return None

    nc_keys = [k for k in slosh_keys if k.endswith(".nc") or k.endswith(".grb2")]
    if not nc_keys:
        return None

    try:
        import xarray as xr
    except ImportError:
        log.warning("build_slosh_features(legacy): xarray not available")
        return None

    surge_vals: dict[str, float] = {}
    for nc_key in nc_keys[:3]:
        local = f"/tmp/slosh_{Path(nc_key).stem}.nc"
        s3.download_file(BUCKET, nc_key, local)
        try:
            ds = xr.open_dataset(local, engine="netcdf4")
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
                dists = np.sqrt((lat_arr - row["lat"])**2 + (lon_arr - row["lon"])**2)
                idx = np.argmin(dists)
                val = float(surge_arr[idx]) if not np.isnan(surge_arr[idx]) else np.nan
                if row["zcta_id"] not in surge_vals or val > surge_vals.get(row["zcta_id"], 0):
                    surge_vals[row["zcta_id"]] = val
        except Exception as exc:
            log.warning("SLOSH legacy read error for %s: %s", nc_key, exc)
        finally:
            if Path(local).exists():
                os.unlink(local)

    if not surge_vals:
        return None

    rows = [{"zcta_id": z, "slosh_max_surge_m": v,
             "slosh_category": _slosh_surge_to_category(v)} for z, v in surge_vals.items()]
    out = pd.DataFrame(rows)
    out = pd.DataFrame({"zcta_id": zcta_ids}).merge(out, on="zcta_id", how="left")
    out["_fs_slosh_max_surge_m"] = np.where(
        out["slosh_max_surge_m"].notna(), "present", _FS_MISSING)
    log.info("build_slosh_features(legacy): %d ZCTAs, %.1f%% with surge data (event=%s)",
             len(out), out["slosh_max_surge_m"].notna().mean() * 100, event)
    return out


def build_sewershed_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Assign sewer_shed_id to each NYC ZCTA via spatial join with NYC DEP sewersheds.

    Requires: raw/nyc_sewersheds/nyc_sewersheds.parquet (attribute table from .gpkg)
    Returns DataFrame: zcta_id, sewer_shed_id, _fs_sewer_shed_id.
    """
    empty = pd.DataFrame({"zcta_id": zcta_ids, "sewer_shed_id": None,
                           "sewershed_name": None,
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

    # Find the ID and name columns
    id_col = next((c for c in sewer_df.columns
                   if any(k in c.lower() for k in ["shed_id", "drainage_id", "district", "id"])),
                  sewer_df.columns[0])
    name_col = next((c for c in sewer_df.columns
                     if any(k in c.lower() for k in ["name", "shed_name", "drainage_name"])),
                    None)
    geom_col = "geometry" if "geometry" in sewer_df.columns else None
    if geom_col is None:
        log.warning("build_sewershed_features: no geometry column in sewershed parquet; returning null")
        return empty

    join_cols = [id_col, geom_col]
    if name_col:
        join_cols.insert(1, name_col)
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

    joined = gpd.sjoin(centroid_gdf, sewer_gdf[join_cols],
                       how="left", predicate="within")
    rename_map = {id_col: "sewer_shed_id"}
    if name_col:
        rename_map[name_col] = "sewershed_name"
    joined = joined.rename(columns=rename_map)
    out_cols = ["zcta_id", "sewer_shed_id"]
    if "sewershed_name" in joined.columns:
        out_cols.append("sewershed_name")
    out = joined[out_cols].drop_duplicates("zcta_id")
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
                          "levee_nearest_km": np.nan, "canal_proximity_m": np.nan,
                          "_fs_levee_condition_rating": _FS_MISSING})

    levee_key = f"raw/usace_levees/{scenario}_levees.parquet"
    levees = s3_read(s3, levee_key)
    if levees is None or levees.empty:
        log.warning("build_levee_features: no levee data for scenario=%s", scenario)
        return empty

    rating_col = next((c for c in levees.columns if "condition" in c.lower() or "rating" in c.lower()), None)
    # USACE NLD ratings are categorical strings -- encode as ordinal
    # Higher = better protection
    _LEVEE_ORDINAL = {
        "Accredited Levee System": 4,
        "Provisionally Accredited Levee (PAL) System": 3,
        "A99": 2,  # area protected by levee under construction
        "No Regulatory Flood Hazard Information Published by FEMA": 1,
        "Non-Accredited Levee System": 0,
    }
    if rating_col:
        levees["_rating_numeric"] = levees[rating_col].map(_LEVEE_ORDINAL)
        # Fallback: try direct numeric coercion for any numeric sources
        if levees["_rating_numeric"].isna().all():
            levees["_rating_numeric"] = pd.to_numeric(levees[rating_col], errors="coerce")
        rating_col = "_rating_numeric"
    lat_col = next((c for c in levees.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in levees.columns if "lon" in c.lower() or "lng" in c.lower()), None)
    counties_col = next((c for c in levees.columns if "counties" in c.lower()), None)

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

    if lat_col and lon_col:
        # Spatial assignment: nearest levee segment to each ZCTA centroid
        for _, crow in centroids.iterrows():
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
            rows.append({
                "zcta_id": crow["zcta_id"],
                "levee_condition_rating": rating,
                "levee_nearest_km": float(dists_km[nearest_idx]),
                "canal_proximity_m": nearest_dist_m,
            })
    elif counties_col:
        # County/parish-based assignment: match levee COUNTIES to ZCTA county
        # Load crosswalk to get ZCTA -> county_name mapping
        xwalk = s3_read(s3, "raw/geocertdb2026/zcta_county_crosswalk.parquet")
        if xwalk is not None and "county_name" in xwalk.columns:
            zcta_county = xwalk[xwalk["zcta_id"].isin(zcta_ids)][["zcta_id", "county_name"]].copy()
            # Normalize county names (strip "Parish", "County" suffix for matching)
            zcta_county["_county_norm"] = (
                zcta_county["county_name"].str.replace(r"\s*(Parish|County)\s*$", "", regex=True).str.strip()
            )
            # Explode levee COUNTIES (comma-separated) and find best rating per county
            levee_counties = levees[[counties_col, rating_col]].copy()
            levee_counties = levee_counties.assign(
                _county_list=levee_counties[counties_col].str.split(r",\s*")
            ).explode("_county_list")
            levee_counties["_county_norm"] = levee_counties["_county_list"].str.strip()
            # Best (max) levee protection rating per county
            county_best = (
                levee_counties.groupby("_county_norm")[rating_col]
                .max()
                .reset_index()
                .rename(columns={rating_col: "levee_condition_rating"})
            )
            # Join to ZCTAs
            merged = zcta_county.merge(county_best, on="_county_norm", how="left")
            for _, row in merged.iterrows():
                rows.append({
                    "zcta_id": row["zcta_id"],
                    "levee_condition_rating": row.get("levee_condition_rating", np.nan),
                    "levee_nearest_km": np.nan,
                    "canal_proximity_m": np.nan,
                })
            log.info("build_levee_features: county-based assignment for %d ZCTAs", len(merged))
        else:
            # Last resort: assign scenario-wide best rating
            best = float(levees[rating_col].max()) if rating_col else np.nan
            for zid in zcta_ids:
                rows.append({
                    "zcta_id": zid,
                    "levee_condition_rating": best,
                    "levee_nearest_km": np.nan,
                    "canal_proximity_m": np.nan,
                })
    else:
        # No spatial or county info -- return empty
        return empty

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


def build_building_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Load Overture building footprint stats per ZCTA from shared cache.

    Cache populated by run_fetch_buildings.py SageMaker job.
    Returns DataFrame: zcta_id, building_count, total_footprint_area_m2,
                       _fs_building_count.
    """
    data_cols = ["building_count", "total_footprint_area_m2"]
    empty = pd.DataFrame({"zcta_id": zcta_ids})
    for col in data_cols:
        empty[col] = np.nan
    empty["_fs_building_count"] = _FS_MISSING

    try:
        cached = s3_read(s3, _BUILDINGS_CACHE_KEY)
        if cached is not None and "zcta_id" in cached.columns:
            cached["zcta_id"] = cached["zcta_id"].astype(str)
            keep = [c for c in cached.columns if c in ["zcta_id"] + data_cols]
            out = pd.DataFrame({"zcta_id": zcta_ids}).merge(
                cached[keep], on="zcta_id", how="left",
            )
            out["_fs_building_count"] = np.where(
                out["building_count"].notna(), "present", _FS_MISSING,
            )
            hit = out["building_count"].notna().sum()
            log.info("build_building_features: %d/%d ZCTAs from cache (%.0f%%)",
                     hit, len(zcta_ids), hit / len(zcta_ids) * 100)
            return out
    except Exception as e:
        log.warning("build_building_features: cache read failed: %s", e)

    log.warning("build_building_features: no cache at %s; returning NaN", _BUILDINGS_CACHE_KEY)
    return empty


def build_depression_features(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Load DEM depression stats per ZCTA from shared cache.

    Cache populated by run_fetch_depressions.py SageMaker job.
    Returns DataFrame: zcta_id, depression_count, depression_volume_m3,
                       max_depression_depth_m, depression_area_m2,
                       _fs_depression_count.
    """
    data_cols = ["depression_count", "depression_volume_m3",
                 "max_depression_depth_m", "depression_area_m2"]
    empty = pd.DataFrame({"zcta_id": zcta_ids})
    for col in data_cols:
        empty[col] = np.nan
    empty["_fs_depression_count"] = _FS_MISSING

    try:
        cached = s3_read(s3, _DEPRESSIONS_CACHE_KEY)
        if cached is not None and "zcta_id" in cached.columns:
            cached["zcta_id"] = cached["zcta_id"].astype(str)
            keep = [c for c in cached.columns if c in ["zcta_id"] + data_cols]
            out = pd.DataFrame({"zcta_id": zcta_ids}).merge(
                cached[keep], on="zcta_id", how="left",
            )
            out["_fs_depression_count"] = np.where(
                out["depression_count"].notna(), "present", _FS_MISSING,
            )
            hit = out["depression_count"].notna().sum()
            log.info("build_depression_features: %d/%d ZCTAs from cache (%.0f%%)",
                     hit, len(zcta_ids), hit / len(zcta_ids) * 100)
            return out
    except Exception as e:
        log.warning("build_depression_features: cache read failed: %s", e)

    log.warning("build_depression_features: no cache at %s; returning NaN", _DEPRESSIONS_CACHE_KEY)
    return empty


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

    _aws = get_aws_credentials()
    _aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

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

    # Post-assembly validation (Layer 2)
    try:
        from validate_assembly import validate_post_assembly, log_validation_report
        report = validate_post_assembly(df, args.scenario)
        passed = log_validation_report(report)
        if not passed:
            log.error("Post-assembly validation FAILED -- refusing to upload broken parquet")
            sys.exit(1)
    except ImportError:
        log.warning("validate_assembly not available -- skipping post-assembly validation")

    # Semantic aliases: latitude/longitude are ZCTA geometric centroids
    # (TIGER/Line 2022 polygon centroids, NOT population-weighted).
    # Add explicit centroid_lat/centroid_lon so downstream consumers
    # (render_zcta_maps, spatial validation) can reference unambiguous names.
    if "latitude" in df.columns and "centroid_lat" not in df.columns:
        df["centroid_lat"] = df["latitude"]
    if "longitude" in df.columns and "centroid_lon" not in df.columns:
        df["centroid_lon"] = df["longitude"]

    s3_upload(df, OUTPUT_KEYS[args.scenario], s3)
    log.info("build_event_dataset complete: %s", args.scenario)


if __name__ == "__main__":
    main()
