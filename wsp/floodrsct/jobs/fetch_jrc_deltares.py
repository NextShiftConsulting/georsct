#!/usr/bin/env python3
"""
fetch_jrc_deltares.py -- SageMaker job: fetch JRC + Deltares shared features.

Extracts two shared feature layers from Microsoft Planetary Computer STAC API
for the ~794 ZCTA centroids in the s035 universe (5 scenarios):

  1. JRC Global Surface Water (1984-2020): water occurrence stats
  2. Deltares Global Flood Maps: modeled depth at RP 10/50/100

Both are scenario-independent (keyed by ZCTA, not by event), so they live
in the shared/ prefix and are built once for all scenarios.

Processes per-scenario to keep STAC bounding boxes small (city-scale, not
continental). Results are deduplicated and merged into single shared parquets.

Source:
  JRC: https://planetarycomputer.microsoft.com/dataset/jrc-gsw
  Deltares: https://planetarycomputer.microsoft.com/dataset/deltares-floods

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_jrc_water_occurrence_pct.parquet
  s3://swarm-floodrsct-data/processed/shared/zcta_deltares_depth.parquet

Usage:
    python fetch_jrc_deltares.py
    python fetch_jrc_deltares.py --jrc-only
    python fetch_jrc_deltares.py --deltares-only
"""

import argparse
import logging
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Self-install floodcaster + deps from mounted wheels (runs before any import)
# ---------------------------------------------------------------------------
_WHEELS = "/opt/ml/processing/input/wheels"


def _ensure_floodcaster():
    """Install floodcaster + its STAC deps, fail loudly on any error."""
    # Test the FULL import chain, not just the top-level package.
    # floodcaster.stac needs planetary_computer, pystac_client, rasterio
    # at module level -- if any are missing, Python reports it as
    # "No module named 'floodcaster.stac'" which hides the real cause.
    try:
        from floodcaster.stac import jrc_centroid_occurrence  # noqa: F401
        from floodcaster.batch import deltares_centroid_depth  # noqa: F401
        return  # everything importable
    except Exception as e:
        print(f"floodcaster import failed: {e}", flush=True)

    # Diagnose which dep is missing
    for pkg in ["rasterio", "planetary_computer", "pystac_client",
                "geopandas", "duckdb", "shapely"]:
        try:
            __import__(pkg)
            print(f"  {pkg}: OK", flush=True)
        except ImportError:
            print(f"  {pkg}: MISSING", flush=True)

    # Install everything from wheels + PyPI
    print(f"Installing floodcaster + deps from {_WHEELS}...", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--find-links", _WHEELS,
        "sphere-core", "sphere-data", "sphere-flood", "floodcaster",
        "planetary-computer", "pystac-client",
    ])

    # Verify the full import chain
    from floodcaster.stac import jrc_centroid_occurrence  # noqa: F401
    from floodcaster.batch import deltares_centroid_depth  # noqa: F401
    print("floodcaster installed and importable", flush=True)


_ensure_floodcaster()
# ---------------------------------------------------------------------------

import boto3
import numpy as np
import pandas as pd
from swarm_auth import get_aws_credentials

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"
CENTROID_KEY = "raw/geocertdb2026/zcta_features_labels.parquet"
JRC_CACHE_KEY = "processed/shared/zcta_jrc_water_occurrence_pct.parquet"
DELTARES_CACHE_KEY = "processed/shared/zcta_deltares_depth.parquet"

# s035 scenario -> event_features key (for ZCTA ID extraction)
SCENARIO_EVENT_KEYS = {
    "houston": "processed/houston/houston_event_features.parquet",
    "new_orleans": "processed/new_orleans/no_event_features.parquet",
    "nyc": "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_read(s3, key: str):
    """Read parquet from S3; return None if missing."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        log.warning("S3 key not found: %s", key)
        return None
    except Exception as e:
        log.warning("Could not read %s: %s", key, e)
        return None


def s3_write_parquet(s3, df: pd.DataFrame, key: str) -> None:
    """Write DataFrame as parquet to S3."""
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    log.info("Uploaded %d rows x %d cols to s3://%s/%s",
             len(df), len(df.columns), BUCKET, key)


# ---------------------------------------------------------------------------
# ZCTA universe + centroid loading
# ---------------------------------------------------------------------------

def load_scenario_zctas(s3) -> dict[str, list[str]]:
    """Load unique ZCTA IDs per scenario from event_features parquets."""
    scenario_zctas = {}
    for scenario, key in SCENARIO_EVENT_KEYS.items():
        df = s3_read(s3, key)
        if df is None:
            log.warning("Missing event_features for %s, skipping", scenario)
            continue
        zcta_col = next((c for c in df.columns if "zcta" in c.lower()), None)
        if zcta_col is None:
            log.warning("No zcta column in %s, skipping", scenario)
            continue
        ids = sorted(df[zcta_col].astype(str).unique().tolist())
        scenario_zctas[scenario] = ids
        log.info("  %s: %d unique ZCTAs", scenario, len(ids))
    return scenario_zctas


def load_centroids(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Load ZCTA centroids from geocertdb2026, filtered to target IDs."""
    static = s3_read(s3, CENTROID_KEY)
    if static is None:
        raise RuntimeError(f"geocertdb2026 not found at s3://{BUCKET}/{CENTROID_KEY}")

    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next(
        (c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()),
        None,
    )
    if not all([zcta_col, lat_col, lon_col]):
        raise RuntimeError(
            f"Missing zcta/lat/lon in geocertdb2026 cols: {list(static.columns)}"
        )

    centroids = (
        static[[zcta_col, lat_col, lon_col]]
        .rename(columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"})
        .dropna(subset=["lat", "lon"])
    )
    centroids["zcta_id"] = centroids["zcta_id"].astype(str)
    centroids = centroids[centroids["zcta_id"].isin(zcta_ids)]
    log.info("Loaded %d ZCTA centroids (filtered from geocertdb2026)", len(centroids))
    return centroids


# ---------------------------------------------------------------------------
# JRC extraction (per-scenario batching)
# ---------------------------------------------------------------------------

def fetch_jrc(s3, scenario_zctas: dict[str, list[str]], all_centroids: pd.DataFrame) -> pd.DataFrame:
    """Fetch JRC Global Surface Water occurrence, batched per scenario bbox."""
    log.info("=== JRC Global Surface Water ===")
    from floodcaster.stac import jrc_centroid_occurrence

    parts = []
    for scenario, zcta_ids in scenario_zctas.items():
        centroids = all_centroids[all_centroids["zcta_id"].isin(zcta_ids)]
        if centroids.empty:
            log.warning("JRC: no centroids for %s, skipping", scenario)
            continue

        t0 = time.time()
        log.info("JRC: %s (%d centroids)...", scenario, len(centroids))
        try:
            result = jrc_centroid_occurrence(centroids, id_col="zcta_id")
            elapsed = time.time() - t0
            coverage = result["jrc_occurrence_mean"].notna().mean() * 100
            log.info("JRC: %s done -- %d rows, %.1f%% coverage, %.0f sec",
                     scenario, len(result), coverage, elapsed)
            parts.append(result)
        except Exception as e:
            log.warning("JRC: %s STAC extraction failed: %s", scenario, e)

    if not parts:
        raise RuntimeError("JRC: all scenarios failed")

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")
    log.info("JRC combined: %d unique ZCTAs", len(combined))

    s3_write_parquet(s3, combined, JRC_CACHE_KEY)
    return combined


# ---------------------------------------------------------------------------
# Deltares extraction (per-scenario batching)
# ---------------------------------------------------------------------------

def _fetch_deltares_scenario(
    centroids: pd.DataFrame,
    scenario: str,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """Fetch Deltares depth for one scenario with retry + backoff.

    Azure Blob Storage (deltaresfloodssa) can be intermittently unreachable
    from SageMaker.  Retry with exponential backoff to handle transient
    connectivity failures.
    """
    from floodcaster.batch import deltares_centroid_depth

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            result = deltares_centroid_depth(
                centroids, id_col="zcta_id", return_periods=[10, 50, 100],
            )
            elapsed = time.time() - t0
            coverage = result["deltares_depth_ft_rp100"].notna().mean() * 100
            log.info(
                "Deltares: %s done (attempt %d) -- %d rows, %.1f%% coverage, %.0f sec",
                scenario, attempt, len(result), coverage, elapsed,
            )
            # Coverage check: if ALL rows are NaN, treat as failure and retry.
            # The Deltares Global Flood Maps cover major US floodplains --
            # 0% coverage means the NetCDF read failed silently.
            if coverage == 0.0 and attempt < max_retries:
                log.warning(
                    "Deltares: %s 0%% coverage on attempt %d -- "
                    "likely transient Azure read failure, retrying",
                    scenario, attempt,
                )
                time.sleep(2 ** attempt)
                continue
            return result
        except Exception as e:
            log.warning(
                "Deltares: %s attempt %d failed: %s", scenario, attempt, e,
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                log.error("Deltares: %s all %d attempts exhausted", scenario, max_retries)
    return None


def fetch_deltares(s3, scenario_zctas: dict[str, list[str]], all_centroids: pd.DataFrame) -> pd.DataFrame:
    """Fetch Deltares Global Flood Maps depth, batched per scenario bbox."""
    log.info("=== Deltares Global Flood Maps ===")

    parts = []
    failed_scenarios = []
    for scenario, zcta_ids in scenario_zctas.items():
        centroids = all_centroids[all_centroids["zcta_id"].isin(zcta_ids)]
        if centroids.empty:
            log.warning("Deltares: no centroids for %s, skipping", scenario)
            continue

        log.info("Deltares: %s (%d centroids)...", scenario, len(centroids))
        result = _fetch_deltares_scenario(centroids, scenario)
        if result is not None:
            parts.append(result)
        else:
            failed_scenarios.append(scenario)

    if not parts:
        raise RuntimeError("Deltares: all scenarios failed")

    if failed_scenarios:
        log.warning("Deltares: %d scenarios failed: %s", len(failed_scenarios), failed_scenarios)

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["zcta_id"], keep="last")

    # Final coverage report
    total = len(combined)
    non_null = combined["deltares_depth_ft_rp100"].notna().sum()
    log.info(
        "Deltares combined: %d unique ZCTAs, %d non-null rp100 (%.1f%%)",
        total, non_null, non_null / max(total, 1) * 100,
    )
    if non_null / max(total, 1) < 0.10:
        log.warning(
            "Deltares: <10%% coverage -- Azure connectivity likely degraded. "
            "Consider re-running with --deltares-only."
        )

    s3_write_parquet(s3, combined, DELTARES_CACHE_KEY)
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch JRC + Deltares shared features")
    parser.add_argument("--jrc-only", action="store_true", help="Fetch JRC only")
    parser.add_argument("--deltares-only", action="store_true", help="Fetch Deltares only")
    args = parser.parse_args()

    do_jrc = not args.deltares_only
    do_deltares = not args.jrc_only

    log.info("fetch_jrc_deltares.py starting (jrc=%s, deltares=%s)", do_jrc, do_deltares)

    aws = get_aws_credentials()
    s3 = boto3.client("s3", **aws)

    # Collect ZCTA universe from s035 event_features
    scenario_zctas = load_scenario_zctas(s3)
    all_zcta_ids = sorted(set(
        z for ids in scenario_zctas.values() for z in ids
    ))
    log.info("Total unique ZCTAs across %d scenarios: %d",
             len(scenario_zctas), len(all_zcta_ids))

    # Load centroids for the full universe
    all_centroids = load_centroids(s3, all_zcta_ids)

    if do_jrc:
        jrc_df = fetch_jrc(s3, scenario_zctas, all_centroids)
        log.info("JRC columns: %s", list(jrc_df.columns))

    if do_deltares:
        deltares_df = fetch_deltares(s3, scenario_zctas, all_centroids)
        log.info("Deltares columns: %s", list(deltares_df.columns))

    log.info("fetch_jrc_deltares.py complete")


if __name__ == "__main__":
    main()
