#!/usr/bin/env python3
"""run_fetch_hydrology.py -- Extract DEM-derived hydrology features per ZCTA.

Fetches Copernicus GLO-30 DEM from Microsoft Planetary Computer and computes
four topographic indices within ~1 km of each ZCTA centroid:

  - hand_mean_m: Height Above Nearest Drainage (mean)
  - gfi_mean: Geomorphic Flood Index (mean)
  - twi_mean: Topographic Wetness Index (mean)
  - spi_mean: Stream Power Index (mean)

Output is scenario-independent (keyed by ZCTA, not by event), so results live
in processed/shared/ as per-scenario files. Each scenario writes its own file
(e.g. zcta_hydrology_houston.parquet) -- no shared cache, no merge step, no
race condition. Scenarios can run in parallel safely.

Source:
  Copernicus DEM GLO-30 via Microsoft Planetary Computer STAC

Output:
  s3://swarm-floodrsct-data/processed/shared/zcta_hydrology_{scenario}.parquet
  s3://swarm-floodrsct-data/results/s035/hydrology_extraction_{scenario}.json

Usage:
    python run_fetch_hydrology.py --scenario houston --upload
    python run_fetch_hydrology.py --scenario houston --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# S3 infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client
from _s3_result import upload_json_result

# Output keys -- one file per scenario, no shared merge
HYDROLOGY_KEY_TEMPLATE = "processed/shared/zcta_hydrology_{scenario}.parquet"
OUTPUT_COLUMNS = ["zcta_id", "hand_mean_m", "twi_mean", "gfi_mean", "spi_mean"]

# ZCTA centroids source
STATIC_KEY = "raw/geocertdb2026/zcta_features_labels.parquet"

# Scenario event_features keys (for ZCTA ID extraction)
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

def s3_read_parquet(s3, key: str):
    """Read parquet from S3; return None if missing."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
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


def get_git_hash() -> str:
    """Return current git hash or 'unknown'."""
    gh = os.environ.get("S035_GIT_HASH")
    if gh:
        return gh
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def get_scenario_zctas(s3, scenario: str) -> list[str]:
    """Get unique ZCTA IDs for a scenario from its processed event features."""
    key = SCENARIO_EVENT_KEYS[scenario]
    log.info("Loading ZCTAs from s3://%s/%s", BUCKET, key)
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()), columns=["zcta_id"])
    zctas = sorted(df["zcta_id"].astype(str).unique().tolist())
    log.info("Scenario %s: %d unique ZCTAs", scenario, len(zctas))
    return zctas


def load_centroids(s3, zcta_ids: list[str]) -> pd.DataFrame:
    """Load ZCTA centroids from geocertdb2026 static features."""
    log.info("Loading centroids from s3://%s/%s", BUCKET, STATIC_KEY)
    resp = s3.get_object(Bucket=BUCKET, Key=STATIC_KEY)
    static = pd.read_parquet(io.BytesIO(resp["Body"].read()))

    # Find columns by name pattern
    zcta_col = next((c for c in static.columns if "zcta" in c.lower()), None)
    lat_col = next((c for c in static.columns if "lat" in c.lower()), None)
    lon_col = next(
        (c for c in static.columns if "lon" in c.lower() or "lng" in c.lower()),
        None,
    )
    if not all([zcta_col, lat_col, lon_col]):
        raise RuntimeError(
            f"Cannot find zcta/lat/lon columns in {STATIC_KEY}. "
            f"Available: {list(static.columns)[:20]}"
        )

    static = static[[zcta_col, lat_col, lon_col]].rename(
        columns={zcta_col: "zcta_id", lat_col: "lat", lon_col: "lon"},
    )
    static["zcta_id"] = static["zcta_id"].astype(str)

    centroids = static[static["zcta_id"].isin(zcta_ids)].dropna(subset=["lat", "lon"])
    log.info("Centroids matched: %d / %d ZCTAs", len(centroids), len(zcta_ids))

    if centroids.empty:
        raise RuntimeError(
            f"No centroids found for {len(zcta_ids)} ZCTAs. "
            f"Check that {STATIC_KEY} contains matching ZCTA IDs."
        )

    return centroids


def extract_hydrology(centroids: pd.DataFrame) -> pd.DataFrame:
    """Run floodcaster hydrology extraction on centroid locations.

    Returns DataFrame with columns: zcta_id, hand_mean_m, twi_mean, gfi_mean, spi_mean.
    """
    from floodcaster.batch import hydrology_centroid_stats

    log.info("Extracting hydrology for %d centroids via Planetary Computer...", len(centroids))
    t0 = time.time()
    extracted = hydrology_centroid_stats(centroids, id_col="zcta_id")
    elapsed = time.time() - t0
    log.info("Extraction complete: %d rows in %.1fs", len(extracted), elapsed)

    # Verify output columns
    for col in OUTPUT_COLUMNS:
        if col not in extracted.columns and col != "zcta_id":
            log.warning("Missing column %s in extraction output", col)

    n_valid = extracted["hand_mean_m"].notna().sum() if "hand_mean_m" in extracted.columns else 0
    log.info("Valid HAND values: %d / %d (%.1f%%)",
             n_valid, len(extracted),
             100 * n_valid / len(extracted) if len(extracted) > 0 else 0)

    return extracted


def write_scenario_cache(s3, scenario: str, data: pd.DataFrame) -> int:
    """Write per-scenario hydrology file to S3. Returns row count."""
    data["zcta_id"] = data["zcta_id"].astype(str)
    keep_cols = [c for c in OUTPUT_COLUMNS if c in data.columns]
    out = data[keep_cols].drop_duplicates(subset=["zcta_id"], keep="last")
    key = HYDROLOGY_KEY_TEMPLATE.format(scenario=scenario)
    s3_write_parquet(s3, out, key)
    return len(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scenario: str, upload: bool, dry_run: bool) -> None:
    t0 = time.time()
    s3 = get_s3_client()
    git_hash = get_git_hash()

    print("=" * 60)
    print(f"  HYDROLOGY EXTRACTION -- {scenario}")
    print("=" * 60)
    sys.stdout.flush()

    # 1. Get scenario ZCTAs
    zcta_ids = get_scenario_zctas(s3, scenario)

    # 2. Check if per-scenario cache already exists
    cache_key = HYDROLOGY_KEY_TEMPLATE.format(scenario=scenario)
    cached = s3_read_parquet(s3, cache_key)
    if cached is not None and len(cached) > 0:
        log.info("Per-scenario cache exists: %s (%d rows). Skipping.", cache_key, len(cached))
        evidence = {
            "scenario": scenario,
            "status": "CACHE_HIT",
            "n_zctas_requested": len(zcta_ids),
            "n_zctas_new": 0,
            "n_zctas_in_cache": len(cached),
            "git_hash": git_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        if dry_run:
            log.info("DRY RUN: would extract %d ZCTAs for %s", len(zcta_ids), scenario)
            return

        # 3. Load centroids
        centroids = load_centroids(s3, zcta_ids)

        # 4. Extract hydrology features
        extracted = extract_hydrology(centroids)

        # 5. Write per-scenario file (no merge, no race)
        n_written = write_scenario_cache(s3, scenario, extracted)

        elapsed = time.time() - t0
        n_valid = int(extracted["hand_mean_m"].notna().sum()) if "hand_mean_m" in extracted.columns else 0

        evidence = {
            "scenario": scenario,
            "status": "EXTRACTED",
            "n_zctas_requested": len(zcta_ids),
            "n_zctas_extracted": len(extracted),
            "n_valid_hand": n_valid,
            "coverage_pct": round(100 * n_valid / len(extracted), 1) if len(extracted) > 0 else 0,
            "n_zctas_in_cache": n_written,
            "elapsed_sec": round(elapsed, 1),
            "git_hash": git_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Print summary
    print("\n" + "=" * 60)
    print(f"  RESULT: {scenario}")
    for k, v in evidence.items():
        print(f"    {k}: {v}")
    print("=" * 60)
    sys.stdout.flush()

    # Upload evidence JSON
    if upload:
        key = f"results/s035/hydrology_extraction_{scenario}.json"
        upload_json_result(s3, BUCKET, key, evidence, git_hash=git_hash)

    print(json.dumps(evidence, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DEM-derived hydrology features per ZCTA"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS,
                        help="Scenario to process (run one at a time)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload results to S3")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without extracting")
    args = parser.parse_args()

    run(args.scenario, args.upload, args.dry_run)


if __name__ == "__main__":
    main()
