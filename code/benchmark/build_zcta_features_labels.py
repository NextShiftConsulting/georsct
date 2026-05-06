#!/usr/bin/env python3
"""
build_zcta_features_labels.py -- Assemble the canonical features+labels parquet.

This is the SINGLE SOURCE parquet for all downstream consumers:
  - Solver training (PCA32, Spatial Lag, GNN)
  - OOF prediction generation
  - N-ceiling estimation
  - Spatial lag computation (build_spatial_lags.py)
  - Allocator inference
  - GeoParquet assembly (build_geoparquet.py adds geometry + splits on top)

Merges:
  Base (69 cols):  33 ACS features + 27 targets + metadata
  + SVI:           5 columns  (svi_socioeconomic, svi_household_disability, ...)
  + HIFLD:         6 columns  (hifld_n_hospitals, hifld_nearest_hospital_km, ...)
  + Flood:         3-4 columns (flood_pct_zone_a, flood_pct_zone_x500, flood_pct_zone_x, flood_sfha)
  + Drive times:   2 columns  (drive_min_to_nearest_hospital, drive_min_to_county_seat)

Output: zcta_features_labels.parquet (~85 cols, 31,789 rows)

v23.001 = original 69-col (33 ACS + 27 targets + metadata)
v23.002 = enriched (v23.001 + SVI + HIFLD + flood + drive times)

S3 paths (all under s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/):
  Input:   zcta_features_labels.parquet  (v23.001, 69 cols)
  Enrich:  svi_zcta.parquet, hifld_zcta.parquet, flood_zones_zcta.parquet, drive_times_zcta.parquet
  Output:  zcta_features_labels.parquet  (v23.002, ~85 cols — overwrites v23.001)

Usage:
    python build_zcta_features_labels.py --dry-run    # local build, no upload
    python build_zcta_features_labels.py              # build + upload to S3
    python build_zcta_features_labels.py --local-dir /tmp/geo_data  # use local files
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

BUCKET = "swarm-yrsn-datasets"
PREFIX = "rsct_curriculum/series_018/processed"
REGION = "us-east-1"
AWS_PROFILE = "nsc-swarm"

# S3 keys — input is v23.001, output overwrites as v23.002
BASE_KEY = f"{PREFIX}/zcta_features_labels.parquet"
SVI_KEY = f"{PREFIX}/svi_zcta.parquet"
HIFLD_KEY = f"{PREFIX}/hifld_zcta.parquet"
FLOOD_KEY = f"{PREFIX}/flood_zones_zcta.parquet"
DRIVE_KEY = f"{PREFIX}/drive_times_zcta.parquet"
OUTPUT_KEY = f"{PREFIX}/zcta_features_labels.parquet"
PROVENANCE_KEY = f"{PREFIX}/zcta_features_labels_provenance.json"


def download(s3, key: str, local_dir: Path) -> Path:
    """Download a file from S3 to local_dir, return local path."""
    local = local_dir / Path(key).name
    if local.exists():
        log.info("  Already local: %s", local)
        return local
    log.info("  Downloading s3://%s/%s", BUCKET, key)
    s3.download_file(BUCKET, key, str(local))
    return local


def load_enrichment(path: Path, name: str) -> pd.DataFrame | None:
    """Load an enrichment parquet, normalize zcta_id, return or None."""
    if not path.exists():
        log.warning("  %s not found at %s -- SKIPPING", name, path)
        return None
    df = pd.read_parquet(path)
    # Normalize join key
    for col in ("zcta_id", "zcta"):
        if col in df.columns:
            df = df.rename(columns={col: "zcta_id"})
            break
    df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)
    log.info("  Loaded %s: %d rows, %d columns (%s)",
             name, len(df), len(df.columns),
             ", ".join(c for c in df.columns if c != "zcta_id"))
    return df


def main():
    parser = argparse.ArgumentParser(description="Assemble enriched features+labels parquet")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build locally, don't upload to S3")
    parser.add_argument("--local-dir", type=str, default=None,
                        help="Use local directory instead of downloading from S3")
    parser.add_argument("--output", type=str, default="/tmp/zcta_features_labels.parquet",
                        help="Local output path")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    if args.local_dir:
        local_dir = Path(args.local_dir)
        s3 = None
    else:
        session = boto3.Session(profile_name=AWS_PROFILE, region_name=REGION)
        s3 = session.client("s3")
        local_dir = Path("/tmp/geo_build")
        local_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load base parquet --
    log.info("=== LOADING BASE ===")
    if s3:
        base_path = download(s3, BASE_KEY, local_dir)
    else:
        base_path = local_dir / "zcta_features_labels.parquet"
    df = pd.read_parquet(base_path)
    df["zcta_id"] = df["zcta_id"].astype(str).str.zfill(5)
    log.info("  Base: %d rows, %d columns", len(df), len(df.columns))
    base_cols = set(df.columns)

    # -- 2. Download enrichment parquets --
    log.info("\n=== LOADING ENRICHMENT LAYERS ===")
    enrichment_keys = {
        "CDC SVI": SVI_KEY,
        "HIFLD facilities": HIFLD_KEY,
        "FEMA flood zones": FLOOD_KEY,
        "Drive times": DRIVE_KEY,
    }
    enrichment_dfs = {}
    for name, key in enrichment_keys.items():
        if s3:
            path = download(s3, key, local_dir)
        else:
            path = local_dir / Path(key).name
        enrichment_dfs[name] = load_enrichment(path, name)

    # -- 3. Merge --
    log.info("\n=== MERGING ===")
    added_cols = []
    for name, edf in enrichment_dfs.items():
        if edf is None:
            continue
        new_cols = [c for c in edf.columns if c != "zcta_id"]
        # Check for collisions
        collisions = [c for c in new_cols if c in df.columns]
        if collisions:
            log.warning("  %s: column collision, dropping from base: %s", name, collisions)
            df = df.drop(columns=collisions)
        before = len(df)
        df = df.merge(edf, on="zcta_id", how="left")
        after = len(df)
        if after != before:
            log.error("  %s: row count changed %d -> %d (duplicate zcta_id?)", name, before, after)
            sys.exit(1)
        added_cols.extend(new_cols)
        log.info("  + %s: %d columns merged", name, len(new_cols))

    # -- 4. Column ordering --
    # metadata -> acs features -> enrichment -> targets
    meta_cols = ["zcta_id", "latitude", "longitude", "county_name", "state", "population"]
    split_cols = [c for c in df.columns if c.startswith("split_")]
    acs_cols = sorted(c for c in df.columns if c.startswith("acs_"))
    svi_cols = sorted(c for c in df.columns if c.startswith("svi_"))
    flood_cols = sorted(c for c in df.columns if c.startswith("flood_"))
    hifld_cols = sorted(c for c in df.columns if c.startswith("hifld_"))
    drive_cols = sorted(c for c in df.columns if c.startswith("drive_"))
    target_cols = sorted(c for c in df.columns if c.startswith("target_"))

    ordered = (
        [c for c in meta_cols if c in df.columns]
        + [c for c in split_cols if c in df.columns]
        + acs_cols
        + svi_cols
        + flood_cols
        + hifld_cols
        + drive_cols
        + target_cols
    )
    # Catch any columns we missed
    remaining = [c for c in df.columns if c not in ordered]
    if remaining:
        log.warning("  Unclassified columns (appended at end): %s", remaining)
        ordered.extend(remaining)

    df = df[ordered]

    # -- 5. Summary --
    log.info("\n=== RESULT ===")
    log.info("  Rows:       %d", len(df))
    log.info("  Columns:    %d (was %d base + %d enrichment)", len(df.columns), len(base_cols), len(added_cols))
    log.info("  Metadata:   %s", [c for c in meta_cols if c in df.columns])
    log.info("  Splits:     %d", len(split_cols))
    log.info("  ACS:        %d", len(acs_cols))
    log.info("  SVI:        %d", len(svi_cols))
    log.info("  Flood:      %d", len(flood_cols))
    log.info("  HIFLD:      %d", len(hifld_cols))
    log.info("  Drive:      %d", len(drive_cols))
    log.info("  Targets:    %d", len(target_cols))

    # NaN report for enrichment columns
    log.info("\n=== ENRICHMENT COVERAGE ===")
    for col in svi_cols + flood_cols + hifld_cols + drive_cols:
        n_valid = df[col].notna().sum()
        log.info("  %-40s %d / %d (%.1f%%)", col, n_valid, len(df), 100 * n_valid / len(df))

    # -- 6. Save --
    output = Path(args.output)
    log.info("\nWriting to %s", output)
    df.to_parquet(output, index=False)
    size_mb = output.stat().st_size / (1024 * 1024)
    log.info("  Size: %.1f MB", size_mb)

    # -- 7. Upload --
    if not args.dry_run and s3:
        log.info("Uploading to s3://%s/%s", BUCKET, OUTPUT_KEY)
        s3.upload_file(str(output), BUCKET, OUTPUT_KEY)

        provenance = {
            "operation": "build_zcta_features_labels",
            "timestamp": timestamp,
            "base_source": f"s3://{BUCKET}/{BASE_KEY}",
            "enrichment_sources": {
                name: f"s3://{BUCKET}/{key}"
                for name, key in enrichment_keys.items()
                if enrichment_dfs.get(name) is not None
            },
            "output": f"s3://{BUCKET}/{OUTPUT_KEY}",
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_base_columns": len(base_cols),
            "n_enrichment_columns": len(added_cols),
            "enrichment_columns": added_cols,
            "file_size_mb": round(size_mb, 1),
            "columns": list(df.columns),
        }
        s3.put_object(
            Bucket=BUCKET, Key=PROVENANCE_KEY,
            Body=json.dumps(provenance, indent=2),
            ContentType="application/json",
        )
        log.info("Provenance: s3://%s/%s", BUCKET, PROVENANCE_KEY)
    else:
        log.info("[DRY RUN] Skipping S3 upload.")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
