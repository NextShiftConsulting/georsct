#!/usr/bin/env python3
"""
build_release_package.py -- Build the full GeoCert release package.

Takes the validated GeoParquet and produces all release artifacts:
  - geocert_simplified_001.geoparquet  (renamed copy)
  - geocert_table.parquet              (no geometry, lightweight)
  - geocert_schema.json                (column metadata)
  - build_manifest.json                  (provenance + stats)
  - geocert_checksums.sha256           (integrity verification)

Usage:
    python build_release_package.py --geoparquet /tmp/geocert.geoparquet
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geoparquet", type=str, default="/tmp/geocert.geoparquet")
    parser.add_argument("--output-dir", type=str, default="/tmp/geocert_release")
    args = parser.parse_args()

    src = Path(args.geoparquet)
    out = Path(args.output_dir)

    if not src.exists():
        print(f"ERROR: GeoParquet not found at {src}", file=sys.stderr)
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    # -- 1. Load GeoParquet --
    print(f"Loading {src} ...")
    geo = gpd.read_parquet(src)
    print(f"  {len(geo)} rows, {len(geo.columns)} columns")
    print(f"  CRS: {geo.crs}")

    # -- 2. Copy as named GeoParquet --
    geo_out = out / "geocert_simplified_001.geoparquet"
    shutil.copy2(src, geo_out)
    print(f"Copied -> {geo_out}")

    # -- 3. Table parquet (no geometry) --
    table_out = out / "geocert_table.parquet"
    table_cols = [c for c in geo.columns if c != "geometry"]
    table_df = pd.DataFrame(geo[table_cols])
    table_df.to_parquet(table_out, index=False)
    print(f"Table parquet -> {table_out} ({table_out.stat().st_size / 1e6:.1f} MB)")

    # -- 4. Schema JSON --
    schema = {
        "dataset": "GeoCert",
        "version": "23.001",
        "crs": "EPSG:4326",
        "n_rows": len(geo),
        "n_columns": len(geo.columns),
        "geometry_simplification_tolerance_degrees": 0.001,
        "columns": {},
    }

    # Classify columns
    meta_cols = ["zcta_id", "county_name", "state", "latitude", "longitude"]
    split_cols = ["split_imputation", "split_extrapolation", "split_superres"]
    coverage_cols = ["has_cdc_places", "has_income", "has_home_value"]
    acs_cols = sorted(c for c in geo.columns if c.startswith("acs_"))
    target_cols = sorted(c for c in geo.columns if c.startswith("target_"))

    for col in geo.columns:
        if col == "geometry":
            info = {
                "dtype": "geometry",
                "group": "geometry",
                "description": "ZCTA boundary polygon (Census TIGER/Line 2022, simplified 0.001 deg)",
            }
        else:
            series = geo[col]
            dtype_str = str(series.dtype)
            n_null = int(series.isna().sum())
            info = {
                "dtype": dtype_str,
                "n_null": n_null,
                "pct_null": round(n_null / len(geo) * 100, 2),
            }

            if col in meta_cols:
                info["group"] = "metadata"
            elif col in split_cols:
                info["group"] = "split"
                info["values"] = sorted(series.dropna().unique().tolist())
            elif col in coverage_cols:
                info["group"] = "coverage"
                info["values"] = [True, False]
            elif col in acs_cols:
                info["group"] = "feature"
                info["source"] = "ACS 2022 5-Year Estimates"
            elif col in target_cols:
                info["group"] = "target"
                if n_null == 0:
                    info["min"] = round(float(series.min()), 4)
                    info["max"] = round(float(series.max()), 4)
                    info["mean"] = round(float(series.mean()), 4)
                else:
                    info["min"] = round(float(series.min(skipna=True)), 4)
                    info["max"] = round(float(series.max(skipna=True)), 4)
                    info["mean"] = round(float(series.mean(skipna=True)), 4)
            else:
                info["group"] = "other"

        schema["columns"][col] = info

    schema_out = out / "geocert_schema.json"
    schema_out.write_text(json.dumps(schema, indent=2))
    print(f"Schema -> {schema_out}")

    # -- 5. Build manifest --
    manifest = {
        "dataset": "GeoCert",
        "version": "23.001",
        "build_timestamp": timestamp,
        "description": "27 geospatial regression tasks across 31,789 US ZCTAs",
        "sources": {
            "features_labels": "s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/zcta_features_labels.parquet",
            "splits": "s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/geocert_splits.parquet",
            "boundaries": "https://www2.census.gov/geo/tiger/TIGER2022/ZCTA520/tl_2022_us_zcta520.zip",
        },
        "geometry": {
            "source": "Census TIGER/Line 2022 ZCTA boundaries",
            "crs": "EPSG:4326",
            "simplification_tolerance_degrees": 0.001,
        },
        "stats": {
            "n_zctas": len(geo),
            "n_columns_geo": len(geo.columns),
            "n_columns_table": len(table_cols),
            "n_acs_features": len(acs_cols),
            "n_targets": len(target_cols),
            "n_split_columns": len(split_cols),
            "n_coverage_flags": len(coverage_cols),
        },
        "coverage": {
            "has_cdc_places_true": int(geo["has_cdc_places"].sum()),
            "has_cdc_places_false": int((~geo["has_cdc_places"]).sum()),
            "has_income_true": int(geo["has_income"].sum()),
            "has_income_false": int((~geo["has_income"]).sum()),
            "has_home_value_true": int(geo["has_home_value"].sum()),
            "has_home_value_false": int((~geo["has_home_value"]).sum()),
        },
        "evaluation_protocols": {
            "imputation": "County-holdout 5-fold CV + test. Geographic interpolation.",
            "extrapolation": "State-holdout 4-fold CV + test. Distribution shift.",
            "super_resolution": "County-aggregated train -> ZCTA-level prediction.",
        },
        "licensing": {
            "CDC_PLACES": "Public domain (US government work)",
            "ACS": "Public domain (US government work)",
            "VIIRS": "Public domain (US government work)",
            "USGS_NED": "Public domain (US government work)",
            "Hansen_GFC": "CC-BY-4.0",
            "Census_TIGER": "Public domain (US government work)",
        },
        "files": {},
    }

    # File sizes (manifest itself added after)
    for f in [geo_out, table_out, schema_out]:
        manifest["files"][f.name] = {
            "size_bytes": f.stat().st_size,
            "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
        }

    manifest_out = out / "build_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest -> {manifest_out}")

    # -- 6. Checksums --
    checksum_out = out / "geocert_checksums.sha256"
    lines = []
    for f in sorted(out.glob("*")):
        if f.name == "geocert_checksums.sha256":
            continue
        h = sha256_file(f)
        lines.append(f"{h}  {f.name}")
        print(f"  SHA256 {f.name}: {h[:16]}...")

    checksum_out.write_text("\n".join(lines) + "\n")
    print(f"Checksums -> {checksum_out}")

    # -- 7. Summary --
    print()
    print("=== RELEASE PACKAGE ===")
    print(f"  Directory: {out}")
    total_size = sum(f.stat().st_size for f in out.glob("*"))
    print(f"  Total size: {total_size / (1024 * 1024):.1f} MB")
    print(f"  Files:")
    for f in sorted(out.glob("*")):
        print(f"    {f.name:45s} {f.stat().st_size / (1024 * 1024):8.1f} MB")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
