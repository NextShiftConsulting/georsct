#!/usr/bin/env python3
"""
validate_assembly.py -- Three-layer validation for FloodRSCT pipeline.

Layer 1: Interface Contract -- verify fetcher outputs match builder expectations
Layer 2: Post-Assembly     -- coverage thresholds and row-count guards
Layer 3: Data Lock         -- full reconciliation against FEATURE_CONTRACT.yaml

Can be run standalone (Layer 1 + 3) or imported by build_event_dataset.py (Layer 2).

Usage:
    # Pre-assembly: check S3 data matches what builder expects
    python validate_assembly.py --layer interface --scenario houston

    # Post-assembly: validate an assembled parquet
    python validate_assembly.py --layer post --parquet /tmp/houston.parquet

    # Data Lock: full contract reconciliation
    python validate_assembly.py --layer lock --scenario houston

    # All layers
    python validate_assembly.py --all --scenario houston
"""

import argparse
import logging
import sys
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Layer 1: Interface Contract Validation
# ---------------------------------------------------------------------------

# What each fetcher actually writes (column names in parquet on S3)
FETCHER_SCHEMAS = {
    "hurdat2": {
        "prefix": "raw/hurdat2/",
        "file": "storm_tracks.parquet",
        "expected_columns": ["storm_id", "storm_name", "timestamp", "status",
                             "lat", "lon", "max_wind_kt", "min_pressure_mb"],
        "min_rows": 100,
    },
    "noaa_tides": {
        "prefix": "raw/noaa_tides/{event}/",
        "file_pattern": "tidal_surge_*.parquet",
        "expected_columns": ["timestamp", "observed_m", "predicted_m", "surge_m"],
        "min_rows": 10,
    },
    "noaa_mrms": {
        "prefix": "raw/noaa_mrms/{event}/",
        "file_pattern": "*.grib2.gz",
        "min_file_size_bytes": 10_000,
        "min_files": 24,
    },
    "usgs_stn": {
        "prefix": "raw/usgs_stn/",
        "file_pattern": "{event}_hwm.parquet",
        "expected_columns": ["latitude", "longitude", "hwm_id"],
        "min_rows": 1,
    },
    "surge_estimates": {
        "prefix": "raw/surge_estimates/{event}/",
        "file_pattern": "hwm_{event}.parquet",
        "expected_columns": ["latitude", "longitude"],
        "min_rows": 1,
    },
    "nlcd": {
        "prefix": "raw/nlcd/impervious_2021/",
        "file_pattern": "*.img",
        "min_file_size_bytes": 1_000_000,
        "min_files": 1,
    },
    "dem_3dep": {
        "prefix": "raw/dem/3dep/",
        "file_pattern": "*.tif",
        "min_file_size_bytes": 100_000,
        "min_files": 1,
    },
    "geocertdb2026": {
        "prefix": "raw/geocertdb2026/",
        "file": "zcta_features_labels.parquet",
        "expected_columns": ["zcta_id", "latitude", "longitude", "state"],
        "min_rows": 30_000,
    },
    "geocertdb2026_crosswalk": {
        "prefix": "raw/geocertdb2026/",
        "file": "zcta_county_crosswalk.parquet",
        "expected_columns": ["zcta_id"],
        "min_rows": 30_000,
    },
    "openfema_nfip": {
        "prefix": "raw/openfema/",
        "file_pattern": "nfip_claims_dr*.parquet",
        "min_rows": 1,
    },
}

# What the builder expects vs what the fetcher writes
INTERFACE_CHECKS = [
    {
        "builder_function": "compute_storm_proximity",
        "fetcher": "hurdat2",
        "column_map": {
            "timestamp": "timestamp",       # builder uses "timestamp" (fixed from datetime_utc)
            "lat": "lat",
            "lon": "lon",
            "max_wind_kt": "max_wind_kt",   # builder derives category from this
        },
        "note": "Builder derives Saffir-Simpson category from max_wind_kt",
    },
    {
        "builder_function": "aggregate_tides",
        "fetcher": "noaa_tides",
        "column_map": {
            "observed_m": "observed_m",     # builder uses "observed_m" (fixed from water_level_m)
            "surge_m": "surge_m",
        },
        "note": "No station_id column in parquet; extracted from S3 key",
    },
    {
        "builder_function": "aggregate_mrms_rainfall",
        "fetcher": "noaa_mrms",
        "format_check": "gzip",             # files are .grib2.gz, need decompression
        "note": "Builder decompresses gzip before cfgrib (fixed)",
    },
    {
        "builder_function": "build_impervious_features",
        "fetcher": "nlcd",
        "crs_check": "EPSG:5070",
        "note": "Builder reprojects centroid to raster CRS (fixed)",
    },
]


def validate_interface(s3, scenario: str, events: list[str]) -> list[dict]:
    """Layer 1: Check that fetcher outputs on S3 match builder expectations."""
    results = []

    for name, schema in FETCHER_SCHEMAS.items():
        prefix = schema["prefix"]
        check = {"dataset": name, "status": "UNKNOWN", "details": []}

        # Handle event-templated prefixes
        if "{event}" in prefix:
            for event in events:
                event_prefix = prefix.format(event=event)
                event_check = _check_s3_prefix(s3, event_prefix, schema, event)
                event_check["dataset"] = f"{name}/{event}"
                results.append(event_check)
            continue

        # Non-event datasets
        if "file" in schema:
            key = prefix + schema["file"]
            try:
                resp = s3.head_object(Bucket=BUCKET, Key=key)
                size = resp["ContentLength"]
                check["status"] = "PRESENT"
                check["details"].append(f"size={size:,} bytes")

                # Check columns if parquet
                if key.endswith(".parquet") and "expected_columns" in schema:
                    col_check = _check_parquet_columns(s3, key, schema)
                    check["details"].extend(col_check)
            except s3.exceptions.ClientError:
                check["status"] = "MISSING"
                check["details"].append(f"s3://{BUCKET}/{key} not found")
        else:
            check = _check_s3_prefix(s3, prefix, schema, None)
            check["dataset"] = name

        results.append(check)

    return results


def _check_s3_prefix(s3, prefix: str, schema: dict, event: str | None) -> dict:
    """Check files under an S3 prefix against schema expectations."""
    check = {"dataset": prefix, "status": "UNKNOWN", "details": []}

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    contents = resp.get("Contents", [])

    if not contents:
        check["status"] = "MISSING"
        check["details"].append(f"no objects under s3://{BUCKET}/{prefix}")
        return check

    # File count
    min_files = schema.get("min_files", 1)
    if len(contents) < min_files:
        check["status"] = "INCOMPLETE"
        check["details"].append(f"only {len(contents)} files (expected >= {min_files})")
    else:
        check["status"] = "PRESENT"
        check["details"].append(f"{len(contents)} files")

    # File size check (for detecting stubs)
    min_size = schema.get("min_file_size_bytes", 0)
    if min_size > 0:
        small_files = [o for o in contents if o["Size"] < min_size]
        if small_files:
            check["status"] = "CORRUPT"
            check["details"].append(
                f"{len(small_files)} files below {min_size} bytes (stubs?)"
            )

    return check


def _check_parquet_columns(s3, key: str, schema: dict) -> list[str]:
    """Download parquet header and check expected columns exist."""
    import tempfile
    details = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            s3.download_file(BUCKET, key, tmp.name)
            df = pd.read_parquet(tmp.name, columns=None)
            actual_cols = set(df.columns)

            for expected in schema.get("expected_columns", []):
                if expected not in actual_cols:
                    details.append(f"MISSING column: {expected}")

            if "min_rows" in schema and len(df) < schema["min_rows"]:
                details.append(
                    f"only {len(df)} rows (expected >= {schema['min_rows']})"
                )

            if not details:
                details.append(f"schema OK ({len(df)} rows, {len(actual_cols)} cols)")
            Path(tmp.name).unlink(missing_ok=True)
    except Exception as exc:
        details.append(f"could not read: {exc}")

    return details


# ---------------------------------------------------------------------------
# Layer 2: Post-Assembly Validation (importable by build_event_dataset.py)
# ---------------------------------------------------------------------------

# Minimum expected non-null rates per column group, per scenario
# These thresholds flag "something is broken" not "data is sparse"
COVERAGE_THRESHOLDS = {
    "rainfall_total_mm":      {"houston": 0.5, "southwest_florida": 0.5, "nyc": 0.5,
                               "new_orleans": 0.5, "riverside_coachella": 0.3},
    "max_surge_m":            {"houston": 0.3, "southwest_florida": 0.3, "new_orleans": 0.3,
                               "nyc": 0.0, "riverside_coachella": 0.0},
    "storm_min_dist_km":      {"houston": 0.9, "southwest_florida": 0.9, "nyc": 0.9,
                               "new_orleans": 0.9, "riverside_coachella": 0.5},
    "impervious_pct":         {"houston": 0.5, "nyc": 0.5,
                               "southwest_florida": 0.0, "new_orleans": 0.0,
                               "riverside_coachella": 0.0},
    "elevation_m_msl":        {"southwest_florida": 0.5, "new_orleans": 0.5,
                               "houston": 0.0, "nyc": 0.0, "riverside_coachella": 0.0},
    "storm_landfall_category": {"houston": 0.9, "southwest_florida": 0.9,
                                "nyc": 0.5, "new_orleans": 0.9,
                                "riverside_coachella": 0.3},
}


def validate_post_assembly(df: pd.DataFrame, scenario: str) -> dict:
    """Layer 2: Validate an assembled DataFrame. Returns report dict.

    Call this from build_event_dataset.py after assembly, before writing output.
    """
    report = {
        "scenario": scenario,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_events": df["event"].nunique() if "event" in df.columns else 0,
        "coverage": {},
        "warnings": [],
        "errors": [],
    }

    # Row count sanity
    if len(df) == 0:
        report["errors"].append("EMPTY: zero rows in assembled DataFrame")
        return report

    # Per-column coverage
    for col in df.columns:
        if col.startswith("_fs_") or col in ("zcta_id", "event", "scenario"):
            continue
        non_null_rate = df[col].notna().mean()
        report["coverage"][col] = round(float(non_null_rate), 3)

    # Check against thresholds
    for col, thresholds in COVERAGE_THRESHOLDS.items():
        if col not in df.columns:
            continue
        threshold = thresholds.get(scenario, 0.0)
        if threshold == 0.0:
            continue
        actual = df[col].notna().mean()
        if actual < threshold:
            msg = (f"COVERAGE FAIL: {col} has {actual:.1%} non-null "
                   f"(threshold: {threshold:.0%} for {scenario})")
            report["errors"].append(msg)
        elif actual < threshold * 1.5:
            msg = (f"COVERAGE WARN: {col} has {actual:.1%} non-null "
                   f"(threshold: {threshold:.0%} for {scenario})")
            report["warnings"].append(msg)

    # Duplicate check
    if "zcta_id" in df.columns and "event" in df.columns:
        dupes = df.duplicated(subset=["zcta_id", "event"]).sum()
        if dupes > 0:
            report["errors"].append(f"DUPLICATE: {dupes} duplicate (zcta_id, event) rows")

    # All-NaN column check
    all_nan_cols = [c for c in df.columns
                    if not c.startswith("_fs_") and c not in ("zcta_id", "event", "scenario")
                    and df[c].isna().all()]
    if all_nan_cols:
        report["warnings"].append(f"ALL-NaN columns: {all_nan_cols}")

    # Constant columns (suspicious)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].notna().any() and df[col].nunique() == 1:
            report["warnings"].append(f"CONSTANT: {col} = {df[col].dropna().iloc[0]} for all rows")

    return report


def log_validation_report(report: dict) -> bool:
    """Log a validation report. Returns True if no errors."""
    log.info("=== VALIDATION: %s ===", report["scenario"])
    log.info("  Rows: %d, Columns: %d, Events: %d",
             report["n_rows"], report["n_columns"], report["n_events"])

    if report["errors"]:
        for err in report["errors"]:
            log.error("  %s", err)

    if report["warnings"]:
        for warn in report["warnings"]:
            log.warning("  %s", warn)

    # Coverage summary: show worst columns
    if report["coverage"]:
        worst = sorted(report["coverage"].items(), key=lambda x: x[1])[:10]
        log.info("  Lowest coverage:")
        for col, rate in worst:
            marker = " <<< BELOW THRESHOLD" if rate < 0.1 else ""
            log.info("    %-35s %.1f%%%s", col, rate * 100, marker)

    n_errors = len(report["errors"])
    n_warnings = len(report["warnings"])
    if n_errors == 0:
        log.info("  PASS (%d warnings)", n_warnings)
    else:
        log.error("  FAIL (%d errors, %d warnings)", n_errors, n_warnings)

    return n_errors == 0


# ---------------------------------------------------------------------------
# Layer 3: Data Lock Validation (against FEATURE_CONTRACT.yaml)
# ---------------------------------------------------------------------------

def validate_data_lock(
    s3, contract_path: str, scenario: str, parquet_path: str | None = None
) -> dict:
    """Layer 3: Validate assembled data against FEATURE_CONTRACT.yaml."""
    report = {
        "scenario": scenario,
        "contract_checks": [],
        "s3_checks": [],
        "errors": [],
        "warnings": [],
    }

    # Load contract
    with open(contract_path) as f:
        contract = yaml.safe_load(f)

    features = contract.get("features", [])
    scenario_features = [
        feat for feat in features
        if scenario in feat.get("scenarios", [])
    ]
    log.info("Contract has %d features for scenario %s", len(scenario_features), scenario)

    # Check S3 raw paths exist
    for feat in scenario_features:
        raw_path = feat.get("raw_s3_path")
        if not raw_path or raw_path == "null":
            continue

        name = feat["feature_name"]
        # Template event placeholder
        if "{event}" in raw_path:
            report["contract_checks"].append({
                "feature": name,
                "raw_path": raw_path,
                "status": "TEMPLATED (checked per-event in Layer 1)",
            })
            continue

        try:
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=raw_path, MaxKeys=1)
            exists = len(resp.get("Contents", [])) > 0
        except Exception:
            exists = False

        status = "PRESENT" if exists else "MISSING"
        if not exists:
            temporal = feat.get("temporal_class", "")
            if temporal == "operational":
                status = "EXPECTED_MISSING (operational)"
            else:
                report["errors"].append(f"RAW DATA MISSING: {name} at {raw_path}")

        report["contract_checks"].append({
            "feature": name,
            "raw_path": raw_path,
            "status": status,
        })

    # Check assembled parquet columns against contract
    if parquet_path:
        df = pd.read_parquet(parquet_path)
        actual_cols = set(df.columns)

        for feat in scenario_features:
            col = feat.get("output_column")
            if not col:
                continue
            if col in actual_cols:
                non_null = df[col].notna().mean()
                report["contract_checks"].append({
                    "feature": feat["feature_name"],
                    "column": col,
                    "status": "PRESENT",
                    "coverage": f"{non_null:.1%}",
                })
            else:
                temporal = feat.get("temporal_class", "")
                if temporal == "operational":
                    report["warnings"].append(
                        f"Column {col} missing (operational -- expected)"
                    )
                else:
                    report["errors"].append(
                        f"CONTRACT VIOLATION: column {col} ({feat['feature_name']}) "
                        f"missing from assembled parquet"
                    )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SCENARIO_EVENTS = {
    "houston": ["harvey2017", "imelda2019", "beryl2024"],
    "new_orleans": ["ida2021"],
    "nyc": ["ida2021_nyc", "henri2021"],
    "riverside_coachella": ["hilary2023", "tropical_storm_2023_03"],
    "southwest_florida": ["ian2022", "helene2024", "milton2024"],
}


def main():
    parser = argparse.ArgumentParser(description="FloodRSCT pipeline validation")
    parser.add_argument("--layer", choices=["interface", "post", "lock", "all"],
                        default="all")
    parser.add_argument("--scenario", default="houston")
    parser.add_argument("--parquet", default=None,
                        help="Path to assembled parquet (for post/lock layers)")
    parser.add_argument("--contract",
                        default=str(Path(__file__).parent.parent / "FEATURE_CONTRACT.yaml"),
                        help="Path to FEATURE_CONTRACT.yaml")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name="us-east-1")
    events = SCENARIO_EVENTS.get(args.scenario, [])
    all_pass = True

    if args.layer in ("interface", "all"):
        log.info("\n========== LAYER 1: INTERFACE CONTRACT ==========")
        results = validate_interface(s3, args.scenario, events)
        for r in results:
            status = r["status"]
            marker = "PASS" if status == "PRESENT" else status
            log.info("  %-40s %s  %s", r["dataset"], marker, "; ".join(r["details"]))
            if status in ("MISSING", "CORRUPT"):
                all_pass = False

    if args.layer in ("post", "all") and args.parquet:
        log.info("\n========== LAYER 2: POST-ASSEMBLY ==========")
        df = pd.read_parquet(args.parquet)
        report = validate_post_assembly(df, args.scenario)
        passed = log_validation_report(report)
        if not passed:
            all_pass = False

    if args.layer in ("lock", "all"):
        log.info("\n========== LAYER 3: DATA LOCK ==========")
        report = validate_data_lock(s3, args.contract, args.scenario, args.parquet)
        n_errors = len(report["errors"])
        n_warnings = len(report["warnings"])
        for check in report["contract_checks"]:
            log.info("  %-35s %s", check["feature"], check["status"])
        for err in report["errors"]:
            log.error("  %s", err)
        for warn in report["warnings"]:
            log.warning("  %s", warn)
        if n_errors > 0:
            log.error("  DATA LOCK: FAIL (%d errors, %d warnings)", n_errors, n_warnings)
            all_pass = False
        else:
            log.info("  DATA LOCK: PASS (%d warnings)", n_warnings)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
