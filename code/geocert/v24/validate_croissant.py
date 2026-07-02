#!/usr/bin/env python3
"""
validate_croissant.py -- Pre-flight validator: parquet vs Croissant manifest.

Reads croissant.json, extracts all promised columns per recordSet, then checks
that the corresponding parquet file contains every column with correct dtype
and acceptable null rates.

GATE: This MUST pass before any upload to HuggingFace or S3 release prefix.

Usage:
    python validate_croissant.py --croissant ../../../evidence/specifications/croissant.json \
                                 --parquet /tmp/georsct_table.parquet

    # Or validate all files declared in the manifest:
    python validate_croissant.py --croissant ../../../evidence/specifications/croissant.json \
                                 --parquet-dir /tmp/

Exit code 0 = all checks pass. Exit code 1 = violations found.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# Maximum allowed null fraction per column (0.05 = 5%)
# Columns with known higher null rates can be exempted below
MAX_NULL_FRACTION = 0.05

# Columns with known higher null rates (with their max allowed fraction)
KNOWN_NULL_EXEMPTIONS = {
    "svi_socioeconomic": 0.01,
    "svi_household_disability": 0.01,
    "svi_minority_language": 0.01,
    "svi_housing_transport": 0.01,
    "svi_overall": 0.01,
    "target_home_value": 0.02,
    "target_income": 0.02,
    # ACS structural nulls: suppressed by Census for small ZCTAs
    "acs_median_home_value": 0.08,
    "acs_median_rent": 0.20,
    # CDC targets: ~260/31789 = 0.82%
    **{f"target_{t}": 0.01 for t in [
        "annual_checkup", "arthritis", "asthma", "binge_drinking",
        "bp_medicated", "cancer", "cholesterol_screening",
        "chronic_kidney_disease", "copd", "coronary_heart_disease",
        "dental_visit", "diabetes", "high_blood_pressure",
        "high_cholesterol", "mental_health_not_good", "obesity",
        "physical_health_not_good", "physical_inactivity",
        "sleep_less_7hr", "smoking", "stroke",
    ]},
}

# Croissant dataType -> pandas dtype families
DTYPE_MAP = {
    "sc:Float": {"float64", "float32", "Float64"},
    "sc:Integer": {"int64", "int32", "Int64", "float64"},  # float64 if nulls
    "sc:Text": {"object", "string", "category", "geometry", "geoarrow.wkb"},
    "sc:Boolean": {"bool", "boolean", "object"},
    "sc:GeoShape": {"geometry", "object", "geoarrow.wkb"},
}


def extract_manifest_columns(croissant: dict) -> dict:
    """Extract {recordset_name: [{"name": col, "dataType": type}, ...]} from Croissant."""
    result = {}
    for rs in croissant.get("recordSet", []):
        rs_name = rs.get("name", rs.get("@id", "unknown"))
        fields = []
        for f in rs.get("field", []):
            name = f.get("name")
            dtype = f.get("dataType", "unknown")
            source = f.get("source", {})
            file_ref = source.get("fileObject", {}).get("@id", "")
            fields.append({
                "name": name,
                "dataType": dtype,
                "file_ref": file_ref,
            })
        result[rs_name] = fields
    return result


def validate_parquet_against_manifest(
    parquet_path: str,
    manifest_fields: list,
    strict: bool = True,
) -> list:
    """Validate a parquet file against manifest field list.

    Returns list of violation dicts. Empty list = all good.
    """
    violations = []
    df = pd.read_parquet(parquet_path)
    parquet_cols = set(df.columns)
    n_rows = len(df)

    log.info("Validating %s: %d rows x %d columns",
             parquet_path, n_rows, len(parquet_cols))

    manifest_col_names = {f["name"] for f in manifest_fields}

    # Check 1: Missing columns (in manifest but not in parquet)
    for field in manifest_fields:
        col = field["name"]
        if col not in parquet_cols:
            violations.append({
                "type": "MISSING_COLUMN",
                "severity": "CRITICAL",
                "column": col,
                "expected_type": field["dataType"],
                "message": f"Column '{col}' declared in Croissant but missing from parquet",
            })
            continue

        # Check 2: Data type compatibility
        actual_dtype = str(df[col].dtype)
        expected_types = DTYPE_MAP.get(field["dataType"], set())
        if expected_types and actual_dtype not in expected_types:
            violations.append({
                "type": "DTYPE_MISMATCH",
                "severity": "WARNING",
                "column": col,
                "expected_type": field["dataType"],
                "actual_dtype": actual_dtype,
                "message": (f"Column '{col}' type mismatch: "
                            f"Croissant={field['dataType']}, "
                            f"parquet={actual_dtype}"),
            })

        # Check 3: Null rate
        null_count = int(df[col].isna().sum())
        null_frac = null_count / n_rows if n_rows > 0 else 0
        max_allowed = KNOWN_NULL_EXEMPTIONS.get(col, MAX_NULL_FRACTION)

        if null_frac > max_allowed:
            severity = "CRITICAL" if null_frac > 0.5 else "WARNING"
            violations.append({
                "type": "EXCESSIVE_NULLS",
                "severity": severity,
                "column": col,
                "null_count": null_count,
                "null_fraction": round(null_frac, 4),
                "max_allowed": max_allowed,
                "message": (f"Column '{col}' has {null_count}/{n_rows} nulls "
                            f"({null_frac:.1%} > {max_allowed:.1%} allowed)"),
            })

    # Check 4: Extra columns (in parquet but not in manifest) -- informational
    extra_cols = parquet_cols - manifest_col_names
    if extra_cols:
        for col in sorted(extra_cols):
            violations.append({
                "type": "EXTRA_COLUMN",
                "severity": "INFO",
                "column": col,
                "message": f"Column '{col}' in parquet but not in Croissant manifest",
            })

    return violations


def main():
    parser = argparse.ArgumentParser(
        description="Validate parquet files against Croissant manifest"
    )
    parser.add_argument("--croissant", required=True,
                        help="Path to croissant.json")
    parser.add_argument("--parquet", default=None,
                        help="Path to a specific parquet file to validate")
    parser.add_argument("--parquet-dir", default=None,
                        help="Directory containing parquet files to validate")
    parser.add_argument("--record-set", default="georsct-main",
                        help="Which recordSet to validate against (default: georsct-main)")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on warnings too (not just critical)")
    args = parser.parse_args()

    # Load Croissant
    with open(args.croissant) as f:
        croissant = json.load(f)

    manifest = extract_manifest_columns(croissant)
    log.info("Croissant recordSets: %s", list(manifest.keys()))

    if args.record_set not in manifest:
        log.error("RecordSet '%s' not found in Croissant. Available: %s",
                  args.record_set, list(manifest.keys()))
        sys.exit(1)

    fields = manifest[args.record_set]
    log.info("Manifest '%s': %d fields declared", args.record_set, len(fields))
    for f in fields:
        log.info("  %-40s %s", f["name"], f["dataType"])

    # Determine parquet file(s) to validate
    parquet_files = []
    if args.parquet:
        parquet_files.append(args.parquet)
    elif args.parquet_dir:
        pdir = Path(args.parquet_dir)
        parquet_files = [str(p) for p in pdir.glob("*.parquet")]
    else:
        log.error("Specify --parquet or --parquet-dir")
        sys.exit(1)

    if not parquet_files:
        log.error("No parquet files found")
        sys.exit(1)

    # Validate each file
    all_violations = []
    for pf in parquet_files:
        log.info("")
        log.info("=" * 60)
        violations = validate_parquet_against_manifest(pf, fields, args.strict)
        all_violations.extend(violations)

        if not violations:
            log.info("PASS: %s matches Croissant manifest", pf)
        else:
            for v in violations:
                level = {"CRITICAL": logging.ERROR,
                         "WARNING": logging.WARNING,
                         "INFO": logging.INFO}.get(v["severity"], logging.INFO)
                log.log(level, "[%s] %s", v["severity"], v["message"])

    # Summary
    log.info("")
    log.info("=" * 60)
    critical = [v for v in all_violations if v["severity"] == "CRITICAL"]
    warnings = [v for v in all_violations if v["severity"] == "WARNING"]
    infos = [v for v in all_violations if v["severity"] == "INFO"]

    log.info("SUMMARY: %d critical, %d warnings, %d info",
             len(critical), len(warnings), len(infos))

    if critical:
        log.error("FAIL: %d critical violations -- upload BLOCKED", len(critical))
        for v in critical:
            log.error("  %s: %s", v["column"], v["message"])
        sys.exit(1)
    elif warnings and args.strict:
        log.error("FAIL (strict mode): %d warnings", len(warnings))
        sys.exit(1)
    else:
        log.info("PASS: Parquet matches Croissant manifest")
        sys.exit(0)


if __name__ == "__main__":
    main()
