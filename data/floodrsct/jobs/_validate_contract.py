"""
_validate_contract.py -- Contract-driven three-layer validation for FloodRSCT.

Reads FEATURE_CONTRACT.yaml and validates:
  Layer 1: Interface contract (raw data exists, schemas match)
  Layer 2: Post-assembly (coverage thresholds, plausibility, dedup)
  Layer 3: Data lock (full reconciliation against contract)

Importable by build_event_dataset.py (Layers 1+2) and by the standalone
validate_data_lock_a.py runner (Layer 3).

Usage as standalone:
    python _validate_contract.py --scenario houston
    python _validate_contract.py --all
"""

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# Silence botocore credential chain chatter
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

BUCKET = "swarm-floodrsct-data"
CONTRACT_PATH = Path(__file__).parent.parent / "FEATURE_CONTRACT.yaml"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class ValidationResult:
    feature: str
    layer: int
    status: Status
    message: str
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[L{self.layer}] [{self.status.value:4s}] {self.feature}: {self.message}"


# ---------------------------------------------------------------------------
# Contract loader
# ---------------------------------------------------------------------------

def load_contract(path: Optional[Path] = None) -> list[dict]:
    """Load FEATURE_CONTRACT.yaml and return the features list."""
    p = path or CONTRACT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Contract not found: {p}")
    with open(p) as f:
        doc = yaml.safe_load(f)
    features = doc.get("features", [])
    log.info("Loaded contract: %d features from %s", len(features), p.name)
    return features


def features_for_scenario(contract: list[dict], scenario: str) -> list[dict]:
    """Filter contract to features relevant to a scenario."""
    result = []
    for feat in contract:
        scenarios = feat.get("scenarios", [])
        if scenarios is None:
            # Global feature (e.g., hurdat2 applies to all)
            result.append(feat)
        elif scenario in scenarios:
            result.append(feat)
    return result


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _list_s3_keys(s3, prefix: str) -> list[dict]:
    """List objects under prefix. Returns [{Key, Size}, ...]."""
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append({"Key": obj["Key"], "Size": obj["Size"]})
    return objects


def _s3_prefix_has_files(s3, prefix: str) -> tuple[bool, int, int]:
    """Check if prefix has files. Returns (exists, file_count, total_bytes)."""
    objects = _list_s3_keys(s3, prefix)
    total = sum(o["Size"] for o in objects)
    return len(objects) > 0, len(objects), total


def _read_parquet_schema(s3, key: str) -> Optional[list[str]]:
    """Read just the column names from a parquet file on S3."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        return list(df.columns)
    except Exception:
        return None


def _read_parquet_df(s3, key: str) -> Optional[pd.DataFrame]:
    """Read a full parquet from S3."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Layer 1: Interface Contract Validation (pre-assembly gate)
# ---------------------------------------------------------------------------

# Known column-name expectations per build function.
# Maps build_function -> list of columns it tries to read from raw data.
EXPECTED_RAW_COLUMNS = {
    "aggregate_tides": {
        "file_pattern": "raw/noaa_tides/",
        "required_columns": ["observed_m", "predicted_m"],
        "note": "Builder historically expected water_level_m; fetcher writes observed_m",
    },
    "compute_storm_proximity": {
        "file_pattern": "raw/hurdat2/storm_tracks.parquet",
        "required_columns": ["storm_id", "lat", "lon", "max_wind_kt", "timestamp"],
        "note": "Builder expects category column but fetcher writes max_wind_kt + status",
    },
    "aggregate_mrms_rainfall": {
        "file_pattern": "raw/noaa_mrms/",
        "required_columns": [],
        "note": "GRIB2 files; column check N/A. Check file extension (.grib2.gz).",
    },
    "aggregate_hwm": {
        "file_pattern": "raw/surge_estimates/",
        "required_columns": ["latitude", "longitude", "elev_ft"],
        "note": "Also checks raw/usgs_stn/ as fallback path. Builder uses elev_ft directly.",
    },
}


# Scenario -> concrete event names (matches build_event_dataset.py event_map dicts)
# NOTE: build_event_dataset.py uses "ida2021" internally for both NYC and NOLA,
# but MRMS/tides fetchers stored data under "ida2021_nyc". The NYC builder
# remaps "ida2021" -> "ida2021_nyc" for MRMS (line 796). NOLA builder does NOT
# call MRMS at all. This mapping uses the S3 event keys (what the fetchers wrote).
SCENARIO_EVENTS = {
    "houston": ["harvey2017", "imelda2019", "beryl2024"],
    "new_orleans": ["ida2021_nyc"],  # S3 key; builder uses "ida2021" internally
    "nyc": ["ida2021_nyc"],
    "riverside_coachella": ["hilary2023"],
    "southwest_florida": ["ian2022"],
}


def _resolve_raw_paths(raw_path: str, scenario: str) -> list[str]:
    """Expand {event}, {scenario}, and 'or' alternatives into concrete S3 prefixes."""
    # Handle "path_a or path_b" alternatives -- check each branch
    if " or " in raw_path:
        alternatives = [p.strip() for p in raw_path.split(" or ")]
        result = []
        for alt in alternatives:
            result.extend(_resolve_raw_paths(alt, scenario))
        return result

    if "{event}" in raw_path:
        events = SCENARIO_EVENTS.get(scenario, [])
        if not events:
            return [raw_path.replace("{event}", scenario)]
        return [raw_path.replace("{event}", e).replace("{scenario}", scenario)
                for e in events]
    if "{scenario}" in raw_path:
        return [raw_path.replace("{scenario}", scenario)]
    if "{dr_number}" in raw_path:
        # OpenFEMA: just check the prefix up to the variable part
        return [raw_path.split("{")[0]]
    return [raw_path]


def validate_layer1(
    s3, scenario: str, contract: list[dict]
) -> list[ValidationResult]:
    """Pre-assembly gate: verify raw data exists and schemas match.

    Returns list of ValidationResult (one per feature).
    """
    features = features_for_scenario(contract, scenario)
    results = []

    for feat in features:
        name = feat["feature_name"]
        source_type = feat.get("source_type", "")
        raw_path = feat.get("raw_s3_path")
        build_fn = feat.get("build_function")

        # Skip operational features -- no raw data expected
        if source_type == "operational":
            results.append(ValidationResult(
                feature=name, layer=1, status=Status.SKIP,
                message="operational -- no raw data expected",
            ))
            continue

        # Skip features with no raw path (hand_coded, etc.)
        if not raw_path or raw_path == "null":
            results.append(ValidationResult(
                feature=name, layer=1, status=Status.SKIP,
                message="no raw_s3_path in contract",
            ))
            continue

        # Resolve placeholders into concrete S3 prefixes
        resolved_paths = _resolve_raw_paths(raw_path, scenario)
        total_file_count = 0
        total_bytes = 0
        any_exists = False

        for rp in resolved_paths:
            exists, fc, tb = _s3_prefix_has_files(s3, rp)
            if not exists and rp.endswith("/"):
                exists, fc, tb = _s3_prefix_has_files(s3, rp.rstrip("/"))
            if exists:
                any_exists = True
                total_file_count += fc
                total_bytes += tb

        # Check 1: raw data exists on S3
        exists = any_exists
        file_count = total_file_count

        if not exists:
            checked = ", ".join(resolved_paths)
            results.append(ValidationResult(
                feature=name, layer=1, status=Status.FAIL,
                message=f"no files at {checked}",
                details={"checked_paths": resolved_paths},
            ))
            continue

        # Check 2: column names (parquet files only)
        # Try each resolved parquet path until one is readable
        cols = None
        parquet_path = None
        for rp in resolved_paths:
            if rp.endswith(".parquet"):
                cols = _read_parquet_schema(s3, rp)
                if cols is not None:
                    parquet_path = rp
                    break
        if parquet_path is None and raw_path.endswith(".parquet"):
            parquet_path = raw_path  # fallback for error message
        if parquet_path and cols is None and any(rp.endswith(".parquet") for rp in resolved_paths):
            # Parquet expected but none readable -- only warn if data exists via other path
            if file_count > 0:
                pass  # data found via non-parquet path; skip schema check
            else:
                results.append(ValidationResult(
                    feature=name, layer=1, status=Status.WARN,
                    message=f"could not read schema from any parquet path",
                    details={"file_count": file_count, "total_bytes": total_bytes},
                ))
                continue
        if cols is not None:

            # Check expected columns if we know them
            if build_fn and build_fn in EXPECTED_RAW_COLUMNS:
                spec = EXPECTED_RAW_COLUMNS[build_fn]
                missing = [c for c in spec["required_columns"] if c not in cols]
                if missing:
                    results.append(ValidationResult(
                        feature=name, layer=1, status=Status.FAIL,
                        message=f"missing columns {missing} in {parquet_path}",
                        details={
                            "expected": spec["required_columns"],
                            "actual": cols,
                            "note": spec.get("note", ""),
                        },
                    ))
                    continue

        # Check 3: file size guard (catch HTML stubs, placeholders, empty files)
        if file_count > 0:
            all_objects = []
            for rp in resolved_paths:
                all_objects.extend(_list_s3_keys(s3, rp))
            tiny = [o for o in all_objects if o["Size"] < 200]
            real = [o for o in all_objects if o["Size"] >= 200]
            if tiny and real:
                results.append(ValidationResult(
                    feature=name, layer=1, status=Status.WARN,
                    message=f"{len(tiny)} file(s) < 200 bytes (possible stubs)",
                    details={"tiny_files": [t["Key"] for t in tiny[:5]]},
                ))
                # Don't skip -- real data files exist alongside stubs
            elif tiny and not real:
                results.append(ValidationResult(
                    feature=name, layer=1, status=Status.FAIL,
                    message=f"all {len(tiny)} file(s) < 200 bytes (placeholders only)",
                    details={"tiny_files": [t["Key"] for t in tiny[:5]]},
                ))
                continue

        results.append(ValidationResult(
            feature=name, layer=1, status=Status.PASS,
            message=f"{file_count} files, {total_bytes / 1e6:.1f} MB",
            details={"file_count": file_count, "total_bytes": total_bytes},
        ))

    return results


# ---------------------------------------------------------------------------
# Layer 2: Post-Assembly Validation (inline checks for builder)
# ---------------------------------------------------------------------------

# Coverage thresholds: minimum non-null fraction per output column
COVERAGE_THRESHOLDS = {
    # Event-window features
    "rainfall_total_mm": 0.50,
    "max_rainfall_mm": 0.50,
    "total_rainfall_mm": 0.50,
    "tidal_surge_max_m": 0.20,
    "storm_distance_km": 0.90,
    "storm_min_dist_km": 0.90,
    # Post-event labels (sparse by nature)
    "hwm_max_ft": 0.05,
    "flood_311_count": 0.05,
    "nfip_event_claims": 0.10,
    # Static features (should be complete)
    "impervious_pct": 0.80,
    "elevation_m_msl": 0.90,
    "elevation_mean_m": 0.90,
}

# Plausibility bounds: (min, max) for physical quantities
PLAUSIBILITY_BOUNDS = {
    "rainfall_total_mm": (0, 3000),       # Harvey max ~1500mm, generous upper
    "max_rainfall_mm": (0, 500),           # single-hour max
    "tidal_surge_max_m": (-2, 10),         # negative = below predicted
    "storm_distance_km": (0, 5000),
    "storm_min_dist_km": (0, 5000),
    "elevation_m_msl": (-100, 5000),       # NOLA can be below sea level
    "elevation_mean_m": (-100, 5000),
    "impervious_pct": (0, 100),
    "hwm_max_ft": (0, 50),
    "slosh_max_surge_m": (0, 15),
}


def validate_layer2(
    df: pd.DataFrame, scenario: str
) -> list[ValidationResult]:
    """Post-assembly validation. Call after build_{scenario}() returns.

    Checks coverage thresholds, plausibility, and dedup.
    Returns list of ValidationResult.
    """
    results = []

    # Check 1: duplicate (zcta_id, event) rows
    if "zcta_id" in df.columns and "event" in df.columns:
        dup_count = df.duplicated(subset=["zcta_id", "event"]).sum()
        if dup_count > 0:
            results.append(ValidationResult(
                feature="(zcta_id, event) uniqueness", layer=2,
                status=Status.FAIL,
                message=f"{dup_count} duplicate rows",
            ))
        else:
            results.append(ValidationResult(
                feature="(zcta_id, event) uniqueness", layer=2,
                status=Status.PASS,
                message=f"{len(df)} rows, all unique",
            ))

    # Check 2: coverage thresholds
    for col, threshold in COVERAGE_THRESHOLDS.items():
        if col not in df.columns:
            continue
        non_null = df[col].notna().mean()
        if non_null < threshold:
            results.append(ValidationResult(
                feature=col, layer=2, status=Status.FAIL,
                message=f"non-null {non_null:.1%} < threshold {threshold:.0%}",
                details={"non_null_rate": non_null, "threshold": threshold},
            ))
        else:
            results.append(ValidationResult(
                feature=col, layer=2, status=Status.PASS,
                message=f"non-null {non_null:.1%} >= {threshold:.0%}",
            ))

    # Check 3: plausibility bounds
    for col, (lo, hi) in PLAUSIBILITY_BOUNDS.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        below = (vals < lo).sum()
        above = (vals > hi).sum()
        if below > 0 or above > 0:
            results.append(ValidationResult(
                feature=col, layer=2, status=Status.WARN,
                message=f"{below} below {lo}, {above} above {hi}",
                details={
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "bounds": (lo, hi),
                },
            ))
        else:
            results.append(ValidationResult(
                feature=col, layer=2, status=Status.PASS,
                message=f"all values in [{lo}, {hi}]",
                details={"min": float(vals.min()), "max": float(vals.max())},
            ))

    # Check 4: row count sanity
    if len(df) == 0:
        results.append(ValidationResult(
            feature="row_count", layer=2, status=Status.FAIL,
            message="empty dataframe",
        ))
    elif len(df) < 10:
        results.append(ValidationResult(
            feature="row_count", layer=2, status=Status.WARN,
            message=f"only {len(df)} rows -- suspiciously low",
        ))
    else:
        results.append(ValidationResult(
            feature="row_count", layer=2, status=Status.PASS,
            message=f"{len(df)} rows",
        ))

    return results


# ---------------------------------------------------------------------------
# Layer 3: Data Lock Validation (standalone reconciliation)
# ---------------------------------------------------------------------------

OUTPUT_KEYS = {
    "houston": "processed/houston/houston_event_features.parquet",
    "new_orleans": "processed/new_orleans/no_event_features.parquet",
    "nyc": "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
}


def validate_layer3(
    s3, scenario: str, contract: list[dict]
) -> list[ValidationResult]:
    """Data lock validation: reconcile output parquet against contract.

    Checks:
    - Every feature in contract has a column in output
    - temporal_class boundary (no post_event/operational as inputs)
    - Schema completeness
    """
    results = []
    output_key = OUTPUT_KEYS.get(scenario)
    if not output_key:
        results.append(ValidationResult(
            feature="output_parquet", layer=3, status=Status.FAIL,
            message=f"unknown scenario: {scenario}",
        ))
        return results

    df = _read_parquet_df(s3, output_key)
    if df is None:
        results.append(ValidationResult(
            feature="output_parquet", layer=3, status=Status.FAIL,
            message=f"not found: s3://{BUCKET}/{output_key}",
        ))
        return results

    output_cols = set(df.columns)
    results.append(ValidationResult(
        feature="output_parquet", layer=3, status=Status.PASS,
        message=f"{len(df)} rows x {len(df.columns)} columns",
    ))

    # Check each feature in contract
    features = features_for_scenario(contract, scenario)
    for feat in features:
        name = feat["feature_name"]
        output_col = feat.get("output_column", name)
        source_type = feat.get("source_type", "")
        temporal_class = feat.get("temporal_class", "")

        if source_type == "operational":
            results.append(ValidationResult(
                feature=name, layer=3, status=Status.SKIP,
                message="operational -- not expected in output",
            ))
            continue

        if output_col not in output_cols:
            results.append(ValidationResult(
                feature=name, layer=3, status=Status.FAIL,
                message=f"column '{output_col}' missing from output",
            ))
            continue

        # Column exists -- check non-null rate
        non_null = df[output_col].notna().mean()
        if non_null == 0:
            results.append(ValidationResult(
                feature=name, layer=3, status=Status.FAIL,
                message=f"column '{output_col}' is 100% null",
                details={"temporal_class": temporal_class},
            ))
        elif non_null < 0.01 and temporal_class not in ("post_event",):
            results.append(ValidationResult(
                feature=name, layer=3, status=Status.WARN,
                message=f"column '{output_col}' is {non_null:.1%} non-null",
                details={"temporal_class": temporal_class},
            ))
        else:
            results.append(ValidationResult(
                feature=name, layer=3, status=Status.PASS,
                message=f"'{output_col}' {non_null:.1%} non-null",
            ))

    # Leakage check: flag if post_event or operational columns are used as
    # model input features (they should only be labels/outcomes)
    post_event_cols = [
        f["output_column"] for f in features
        if f.get("temporal_class") in ("post_event", "operational")
        and f.get("output_column") in output_cols
    ]
    if post_event_cols:
        results.append(ValidationResult(
            feature="leakage_gate", layer=3, status=Status.WARN,
            message=f"post_event/operational columns present: {post_event_cols}. "
                    "Ensure these are used as labels only, not model inputs.",
        ))

    # Run Layer 2 checks on the output
    l2_results = validate_layer2(df, scenario)
    results.extend(l2_results)

    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    results: list[ValidationResult],
    scenario: str = "",
) -> tuple[int, int, int]:
    """Print a formatted validation report. Returns (pass, fail, warn) counts."""
    pass_n = sum(1 for r in results if r.status == Status.PASS)
    fail_n = sum(1 for r in results if r.status == Status.FAIL)
    warn_n = sum(1 for r in results if r.status == Status.WARN)
    skip_n = sum(1 for r in results if r.status == Status.SKIP)

    header = f"Validation Report: {scenario}" if scenario else "Validation Report"
    print(f"\n{'=' * 72}")
    print(header)
    print(f"{'=' * 72}")

    for layer in (1, 2, 3):
        layer_results = [r for r in results if r.layer == layer]
        if not layer_results:
            continue
        print(f"\n--- Layer {layer} ---")
        for r in layer_results:
            print(f"  {r}")

    print(f"\n{'=' * 72}")
    print(f"PASS: {pass_n}  FAIL: {fail_n}  WARN: {warn_n}  SKIP: {skip_n}")
    if fail_n > 0:
        print("VERDICT: BLOCKED -- fix FAILs before proceeding")
    elif warn_n > 0:
        print("VERDICT: CONDITIONAL PASS -- review WARNs")
    else:
        print("VERDICT: CLEAR")
    print(f"{'=' * 72}\n")

    return pass_n, fail_n, warn_n


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Contract-driven validation for FloodRSCT"
    )
    parser.add_argument(
        "--scenario",
        choices=list(OUTPUT_KEYS.keys()),
        help="Validate a single scenario",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Validate all scenarios",
    )
    parser.add_argument(
        "--layer", type=int, choices=[1, 2, 3],
        help="Run only a specific layer (default: all available)",
    )
    parser.add_argument(
        "--contract", type=str, default=None,
        help="Path to FEATURE_CONTRACT.yaml (default: auto-detect)",
    )
    args = parser.parse_args()

    if not args.scenario and not args.all:
        parser.error("specify --scenario or --all")

    contract_path = Path(args.contract) if args.contract else None
    contract = load_contract(contract_path)

    from swarm_auth import get_aws_credentials
    aws = get_aws_credentials()
    aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **aws)

    scenarios = list(OUTPUT_KEYS.keys()) if args.all else [args.scenario]
    total_fails = 0

    for scenario in scenarios:
        all_results = []

        # Layer 1
        if args.layer is None or args.layer == 1:
            log.info("Running Layer 1 for %s ...", scenario)
            l1 = validate_layer1(s3, scenario, contract)
            all_results.extend(l1)

        # Layer 3 (includes Layer 2 inline)
        if args.layer is None or args.layer >= 2:
            log.info("Running Layer 3 for %s ...", scenario)
            l3 = validate_layer3(s3, scenario, contract)
            all_results.extend(l3)

        _, fails, _ = print_report(all_results, scenario)
        total_fails += fails

    sys.exit(1 if total_fails > 0 else 0)


if __name__ == "__main__":
    main()
