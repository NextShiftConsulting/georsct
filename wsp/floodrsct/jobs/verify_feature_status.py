#!/usr/bin/env python3
"""verify_feature_status.py -- Derive feature lifecycle state from artifacts.

Checks S3 caches, code presence, contract entries, and consumer wiring
to determine each feature's lifecycle stage:

  proposed    -> no contract entry
  contracted  -> FEATURE_CONTRACT.yaml entry exists
  implemented -> job script + launcher exist (extracted type only)
  extracted   -> S3 cache parquet exists with data
  validated   -> coverage > min_coverage_pct
  integrated  -> consumer function wired into scenario assemblers

Usage:
    python verify_feature_status.py --all
    python verify_feature_status.py --feature buildings
    python verify_feature_status.py --all --json
    python verify_feature_status.py --next-steps
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

SERIES_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = SERIES_DIR / "jobs"
SCRIPTS_DIR = SERIES_DIR / "scripts"
REGISTRY_PATH = SERIES_DIR / "FEATURE_REGISTRY.yaml"
CONTRACT_PATH = SERIES_DIR / "FEATURE_CONTRACT.yaml"
DATASET_SCRIPT = JOBS_DIR / "build_event_dataset.py"

SCENARIOS = ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"]

BUCKET = "swarm-floodrsct-data"


def _get_s3():
    """Get S3 client via swarm_auth."""
    try:
        from swarm_auth import get_aws_credentials
        import boto3
        return boto3.client("s3", region_name="us-east-1", **get_aws_credentials())
    except ImportError:
        import boto3
        return boto3.client("s3", region_name="us-east-1")


def load_registry() -> list[dict]:
    """Load feature registry."""
    with open(REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("features", [])


def load_contract_functions() -> set[str]:
    """Extract all build_function values from FEATURE_CONTRACT.yaml."""
    if not CONTRACT_PATH.exists():
        return set()
    with open(CONTRACT_PATH, encoding="utf-8", errors="replace") as f:
        text = f.read()
    fns = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("build_function:"):
            val = line.split(":", 1)[1].strip()
            if val and val != "null":
                fns.add(val)
    return fns


def load_dataset_script_text() -> str:
    """Read build_event_dataset.py for consumer checks."""
    if DATASET_SCRIPT.exists():
        return DATASET_SCRIPT.read_text(encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# Lifecycle checks
# ---------------------------------------------------------------------------

def check_contracted(feat: dict, contract_fns: set[str]) -> tuple[str, str]:
    """Check if feature has FEATURE_CONTRACT.yaml entry."""
    fn = feat.get("build_function", "")
    if fn in contract_fns:
        return "PASS", f"build_function={fn} in contract"
    return "FAIL", f"build_function={fn} NOT in contract"


def check_implemented(feat: dict) -> tuple[str, str]:
    """Check if job script + launcher exist (extracted type only)."""
    if feat.get("type") != "extracted":
        return "SKIP", "derived feature (no dedicated job)"

    job = feat.get("job_script")
    launcher = feat.get("launcher_script")

    if not job and not launcher:
        # Inline extraction (impervious, cropland, etc.)
        return "SKIP", "inline extraction (no dedicated job script)"

    issues = []
    if job and not (JOBS_DIR / job).exists():
        issues.append(f"job_script {job} missing")
    if launcher and not (SCRIPTS_DIR / launcher).exists():
        issues.append(f"launcher_script {launcher} missing")

    if issues:
        return "FAIL", "; ".join(issues)
    return "PASS", f"job={job}, launcher={launcher}"


def check_extracted(feat: dict, s3) -> tuple[str, str]:
    """Check if S3 cache parquet exists and has rows."""
    cache_key = feat.get("cache_key")
    if not cache_key:
        return "SKIP", "no cache_key (derived inline)"

    try:
        resp = s3.head_object(Bucket=BUCKET, Key=cache_key)
        size_kb = resp["ContentLength"] / 1024
        if size_kb < 1:
            return "WARN", f"cache exists but tiny ({size_kb:.0f} KB)"
        return "PASS", f"cache exists ({size_kb:.0f} KB)"
    except s3.exceptions.ClientError:
        return "FAIL", f"s3://{BUCKET}/{cache_key} does not exist"


def check_validated(feat: dict, s3) -> tuple[str, str]:
    """Check coverage by reading cache parquet row count vs expected."""
    cache_key = feat.get("cache_key")
    if not cache_key:
        return "SKIP", "no cache_key"

    try:
        import pandas as pd
        obj = s3.get_object(Bucket=BUCKET, Key=cache_key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

        if df.empty:
            return "FAIL", "cache parquet is empty"

        n_rows = len(df)
        cols = feat.get("output_columns", [])
        coverage = {}
        for col in cols:
            if col in df.columns:
                pct = df[col].notna().mean() * 100
                coverage[col] = round(pct, 1)
            else:
                coverage[col] = 0.0

        min_req = feat.get("min_coverage_pct", 80)
        min_actual = min(coverage.values()) if coverage else 0.0

        detail = f"{n_rows} rows, min_coverage={min_actual:.0f}% (req={min_req}%)"
        if min_actual >= min_req:
            return "PASS", detail
        elif min_actual > 0:
            return "WARN", detail
        else:
            return "FAIL", detail

    except s3.exceptions.ClientError:
        return "FAIL", "cache not readable"
    except Exception as e:
        return "WARN", f"could not validate: {e}"


def check_integrated(feat: dict, ds_text: str) -> tuple[str, str]:
    """Check if consumer function exists and is wired into assemblers."""
    fn = feat.get("build_function", "")
    if not fn:
        return "FAIL", "no build_function defined"

    # Check function definition exists
    def_pattern = f"def {fn}("
    if def_pattern not in ds_text:
        return "FAIL", f"{fn}() not defined in build_event_dataset.py"

    # Check wiring into scenario assemblers
    # Look for function calls like: build_building_features(s3,
    call_pattern = f"{fn}(s3,"
    call_count = ds_text.count(call_pattern)

    scenarios = feat.get("scenarios", [])
    if scenarios == "all":
        expected = 5
    else:
        expected = len(scenarios)

    if call_count >= expected:
        return "PASS", f"{fn}() called {call_count}x (need {expected})"
    elif call_count > 0:
        return "WARN", f"{fn}() called {call_count}x but need {expected}"
    else:
        return "FAIL", f"{fn}() defined but never called in assemblers"


# ---------------------------------------------------------------------------
# Lifecycle stage derivation
# ---------------------------------------------------------------------------

STAGES = ["contracted", "implemented", "extracted", "validated", "integrated"]


def derive_stage(checks: dict[str, tuple[str, str]]) -> str:
    """Derive highest completed lifecycle stage from check results."""
    # Walk stages in reverse; highest passing stage wins
    for stage in reversed(STAGES):
        status, _ = checks.get(stage, ("SKIP", ""))
        if status == "PASS":
            return stage
    # If contracted passes but nothing else
    if checks.get("contracted", ("FAIL", ""))[0] == "PASS":
        return "contracted"
    return "proposed"


def verify_feature(feat: dict, s3, contract_fns: set[str],
                   ds_text: str, check_s3: bool = True) -> dict:
    """Run all lifecycle checks for one feature."""
    fid = feat["feature_id"]

    checks = {}
    checks["contracted"] = check_contracted(feat, contract_fns)
    checks["implemented"] = check_implemented(feat)

    if check_s3:
        checks["extracted"] = check_extracted(feat, s3)
        checks["validated"] = check_validated(feat, s3)
    else:
        checks["extracted"] = ("SKIP", "S3 check disabled")
        checks["validated"] = ("SKIP", "S3 check disabled")

    checks["integrated"] = check_integrated(feat, ds_text)

    stage = derive_stage(checks)

    return {
        "feature_id": fid,
        "title": feat.get("title", ""),
        "type": feat.get("type", ""),
        "stage": stage,
        "checks": {k: {"status": v[0], "detail": v[1]} for k, v in checks.items()},
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

STAGE_ICONS = {
    "proposed": "[ ]",
    "contracted": "[C]",
    "implemented": "[I]",
    "extracted": "[E]",
    "validated": "[V]",
    "integrated": "[*]",
}

STATUS_MARKERS = {
    "PASS": "  OK",
    "FAIL": "FAIL",
    "WARN": "WARN",
    "SKIP": "SKIP",
}


def print_summary(results: list[dict]) -> None:
    """Print human-readable summary table."""
    print("\n=== FEATURE LIFECYCLE STATUS ===\n")
    print(f"{'Feature':<20} {'Type':<10} {'Stage':<14} {'Contracted':<12} "
          f"{'Implemented':<13} {'Extracted':<11} {'Validated':<11} {'Integrated':<12}")
    print("-" * 110)

    for r in results:
        fid = r["feature_id"]
        ftype = r["type"]
        stage = f"{STAGE_ICONS.get(r['stage'], '[ ]')} {r['stage']}"
        cols = []
        for s in STAGES:
            c = r["checks"].get(s, {})
            cols.append(STATUS_MARKERS.get(c.get("status", "SKIP"), "    "))

        print(f"{fid:<20} {ftype:<10} {stage:<14} "
              f"{cols[0]:<12} {cols[1]:<13} {cols[2]:<11} {cols[3]:<11} {cols[4]:<12}")

    # Summary counts
    print()
    stage_counts = {}
    for r in results:
        s = r["stage"]
        stage_counts[s] = stage_counts.get(s, 0) + 1

    total = len(results)
    for stage_name in ["integrated", "validated", "extracted", "implemented", "contracted", "proposed"]:
        count = stage_counts.get(stage_name, 0)
        if count > 0:
            print(f"  {stage_name}: {count}/{total}")


def print_next_steps(results: list[dict]) -> None:
    """Print actionable next steps for features not yet fully integrated."""
    print("\n=== NEXT STEPS ===\n")

    for r in results:
        stage = r["stage"]
        if stage == "integrated":
            continue

        fid = r["feature_id"]
        checks = r["checks"]

        blockers = []
        for s in STAGES:
            c = checks.get(s, {})
            if c.get("status") == "FAIL":
                blockers.append(f"  - {s}: {c['detail']}")

        if blockers:
            print(f"{fid} (current: {stage}):")
            for b in blockers:
                print(b)
            print()


def print_details(results: list[dict]) -> None:
    """Print detailed check results."""
    for r in results:
        print(f"\n--- {r['feature_id']} ({r['type']}) ---")
        print(f"  Stage: {r['stage']}")
        for s in STAGES:
            c = r["checks"].get(s, {})
            marker = STATUS_MARKERS.get(c.get("status", "SKIP"), "    ")
            print(f"  {s:<14} {marker}  {c.get('detail', '')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive feature lifecycle state from S3 artifacts and code presence")
    parser.add_argument("--feature", help="Check single feature by ID")
    parser.add_argument("--all", action="store_true", help="Check all features")
    parser.add_argument("--next-steps", action="store_true",
                        help="Show actionable blockers for incomplete features")
    parser.add_argument("--details", action="store_true",
                        help="Show detailed check results per feature")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-s3", action="store_true",
                        help="Skip S3 checks (offline mode)")
    args = parser.parse_args()

    if not args.feature and not args.all and not args.next_steps:
        parser.error("Specify --feature <id>, --all, or --next-steps")

    features = load_registry()
    contract_fns = load_contract_functions()
    ds_text = load_dataset_script_text()

    s3 = None if args.no_s3 else _get_s3()

    if args.feature:
        targets = [f for f in features if f["feature_id"] == args.feature]
        if not targets:
            print(f"Feature '{args.feature}' not in registry. "
                  f"Known: {[f['feature_id'] for f in features]}")
            sys.exit(1)
    else:
        targets = features

    results = []
    for feat in targets:
        r = verify_feature(feat, s3, contract_fns, ds_text,
                           check_s3=not args.no_s3)
        results.append(r)

    if args.json:
        print(json.dumps(results, indent=2))
    elif args.next_steps:
        print_summary(results)
        print_next_steps(results)
    elif args.details:
        print_details(results)
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
