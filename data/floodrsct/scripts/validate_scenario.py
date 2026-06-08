#!/usr/bin/env python3
"""
validate_scenario.py -- Standardized scenario validation for FloodRSCT.

Runs 7 checks against an assembled event-features parquet and its
supporting data layers.  Designed to be the single validation entry
point that all scenario SMEs call from their ``validate`` command.

Usage:
    python validate_scenario.py --scenario houston
    python validate_scenario.py --scenario all
    python validate_scenario.py --scenario houston --upload
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET = "swarm-floodrsct-data"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
CONFIG_DIR = REPO_DIR / "configs"
RESULTS_DIR = SCRIPT_DIR / "results" / "s035"

ALL_SCENARIOS = [
    "houston",
    "southwest_florida",
    "nyc",
    "new_orleans",
    "riverside_coachella",
]

# Scenario name -> S3 parquet key
PARQUET_KEYS: dict[str, str] = {
    "houston": "processed/houston/houston_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
    "nyc": "processed/nyc/nyc_event_features.parquet",
    "new_orleans": "processed/new_orleans/no_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
}

CROSSWALK_KEY = "raw/geocertdb2026/zcta_county_crosswalk.parquet"
ADJACENCY_KEY = "raw/geocertdb2026/zcta_adjacency.parquet"

NULL_RATE_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _get_s3_client():
    """Create S3 client using P18 credential triage."""
    import boto3
    from swarm_auth import get_aws_credentials

    return boto3.client("s3", **get_aws_credentials())


def _s3_read(s3, key: str) -> Optional[pd.DataFrame]:
    """Read a parquet from S3. Returns None if missing."""
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        log.warning("S3 key not found: %s", key)
        return None
    except Exception as exc:
        log.warning("Could not read %s: %s", key, exc)
        return None


def _s3_upload(s3, data: bytes, key: str) -> None:
    """Upload bytes to S3."""
    s3.put_object(Bucket=BUCKET, Key=key, Body=data)
    log.info("Uploaded to s3://%s/%s", BUCKET, key)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(scenario: str) -> dict:
    """Load and return scenario YAML config.

    Args:
        scenario: Scenario name matching a file in configs/.

    Returns:
        Parsed config dict.
    """
    path = CONFIG_DIR / f"{scenario}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


def _extract_fips(cfg: dict) -> list[str]:
    """Extract county FIPS codes from a scenario config.

    Handles the three config shapes: ``county_fips_list``,
    ``county_fips``, and ``borough_counties`` (NYC).

    Args:
        cfg: Parsed scenario config.

    Returns:
        Sorted list of 5-digit FIPS strings.
    """
    if "county_fips_list" in cfg:
        return sorted(cfg["county_fips_list"])
    if "county_fips" in cfg:
        raw = cfg["county_fips"]
        if isinstance(raw, list):
            return sorted(raw)
        return [raw]
    if "borough_counties" in cfg:
        return sorted(cfg["borough_counties"].values())
    raise KeyError("No county FIPS field found in config")


def _extract_required_columns(cfg: dict) -> list[str]:
    """Return the mvd.required_columns list from config."""
    mvd = cfg.get("mvd", {})
    return mvd.get("required_columns", [])


def _extract_min_zctas(cfg: dict) -> int:
    """Return mvd.min_zctas (or min_sewersheds for NYC)."""
    mvd = cfg.get("mvd", {})
    return mvd.get("min_zctas", mvd.get("min_sewersheds", 0))


def _extract_event_names(cfg: dict) -> list[str]:
    """Return the list of event keys from config."""
    return sorted(cfg.get("events", {}).keys())


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_fips_match(
    df: pd.DataFrame,
    crosswalk: pd.DataFrame,
    expected_fips: list[str],
) -> dict[str, Any]:
    """Check 1: FIPS match between config and assembled data.

    Args:
        df: Event-features dataframe (must have ``zcta_id``).
        crosswalk: ZCTA-to-county crosswalk.
        expected_fips: FIPS from scenario config.

    Returns:
        Check result dict.
    """
    zctas_in_data = set(df["zcta_id"].unique())
    xw = crosswalk[crosswalk["zcta_id"].isin(zctas_in_data)]
    actual_fips = sorted(xw["county_fips"].unique().tolist())

    extra = sorted(set(actual_fips) - set(expected_fips))
    missing = sorted(set(expected_fips) - set(actual_fips))
    status = "PASS" if not extra and not missing else "FAIL"

    return {
        "status": status,
        "expected": expected_fips,
        "actual": actual_fips,
        "extra": extra,
        "missing": missing,
    }


def check_event_coverage(
    df: pd.DataFrame,
    expected_events: list[str],
) -> dict[str, Any]:
    """Check 2: Every ZCTA appears in every event.

    Args:
        df: Event-features dataframe.
        expected_events: Event keys from config.

    Returns:
        Check result dict.
    """
    actual_events = sorted(df["event"].unique().tolist())
    zcta_per_event = df.groupby("event")["zcta_id"].apply(set)

    if len(zcta_per_event) < 2:
        return {
            "status": "WARN",
            "details": f"Only {len(actual_events)} event(s) in data",
            "expected_events": expected_events,
            "actual_events": actual_events,
        }

    all_zctas = set().union(*zcta_per_event.values)
    gaps: dict[str, list[str]] = {}
    for evt, zctas in zcta_per_event.items():
        missing = sorted(all_zctas - zctas)
        if missing:
            gaps[evt] = missing

    if gaps:
        total_missing = sum(len(v) for v in gaps.values())
        return {
            "status": "FAIL",
            "details": f"{total_missing} ZCTA-event gaps across {len(gaps)} event(s)",
            "gaps": {k: v[:10] for k, v in gaps.items()},
        }

    return {
        "status": "PASS",
        "details": f"All {len(all_zctas)} ZCTAs present in all {len(actual_events)} events",
    }


def check_feature_nulls(
    df: pd.DataFrame,
    crosswalk: pd.DataFrame,
    required_columns: list[str],
) -> dict[str, Any]:
    """Check 3: Per-county null rates for required features.

    Args:
        df: Event-features dataframe.
        crosswalk: ZCTA-to-county crosswalk.
        required_columns: Columns from mvd.required_columns.

    Returns:
        Check result dict.
    """
    present_cols = [c for c in required_columns if c in df.columns]
    missing_cols = [c for c in required_columns if c not in df.columns]

    if missing_cols:
        return {
            "status": "FAIL",
            "details": f"Required columns missing from data: {missing_cols}",
            "missing_columns": missing_cols,
            "by_county": {},
        }

    merged = df.merge(
        crosswalk[["zcta_id", "county_fips"]],
        on="zcta_id",
        how="left",
    )

    by_county: dict[str, dict[str, float]] = {}
    worst_status = "PASS"

    for fips, grp in merged.groupby("county_fips"):
        county_nulls: dict[str, float] = {}
        for col in present_cols:
            rate = float(grp[col].isna().mean())
            if rate > 0:
                county_nulls[col] = round(rate, 4)
        if county_nulls:
            by_county[str(fips)] = county_nulls
            max_rate = max(county_nulls.values())
            if max_rate == 1.0:
                worst_status = "FAIL"  # 100% null = column entirely absent
            elif max_rate > NULL_RATE_THRESHOLD and worst_status != "FAIL":
                worst_status = "WARN"  # >30% null = structural gap (e.g. pre-MRMS)

    return {
        "status": worst_status,
        "by_county": by_county,
    }


def check_target_availability(
    df: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> dict[str, Any]:
    """Check 4: Target columns exist and report null rates per county.

    Args:
        df: Event-features dataframe.
        crosswalk: ZCTA-to-county crosswalk.

    Returns:
        Check result dict.
    """
    target_candidates = [
        "nfip_event_claim_count",
        "obs_nfip_event_claims",
        "hwm_count",
        "flood_311_count",
    ]
    found = [c for c in target_candidates if c in df.columns]

    if not found:
        return {
            "status": "FAIL",
            "details": "No target columns found in data",
            "checked": target_candidates,
        }

    merged = df.merge(
        crosswalk[["zcta_id", "county_fips"]],
        on="zcta_id",
        how="left",
    )

    details: dict[str, dict[str, float]] = {}
    worst_status = "PASS"

    for col in found:
        per_county: dict[str, float] = {}
        for fips, grp in merged.groupby("county_fips"):
            rate = float(grp[col].isna().mean())
            per_county[str(fips)] = round(rate, 4)
        details[col] = per_county
        max_rate = max(per_county.values()) if per_county else 0.0
        if max_rate == 1.0:
            worst_status = "FAIL"
        elif max_rate > NULL_RATE_THRESHOLD and worst_status != "FAIL":
            worst_status = "WARN"

    return {"status": worst_status, "details": details}


def check_adjacency(
    df: pd.DataFrame,
    adjacency: pd.DataFrame,
) -> dict[str, Any]:
    """Check 5: Adjacency connectivity for scenario ZCTAs.

    Args:
        df: Event-features dataframe.
        adjacency: Full ZCTA adjacency edgelist.

    Returns:
        Check result dict.
    """
    scenario_zctas = set(df["zcta_id"].unique())

    adj = adjacency[
        adjacency["zcta_id_1"].isin(scenario_zctas)
        & adjacency["zcta_id_2"].isin(scenario_zctas)
    ]
    n_edges = len(adj)

    connected_zctas = set(adj["zcta_id_1"].unique()) | set(
        adj["zcta_id_2"].unique()
    )
    isolated = sorted(scenario_zctas - connected_zctas)
    n_isolated = len(isolated)

    if n_isolated == 0:
        status = "PASS"
    elif n_isolated <= 3:
        status = "WARN"  # small number of island/boundary ZCTAs acceptable
    else:
        status = "FAIL"

    result: dict[str, Any] = {
        "status": status,
        "n_edges": n_edges,
        "n_isolated": n_isolated,
    }
    if isolated:
        result["isolated_zctas"] = isolated[:20]
    return result


def check_mvd_thresholds(
    df: pd.DataFrame,
    min_required: int,
) -> dict[str, Any]:
    """Check 6: Minimum viable dataset ZCTA count.

    Args:
        df: Event-features dataframe.
        min_required: mvd.min_zctas from config.

    Returns:
        Check result dict.
    """
    actual = int(df["zcta_id"].nunique())
    status = "PASS" if actual >= min_required else "FAIL"
    return {
        "status": status,
        "min_required": min_required,
        "actual": actual,
    }


def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Check 7: No duplicate (zcta_id, event) rows.

    Args:
        df: Event-features dataframe.

    Returns:
        Check result dict.
    """
    dupes = df.duplicated(subset=["zcta_id", "event"])
    n_duplicates = int(dupes.sum())
    status = "PASS" if n_duplicates == 0 else "FAIL"
    return {"status": status, "n_duplicates": n_duplicates}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_overall(checks: dict[str, dict]) -> tuple[str, list[str]]:
    """Compute overall status and blocker list from check results.

    Args:
        checks: Dict mapping check name to its result dict.

    Returns:
        Tuple of (overall_status, blocker_list).
    """
    blockers = [name for name, c in checks.items() if c["status"] == "FAIL"]
    if blockers:
        return "FAIL", blockers
    warns = [name for name, c in checks.items() if c["status"] == "WARN"]
    if warns:
        return "WARN", []
    return "PASS", []


def validate_scenario(scenario: str, s3=None) -> dict[str, Any]:
    """Run all 7 validation checks for a scenario.

    Args:
        scenario: Scenario name (e.g. 'houston').
        s3: Optional pre-built S3 client.

    Returns:
        Full validation report dict.
    """
    if s3 is None:
        s3 = _get_s3_client()

    log.info("=== Validating scenario: %s ===", scenario)

    # Load config
    cfg = load_config(scenario)
    expected_fips = _extract_fips(cfg)
    required_columns = _extract_required_columns(cfg)
    min_zctas = _extract_min_zctas(cfg)
    expected_events = _extract_event_names(cfg)

    # Load data from S3
    parquet_key = PARQUET_KEYS[scenario]
    log.info("Reading parquet: s3://%s/%s", BUCKET, parquet_key)
    df = _s3_read(s3, parquet_key)
    if df is None:
        return _empty_report(scenario, f"Parquet not found: {parquet_key}")

    log.info("Reading crosswalk: s3://%s/%s", BUCKET, CROSSWALK_KEY)
    crosswalk = _s3_read(s3, CROSSWALK_KEY)
    if crosswalk is None:
        return _empty_report(scenario, f"Crosswalk not found: {CROSSWALK_KEY}")

    log.info("Reading adjacency: s3://%s/%s", BUCKET, ADJACENCY_KEY)
    adjacency = _s3_read(s3, ADJACENCY_KEY)
    if adjacency is None:
        return _empty_report(scenario, f"Adjacency not found: {ADJACENCY_KEY}")

    # Normalize column names
    if "zcta_id" not in df.columns:
        id_col = _find_id_column(df)
        if id_col:
            df = df.rename(columns={id_col: "zcta_id"})
        else:
            return _empty_report(scenario, "No zcta_id column found in parquet")

    # Ensure string types for join keys
    df["zcta_id"] = df["zcta_id"].astype(str)
    crosswalk["zcta_id"] = crosswalk["zcta_id"].astype(str)
    crosswalk["county_fips"] = crosswalk["county_fips"].astype(str)
    adjacency["zcta_id_1"] = adjacency["zcta_id_1"].astype(str)
    adjacency["zcta_id_2"] = adjacency["zcta_id_2"].astype(str)

    # Run checks
    checks: dict[str, dict] = {}
    checks["fips_match"] = check_fips_match(df, crosswalk, expected_fips)
    checks["event_coverage"] = check_event_coverage(df, expected_events)
    checks["feature_nulls"] = check_feature_nulls(
        df, crosswalk, required_columns,
    )
    checks["target_availability"] = check_target_availability(df, crosswalk)
    checks["adjacency"] = check_adjacency(df, adjacency)
    checks["mvd_thresholds"] = check_mvd_thresholds(df, min_zctas)
    checks["duplicates"] = check_duplicates(df)

    overall, blockers = _resolve_overall(checks)

    report = {
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_zctas": int(df["zcta_id"].nunique()),
        "n_rows": len(df),
        "n_events": int(df["event"].nunique()),
        "checks": checks,
        "overall": overall,
        "blockers": blockers,
    }

    _print_report(report)
    return report


def _find_id_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic search for the ZCTA ID column."""
    candidates = ["zcta_id", "ZCTA", "zcta", "sewershed_id", "GEOID"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _empty_report(scenario: str, reason: str) -> dict[str, Any]:
    """Return a FAIL report when data cannot be loaded."""
    log.error("ABORT: %s -- %s", scenario, reason)
    return {
        "scenario": scenario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_zctas": 0,
        "n_rows": 0,
        "n_events": 0,
        "checks": {},
        "overall": "FAIL",
        "blockers": [f"data_load: {reason}"],
    }


def _print_report(report: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    scenario = report["scenario"]
    overall = report["overall"]
    marker = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}

    print()
    print("=" * 60)
    print(f"  {scenario.upper()} -- {marker.get(overall, overall)}")
    print(f"  {report['n_zctas']} ZCTAs, {report['n_rows']} rows, "
          f"{report['n_events']} events")
    print("=" * 60)

    for name, result in report.get("checks", {}).items():
        status = result.get("status", "?")
        tag = marker.get(status, status)
        line = f"  {tag} {name}"

        if name == "fips_match":
            extra = result.get("extra", [])
            missing = result.get("missing", [])
            if extra or missing:
                line += f"  extra={extra} missing={missing}"
        elif name == "feature_nulls" and status != "PASS":
            by_county = result.get("by_county", {})
            for fips, cols in list(by_county.items())[:5]:
                worst_col = max(cols, key=cols.get)
                line += f"\n         {fips}: {worst_col}={cols[worst_col]:.1%}"
        elif name == "adjacency":
            n_iso = result.get("n_isolated", 0)
            n_edges = result.get("n_edges", 0)
            line += f"  edges={n_edges}, isolated={n_iso}"
        elif name == "mvd_thresholds":
            line += (f"  required={result.get('min_required')}, "
                     f"actual={result.get('actual')}")
        elif name == "duplicates":
            n_dup = result.get("n_duplicates", 0)
            if n_dup > 0:
                line += f"  n_duplicates={n_dup}"

        print(line)

    if report.get("blockers"):
        print()
        print("  BLOCKERS:")
        for b in report["blockers"]:
            print(f"    - {b}")

    print("=" * 60)
    print()


def _save_report(report: dict[str, Any], upload: bool, s3=None) -> Path:
    """Write report JSON locally and optionally to S3.

    Args:
        report: Validation report dict.
        upload: Whether to upload to S3.
        s3: Optional pre-built S3 client.

    Returns:
        Local path where the JSON was written.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scenario = report["scenario"]
    local_path = RESULTS_DIR / f"validation_{scenario}.json"
    payload = json.dumps(report, indent=2, default=str)

    with open(local_path, "w") as fh:
        fh.write(payload)
    log.info("Wrote %s", local_path)

    if upload:
        if s3 is None:
            s3 = _get_s3_client()
        s3_key = f"results/s035/validation_{scenario}.json"
        _s3_upload(s3, payload.encode(), s3_key)

    return local_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Validate a FloodRSCT scenario dataset",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario name or 'all'",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload validation JSON to S3",
    )
    args = parser.parse_args()

    scenarios = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]

    s3 = _get_s3_client()
    any_fail = False

    for scenario in scenarios:
        if scenario not in ALL_SCENARIOS:
            log.error("Unknown scenario: %s", scenario)
            log.error("Valid scenarios: %s", ALL_SCENARIOS)
            sys.exit(1)

        report = validate_scenario(scenario, s3=s3)
        _save_report(report, upload=args.upload, s3=s3)

        if report["overall"] == "FAIL":
            any_fail = True

    if any_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
