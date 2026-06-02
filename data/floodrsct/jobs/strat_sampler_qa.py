#!/usr/bin/env python3
"""
strat_sampler_qa.py -- Stratified sampling QA for assembled event datasets.

Jumps across scenarios, events, and column families to validate:
  1. Value ranges (physical plausibility)
  2. Cross-column consistency (rainfall > 0 implies MRMS coverage > 0)
  3. Schema alignment (all scenarios share the same column set)
  4. Distribution sanity (no scenario is all-zero or all-constant)
  5. Key-integrity (no duplicate zcta_id x event, no orphan ZCTAs)

Designed to catch silent data corruption that per-scenario validators miss --
the kind where one scenario looks fine in isolation but is wrong relative to
the others.

Usage:
    python strat_sampler_qa.py                    # all available scenarios
    python strat_sampler_qa.py --scenario houston  # single scenario
    python strat_sampler_qa.py --verbose           # print every sample probe
"""

import argparse
import logging
import sys
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

BUCKET = "swarm-floodrsct-data"

SCENARIO_PARQUETS = {
    "houston":             "processed/houston/houston_event_features.parquet",
    "new_orleans":         "processed/new_orleans/no_event_features.parquet",
    "nyc":                 "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella":  "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida":   "processed/southwest_florida/swfl_event_features.parquet",
}

# ---------------------------------------------------------------------------
# Value range checks -- (column, min, max, description)
# These are physical/domain constraints, not statistical.
# ---------------------------------------------------------------------------
RANGE_CHECKS = [
    # Geography
    ("latitude",            20.0,   50.0,   "CONUS latitude"),
    ("longitude",          -130.0, -60.0,   "CONUS longitude"),
    ("elevation_m_msl",    -10.0,  3000.0,  "elevation MSL"),
    # Weather
    ("rainfall_total_mm",    0.0,  2000.0,  "total rainfall"),
    ("max_water_level_m",   -5.0,   15.0,   "tidal water level"),
    ("max_surge_m",         -2.0,   10.0,   "storm surge"),
    # Storm
    ("storm_min_dist_km",    0.0,  3000.0,  "distance to storm track"),
    ("storm_landfall_category", 0.0, 5.0,   "Saffir-Simpson category"),
    # Hydro
    ("peak_stage_ft",        0.0,  100.0,   "gauge stage"),
    ("peak_flow_cfs",        0.0,  1e7,     "gauge flow"),
    # SVI / demographics
    ("svi_overall",          0.0,    1.0,   "SVI percentile"),
    ("acs_total_pop",        0.0,  1e6,     "ZCTA population"),
    ("acs_median_hh_income", 0.0,  500000,  "median household income"),
    # NFIP (temporally-gated historical)
    ("nfip_historical_frequency", 0.0, 1e5,  "NFIP historical claim count"),
    ("nfip_historical_severity",  0.0, 1e7,  "NFIP historical mean loss/claim $"),
    # Coverage
    ("obs_mrms_coverage_pct", 0.0,  1.0,    "MRMS coverage fraction"),
    # TWI
    ("twi_twi",              0.0,   30.0,   "topographic wetness index"),
    # Flood zones (stored as percentages 0-100, not fractions)
    ("flood_pct_zone_a",     0.0,  100.0,   "pct SFHA zone A"),
    ("flood_pct_zone_x",     0.0,  100.0,   "pct zone X"),
    ("flood_pct_zone_x500",  0.0,  100.0,   "pct zone X500"),
    # Slope
    ("slope_mean_pct",       0.0,  100.0,   "mean slope %"),
]

# ---------------------------------------------------------------------------
# Cross-column consistency rules
# If condition_col has a truthy/non-null value, then check_col should too.
# ---------------------------------------------------------------------------
CONSISTENCY_RULES = [
    {
        "name": "rainfall implies MRMS coverage",
        "condition": lambda df: df["rainfall_total_mm"].notna() & (df["rainfall_total_mm"] > 0),
        "check": lambda df: df["obs_mrms_coverage_pct"].notna() & (df["obs_mrms_coverage_pct"] > 0),
        "severity": "ERROR",
    },
    {
        "name": "NFIP claims implies non-negative loss",
        "condition": lambda df: df["nfip_event_claim_count"].notna() & (df["nfip_event_claim_count"] > 0),
        "check": lambda df: df["nfip_event_total_loss"].notna() & (df["nfip_event_total_loss"] >= 0),
        "severity": "ERROR",
    },
    {
        "name": "latitude implies longitude",
        "condition": lambda df: df["latitude"].notna(),
        "check": lambda df: df["longitude"].notna(),
        "severity": "ERROR",
    },
    {
        "name": "flood zone A + X + X500 <= 100%",
        "condition": lambda df: (
            df["flood_pct_zone_a"].notna() &
            df["flood_pct_zone_x"].notna() &
            df["flood_pct_zone_x500"].notna()
        ),
        "check": lambda df: (
            df["flood_pct_zone_a"] + df["flood_pct_zone_x"] + df["flood_pct_zone_x500"] <= 100.1
        ),
        "severity": "WARN",
    },
    {
        "name": "SVI components all present or all absent",
        "condition": lambda df: df["svi_overall"].notna(),
        "check": lambda df: (
            df["svi_socioeconomic"].notna() &
            df["svi_household_disability"].notna() &
            df["svi_minority_language"].notna() &
            df["svi_housing_transport"].notna()
        ),
        "severity": "ERROR",
    },
]


# ---------------------------------------------------------------------------
# Column families -- groups that should move together
# ---------------------------------------------------------------------------
COLUMN_FAMILIES = {
    "acs": [c for c in [
        "acs_total_pop", "acs_median_hh_income", "acs_median_age",
        "acs_pct_below_poverty", "acs_pct_owner_occupied",
    ]],
    "svi": ["svi_overall", "svi_socioeconomic", "svi_household_disability",
            "svi_minority_language", "svi_housing_transport"],
    "flood_zone": ["flood_pct_zone_a", "flood_pct_zone_x", "flood_pct_zone_x500"],
    "nfip_event": ["nfip_event_claim_count", "nfip_event_total_loss"],
    "storm": ["storm_min_dist_km", "storm_landfall_category"],
    "weather": ["rainfall_total_mm", "obs_mrms_coverage_pct"],
    "hydro": ["peak_stage_ft", "peak_flow_cfs"],
    "twi": ["twi_twi", "twi_acc_twi", "twi_tot_twi"],
    "target_health": ["target_diabetes", "target_obesity", "target_asthma",
                       "target_copd", "target_high_blood_pressure"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_parquet(s3, key: str) -> Optional[pd.DataFrame]:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except Exception as e:
        log.warning("Cannot load %s: %s", key, e)
        return None


class QAResult:
    """Single QA check result."""
    def __init__(self, scenario: str, check: str, status: str, message: str):
        self.scenario = scenario
        self.check = check
        self.status = status   # PASS, FAIL, WARN, SKIP
        self.message = message

    def __str__(self):
        return f"  [{self.status:4s}] {self.scenario:22s} {self.check}: {self.message}"


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_ranges(df: pd.DataFrame, scenario: str) -> list[QAResult]:
    """Check that numeric columns fall within physically plausible ranges."""
    results = []
    for col, lo, hi, desc in RANGE_CHECKS:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        below = (vals < lo).sum()
        above = (vals > hi).sum()
        if below > 0 or above > 0:
            results.append(QAResult(
                scenario, f"range:{col}", "FAIL",
                f"{desc}: {below} below {lo}, {above} above {hi} "
                f"(min={vals.min():.4g}, max={vals.max():.4g})"
            ))
        else:
            results.append(QAResult(
                scenario, f"range:{col}", "PASS",
                f"{desc}: [{vals.min():.4g}, {vals.max():.4g}] within [{lo}, {hi}]"
            ))
    return results


def probe_consistency(df: pd.DataFrame, scenario: str) -> list[QAResult]:
    """Check cross-column logical consistency."""
    results = []
    for rule in CONSISTENCY_RULES:
        name = rule["name"]
        severity = rule["severity"]
        try:
            cond_mask = rule["condition"](df)
            if cond_mask.sum() == 0:
                results.append(QAResult(scenario, f"consistency:{name}", "SKIP",
                                        "condition never true"))
                continue
            check_mask = rule["check"](df)
            violations = cond_mask & ~check_mask
            n_viol = violations.sum()
            n_cond = cond_mask.sum()
            if n_viol > 0:
                results.append(QAResult(
                    scenario, f"consistency:{name}",
                    "FAIL" if severity == "ERROR" else "WARN",
                    f"{n_viol}/{n_cond} rows violate rule"
                ))
            else:
                results.append(QAResult(scenario, f"consistency:{name}", "PASS",
                                        f"0/{n_cond} violations"))
        except KeyError as e:
            results.append(QAResult(scenario, f"consistency:{name}", "SKIP",
                                    f"missing column: {e}"))
    return results


def probe_family_coherence(df: pd.DataFrame, scenario: str) -> list[QAResult]:
    """Within a column family, all members should have similar non-null rates."""
    results = []
    for family, cols in COLUMN_FAMILIES.items():
        present = [c for c in cols if c in df.columns]
        if len(present) < 2:
            continue
        rates = {c: df[c].notna().mean() for c in present}
        lo = min(rates.values())
        hi = max(rates.values())
        spread = hi - lo
        if spread > 0.4:
            worst = min(rates, key=rates.get)
            best = max(rates, key=rates.get)
            results.append(QAResult(
                scenario, f"family:{family}", "WARN",
                f"coverage spread {spread:.0%} -- "
                f"{best}={rates[best]:.0%} vs {worst}={rates[worst]:.0%}"
            ))
        else:
            results.append(QAResult(
                scenario, f"family:{family}", "PASS",
                f"coverage spread {spread:.0%} across {len(present)} columns"
            ))
    return results


def probe_keys(df: pd.DataFrame, scenario: str) -> list[QAResult]:
    """Check primary key integrity and mandatory columns."""
    results = []

    # zcta_id must be present and non-null
    if "zcta_id" not in df.columns:
        results.append(QAResult(scenario, "key:zcta_id", "FAIL", "column missing"))
        return results
    null_ids = df["zcta_id"].isna().sum()
    if null_ids > 0:
        results.append(QAResult(scenario, "key:zcta_id", "FAIL",
                                f"{null_ids} null zcta_id values"))
    else:
        results.append(QAResult(scenario, "key:zcta_id", "PASS",
                                f"{len(df)} rows, 0 null IDs"))

    # event must be present
    if "event" in df.columns:
        null_events = df["event"].isna().sum()
        if null_events > 0:
            results.append(QAResult(scenario, "key:event", "FAIL",
                                    f"{null_events} null event values"))
        else:
            events = df["event"].unique()
            results.append(QAResult(scenario, "key:event", "PASS",
                                    f"events: {list(events)}"))

    # Duplicate (zcta_id, event)
    if "zcta_id" in df.columns and "event" in df.columns:
        dupes = df.duplicated(subset=["zcta_id", "event"]).sum()
        if dupes > 0:
            results.append(QAResult(scenario, "key:duplicates", "FAIL",
                                    f"{dupes} duplicate (zcta_id, event) rows"))
        else:
            results.append(QAResult(scenario, "key:duplicates", "PASS",
                                    f"0 duplicates across {len(df)} rows"))

    return results


def probe_constants(df: pd.DataFrame, scenario: str) -> list[QAResult]:
    """Flag numeric columns where every non-null value is identical."""
    results = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = []
    for col in num_cols:
        vals = df[col].dropna()
        if len(vals) >= 5 and vals.nunique() == 1:
            constant_cols.append((col, vals.iloc[0]))

    if constant_cols:
        col_list = ", ".join(f"{c}={v}" for c, v in constant_cols[:5])
        suffix = f" (+{len(constant_cols)-5} more)" if len(constant_cols) > 5 else ""
        results.append(QAResult(scenario, "constant_columns", "WARN",
                                f"{len(constant_cols)} constant cols: {col_list}{suffix}"))
    else:
        results.append(QAResult(scenario, "constant_columns", "PASS",
                                "no constant numeric columns"))
    return results


def probe_stratified_sample(df: pd.DataFrame, scenario: str,
                            seed: int = 42, n_samples: int = 3) -> list[QAResult]:
    """Pick random rows, print a diagnostic fingerprint for eyeball QA."""
    results = []
    n_sample = min(n_samples, len(df))
    if n_sample == 0:
        return results

    sample = df.sample(n=n_sample, random_state=seed)
    key_cols = ["zcta_id", "event", "rainfall_total_mm", "nfip_event_claim_count",
                "svi_overall", "elevation_m_msl", "storm_min_dist_km"]
    present = [c for c in key_cols if c in sample.columns]

    for _, row in sample.iterrows():
        vals = " | ".join(f"{c}={row[c]}" for c in present)
        results.append(QAResult(scenario, "sample", "INFO", vals))
    return results


# ---------------------------------------------------------------------------
# Cross-scenario checks
# ---------------------------------------------------------------------------

def probe_schema_alignment(dfs: dict[str, pd.DataFrame]) -> list[QAResult]:
    """All scenarios should share the same column set."""
    results = []
    if len(dfs) < 2:
        return results

    names = list(dfs.keys())
    ref_name = names[0]
    ref_cols = set(dfs[ref_name].columns)

    for name in names[1:]:
        other_cols = set(dfs[name].columns)
        only_ref = ref_cols - other_cols
        only_other = other_cols - ref_cols
        if only_ref or only_other:
            msg_parts = []
            if only_ref:
                msg_parts.append(f"only in {ref_name}: {sorted(only_ref)[:5]}")
            if only_other:
                msg_parts.append(f"only in {name}: {sorted(only_other)[:5]}")
            results.append(QAResult(
                "cross", f"schema:{ref_name} vs {name}", "WARN",
                "; ".join(msg_parts)
            ))
        else:
            results.append(QAResult(
                "cross", f"schema:{ref_name} vs {name}", "PASS",
                f"identical column sets ({len(ref_cols)} cols)"
            ))
    return results


def probe_cross_scenario_coverage(dfs: dict[str, pd.DataFrame]) -> list[QAResult]:
    """Compare non-null rates across scenarios for key columns."""
    results = []
    key_cols = ["rainfall_total_mm", "svi_overall", "acs_total_pop",
                "nfip_event_claim_count", "storm_min_dist_km",
                "elevation_m_msl", "flood_pct_zone_a"]

    for col in key_cols:
        rates = {}
        for name, df in dfs.items():
            if col in df.columns:
                rates[name] = df[col].notna().mean()
        if len(rates) < 2:
            continue

        lo_name = min(rates, key=rates.get)
        hi_name = max(rates, key=rates.get)
        spread = rates[hi_name] - rates[lo_name]

        if spread > 0.5:
            results.append(QAResult(
                "cross", f"coverage:{col}", "WARN",
                f"spread={spread:.0%} -- {lo_name}={rates[lo_name]:.0%} vs "
                f"{hi_name}={rates[hi_name]:.0%}"
            ))
        else:
            results.append(QAResult(
                "cross", f"coverage:{col}", "PASS",
                f"spread={spread:.0%} across {len(rates)} scenarios"
            ))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_qa(s3, scenarios: list[str], verbose: bool = False,
           seed: int = 42, n_samples: int = 3) -> int:
    """Run full stratified QA. Returns number of FAILs."""
    log.info("QA seed=%d, n_samples=%d", seed, n_samples)
    dfs: dict[str, pd.DataFrame] = {}

    # Load all available parquets
    for name in scenarios:
        key = SCENARIO_PARQUETS.get(name)
        if not key:
            log.warning("No parquet path configured for %s", name)
            continue
        df = _load_parquet(s3, key)
        if df is not None:
            dfs[name] = df
            log.info("Loaded %s: %d rows x %d cols", name, len(df), len(df.columns))

    if not dfs:
        log.error("No scenario parquets available. Run build_event_dataset first.")
        return 1

    all_results: list[QAResult] = []

    # Per-scenario probes
    for name, df in dfs.items():
        log.info("\n--- Probing %s ---", name)
        all_results.extend(probe_keys(df, name))
        all_results.extend(probe_ranges(df, name))
        all_results.extend(probe_consistency(df, name))
        all_results.extend(probe_family_coherence(df, name))
        all_results.extend(probe_constants(df, name))
        all_results.extend(probe_stratified_sample(df, name, seed=seed, n_samples=n_samples))

    # Cross-scenario probes
    if len(dfs) >= 2:
        log.info("\n--- Cross-scenario probes ---")
        all_results.extend(probe_schema_alignment(dfs))
        all_results.extend(probe_cross_scenario_coverage(dfs))

    # Report
    print(f"\n{'=' * 78}")
    print(f"  STRATIFIED SAMPLER QA REPORT")
    print(f"  Scenarios: {list(dfs.keys())}")
    print(f"  Total probes: {len(all_results)}")
    print(f"{'=' * 78}\n")

    fails = [r for r in all_results if r.status == "FAIL"]
    warns = [r for r in all_results if r.status == "WARN"]
    passes = [r for r in all_results if r.status == "PASS"]
    infos = [r for r in all_results if r.status == "INFO"]

    if fails:
        print("FAILURES:")
        for r in fails:
            print(r)
        print()

    if warns:
        print("WARNINGS:")
        for r in warns:
            print(r)
        print()

    if verbose:
        print("PASSES:")
        for r in passes:
            print(r)
        print()

    if infos:
        print("SAMPLE FINGERPRINTS:")
        for r in infos:
            print(r)
        print()

    print(f"{'=' * 78}")
    print(f"  PASS: {len(passes)}  FAIL: {len(fails)}  WARN: {len(warns)}  "
          f"SKIP: {len(all_results) - len(passes) - len(fails) - len(warns) - len(infos)}  "
          f"INFO: {len(infos)}")
    verdict = "FAIL" if fails else ("REVIEW" if warns else "CLEAN")
    print(f"  VERDICT: {verdict}")
    print(f"{'=' * 78}\n")

    return all_results, len(fails)


def results_to_json(results: list[QAResult], seed: int, n_samples: int) -> dict:
    """Serialize QA results for experiment evidence."""
    import json
    from datetime import datetime, timezone

    summary = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0, "INFO": 0}
    for r in results:
        summary[r.status] = summary.get(r.status, 0) + 1

    return {
        "qa_type": "strat_sampler_qa",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_samples": n_samples,
        "summary": summary,
        "verdict": "FAIL" if summary["FAIL"] > 0 else (
            "REVIEW" if summary["WARN"] > 0 else "CLEAN"),
        "checks": [
            {
                "scenario": r.scenario,
                "check": r.check,
                "status": r.status,
                "message": r.message,
            }
            for r in results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Stratified sampler QA")
    parser.add_argument("--scenario", default=None,
                        choices=list(SCENARIO_PARQUETS.keys()),
                        help="Run QA for a single scenario (default: all available)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stratified sampling (default: 42)")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Number of sample rows per scenario (default: 3)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all PASS results too")
    parser.add_argument("--json-out", default=None,
                        help="Write JSON evidence to S3 key (e.g. evidence/qa_seed42.json)")
    args = parser.parse_args()

    import json
    import boto3
    from swarm_auth import get_aws_credentials
    s3 = boto3.client("s3", region_name="us-east-1", **get_aws_credentials())

    scenarios = [args.scenario] if args.scenario else list(SCENARIO_PARQUETS.keys())
    all_results, n_fails = run_qa(s3, scenarios, verbose=args.verbose,
                                  seed=args.seed, n_samples=args.n_samples)

    if args.json_out:
        evidence = results_to_json(all_results, args.seed, args.n_samples)
        body = json.dumps(evidence, indent=2)
        if args.json_out.startswith("s3://"):
            parts = args.json_out.replace("s3://", "").split("/", 1)
            s3.put_object(Bucket=parts[0], Key=parts[1], Body=body.encode())
            log.info("Evidence written to %s", args.json_out)
        else:
            s3.put_object(Bucket=BUCKET, Key=args.json_out, Body=body.encode())
            log.info("Evidence written to s3://%s/%s", BUCKET, args.json_out)
    sys.exit(1 if n_fails > 0 else 0)


if __name__ == "__main__":
    main()
