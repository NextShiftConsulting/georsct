#!/usr/bin/env python3
"""
audit_mode_c3_missingness.py -- GeoRSCT Mode C.3: Spatial Missingness Bias.

Checks whether null patterns in outcome features correlate with
spatial accessibility or population density. If HWM marks are only
placed at road-accessible locations, the null pattern is informative
(not random) and models that ignore it will be biased.

Checks:
  1. HWM missingness vs population: do ZCTAs with HWM marks have
     systematically different population than ZCTAs without?
  2. 311 missingness: are null 311 counts concentrated in low-population
     ZCTAs (self-selection bias)?
  3. NFIP censoring: does NFIP null rate differ between high-income
     and low-income ZCTAs (policy purchase bias)?

Uses Welch's t-test to check whether the population/income mean
differs between null vs non-null groups. A significant difference
indicates systematic missingness.

Usage:
    python audit_mode_c3_missingness.py --scenario houston
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    AuditResult, SCENARIOS, get_s3_client, load_processed_parquet,
    write_evidence,
)

# Population proxy columns (from ACS via geocertdb2026)
POP_COLS = ["acs_total_pop", "acs_pop_total", "total_pop"]
INCOME_COLS = ["acs_median_income", "acs_med_income", "median_income"]

# Outcome columns to test for missingness bias
OUTCOME_TESTS = [
    {
        "outcome": "hwm_max_ft",
        "bias_type": "road_accessibility",
        "covariate_family": "population",
        "scenarios": ["houston", "southwest_florida", "nyc", "riverside_coachella"],
    },
    {
        "outcome": "flood_311_count",
        "bias_type": "self_selection",
        "covariate_family": "population",
        "scenarios": ["houston", "nyc"],
    },
    {
        "outcome": "nfip_event_claims",
        "bias_type": "policy_purchase",
        "covariate_family": "income",
        "scenarios": ["houston", "new_orleans", "nyc",
                      "southwest_florida", "riverside_coachella"],
    },
]


def _find_col(df, candidates: list[str]) -> str | None:
    """Find first matching column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _welch_t(group1, group2):
    """Welch's t-test (unequal variance). Returns t-stat and approx p."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    se = np.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0, 1.0

    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df_ws = num / den if den > 0 else 1

    # Approximate p-value using normal (good enough for audit)
    p_approx = 2 * (1 - 0.5 * (1 + np.sign(abs(t_stat))
                * (1 - np.exp(-0.717 * abs(t_stat)
                              - 0.416 * t_stat**2))))
    # Clamp
    p_approx = max(0.0, min(1.0, p_approx))

    return float(t_stat), float(p_approx)


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)
    results = []

    pop_col = _find_col(df, POP_COLS)
    income_col = _find_col(df, INCOME_COLS)

    for test in OUTCOME_TESTS:
        if scenario not in test["scenarios"]:
            continue

        outcome_col = test["outcome"]
        if outcome_col not in df.columns:
            continue

        # Select covariate
        if test["covariate_family"] == "population":
            cov_col = pop_col
        elif test["covariate_family"] == "income":
            cov_col = income_col
        else:
            cov_col = None

        if cov_col is None:
            results.append(AuditResult(
                audit_id="mode_C3", scenario=scenario, probe="missingness",
                status="SKIP",
                detail={
                    "outcome": outcome_col,
                    "bias_type": test["bias_type"],
                    "error": f"No {test['covariate_family']} column found",
                    "checked": POP_COLS if test["covariate_family"] == "population"
                               else INCOME_COLS,
                },
                min_support=min_support, timestamp=ts,
            ))
            continue

        # Split: outcome present vs absent
        has_outcome = df[outcome_col].notna() & (df[outcome_col] > 0)
        group_present = df.loc[has_outcome, cov_col].dropna().values
        group_absent = df.loc[~has_outcome, cov_col].dropna().values

        if len(group_present) < 2 or len(group_absent) < 2:
            results.append(AuditResult(
                audit_id="mode_C3", scenario=scenario, probe="missingness",
                status="SKIP",
                detail={
                    "outcome": outcome_col,
                    "covariate": cov_col,
                    "n_present": len(group_present),
                    "n_absent": len(group_absent),
                    "error": "Insufficient samples for t-test",
                },
                min_support=min_support, timestamp=ts,
            ))
            continue

        t_stat, p_val = _welch_t(group_present, group_absent)

        # Significant at p < 0.05 means systematic missingness
        is_biased = p_val < 0.05
        status = "FAIL" if is_biased else "PASS"

        results.append(AuditResult(
            audit_id="mode_C3", scenario=scenario, probe="missingness",
            status=status,
            detail={
                "outcome": outcome_col,
                "bias_type": test["bias_type"],
                "covariate": cov_col,
                "n_with_outcome": len(group_present),
                "n_without_outcome": len(group_absent),
                "mean_with": round(float(np.mean(group_present)), 2),
                "mean_without": round(float(np.mean(group_absent)), 2),
                "t_statistic": round(t_stat, 4),
                "p_value_approx": round(p_val, 4),
                "significant_at_05": is_biased,
                "interpretation": (
                    f"ZCTAs with {outcome_col} have systematically "
                    f"{'higher' if t_stat > 0 else 'lower'} "
                    f"{test['covariate_family']} than ZCTAs without"
                    if is_biased else
                    f"No significant {test['covariate_family']} difference "
                    f"between ZCTAs with/without {outcome_col}"
                ),
            },
            min_support=min_support, timestamp=ts,
        ))

    if not results:
        results.append(AuditResult(
            audit_id="mode_C3", scenario=scenario, probe="missingness",
            status="SKIP",
            detail={"note": "No testable outcome columns for this scenario"},
            min_support=min_support, timestamp=ts,
        ))

    write_evidence(results, "mode_c3_missingness", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode C.3: Spatial missingness bias detection"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
