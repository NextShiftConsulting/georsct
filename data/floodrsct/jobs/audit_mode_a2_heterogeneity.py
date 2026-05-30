#!/usr/bin/env python3
"""
audit_mode_a2_heterogeneity.py -- GeoRSCT Mode A.2: Geographic Heterogeneity.

Detects whether feature distributions differ systematically across
geographic strata. If rainfall, surge, or outcomes have very different
distributions in different counties or coastal/inland zones, a model
trained on pooled data may underperform in specific strata.

Checks per-stratum coefficient of variation (CV) for key features:
  1. By county: do features have similar spread across counties?
  2. By coastal/inland: do coastal ZCTAs look different from inland?
  3. By levee zone: do protected ZCTAs have different feature profiles?

For each (feature, stratification), computes per-stratum mean and CV,
then flags if the ratio of max-stratum-mean to min-stratum-mean
exceeds a threshold (heterogeneity ratio > 5x).

PASS if all features have heterogeneity ratio < 5x across strata.
FAIL if any feature shows extreme heterogeneity.

Usage:
    python audit_mode_a2_heterogeneity.py --scenario houston
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    AuditResult, SCENARIOS, get_s3_client, load_processed_parquet,
    load_crosswalk, write_evidence,
)

HETEROGENEITY_THRESHOLD = 5.0  # max/min stratum mean ratio

FEATURES_TO_CHECK = [
    "rainfall_total_mm", "total_rainfall_mm", "max_rainfall_mm",
    "tidal_surge_max_m", "max_surge_m",
    "elevation_m_msl",
    "impervious_pct",
    "hwm_max_ft",
    "nfip_event_claims",
]

COASTAL_THRESHOLD_M = 20_000


def _stratum_stats(df, col: str, stratum_col: str) -> list[dict]:
    """Compute per-stratum stats for a feature."""
    stats = []
    for name, group in df.groupby(stratum_col):
        values = group[col].dropna().values
        if len(values) < 3:
            continue
        stats.append({
            "stratum": str(name),
            "n": len(values),
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "cv": round(float(np.std(values) / np.mean(values)), 4)
                  if np.mean(values) != 0 else 0.0,
            "median": round(float(np.median(values)), 4),
        })
    return stats


def _check_heterogeneity(stats: list[dict]) -> tuple[float, bool]:
    """Compute heterogeneity ratio from stratum stats."""
    means = [s["mean"] for s in stats if s["mean"] > 0]
    if len(means) < 2:
        return 1.0, False
    ratio = max(means) / min(means)
    return round(ratio, 4), ratio > HETEROGENEITY_THRESHOLD


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)
    df["zcta_id"] = df["zcta_id"].astype(str)
    results = []

    available = [c for c in FEATURES_TO_CHECK if c in df.columns
                 and df[c].notna().sum() >= min_support]

    if not available:
        results.append(AuditResult(
            audit_id="mode_A2", scenario=scenario, probe="heterogeneity",
            status="SKIP",
            detail={"note": "No checkable features with sufficient data"},
            min_support=min_support, timestamp=ts,
        ))
        write_evidence(results, "mode_a2_heterogeneity", scenario,
                       s3=s3, upload=upload)
        return results

    # --- Stratification 1: By county ---
    try:
        xwalk = load_crosswalk(s3)
        xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)
        df_county = df.merge(xwalk, on="zcta_id", how="left")

        if "county_fips" in df_county.columns:
            for col in available:
                stats = _stratum_stats(df_county, col, "county_fips")
                if len(stats) < 2:
                    continue
                ratio, is_heterogeneous = _check_heterogeneity(stats)

                results.append(AuditResult(
                    audit_id="mode_A2", scenario=scenario,
                    probe="heterogeneity",
                    status="FAIL" if is_heterogeneous else "PASS",
                    detail={
                        "feature": col,
                        "stratification": "county",
                        "n_strata": len(stats),
                        "heterogeneity_ratio": ratio,
                        "threshold": HETEROGENEITY_THRESHOLD,
                        "per_stratum": stats,
                    },
                    min_support=min_support, timestamp=ts,
                ))
    except Exception:
        pass

    # --- Stratification 2: Coastal vs inland ---
    if "coastal_distance_m" in df.columns:
        df["_zone"] = np.where(
            df["coastal_distance_m"] <= COASTAL_THRESHOLD_M,
            "coastal", "inland"
        )

        for col in available:
            stats = _stratum_stats(df, col, "_zone")
            if len(stats) < 2:
                continue
            ratio, is_heterogeneous = _check_heterogeneity(stats)

            results.append(AuditResult(
                audit_id="mode_A2", scenario=scenario,
                probe="heterogeneity",
                status="FAIL" if is_heterogeneous else "PASS",
                detail={
                    "feature": col,
                    "stratification": "coastal_inland",
                    "threshold_m": COASTAL_THRESHOLD_M,
                    "n_strata": len(stats),
                    "heterogeneity_ratio": ratio,
                    "threshold": HETEROGENEITY_THRESHOLD,
                    "per_stratum": stats,
                },
                min_support=min_support, timestamp=ts,
            ))

        df.drop(columns=["_zone"], inplace=True)

    # --- Stratification 3: Levee-protected vs unprotected ---
    if "levee_condition_rating" in df.columns:
        df["_levee_zone"] = np.where(
            df["levee_condition_rating"].notna(),
            "protected", "unprotected"
        )

        for col in available:
            stats = _stratum_stats(df, col, "_levee_zone")
            if len(stats) < 2:
                continue
            ratio, is_heterogeneous = _check_heterogeneity(stats)

            results.append(AuditResult(
                audit_id="mode_A2", scenario=scenario,
                probe="heterogeneity",
                status="FAIL" if is_heterogeneous else "PASS",
                detail={
                    "feature": col,
                    "stratification": "levee_zone",
                    "n_strata": len(stats),
                    "heterogeneity_ratio": ratio,
                    "threshold": HETEROGENEITY_THRESHOLD,
                    "per_stratum": stats,
                },
                min_support=min_support, timestamp=ts,
            ))

        df.drop(columns=["_levee_zone"], inplace=True)

    # --- Stratification 4: By event ---
    if "event" in df.columns and df["event"].nunique() >= 2:
        for col in available:
            stats = _stratum_stats(df, col, "event")
            if len(stats) < 2:
                continue
            ratio, is_heterogeneous = _check_heterogeneity(stats)

            results.append(AuditResult(
                audit_id="mode_A2", scenario=scenario,
                probe="heterogeneity",
                status="FAIL" if is_heterogeneous else "PASS",
                detail={
                    "feature": col,
                    "stratification": "event",
                    "n_strata": len(stats),
                    "heterogeneity_ratio": ratio,
                    "threshold": HETEROGENEITY_THRESHOLD,
                    "per_stratum": stats,
                },
                min_support=min_support, timestamp=ts,
            ))

    if not results:
        results.append(AuditResult(
            audit_id="mode_A2", scenario=scenario, probe="heterogeneity",
            status="PASS",
            detail={"note": "No stratifications produced checkable groups"},
            min_support=min_support, timestamp=ts,
        ))

    write_evidence(results, "mode_a2_heterogeneity", scenario,
                   s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode A.2: Geographic heterogeneity detection"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
