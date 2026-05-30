#!/usr/bin/env python3
"""
audit_mode_a1_leakage.py -- GeoRSCT Mode A.1: Autocorrelation Leakage.

Checks whether random train/test splits create spatial leakage by
placing adjacent ZCTAs in different folds. The leakage rate is the
fraction of test ZCTAs that have at least one spatial neighbor in the
training set.

Compares two split strategies:
  1. Random 80/20 split (seed-controlled)
  2. County-blocked split (hold out one county at a time)

If random leakage rate >> county-blocked rate, random splits are
unreliable for this scenario's spatial structure.

PASS if random leakage rate < 0.9 (some leakage is inevitable in
dense urban ZCTAs; flag only extreme cases).

Usage:
    python audit_mode_a1_leakage.py --scenario houston
    python audit_mode_a1_leakage.py --scenario houston --seed 42
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    AuditResult, SCENARIOS, get_s3_client, load_processed_parquet,
    load_adjacency, load_crosswalk, write_evidence,
)

LEAKAGE_THRESHOLD = 0.90


def _compute_leakage_rate(test_zctas: set, train_zctas: set,
                          adj_dict: dict) -> float:
    """Fraction of test ZCTAs with >= 1 neighbor in train set."""
    if not test_zctas:
        return 0.0
    leaked = 0
    for z in test_zctas:
        neighbors = adj_dict.get(z, set())
        if neighbors & train_zctas:
            leaked += 1
    return leaked / len(test_zctas)


def audit(scenario: str, min_support: int, seed: int,
          upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)

    # Load adjacency
    try:
        adj_df = load_adjacency(s3)
    except FileNotFoundError as e:
        result = AuditResult(
            audit_id="mode_A1", scenario=scenario, mode="leakage",
            status="FAIL", detail={"error": str(e)},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "mode_a1_leakage", scenario, s3=s3, upload=upload)
        return [result]

    # Build adjacency dict
    cols = adj_df.columns.tolist()
    src_col, dst_col = cols[0], cols[1]
    adj_df[src_col] = adj_df[src_col].astype(str)
    adj_df[dst_col] = adj_df[dst_col].astype(str)

    adj_dict: dict[str, set] = {}
    for _, row in adj_df.iterrows():
        s, d = row[src_col], row[dst_col]
        adj_dict.setdefault(s, set()).add(d)
        adj_dict.setdefault(d, set()).add(s)

    # Unique ZCTAs
    df["zcta_id"] = df["zcta_id"].astype(str)
    all_zctas = sorted(df["zcta_id"].unique())
    scenario_zctas = set(all_zctas)

    # Filter adjacency to scenario ZCTAs
    scenario_adj = {z: (adj_dict.get(z, set()) & scenario_zctas)
                    for z in scenario_zctas}

    results = []
    rng = np.random.RandomState(seed)

    # --- Random split leakage ---
    n_test = max(1, len(all_zctas) // 5)  # 20%
    indices = rng.permutation(len(all_zctas))
    test_idx = set(indices[:n_test])
    train_idx = set(indices[n_test:])

    test_random = {all_zctas[i] for i in test_idx}
    train_random = {all_zctas[i] for i in train_idx}

    random_leakage = _compute_leakage_rate(test_random, train_random, scenario_adj)

    results.append(AuditResult(
        audit_id="mode_A1", scenario=scenario, mode="leakage",
        status="PASS" if random_leakage < LEAKAGE_THRESHOLD else "FAIL",
        detail={
            "split": "random_80_20",
            "seed": seed,
            "n_train": len(train_random),
            "n_test": len(test_random),
            "leakage_rate": round(random_leakage, 4),
            "threshold": LEAKAGE_THRESHOLD,
        },
        min_support=min_support, timestamp=ts,
        primary_metric="leakage_rate",
        metric_value=round(random_leakage, 4),
        threshold=LEAKAGE_THRESHOLD,
        recommendation=("Use spatial blocked CV instead of random split."
                        if random_leakage >= LEAKAGE_THRESHOLD
                        else "Random split leakage is acceptable."),
    ))

    # --- County-blocked split leakage ---
    try:
        xwalk = load_crosswalk(s3)
        xwalk["zcta_id"] = xwalk["zcta_id"].astype(str)
        zcta_county = dict(zip(xwalk["zcta_id"], xwalk["county_fips"].astype(str)))

        counties = sorted(set(
            zcta_county[z] for z in all_zctas if z in zcta_county
        ))

        county_leakages = []
        for holdout_county in counties:
            test_blocked = {z for z in all_zctas
                           if zcta_county.get(z) == holdout_county}
            train_blocked = scenario_zctas - test_blocked

            if not test_blocked:
                continue

            rate = _compute_leakage_rate(test_blocked, train_blocked, scenario_adj)
            county_leakages.append({
                "holdout_county": holdout_county,
                "n_test": len(test_blocked),
                "leakage_rate": round(rate, 4),
            })

        mean_blocked = (np.mean([c["leakage_rate"] for c in county_leakages])
                        if county_leakages else 0.0)

        reduction = round(random_leakage - float(mean_blocked), 4)
        results.append(AuditResult(
            audit_id="mode_A1", scenario=scenario, mode="leakage",
            status="PASS" if mean_blocked < random_leakage else "FAIL",
            detail={
                "split": "county_blocked",
                "n_counties": len(counties),
                "mean_leakage_rate": round(float(mean_blocked), 4),
                "random_leakage_rate": round(random_leakage, 4),
                "reduction": reduction,
                "per_county": county_leakages,
            },
            min_support=min_support, timestamp=ts,
            primary_metric="leakage_reduction",
            metric_value=reduction,
            recommendation=("County-blocked CV reduces leakage; use as primary split."
                            if mean_blocked < random_leakage
                            else "Blocked CV does not reduce leakage; investigate adjacency."),
        ))
    except Exception as e:
        results.append(AuditResult(
            audit_id="mode_A1", scenario=scenario, mode="leakage",
            status="SKIP",
            detail={"split": "county_blocked", "error": str(e)},
            min_support=min_support, timestamp=ts,
        ))

    write_evidence(results, "mode_a1_leakage", scenario, s3=s3, upload=upload)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode A.1: Autocorrelation leakage check"
    )
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.seed, args.upload)


if __name__ == "__main__":
    main()
