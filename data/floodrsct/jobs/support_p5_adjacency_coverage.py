#!/usr/bin/env python3
"""
audit_a5_adjacency_coverage.py -- Stage 5 Audit A5: Adjacency coverage.

Checks whether the relational propagation probe (G6) has enough
adjacency edges where both endpoints have populated features. The
logical probe computes Fisher discriminant ratio on Queen's contiguity
neighbors; edges where one or both endpoints have null features are
dead edges that reduce statistical power.

Loads the zcta_adjacency.parquet edge list and checks what fraction
of edges have both endpoints present in the processed parquet with
non-null core features.

PASS if edge_coverage >= 0.5 (at least half of adjacency edges are
usable).

Usage:
    python audit_a5_adjacency_coverage.py --scenario houston
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import (
    AuditResult, SCENARIOS, get_s3_client, load_processed_parquet,
    load_adjacency, write_evidence,
)

CORE_FEATURES = ["rainfall_total_mm", "total_rainfall_mm"]
MIN_EDGE_COVERAGE = 0.5


def audit(scenario: str, min_support: int, upload: bool) -> list[AuditResult]:
    s3 = get_s3_client()
    ts = datetime.now(timezone.utc).isoformat()
    df = load_processed_parquet(s3, scenario)

    try:
        adj = load_adjacency(s3)
    except FileNotFoundError as e:
        result = AuditResult(
            audit_id="P5", scenario=scenario, probe="relational_propagation",
            status="FAIL",
            detail={"error": str(e)},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "a5_adjacency", scenario, s3=s3, upload=upload)
        return [result]

    # Find rainfall column
    rain_col = None
    for c in CORE_FEATURES:
        if c in df.columns:
            rain_col = c
            break

    # Get set of ZCTAs with non-null core features (across any event)
    df["zcta_id"] = df["zcta_id"].astype(str)
    if rain_col:
        populated = set(df.loc[df[rain_col].notna(), "zcta_id"].unique())
    else:
        populated = set(df["zcta_id"].unique())

    all_zctas = set(df["zcta_id"].unique())

    # Identify adjacency columns (try common naming patterns)
    adj_cols = adj.columns.tolist()
    if len(adj_cols) >= 2:
        src_col, dst_col = adj_cols[0], adj_cols[1]
    else:
        result = AuditResult(
            audit_id="P5", scenario=scenario, probe="relational_propagation",
            status="FAIL",
            detail={"error": f"Adjacency has unexpected columns: {adj_cols}"},
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "a5_adjacency", scenario, s3=s3, upload=upload)
        return [result]

    adj[src_col] = adj[src_col].astype(str)
    adj[dst_col] = adj[dst_col].astype(str)

    # Filter adjacency to edges within this scenario's ZCTAs
    scenario_edges = adj[
        adj[src_col].isin(all_zctas) & adj[dst_col].isin(all_zctas)
    ]

    total_edges = len(scenario_edges)
    if total_edges == 0:
        result = AuditResult(
            audit_id="P5", scenario=scenario, probe="relational_propagation",
            status="FAIL",
            detail={
                "error": "No adjacency edges found for this scenario's ZCTAs",
                "scenario_zctas": len(all_zctas),
                "adjacency_total_edges": len(adj),
            },
            min_support=min_support, timestamp=ts,
        )
        write_evidence([result], "a5_adjacency", scenario, s3=s3, upload=upload)
        return [result]

    # Count live edges (both endpoints populated)
    live_edges = scenario_edges[
        scenario_edges[src_col].isin(populated) &
        scenario_edges[dst_col].isin(populated)
    ]

    coverage = len(live_edges) / total_edges
    status = "PASS" if coverage >= MIN_EDGE_COVERAGE else "FAIL"

    result = AuditResult(
        audit_id="P5", scenario=scenario, probe="relational_propagation",
        status=status,
        detail={
            "scenario_zctas": len(all_zctas),
            "populated_zctas": len(populated),
            "total_edges": total_edges,
            "live_edges": len(live_edges),
            "dead_edges": total_edges - len(live_edges),
            "edge_coverage": round(coverage, 4),
            "threshold": MIN_EDGE_COVERAGE,
            "rain_col": rain_col,
        },
        min_support=min_support, timestamp=ts,
    )

    write_evidence([result], "a5_adjacency", scenario, s3=s3, upload=upload)
    return [result]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A5: Adjacency coverage")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    audit(args.scenario, args.min_support, args.upload)


if __name__ == "__main__":
    main()
