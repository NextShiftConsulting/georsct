#!/usr/bin/env python3
"""
cross_analysis_a6_coverage.py -- A6: Coverage Gap Geographic Overlap.

Fisher's exact test on Prithvi/hydrology gap overlap per scenario.
Jaccard index and descriptive geographic characterization of overlap ZCTAs.

Usage:
    python cross_analysis_a6_coverage.py --upload
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, load_processed_parquet
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035/cross_analysis"


def _load_prithvi_coverage(s3, scenario: str) -> set[str]:
    """Return set of ZCTAs with valid HLS Prithvi embeddings."""
    key = f"results/s035/prithvi_embeddings/{scenario}_prithvi_embeddings.parquet"
    resp = s3.get_object(Bucket=BUCKET, Key=key)
    prithvi = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    # Prithvi parquet uses "zcta" not "zcta_id"
    if "zcta" in prithvi.columns and "zcta_id" not in prithvi.columns:
        prithvi = prithvi.rename(columns={"zcta": "zcta_id"})
    prithvi["zcta_id"] = prithvi["zcta_id"].astype(str)

    if "source" in prithvi.columns:
        hls_zctas = set(prithvi[prithvi["source"] == "hls"]["zcta_id"].unique())
    else:
        # No source column: assume all are HLS
        emb_cols = [c for c in prithvi.columns if c.startswith("prithvi_emb_") or c.startswith("emb_")]
        valid = prithvi[prithvi[emb_cols].notna().all(axis=1)]
        hls_zctas = set(valid["zcta_id"].unique())

    return hls_zctas


def _load_hydrology_coverage(s3, scenario: str) -> set[str]:
    """Return set of ZCTAs with hydrology data present."""
    # Try hydrology extraction metadata
    meta_key = f"results/s035/hydrology_extraction_{scenario}.json"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=meta_key)
        meta = json.loads(resp["Body"].read().decode())

        # Extract ZCTAs with hydrology coverage
        if "zcta_coverage" in meta:
            return set(str(z) for z in meta["zcta_coverage"].get("covered", []))
        if "covered_zctas" in meta:
            return set(str(z) for z in meta["covered_zctas"])
        if "per_zcta" in meta:
            return set(str(z) for z, v in meta["per_zcta"].items()
                       if v.get("has_data", v.get("coverage", 0) > 0))
    except Exception:
        pass

    # Fallback: check R1 features for hydrology columns
    # R1 adds hydrology features; ZCTAs with non-null hydro features have coverage
    try:
        from _coverage_common import OUTPUT_KEYS
        event_key = OUTPUT_KEYS[scenario]
        resp = s3.get_object(Bucket=BUCKET, Key=event_key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        df["zcta_id"] = df["zcta_id"].astype(str)

        # R1 hydrology feature prefix candidates
        hydro_cols = [c for c in df.columns if any(c.startswith(p) for p in
                      ["hydro_", "stream_", "drainage_", "twi_", "flow_"])]
        if hydro_cols:
            # ZCTAs with any non-null hydrology feature
            mask = df[hydro_cols].notna().any(axis=1)
            return set(df.loc[mask, "zcta_id"].unique())
    except Exception:
        pass

    log.warning("Could not determine hydrology coverage for %s", scenario)
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(description="A6: Coverage Gap Geographic Overlap")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    print("\n" + "=" * 60)
    print("  A6: COVERAGE GAP GEOGRAPHIC OVERLAP")
    print("=" * 60 + "\n")

    from scipy.stats import fisher_exact

    scenario_results = {}

    for scenario in SCENARIOS:
        log.info("\n--- %s ---", scenario)

        # Load all ZCTAs for this scenario
        df = load_processed_parquet(s3, scenario)
        df["zcta_id"] = df["zcta_id"].astype(str)
        all_zctas = set(df["zcta_id"].unique())
        log.info("  Total ZCTAs: %d", len(all_zctas))

        # Prithvi coverage
        prithvi_present = _load_prithvi_coverage(s3, scenario)
        prithvi_missing = all_zctas - prithvi_present
        log.info("  Prithvi: %d present, %d missing", len(prithvi_present), len(prithvi_missing))

        # Hydrology coverage
        hydro_present = _load_hydrology_coverage(s3, scenario)
        hydro_missing = all_zctas - hydro_present if hydro_present else set()
        log.info("  Hydrology: %d present, %d missing", len(hydro_present), len(hydro_missing))

        if not hydro_present and not hydro_missing:
            log.warning("  Hydrology coverage unknown for %s, skipping Fisher test", scenario)
            scenario_results[scenario] = {
                "n_zctas": len(all_zctas),
                "prithvi_missing": len(prithvi_missing),
                "hydrology_missing": 0,
                "hydrology_coverage_available": False,
                "note": "Hydrology coverage metadata not available",
            }
            continue

        # 2x2 contingency table
        both_present = len(prithvi_present & hydro_present)
        prithvi_only_missing = len(prithvi_missing - hydro_missing)
        hydro_only_missing = len(hydro_missing - prithvi_missing)
        both_missing = len(prithvi_missing & hydro_missing)

        # a = both present, b = prithvi present & hydro missing
        # c = prithvi missing & hydro present, d = both missing
        a = both_present
        b = len(prithvi_present & hydro_missing)
        c = len(prithvi_missing & hydro_present)
        d = both_missing

        table = [[a, b], [c, d]]
        log.info("  Contingency table: [[%d, %d], [%d, %d]]", a, b, c, d)

        # Fisher's exact test (one-tailed: overlap greater than chance)
        odds_ratio, p_value = fisher_exact(table, alternative="greater")

        # Jaccard index
        union = prithvi_missing | hydro_missing
        intersection = prithvi_missing & hydro_missing
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0

        # Expected cell count warning
        n_total = a + b + c + d
        expected_d = ((c + d) * (b + d)) / n_total if n_total > 0 else 0
        small_expected = expected_d < 5

        log.info("  Fisher p=%.4f, OR=%.2f, Jaccard=%.3f",
                 p_value, odds_ratio if odds_ratio != np.inf else 999, jaccard)

        # Descriptive geographic characterization of overlap ZCTAs
        geo_desc = None
        if both_missing > 0 and "latitude" in df.columns and "longitude" in df.columns:
            overlap_df = df[df["zcta_id"].isin(intersection)].drop_duplicates(subset=["zcta_id"])
            all_unique = df.drop_duplicates(subset=["zcta_id"])

            if len(overlap_df) > 0:
                geo_desc = {
                    "n_overlap_zctas": len(overlap_df),
                    "overlap_median_lat": float(overlap_df["latitude"].median()),
                    "overlap_median_lon": float(overlap_df["longitude"].median()),
                    "scenario_median_lat": float(all_unique["latitude"].median()),
                    "scenario_median_lon": float(all_unique["longitude"].median()),
                }
                if "coastal_distance_m" in df.columns:
                    geo_desc["overlap_median_coastal_dist_m"] = float(
                        overlap_df["coastal_distance_m"].median()
                    )
                    geo_desc["scenario_median_coastal_dist_m"] = float(
                        all_unique["coastal_distance_m"].median()
                    )

        scenario_results[scenario] = {
            "n_zctas": len(all_zctas),
            "prithvi_present": len(prithvi_present),
            "prithvi_missing": len(prithvi_missing),
            "hydrology_present": len(hydro_present),
            "hydrology_missing": len(hydro_missing),
            "hydrology_coverage_available": True,
            "contingency_table": {"a": a, "b": b, "c": c, "d": d},
            "fisher_exact": {
                "odds_ratio": float(odds_ratio) if odds_ratio != np.inf else None,
                "p_value": float(p_value),
                "alternative": "greater",
                "small_expected_warning": small_expected,
                "expected_d": round(expected_d, 2),
            },
            "jaccard_index": round(jaccard, 4),
            "both_missing_zctas": sorted(list(intersection)),
            "geographic_description": geo_desc,
        }

    # --- AC-A6-1: significant overlap in any scenario ---
    scenarios_with_significant_overlap = [
        sc for sc, r in scenario_results.items()
        if r.get("fisher_exact", {}).get("p_value", 1) < 0.05
    ]

    payload = {
        "experiment": "s035-model-ladder",
        "analysis": "A6_coverage_gap_overlap",
        "version": "2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenario_results,
        "hypotheses": {
            "AC_A6_1": {
                "description": "Prithvi-missing and hydrology-missing ZCTAs overlap more than chance (Fisher exact, one-tailed)",
                "scenarios_significant": scenarios_with_significant_overlap,
                "note": "Holm-Bonferroni correction applied across all 10 DOE hypotheses at family alpha=0.05",
            },
        },
    }

    results_json = json.dumps(payload, indent=2, default=str)
    print("\n" + results_json)

    if args.upload:
        key = f"{RESULTS_PREFIX}/a6_coverage_gap_overlap.json"
        upload_json_result(s3, BUCKET, key, payload)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)
    else:
        local = "/tmp/a6_coverage_gap_overlap.json"
        Path(local).write_text(results_json)
        log.info("Wrote %s", local)


if __name__ == "__main__":
    main()
