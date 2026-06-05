#!/usr/bin/env python3
# =============================================================================
# PROVENANCE:
#   generator: deepseek/deepseek-chat-v3.1 (via OpenRouter)
#   cleanup_by: Martin
#   cleanup_summary: Strip markdown fences, fix obs ceiling column mismatch,
#       fix None format-string crash, genericize spearman_rho signature
#   v2_by: Martin (2026-06-05)
#   v2_summary: Event-level matching instead of ZCTA-level collapse.
#       Each historical event matched to closest synthetic return period
#       before computing Spearman rho(pred, FAST).
#   see: ../exp/s035-model-ladder/SCRIPT_PROVENANCE.yaml
# =============================================================================
"""compute_fast_validation.py -- Phase 7b: FAST external validation.

Compares s035 model-ladder predictions (R0/R1/R2) against FEMA FAST
engineering damage estimates using **event-level matching**.

For each (scenario, event):
  1. Map the historical event to its closest synthetic return period
     (pluvial: FloodSimBench rainfall ARI; surge: SLOSH Saffir-Simpson)
  2. Load FAST ZCTA aggregates for that return period
  3. Load model predictions for that specific event at each level (R0, R1, R2)
  4. Compute Spearman rho(pred, FAST_loss) at each level
  5. Compute Spearman rho(obs_nfip, FAST_loss) as ceiling

This avoids the semantic mismatch of collapsing multiple events with
different intensities into one ZCTA-level average before correlation.

Usage:
    python compute_fast_validation.py --upload
    python compute_fast_validation.py --dry-run
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, get_s3_client, level_prefix
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
FAST_SCENARIOS = ["houston", "nyc", "southwest_florida"]
LEVELS = ["r0", "r1", "r2"]
PRIMARY_TARGET = "obs_nfip_event_claims"
PRIMARY_SOLVER = "histgbdt"

# ---------------------------------------------------------------------------
# Event-to-return-period mapping
# ---------------------------------------------------------------------------
# Each historical event is matched to its closest synthetic hazard scenario.
#
# Houston/NYC: FloodSimBench uses 6-hour design rainfall (mm) to define
# return periods. We match by comparing observed MRMS peak 6-hour totals
# against the FloodSimBench bins. These assignments are documented in
# Appendix D of the paper.
#
# SW Florida: SLOSH MOM is category-based (Saffir-Simpson), not ARI-based.
# Each event is matched to its landfall category.
#
# Assignments are FIXED before Phase 7b runs -- no data-adaptive selection.
# ---------------------------------------------------------------------------

EVENT_RETURN_PERIOD: dict[str, dict[str, str]] = {
    "houston": {
        # Harvey 2017: >1000mm total, peak 6h ~180mm -> 1000yr (181mm bin)
        "harvey2017": "1000yr",
        # Imelda 2019: ~250mm total, peak 6h ~140mm -> 200yr (138mm bin)
        "imelda2019": "200yr",
        # Beryl 2024: ~120mm total, peak 6h ~100mm -> 25yr (98mm bin)
        "beryl2024": "25yr",
    },
    "nyc": {
        # Ida 2021 NYC: ~80mm in 1h (Central Park record), peak 6h ~120mm
        # -> 100yr (123mm bin). FloodSimBench coverage Manhattan only.
        "ida2021": "100yr",
        # Henri 2021: ~50mm total, peak 6h ~45mm -> 1yr (48mm bin)
        "henri2021": "1yr",
    },
    "southwest_florida": {
        # Ian 2022: Cat 4 at landfall (Fort Myers Beach) -> SLOSH cat4
        "ian2022": "cat4",
        # Helene 2024: Cat 4 at landfall (Big Bend) -> SLOSH cat4
        "helene2024": "cat4",
        # Milton 2024: Cat 3 at landfall (Siesta Key) -> SLOSH cat3
        "milton2024": "cat3",
    },
}

# Justification metadata for each assignment (for paper appendix)
EVENT_MATCHING_NOTES: dict[str, dict[str, str]] = {
    "houston": {
        "harvey2017": "Peak 6h MRMS ~180mm; FloodSimBench 1000yr=181mm/6h",
        "imelda2019": "Peak 6h MRMS ~140mm; FloodSimBench 200yr=138mm/6h",
        "beryl2024": "Peak 6h MRMS ~100mm; FloodSimBench 25yr=98mm/6h",
    },
    "nyc": {
        "ida2021": "Peak 6h MRMS ~120mm; FloodSimBench 100yr=123mm/6h (Manhattan only)",
        "henri2021": "Peak 6h MRMS ~45mm; FloodSimBench 1yr=48mm/6h",
    },
    "southwest_florida": {
        "ian2022": "Cat 4 at landfall; SLOSH MOM Category 4 HIGH tide",
        "helene2024": "Cat 4 at landfall; SLOSH MOM Category 4 HIGH tide",
        "milton2024": "Cat 3 at landfall; SLOSH MOM Category 3 HIGH tide",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_parquet(s3, key: str) -> pd.DataFrame | None:
    """Load parquet from S3, return None on failure."""
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(resp["Body"].read()))
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


def load_fast_zcta(s3, scenario: str, return_period: str) -> pd.DataFrame | None:
    """Load FAST ZCTA aggregates for a scenario + return period."""
    key = f"processed/{scenario}/{scenario}_fast_zcta_{return_period}.parquet"
    df = _load_parquet(s3, key)
    if df is None:
        return None
    if "fast_total_loss_zcta" not in df.columns:
        # Try alternative column name
        loss_cols = [c for c in df.columns if "loss" in c.lower() and "total" in c.lower()]
        if loss_cols:
            df = df.rename(columns={loss_cols[0]: "fast_total_loss_zcta"})
        else:
            log.error("Missing fast_total_loss_zcta in %s (cols: %s)", key, list(df.columns))
            return None
    df["zcta_id"] = df["zcta_id"].astype(str)
    return df[["zcta_id", "fast_total_loss_zcta"]].copy()


def load_predictions_by_event(
    s3, level: str, scenario: str, event: str,
) -> pd.DataFrame | None:
    """Load model predictions filtered to a specific event.

    Returns one row per ZCTA with mean prediction across folds
    for the given (target, solver, event) combination.
    """
    key = f"{RESULTS_PREFIX}/{level_prefix(level)}_{scenario}_predictions.parquet"
    df = _load_parquet(s3, key)
    if df is None:
        return None
    mask = (
        (df["target"] == PRIMARY_TARGET)
        & (df["solver"] == PRIMARY_SOLVER)
        & (df["event"] == event)
    )
    df = df.loc[mask]
    if df.empty:
        log.warning("No predictions for %s/%s/%s/%s", level, scenario, event, PRIMARY_SOLVER)
        return None
    # Mean prediction per ZCTA across folds (same event)
    agg = df.groupby("zcta_id", as_index=False).agg(
        y_pred=("y_pred", "mean"),
        y_true=("y_true", "first"),
        n_folds=("fold", "nunique"),
    )
    agg["zcta_id"] = agg["zcta_id"].astype(str)
    return agg


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def spearman_rho(
    fast_df: pd.DataFrame, pred_df: pd.DataFrame, value_col: str, label: str,
) -> dict:
    """Compute Spearman rho between FAST losses and a prediction/observation column.

    Returns dict with rho, p_value, n, or nulls if insufficient data.
    """
    merged = fast_df[["zcta_id", "fast_total_loss_zcta"]].merge(
        pred_df[["zcta_id", value_col]],
        on="zcta_id",
        how="inner",
    )
    merged = merged.dropna(subset=["fast_total_loss_zcta", value_col])
    n = len(merged)
    if n < 3:
        log.warning("Too few matched ZCTAs for %s (n=%d)", label, n)
        return {"rho": None, "p_value": None, "n": n}
    rho, p = stats.spearmanr(merged["fast_total_loss_zcta"], merged[value_col])
    return {"rho": float(rho), "p_value": float(p), "n": n}


# ---------------------------------------------------------------------------
# Event-level validation
# ---------------------------------------------------------------------------

def validate_event(
    s3,
    scenario: str,
    event: str,
    return_period: str,
    pred_cache: dict,
    fast_cache: dict,
) -> dict:
    """Run validation for one (scenario, event) -> return_period pair."""
    row = {
        "scenario": scenario,
        "event": event,
        "matched_return_period": return_period,
        "matching_note": EVENT_MATCHING_NOTES.get(scenario, {}).get(event, ""),
    }

    # Load FAST data for the matched return period (cached)
    fast_key = (scenario, return_period)
    if fast_key not in fast_cache:
        fast_cache[fast_key] = load_fast_zcta(s3, scenario, return_period)
    fast_df = fast_cache[fast_key]

    if fast_df is None:
        row["error"] = f"FAST data not available for {scenario}/{return_period}"
        return row

    row["n_fast_zctas"] = len(fast_df)

    # Load predictions per level, filtered to this event
    for level in LEVELS:
        cache_key = (level, scenario, event)
        if cache_key not in pred_cache:
            pred_cache[cache_key] = load_predictions_by_event(s3, level, scenario, event)
        pred_df = pred_cache[cache_key]

        if pred_df is None:
            row[f"rho_{level}_fast"] = None
            row[f"p_{level}"] = None
            row[f"n_{level}"] = 0
            continue

        result = spearman_rho(fast_df, pred_df, "y_pred", f"{level}/{event} vs FAST")
        row[f"rho_{level}_fast"] = result["rho"]
        row[f"p_{level}"] = result["p_value"]
        row[f"n_{level}"] = result["n"]

    # Ceiling: obs NFIP vs FAST (use y_true from R0 for this event)
    r0_key = ("r0", scenario, event)
    r0_df = pred_cache.get(r0_key)
    if r0_df is not None:
        obs_result = spearman_rho(fast_df, r0_df, "y_true", f"obs/{event} vs FAST")
        row["rho_nfip_obs_fast"] = obs_result["rho"]
        row["p_nfip_obs"] = obs_result["p_value"]
        row["n_obs"] = obs_result["n"]
    else:
        row["rho_nfip_obs_fast"] = None
        row["p_nfip_obs"] = None
        row["n_obs"] = 0

    return row


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------

def check_robustness(table: list[dict]) -> dict:
    """Check whether level ranking is consistent across events."""
    summary = {}

    # Per-level aggregate statistics
    for level in LEVELS:
        key = f"rho_{level}_fast"
        vals = [r[key] for r in table if r.get(key) is not None]
        if vals:
            summary[level] = {
                "mean_rho": round(float(np.mean(vals)), 4),
                "median_rho": round(float(np.median(vals)), 4),
                "min_rho": round(float(np.min(vals)), 4),
                "max_rho": round(float(np.max(vals)), 4),
                "n_events": len(vals),
            }

    # Ranking consistency: does the best level stay best across events?
    rankings = []
    for row in table:
        rhos = {lv: row.get(f"rho_{lv}_fast") for lv in LEVELS}
        rhos = {k: v for k, v in rhos.items() if v is not None}
        if rhos:
            rankings.append(max(rhos, key=rhos.get))
    if rankings:
        from collections import Counter
        counts = Counter(rankings)
        summary["best_level_counts"] = dict(counts)
        summary["ranking_consistent"] = counts.most_common(1)[0][1] == len(rankings)

    # Per-scenario breakdown
    scenarios_seen = sorted(set(r["scenario"] for r in table))
    per_scenario = {}
    for sc in scenarios_seen:
        sc_rows = [r for r in table if r["scenario"] == sc]
        sc_summary = {}
        for level in LEVELS:
            key = f"rho_{level}_fast"
            vals = [r[key] for r in sc_rows if r.get(key) is not None]
            if vals:
                sc_summary[level] = round(float(np.mean(vals)), 4)
        per_scenario[sc] = sc_summary
    summary["per_scenario"] = per_scenario

    return summary


# ---------------------------------------------------------------------------
# ZCTA-collapsed comparison (legacy, for appendix disclosure)
# ---------------------------------------------------------------------------

def collapsed_comparison(s3, pred_cache: dict, fast_cache: dict) -> list[dict]:
    """Run the naive ZCTA-collapsed correlation for comparison.

    This collapses all events into one ZCTA-level average (the v1 approach)
    and reports alongside event-matched results so the reader can see the
    impact of event-level matching.
    """
    rows = []
    for scenario in FAST_SCENARIOS:
        events = list(EVENT_RETURN_PERIOD.get(scenario, {}).keys())
        return_periods = sorted(set(EVENT_RETURN_PERIOD.get(scenario, {}).values()))

        for rp in return_periods:
            fast_key = (scenario, rp)
            fast_df = fast_cache.get(fast_key)
            if fast_df is None:
                continue

            for level in LEVELS:
                # Collect all event predictions and average across events
                all_preds = []
                for event in events:
                    cache_key = (level, scenario, event)
                    pred_df = pred_cache.get(cache_key)
                    if pred_df is not None:
                        all_preds.append(pred_df[["zcta_id", "y_pred", "y_true"]])

                if not all_preds:
                    continue

                combined = pd.concat(all_preds)
                collapsed = combined.groupby("zcta_id", as_index=False).agg(
                    y_pred=("y_pred", "mean"),
                    y_true=("y_true", "first"),
                )

                result = spearman_rho(fast_df, collapsed, "y_pred",
                                      f"collapsed {level}/{scenario}/{rp}")
                rows.append({
                    "scenario": scenario,
                    "return_period": rp,
                    "level": level,
                    "method": "collapsed",
                    "rho": result["rho"],
                    "p_value": result["p_value"],
                    "n": result["n"],
                })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 7b: FAST external validation")
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, skip execution")
    parser.add_argument("--include-collapsed", action="store_true",
                        help="Also run collapsed (v1) comparison for appendix disclosure")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would validate %d event-return_period pairs",
                 sum(len(v) for v in EVENT_RETURN_PERIOD.values()))
        for sc, events in EVENT_RETURN_PERIOD.items():
            for ev, rp in events.items():
                log.info("  %s / %s -> %s", sc, ev, rp)
        log.info("Reads: FAST parquets + prediction parquets for R0/R1/R2")
        log.info("Writes: %s/fast_validation.json", RESULTS_PREFIX)
        return 0

    s3 = get_s3_client()
    pred_cache: dict = {}
    fast_cache: dict = {}
    table: list[dict] = []

    for scenario in FAST_SCENARIOS:
        events = EVENT_RETURN_PERIOD.get(scenario, {})
        for event, return_period in events.items():
            log.info("--- %s / %s -> %s ---", scenario, event, return_period)
            row = validate_event(s3, scenario, event, return_period,
                                 pred_cache, fast_cache)
            table.append(row)

            rho_r2 = row.get("rho_r2_fast")
            rho_obs = row.get("rho_nfip_obs_fast")
            n = row.get("n_r2", "?")
            log.info(
                "  rho(R2,FAST)=%s  rho(obs,FAST)=%s  n=%s",
                f"{rho_r2:.3f}" if rho_r2 is not None else "N/A",
                f"{rho_obs:.3f}" if rho_obs is not None else "N/A",
                n,
            )

    robustness = check_robustness(table)

    result = {
        "phase": "7b_fast_validation",
        "version": "2.0_event_matched",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "Event-level matching: each historical event mapped to closest "
                  "synthetic return period before computing Spearman rho(pred, FAST_loss)",
        "event_mapping": EVENT_RETURN_PERIOD,
        "event_matching_notes": EVENT_MATCHING_NOTES,
        "scenarios": FAST_SCENARIOS,
        "levels": LEVELS,
        "primary_target": PRIMARY_TARGET,
        "primary_solver": PRIMARY_SOLVER,
        "validation_table": table,
        "robustness": robustness,
    }

    # Optional: include collapsed comparison for appendix disclosure
    if args.include_collapsed:
        collapsed = collapsed_comparison(s3, pred_cache, fast_cache)
        result["collapsed_comparison"] = {
            "method": "All events averaged per ZCTA then correlated (v1 approach)",
            "note": "Included for transparency; event-matched is primary",
            "table": collapsed,
        }

    # Write local copy
    out_path = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_path.mkdir(parents=True, exist_ok=True)
    local_file = out_path / "fast_validation.json"
    with open(local_file, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Written to %s", local_file)

    if args.upload:
        key = f"{RESULTS_PREFIX}/fast_validation.json"
        upload_json_result(s3, BUCKET, key, result)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)

    # Summary
    print(f"\n{'='*70}")
    print("  FAST VALIDATION SUMMARY (Event-Matched)")
    print(f"{'='*70}\n")

    for row in table:
        sc = row["scenario"]
        ev = row["event"]
        rp = row["matched_return_period"]
        print(f"  {sc:25s} {ev:15s} -> {rp:8s}")
        for level in LEVELS:
            rho = row.get(f"rho_{level}_fast")
            p = row.get(f"p_{level}")
            n = row.get(f"n_{level}", 0)
            rho_str = f"{rho:.3f}" if rho is not None else "N/A"
            p_str = f"p={p:.3f}" if p is not None else "p=N/A"
            print(f"    {level.upper()}: rho={rho_str}  {p_str}  n={n}")
        rho_obs = row.get("rho_nfip_obs_fast")
        if rho_obs is not None:
            print(f"    OBS: rho={rho_obs:.3f}  (ceiling)")
        print()

    print("--- Robustness ---")
    for level, info in robustness.items():
        if isinstance(info, dict) and "mean_rho" in info:
            print(f"  {level.upper()}: mean rho = {info['mean_rho']:.3f} "
                  f"[{info['min_rho']:.3f}, {info['max_rho']:.3f}] "
                  f"(n_events={info['n_events']})")
    if "ranking_consistent" in robustness:
        print(f"  Ranking consistent across events: {robustness['ranking_consistent']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
