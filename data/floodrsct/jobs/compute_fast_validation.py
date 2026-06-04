#!/usr/bin/env python3
# =============================================================================
# PROVENANCE:
#   generator: deepseek/deepseek-chat-v3.1 (via OpenRouter)
#   cleanup_by: Martin
#   cleanup_summary: Strip markdown fences, fix obs ceiling column mismatch,
#       fix None format-string crash, genericize spearman_rho signature
#   see: ../exp/s035-model-ladder/SCRIPT_PROVENANCE.yaml
# =============================================================================
"""compute_fast_validation.py -- Phase 7b: FAST external validation.

Compares s035 model-ladder predictions (R0/R1/R2) against FEMA FAST
engineering damage estimates. Tests H6: do representation upgrades
capture physical flood damage signal?

For each (scenario, return_period):
  1. Load FAST ZCTA aggregates (fast_total_loss_zcta)
  2. Load model predictions at each level (R0, R1, R2)
  3. Compute Spearman rho(pred, FAST) at each level
  4. Compute Spearman rho(obs_nfip, FAST) as ceiling

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
FAST_SCENARIOS = ["houston", "nyc"]
RETURN_PERIODS = ["10yr", "50yr", "100yr", "500yr"]
LEVELS = ["r0", "r1", "r2"]
PRIMARY_TARGET = "obs_nfip_event_claims"
PRIMARY_SOLVER = "histgbdt"


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
        log.error("Missing fast_total_loss_zcta in %s", key)
        return None
    df["zcta_id"] = df["zcta_id"].astype(str)
    return df[["zcta_id", "fast_total_loss_zcta"]].copy()


def load_predictions(s3, level: str, scenario: str) -> pd.DataFrame | None:
    """Load model predictions, filter to primary target/solver, aggregate per ZCTA."""
    key = f"{RESULTS_PREFIX}/{level_prefix(level)}_{scenario}_predictions.parquet"
    df = _load_parquet(s3, key)
    if df is None:
        return None
    mask = (df["target"] == PRIMARY_TARGET) & (df["solver"] == PRIMARY_SOLVER)
    df = df.loc[mask]
    if df.empty:
        log.warning("No predictions for %s/%s/%s", level, PRIMARY_TARGET, PRIMARY_SOLVER)
        return None
    # Mean prediction per ZCTA across folds
    agg = df.groupby("zcta_id", as_index=False).agg(
        y_pred=("y_pred", "mean"),
        y_true=("y_true", "first"),
    )
    agg["zcta_id"] = agg["zcta_id"].astype(str)
    return agg


def spearman_rho(fast_df: pd.DataFrame, values: pd.Series, label: str):
    """Compute Spearman rho between FAST losses and a value series.

    Returns (rho, p_value, n) or (None, None, 0).
    """
    merged = fast_df[["zcta_id", "fast_total_loss_zcta"]].merge(
        values.rename("val").to_frame().assign(zcta_id=values.index),
        on="zcta_id",
        how="inner",
    )
    merged = merged.dropna(subset=["fast_total_loss_zcta", "val"])
    n = len(merged)
    if n < 3:
        log.warning("Too few matched ZCTAs for %s (n=%d)", label, n)
        return None, None, n
    rho, p = stats.spearmanr(merged["fast_total_loss_zcta"], merged["val"])
    return float(rho), float(p), n


def validate_scenario_rp(s3, scenario: str, rp: str, pred_cache: dict) -> dict:
    """Run validation for one (scenario, return_period) pair."""
    row = {"scenario": scenario, "return_period": rp}

    fast_df = load_fast_zcta(s3, scenario, rp)
    if fast_df is None:
        row["error"] = "FAST data not available"
        return row

    n_fast = len(fast_df)
    row["n_fast_zctas"] = n_fast

    # Load or reuse predictions per level
    for level in LEVELS:
        cache_key = (level, scenario)
        if cache_key not in pred_cache:
            pred_cache[cache_key] = load_predictions(s3, level, scenario)
        pred_df = pred_cache[cache_key]

        if pred_df is None:
            row[f"rho_{level}_fast"] = None
            row[f"p_{level}"] = None
            continue

        vals = pred_df.set_index("zcta_id")["y_pred"]
        rho, p, n = spearman_rho(fast_df, vals, f"{level} vs FAST")
        row[f"rho_{level}_fast"] = rho
        row[f"p_{level}"] = p
        row["n_matched"] = n

    # Ceiling: obs NFIP vs FAST (use y_true from R0)
    r0_df = pred_cache.get(("r0", scenario))
    if r0_df is not None:
        obs_vals = r0_df.set_index("zcta_id")["y_true"]
        rho_obs, p_obs, n_obs = spearman_rho(fast_df, obs_vals, "obs vs FAST")
        row["rho_nfip_obs_fast"] = rho_obs
        row["p_nfip_obs"] = p_obs
    else:
        row["rho_nfip_obs_fast"] = None

    return row


def check_robustness(table: list[dict]) -> dict:
    """Check whether level ranking is consistent across return periods."""
    summary = {}
    for level in LEVELS:
        key = f"rho_{level}_fast"
        vals = [r[key] for r in table if r.get(key) is not None]
        if vals:
            summary[level] = {
                "mean_rho": round(float(np.mean(vals)), 4),
                "min_rho": round(float(np.min(vals)), 4),
                "max_rho": round(float(np.max(vals)), 4),
                "n_comparisons": len(vals),
            }

    # Ranking consistency: does the best level stay best across return periods?
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

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 7b: FAST external validation")
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, skip execution")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would validate %s x %s", FAST_SCENARIOS, RETURN_PERIODS)
        log.info("Reads: FAST parquets + prediction parquets for R0/R1/R2")
        log.info("Writes: %s/fast_validation.json", RESULTS_PREFIX)
        return 0

    s3 = get_s3_client()
    pred_cache: dict = {}
    table: list[dict] = []

    for scenario in FAST_SCENARIOS:
        for rp in RETURN_PERIODS:
            log.info("--- %s / %s ---", scenario, rp)
            row = validate_scenario_rp(s3, scenario, rp, pred_cache)
            table.append(row)
            rho_r2 = row.get("rho_r2_fast")
            rho_obs = row.get("rho_nfip_obs_fast")
            log.info(
                "  rho(R2,FAST)=%s  rho(obs,FAST)=%s  n=%s",
                f"{rho_r2:.3f}" if rho_r2 is not None else "N/A",
                f"{rho_obs:.3f}" if rho_obs is not None else "N/A",
                row.get("n_matched", "?"),
            )

    robustness = check_robustness(table)

    result = {
        "phase": "7b_fast_validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": FAST_SCENARIOS,
        "return_periods": RETURN_PERIODS,
        "primary_target": PRIMARY_TARGET,
        "primary_solver": PRIMARY_SOLVER,
        "validation_table": table,
        "robustness": robustness,
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
    log.info("\n=== FAST Validation Summary ===")
    for level, info in robustness.items():
        if isinstance(info, dict) and "mean_rho" in info:
            log.info("  %s: mean rho = %.3f [%.3f, %.3f] (n=%d)",
                     level, info["mean_rho"], info["min_rho"],
                     info["max_rho"], info["n_comparisons"])
    if "ranking_consistent" in robustness:
        log.info("  Ranking consistent: %s", robustness["ranking_consistent"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
