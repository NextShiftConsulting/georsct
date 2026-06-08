#!/usr/bin/env python3
"""compute_paired_loss.py -- Spatially-blocked paired loss (V4.5 / V4.10).

Implements the contract V4.5 requirement: spatially-blocked paired comparison
using county-level spatial blocks. Two-stage aggregation per V4.10:
  Stage 1: (zcta_id, event) -> per-ZCTA mean across events
  Stage 2: per-ZCTA -> per-county (block) mean

Computes:
  - Per-county paired metric delta (R1 - R0, R2 - R1, R2 - R0)
  - Wilcoxon signed-rank on county-level deltas
  - Exact permutation p-value (when n_blocks <= 20)
  - Block bootstrap 95% CI on mean delta
  - Matched-pairs rank-biserial effect size

Outputs: results/s035/spatially_blocked_paired_loss.json

Usage:
    python compute_paired_loss.py --upload
    python compute_paired_loss.py --dry-run
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jobs"))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, level_prefix, load_crosswalk
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
N_BOOTSTRAP = 2000
SEED = 42

# Target metadata from contract
TARGETS = {
    "obs_nfip_event_claims": {"task": "regression", "metric": "r2"},
    "obs_has_311": {"task": "binary_classification", "metric": "roc_auc"},
    "obs_has_hwm": {"task": "binary_classification", "metric": "roc_auc"},
}

# Comparisons to run
COMPARISONS = [
    ("r0", "r1"),
    ("r1", "r2"),
    ("r0", "r2"),
]


def load_predictions(s3, level: str, scenario: str) -> Optional[pd.DataFrame]:
    """Load prediction parquet from S3 for a given level and scenario.

    Args:
        s3: boto3 S3 client.
        level: Representation level (r0, r1, r2).
        scenario: Scenario name.

    Returns:
        DataFrame with columns [zcta_id, target, solver, fold, y_true, y_pred, event]
        or None if not found.
    """
    prefix = level_prefix(level)
    key = f"{RESULTS_PREFIX}/{prefix}_{scenario}_predictions.parquet"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        log.info("Loaded %s (%d rows)", key, len(df))
        return df
    except Exception as exc:
        log.warning("Could not load %s: %s", key, exc)
        return None


def compute_fold_metric(
    df: pd.DataFrame, target: str, task: str, fold: str,
) -> Optional[float]:
    """Compute metric for one fold of one target.

    Args:
        df: Predictions subset for a single fold.
        target: Target column name.
        task: 'regression' or 'binary_classification'.
        fold: Fold identifier (for logging).

    Returns:
        Metric value or None if insufficient data.
    """
    if len(df) < 5:
        return None
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    if task == "regression":
        return float(r2_score(y_true, y_pred))
    # binary_classification
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_pred))


def two_stage_aggregate(
    preds: pd.DataFrame,
    crosswalk: pd.DataFrame,
    target: str,
    task: str,
    solver: str = "histgbdt",
    fold: str = "fold_0",
) -> pd.DataFrame:
    """Two-stage aggregation: (zcta, event) -> zcta mean -> county mean.

    Per V4.10: first average predictions within each ZCTA across events,
    then average ZCTAs within each county (spatial block).

    Args:
        preds: Prediction DataFrame with zcta_id, target, solver, fold, event, y_true, y_pred.
        crosswalk: ZCTA-county crosswalk with zcta_id and county_fips columns.
        target: Target column name to filter on.
        task: Task type for metric computation.
        solver: Solver name to filter on.
        fold: Fold identifier to filter on.

    Returns:
        DataFrame with one row per county, columns: county_fips, metric_value, n_zctas.
    """
    mask = (
        (preds["target"] == target)
        & (preds["solver"] == solver)
        & (preds["fold"] == fold)
    )
    subset = preds.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame()

    # Stage 1: per-ZCTA mean across events
    zcta_agg = subset.groupby("zcta_id").agg(
        y_true_mean=("y_true", "mean"),
        y_pred_mean=("y_pred", "mean"),
    ).reset_index()

    # Join county
    zcta_agg = zcta_agg.merge(
        crosswalk[["zcta_id", "county_fips"]].drop_duplicates(),
        on="zcta_id",
        how="left",
    )
    zcta_agg = zcta_agg.dropna(subset=["county_fips"])

    # Stage 2: per-county metric
    county_rows = []
    for county, grp in zcta_agg.groupby("county_fips"):
        if len(grp) < 3:
            continue
        y_t = grp["y_true_mean"].values
        y_p = grp["y_pred_mean"].values

        if task == "regression":
            metric_val = float(r2_score(y_t, y_p))
        else:
            if len(np.unique(y_t.round())) < 2:
                continue
            metric_val = float(roc_auc_score(y_t.round(), y_p))

        county_rows.append({
            "county_fips": county,
            "metric_value": metric_val,
            "n_zctas": len(grp),
        })

    return pd.DataFrame(county_rows)


def rank_biserial(deltas: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (effect size for Wilcoxon).

    Args:
        deltas: Array of paired differences.

    Returns:
        Rank-biserial r in [-1, 1].
    """
    n = len(deltas)
    if n == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(deltas))
    r_plus = np.sum(ranks[deltas > 0])
    r_minus = np.sum(ranks[deltas < 0])
    return float((r_plus - r_minus) / (r_plus + r_minus)) if (r_plus + r_minus) > 0 else 0.0


def exact_permutation_pvalue(deltas: np.ndarray) -> Optional[float]:
    """Exact permutation p-value for paired differences.

    Only computed when n_blocks <= 20 (2^20 = 1M permutations max).
    Tests H0: mean(deltas) <= 0 vs H1: mean(deltas) > 0.

    Args:
        deltas: Array of paired differences.

    Returns:
        One-sided p-value or None if n > 20.
    """
    n = len(deltas)
    if n > 20:
        return None
    observed = np.mean(deltas)
    n_perms = 2 ** n
    count_ge = 0
    for i in range(n_perms):
        signs = np.array([(1 if (i >> j) & 1 else -1) for j in range(n)])
        perm_mean = np.mean(signs * np.abs(deltas))
        if perm_mean >= observed:
            count_ge += 1
    return float(count_ge / n_perms)


def block_bootstrap_ci(
    deltas: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05,
) -> dict:
    """Block bootstrap 95% CI on mean delta.

    Resamples county-level deltas with replacement.

    Args:
        deltas: Array of county-level paired differences.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level for CI.

    Returns:
        Dict with ci_lower, ci_upper, boot_mean, boot_se.
    """
    rng = np.random.default_rng(SEED)
    boot_means = np.array([
        np.mean(rng.choice(deltas, size=len(deltas), replace=True))
        for _ in range(n_boot)
    ])
    return {
        "ci_lower": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "boot_mean": float(np.mean(boot_means)),
        "boot_se": float(np.std(boot_means, ddof=1)),
    }


def run_paired_comparison(
    county_a: pd.DataFrame,
    county_b: pd.DataFrame,
) -> dict:
    """Run spatially-blocked paired test between two levels.

    Merges county-level metrics from level A and B, computes deltas (B - A),
    and runs Wilcoxon + permutation + bootstrap.

    Args:
        county_a: County metrics for level A (columns: county_fips, metric_value).
        county_b: County metrics for level B (columns: county_fips, metric_value).

    Returns:
        Dict with test results.
    """
    merged = county_a.merge(
        county_b, on="county_fips", suffixes=("_a", "_b"),
    )
    if len(merged) < 3:
        return {"n_blocks": len(merged), "note": "too few paired blocks"}

    deltas = merged["metric_value_b"].values - merged["metric_value_a"].values
    n_blocks = len(deltas)

    # Wilcoxon signed-rank (one-sided: B > A)
    try:
        w_stat, w_p = stats.wilcoxon(deltas, alternative="greater")
        w_stat, w_p = float(w_stat), float(w_p)
    except ValueError:
        w_stat, w_p = float("nan"), 1.0

    # Effect size
    r_rb = rank_biserial(deltas)

    # Exact permutation (only if n_blocks <= 20)
    perm_p = exact_permutation_pvalue(deltas)

    # Block bootstrap CI
    boot = block_bootstrap_ci(deltas)

    return {
        "n_blocks": n_blocks,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "wilcoxon_W": w_stat,
        "wilcoxon_p_one_sided": w_p,
        "rank_biserial_r": r_rb,
        "exact_permutation_p": perm_p,
        "block_bootstrap_ci": boot,
        "all_positive": bool(np.all(deltas > 0)),
        "n_positive": int(np.sum(deltas > 0)),
        "n_negative": int(np.sum(deltas < 0)),
        "n_zero": int(np.sum(deltas == 0)),
    }


def process_scenario(
    s3,
    scenario: str,
    crosswalk: pd.DataFrame,
) -> list[dict]:
    """Process all comparisons for one scenario.

    Args:
        s3: boto3 S3 client.
        scenario: Scenario name.
        crosswalk: ZCTA-county crosswalk.

    Returns:
        List of per-(target, comparison, fold) result dicts.
    """
    # Load all prediction levels
    preds = {}
    for level in ("r0", "r1", "r2"):
        df = load_predictions(s3, level, scenario)
        if df is not None:
            preds[level] = df

    if len(preds) < 2:
        log.warning("Fewer than 2 levels for %s, skipping", scenario)
        return []

    results = []
    for target, tmeta in TARGETS.items():
        # Check which levels have this target
        avail = {
            lv: df for lv, df in preds.items()
            if target in df["target"].unique()
        }
        if len(avail) < 2:
            continue

        # Get common folds
        fold_sets = [set(df.loc[df["target"] == target, "fold"].unique()) for df in avail.values()]
        common_folds = sorted(set.intersection(*fold_sets))
        if not common_folds:
            continue

        for level_a, level_b in COMPARISONS:
            if level_a not in avail or level_b not in avail:
                continue

            # Per-fold county aggregation, then pool across folds
            all_county_a = []
            all_county_b = []
            for fold in common_folds:
                ca = two_stage_aggregate(
                    avail[level_a], crosswalk, target, tmeta["task"],
                    solver="histgbdt", fold=fold,
                )
                cb = two_stage_aggregate(
                    avail[level_b], crosswalk, target, tmeta["task"],
                    solver="histgbdt", fold=fold,
                )
                if not ca.empty:
                    ca["fold"] = fold
                    all_county_a.append(ca)
                if not cb.empty:
                    cb["fold"] = fold
                    all_county_b.append(cb)

            if not all_county_a or not all_county_b:
                continue

            # Pool across folds: average each county's metric across folds
            pooled_a = (
                pd.concat(all_county_a)
                .groupby("county_fips", as_index=False)["metric_value"]
                .mean()
            )
            pooled_b = (
                pd.concat(all_county_b)
                .groupby("county_fips", as_index=False)["metric_value"]
                .mean()
            )

            test_result = run_paired_comparison(pooled_a, pooled_b)
            test_result.update({
                "scenario": scenario,
                "target": target,
                "task": tmeta["task"],
                "metric": tmeta["metric"],
                "level_a": level_a,
                "level_b": level_b,
                "n_folds_used": len(common_folds),
            })
            results.append(test_result)

    return results


def compute_pooled_results(per_scenario: list[dict]) -> dict:
    """Pool results across scenarios for each comparison and target.

    Args:
        per_scenario: All per-scenario results.

    Returns:
        Dict of pooled test results keyed by (target, comparison).
    """
    pooled = {}
    for target, tmeta in TARGETS.items():
        for level_a, level_b in COMPARISONS:
            key = f"{target}__{level_a}_vs_{level_b}"
            subset = [
                r for r in per_scenario
                if r["target"] == target
                and r["level_a"] == level_a
                and r["level_b"] == level_b
                and r.get("mean_delta") is not None
            ]
            if not subset:
                continue

            # Collect all county deltas from individual scenario results
            all_deltas = [r["mean_delta"] for r in subset]
            n_total_blocks = sum(r.get("n_blocks", 0) for r in subset)

            pooled[key] = {
                "target": target,
                "level_a": level_a,
                "level_b": level_b,
                "n_scenarios": len(subset),
                "n_total_blocks": n_total_blocks,
                "mean_scenario_delta": float(np.mean(all_deltas)),
                "median_scenario_delta": float(np.median(all_deltas)),
                "scenarios_positive": int(np.sum(np.array(all_deltas) > 0)),
                "scenarios_negative": int(np.sum(np.array(all_deltas) < 0)),
            }

    return pooled


def main() -> int:
    """Entry point for spatially-blocked paired loss computation."""
    parser = argparse.ArgumentParser(
        description="V4.5: Spatially-blocked paired loss"
    )
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: spatially-blocked paired loss (V4.5)")
        log.info("  Reads: prediction parquets (r0/r1/r2) + county crosswalk")
        log.info("  Writes: %s/spatially_blocked_paired_loss.json", RESULTS_PREFIX)
        log.info("  Comparisons: %s", COMPARISONS)
        log.info("  Scenarios: %s", SCENARIOS)
        log.info("  Two-stage aggregation: (zcta, event) -> zcta -> county")
        return 0

    s3 = get_s3_client()
    crosswalk = load_crosswalk(s3)
    log.info("Crosswalk: %d rows, %d unique ZCTAs, %d unique counties",
             len(crosswalk),
             crosswalk["zcta_id"].nunique(),
             crosswalk["county_fips"].nunique())

    all_results = []
    for scenario in SCENARIOS:
        log.info("--- Processing %s ---", scenario)
        scenario_results = process_scenario(s3, scenario, crosswalk)
        all_results.extend(scenario_results)
        log.info("  %d comparison cells for %s", len(scenario_results), scenario)

    pooled = compute_pooled_results(all_results)

    output = {
        "phase": "spatially_blocked_paired_loss",
        "contract_gate": "V4.5",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": {
            "description": (
                "County-level spatially-blocked paired comparison. "
                "Two-stage aggregation per V4.10: "
                "(zcta_id, event) -> per-ZCTA mean -> per-county mean."
            ),
            "test": "Wilcoxon signed-rank (one-sided)",
            "effect_size": "Matched-pairs rank-biserial correlation",
            "permutation": "Exact when n_blocks <= 20",
            "bootstrap": f"Block bootstrap ({N_BOOTSTRAP} resamples), 95% CI",
            "spatial_block": "county_fips from zcta_county_crosswalk",
            "reference_solver": "histgbdt",
            "reference_split": "spatial_blocked",
        },
        "n_scenarios": len(SCENARIOS),
        "n_comparisons": len(all_results),
        "per_scenario": all_results,
        "pooled": pooled,
    }

    # Write local
    out_dir = Path("/tmp/paired_loss")
    out_dir.mkdir(parents=True, exist_ok=True)
    local_file = out_dir / "spatially_blocked_paired_loss.json"
    with open(local_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Written to %s", local_file)

    if args.upload:
        key = f"{RESULTS_PREFIX}/spatially_blocked_paired_loss.json"
        upload_json_result(s3, BUCKET, key, output)
        log.info("Uploaded to s3://%s/%s", BUCKET, key)

    # Summary
    print(f"\n{'='*60}")
    print("  SPATIALLY-BLOCKED PAIRED LOSS (V4.5)")
    print(f"{'='*60}\n")
    for r in all_results:
        blocks = r.get("n_blocks", 0)
        delta = r.get("mean_delta")
        p_val = r.get("wilcoxon_p_one_sided")
        r_rb = r.get("rank_biserial_r")
        delta_s = f"{delta:+.4f}" if delta is not None else "N/A"
        p_s = f"{p_val:.4f}" if p_val is not None else "N/A"
        r_s = f"{r_rb:+.3f}" if r_rb is not None else "N/A"
        print(f"  {r['scenario']:25s} {r['target']:25s} "
              f"{r['level_a']}->{r['level_b']}  "
              f"delta={delta_s}  p={p_s}  r={r_s}  "
              f"blocks={blocks}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
