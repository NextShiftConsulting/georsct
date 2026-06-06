#!/usr/bin/env python3
"""
compute_r3_money_table.py -- Phase R3_3: R3 money table + hypothesis tests.

Loads R0/R1/R2/R3 results and produces:
  1. Extended money table: per (scenario, target) cell with R0-R3 metrics
  2. H5: R3 headline vs R2 (Wilcoxon signed-rank)
  3. H6: Admitted vs rejected block leakage (Mann-Whitney U)
  4. H7: Order robustness concordance rate (from R3_1b)
  5. H8: Headline tier vs stabilizer/marginal tier per-block delta (Mann-Whitney U)

Usage:
    python compute_r3_money_table.py --upload
    python compute_r3_money_table.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _coverage_common import BUCKET, SCENARIOS, get_s3_client, level_prefix
from _s3_result import upload_json_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"
N_BOOTSTRAP = 10_000
SEED = 42

PRIMARY_METRIC = {"regression": "r2", "classification": "roc_auc"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(s3, key: str):
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception:
        return None


def _per_fold_metrics(results: dict, target: str, variant: str = None) -> list[float]:
    """Per-fold primary metric for spatial_blocked + histgbdt."""
    runs = results.get("runs", [])
    task_type = None
    for r in runs:
        if r["target"] == target:
            task_type = r.get("task")
            break
    if not task_type:
        return []

    metric_name = PRIMARY_METRIC[task_type]
    vals = []
    for r in runs:
        if r["target"] != target:
            continue
        if r["split"] != "spatial_blocked":
            continue
        if r["solver"] != "histgbdt":
            continue
        if variant is not None and r.get("variant") != variant:
            continue
        v = r["metrics"].get(metric_name)
        if v is not None:
            vals.append(float(v))
    return vals


def _mean_metric(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def _wilcoxon_test(x: list[float], y: list[float]) -> dict:
    """Paired Wilcoxon signed-rank test. Returns stat, p, cohen's d."""
    x, y = np.array(x), np.array(y)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)

    if n < 3:
        return {
            "n_pairs": n,
            "stat": None,
            "p_one_sided": None,
            "cohens_d": None,
            "verdict": "INSUFFICIENT_DATA",
        }

    diffs = x - y
    if np.all(diffs == 0):
        return {
            "n_pairs": n,
            "stat": 0.0,
            "p_one_sided": 1.0,
            "cohens_d": 0.0,
            "verdict": "NO_DIFFERENCE",
        }

    try:
        stat, p_two = stats.wilcoxon(x, y, alternative="two-sided")
        # One-sided: R3 > R2
        _, p_one = stats.wilcoxon(x, y, alternative="greater")
    except ValueError:
        return {
            "n_pairs": n,
            "stat": None,
            "p_one_sided": None,
            "cohens_d": None,
            "verdict": "TEST_FAILED",
        }

    d_mean = float(np.mean(diffs))
    d_std = float(np.std(diffs, ddof=1)) if n > 1 else 1.0
    cohens_d = d_mean / d_std if d_std > 0 else 0.0

    verdict = "INCONCLUSIVE"
    if p_one < 0.05 and cohens_d > 0.2:
        verdict = "IMPROVEMENT"
    elif p_one >= 0.05:
        verdict = "NO_DEGRADATION"

    return {
        "n_pairs": n,
        "stat": float(stat),
        "p_one_sided": float(p_one),
        "cohens_d": round(cohens_d, 4),
        "mean_diff": round(d_mean, 6),
        "verdict": verdict,
    }


def _mannwhitney_test(x: list[float], y: list[float], label: str) -> dict:
    """Mann-Whitney U test for independent samples."""
    x = np.array([v for v in x if v is not None and not np.isnan(v)])
    y = np.array([v for v in y if v is not None and not np.isnan(v)])

    if len(x) < 2 or len(y) < 2:
        return {
            "test": label,
            "n_x": len(x),
            "n_y": len(y),
            "stat": None,
            "p_value": None,
            "verdict": "INSUFFICIENT_DATA",
        }

    try:
        stat, p = stats.mannwhitneyu(x, y, alternative="less")
    except ValueError:
        return {
            "test": label,
            "n_x": len(x),
            "n_y": len(y),
            "stat": None,
            "p_value": None,
            "verdict": "TEST_FAILED",
        }

    return {
        "test": label,
        "n_x": len(x),
        "n_y": len(y),
        "median_x": round(float(np.median(x)), 6),
        "median_y": round(float(np.median(y)), 6),
        "stat": float(stat),
        "p_value": float(p),
        "verdict": "PASS" if p < 0.05 else "FAIL",
    }


def _bootstrap_ci(vals: list[float], n_boot: int = N_BOOTSTRAP) -> dict:
    """Bootstrap 95% CI for the mean."""
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(arr) < 2:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "n": len(arr)}

    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))

    return {
        "mean": round(float(np.mean(arr)), 6),
        "ci_lower": round(float(np.percentile(means, 2.5)), 6),
        "ci_upper": round(float(np.percentile(means, 97.5)), 6),
        "n": len(arr),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase R3_3: R3 money table + H5-H8 hypothesis tests"
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: would load R0/R1/R2/R3 results and admission table")
        log.info("  Tests: H5 (Wilcoxon), H6 (Mann-Whitney), H7 (concordance), H8 (Mann-Whitney)")
        return 0

    s3 = get_s3_client()

    print(f"\n{'='*60}")
    print(f"  S035 PHASE R3_3: R3 MONEY TABLE")
    print(f"{'='*60}\n")

    # Load all results
    all_results = {}
    for level in ("r0", "r1", "r2", "r3"):
        for scenario in SCENARIOS:
            prefix = level_prefix(level) if level != "r3" else "r3"
            key = f"{RESULTS_PREFIX}/{prefix}_{scenario}.json"
            data = _load_json(s3, key)
            if data:
                all_results[f"{level}_{scenario}"] = data
    log.info("Loaded results for %d (level, scenario) pairs", len(all_results))

    # Load admission table + gear summary
    admission_data = _load_json(s3, f"{RESULTS_PREFIX}/r3_block_admission_table.json")
    admission_table = admission_data.get("admission_table", []) if admission_data else []

    gear_data = _load_json(s3, f"{RESULTS_PREFIX}/r3_gear_summary.json")
    gear_entries = gear_data.get("entries", []) if gear_data else []

    # Load order robustness
    order_results = {}
    for scenario in SCENARIOS:
        data = _load_json(s3, f"{RESULTS_PREFIX}/r3_order_robustness_{scenario}.json")
        if data:
            order_results[scenario] = data

    # Build money table rows
    money_rows = []
    h5_evidence = []  # Per-cell H5 tests
    h6_admitted_leakage = []
    h6_rejected_leakage = []
    h8_headline_deltas = []
    h8_other_deltas = []

    for scenario in SCENARIOS:
        r0_data = all_results.get(f"r0_{scenario}")
        if not r0_data:
            continue

        targets = sorted(set(r["target"] for r in r0_data.get("runs", [])))
        for target in targets:
            r0_folds = _per_fold_metrics(r0_data, target)
            r0_mean = _mean_metric(r0_folds)
            if np.isnan(r0_mean):
                continue

            r1_data = all_results.get(f"r1_{scenario}")
            r2_data = all_results.get(f"r2_{scenario}")
            r3_data = all_results.get(f"r3_{scenario}")

            r1_folds = _per_fold_metrics(r1_data, target) if r1_data else []
            r2_folds = _per_fold_metrics(r2_data, target) if r2_data else []
            r3_headline_folds = _per_fold_metrics(r3_data, target, "headline") if r3_data else []
            r3_full_folds = _per_fold_metrics(r3_data, target, "full") if r3_data else []
            r3_stabilized_folds = _per_fold_metrics(r3_data, target, "stabilized") if r3_data else []

            r1_mean = _mean_metric(r1_folds)
            r2_mean = _mean_metric(r2_folds)
            r3h_mean = _mean_metric(r3_headline_folds)
            r3f_mean = _mean_metric(r3_full_folds)
            r3s_mean = _mean_metric(r3_stabilized_folds)

            # Uplift percentages
            def _uplift(new, old):
                if np.isnan(new) or np.isnan(old) or abs(old) < 0.001:
                    return None
                return round((new - old) / max(abs(old), 0.001) * 100, 4)

            row = {
                "scenario": scenario,
                "target": target,
                "r0_metric": round(r0_mean, 6),
                "r1_metric": round(r1_mean, 6) if not np.isnan(r1_mean) else None,
                "r2_metric": round(r2_mean, 6) if not np.isnan(r2_mean) else None,
                "r3_headline_metric": round(r3h_mean, 6) if not np.isnan(r3h_mean) else None,
                "r3_full_metric": round(r3f_mean, 6) if not np.isnan(r3f_mean) else None,
                "r3_stabilized_metric": round(r3s_mean, 6) if not np.isnan(r3s_mean) else None,
                "uplift_r0_r1_pct": _uplift(r1_mean, r0_mean),
                "uplift_r1_r2_pct": _uplift(r2_mean, r1_mean),
                "uplift_r2_r3h_pct": _uplift(r3h_mean, r2_mean),
                "uplift_r2_r3f_pct": _uplift(r3f_mean, r2_mean),
                "r3_headline_ci": _bootstrap_ci(r3_headline_folds),
                "r3_full_ci": _bootstrap_ci(r3_full_folds),
            }
            money_rows.append(row)

            # H5: R3 headline vs R2 (Wilcoxon)
            if r3_headline_folds and r2_folds:
                h5_test = _wilcoxon_test(r3_headline_folds, r2_folds)
                h5_test["scenario"] = scenario
                h5_test["target"] = target
                h5_evidence.append(h5_test)

            # Collect H6 data: leakage by admission status
            for entry in admission_table:
                if entry.get("scenario") != scenario or entry.get("target") != target:
                    continue
                leakage = entry.get("delta_leakage") if "delta_leakage" in entry else None
                if leakage is None:
                    # Try to get from certificate
                    continue
                if entry.get("enforcement_decision") == "EXECUTE":
                    h6_admitted_leakage.append(leakage)
                elif entry.get("enforcement_decision") in ("REJECT", "BLOCK"):
                    h6_rejected_leakage.append(leakage)

            # Collect H8 data: per-block delta by tier
            for entry in admission_table:
                if entry.get("scenario") != scenario or entry.get("target") != target:
                    continue
                delta = entry.get("delta_spatial")
                if delta is None:
                    continue
                tier = entry.get("admission_tier", "")
                if tier == "headline":
                    h8_headline_deltas.append(delta)
                elif tier in ("diagnostic-stabilizer", "marginal"):
                    h8_other_deltas.append(delta)

    # H5: Pooled test across cells
    all_r3h = []
    all_r2 = []
    for scenario in SCENARIOS:
        r2_data = all_results.get(f"r2_{scenario}")
        r3_data = all_results.get(f"r3_{scenario}")
        if not r2_data or not r3_data:
            continue
        targets = sorted(set(r["target"] for r in r2_data.get("runs", [])))
        for target in targets:
            r2_folds = _per_fold_metrics(r2_data, target)
            r3_folds = _per_fold_metrics(r3_data, target, "headline")
            if len(r2_folds) == len(r3_folds) and len(r2_folds) > 0:
                all_r3h.extend(r3_folds)
                all_r2.extend(r2_folds)

    pooled_h5 = _wilcoxon_test(all_r3h, all_r2) if all_r3h else {
        "verdict": "NO_DATA", "n_pairs": 0
    }

    # H6: Mann-Whitney on leakage
    h6_test = _mannwhitney_test(
        h6_admitted_leakage, h6_rejected_leakage,
        "H6: admitted leakage < rejected leakage"
    )

    # H7: Aggregate order robustness from R3_1b
    h7_results = []
    for scenario, data in order_results.items():
        for target_result in data.get("results", []):
            conc = target_result.get("concordance", {})
            h7_results.append({
                "scenario": scenario,
                "target": target_result.get("target"),
                "overall_robustness_rate": conc.get("overall_robustness_rate"),
                "h7_pass": conc.get("h7_pass"),
            })

    h7_pass_count = sum(1 for r in h7_results if r.get("h7_pass"))
    h7_total = len(h7_results)

    # H8: Mann-Whitney on per-block deltas
    h8_test = _mannwhitney_test(
        h8_headline_deltas, h8_other_deltas,
        "H8: headline tier delta > stabilizer/marginal tier delta"
    )
    # Override direction: we want headline > other, so use "greater"
    if h8_headline_deltas and h8_other_deltas:
        x = np.array([v for v in h8_headline_deltas if not np.isnan(v)])
        y = np.array([v for v in h8_other_deltas if not np.isnan(v)])
        if len(x) >= 2 and len(y) >= 2:
            try:
                _, p = stats.mannwhitneyu(x, y, alternative="greater")
                h8_test["p_value_greater"] = float(p)
                h8_test["verdict"] = "PASS" if p < 0.05 else "FAIL"
            except ValueError:
                pass

    # Assemble hypothesis evidence
    hypothesis_evidence = {
        "H5": {
            "statement": "R3 headline >= R2 OR R3 stabilizes (lower sigma) without degradation",
            "pooled_test": pooled_h5,
            "per_cell_tests": h5_evidence,
        },
        "H6": {
            "statement": "Admitted blocks have lower leakage than rejected blocks",
            "test": h6_test,
        },
        "H7": {
            "statement": ">= 80% of blocks receive same verdict under different orderings",
            "per_cell": h7_results,
            "pass_count": h7_pass_count,
            "total": h7_total,
            "verdict": "PASS" if h7_pass_count == h7_total and h7_total > 0 else "FAIL",
        },
        "H8": {
            "statement": "Headline tier blocks contribute higher delta than stabilizer/marginal",
            "test": h8_test,
        },
    }

    # Assemble money table
    money_table = {
        "phase": "R3_3_money_table",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(money_rows),
        "rows": money_rows,
        "hypothesis_evidence": hypothesis_evidence,
    }

    # Write local
    out_dir = Path(__file__).parent.parent / "exp" / "s035-model-ladder" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "r3_money_table.json", "w") as f:
        json.dump(money_table, f, indent=2, default=str)
    with open(out_dir / "r3_hypothesis_evidence.json", "w") as f:
        json.dump(hypothesis_evidence, f, indent=2, default=str)

    if args.upload:
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/r3_money_table.json", money_table)
        upload_json_result(s3, BUCKET, f"{RESULTS_PREFIX}/r3_hypothesis_evidence.json", hypothesis_evidence)
        log.info("Uploaded to S3")

    # Summary
    log.info("\n=== R3_3 Money Table Summary ===")
    log.info("  Cells: %d", len(money_rows))
    log.info("  H5 (R3 vs R2): %s (pooled p=%.4f, d=%.3f)",
             pooled_h5.get("verdict", "N/A"),
             pooled_h5.get("p_one_sided") or 0,
             pooled_h5.get("cohens_d") or 0)
    log.info("  H6 (leakage): %s (p=%.4f)",
             h6_test.get("verdict", "N/A"),
             h6_test.get("p_value") or 0)
    log.info("  H7 (order): %d/%d cells pass", h7_pass_count, h7_total)
    log.info("  H8 (gear): %s (p=%.4f)",
             h8_test.get("verdict", "N/A"),
             h8_test.get("p_value_greater", h8_test.get("p_value")) or 0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
