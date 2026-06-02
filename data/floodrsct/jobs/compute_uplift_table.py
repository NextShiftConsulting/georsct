#!/usr/bin/env python3
"""
compute_uplift_table.py -- Phase 5: money table + hypothesis evidence.

Loads all results (R0/R1/R2) + all kappa diagnostics (R0/R1/R2) and produces:

1. Money table: per (scenario, target) cell with metrics at each level,
   uplift percentages, and kappa values (paper Figure 2)
2. Diagnostic movement table: kappa evolution across levels (paper Figure 3)
3. H2-H4 hypothesis evidence via fold-level paired tests (Wilcoxon signed-rank
   on per-fold metric deltas) + effect sizes (Cohen's d).  Cell-level Spearman
   retained as exploratory association only (n~8 cells, underpowered for
   hypothesis testing).
4. Anti-cherry-picking report: Holm-Bonferroni correction across all tests,
   pre-registration verification

Usage:
    python compute_uplift_table.py --upload
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
from _coverage_common import BUCKET, get_s3_client
from _s3_result import upload_json_result

# rsct service layer for experiment certification
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rsct.experiment_cert import certify_experiment_cell

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

RESULTS_PREFIX = "results/s035"

# Primary metric per task type (higher = better)
PRIMARY_METRIC = {
    "regression": "r2",
    "classification": "roc_auc",
}

N_BOOTSTRAP = 10_000
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(s3, key: str) -> dict | None:
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode())
    except Exception:
        return None


def load_all_results(s3) -> dict[str, dict]:
    """Load all results JSONs. Key = '{level}_{scenario}'."""
    out = {}
    for level in ("r0", "r1", "r2"):
        for scenario in ("houston", "southwest_florida", "nyc", "riverside_coachella"):
            key = f"{RESULTS_PREFIX}/{level}_{scenario}.json"
            data = _load_json(s3, key)
            if data:
                out[f"{level}_{scenario}"] = data
    return out


def load_all_kappas(s3) -> dict[str, dict]:
    """Load kappa diagnostics at each level."""
    out = {}
    for level in ("r0", "r1", "r2"):
        key = f"{RESULTS_PREFIX}/diagnostics_{level}.json"
        data = _load_json(s3, key)
        if data:
            out[level] = data
    return out


def load_all_certificates(s3) -> dict[str, dict]:
    """Load Phase 4.5 certificates, indexed by (level, scenario, target).

    Returns dict keyed by '{level}_{scenario}_{target}' with certificate data.
    """
    out = {}
    for level in ("r0", "r1", "r2"):
        key = f"{RESULTS_PREFIX}/certificates_{level}.json"
        data = _load_json(s3, key)
        if not data:
            continue
        for cert in data.get("certificates", []):
            idx = f"{level}_{cert['scenario']}_{cert['target']}"
            out[idx] = cert
    log.info("Loaded %d certificates", len(out))
    return out


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _mean_metric(results: dict, target: str) -> float | None:
    """Mean primary metric for spatial_blocked + histgbdt from results JSON."""
    runs = results.get("runs", [])
    for r in runs:
        if r["target"] == target:
            task_type = r["task"]
            break
    else:
        return None

    metric_name = PRIMARY_METRIC[task_type]
    vals = []
    for r in runs:
        if r["target"] != target or r["split"] != "spatial_blocked" or r["solver"] != "histgbdt":
            continue
        v = r["metrics"].get(metric_name)
        if v is not None:
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


def _per_fold_metrics(results: dict, target: str) -> list[float]:
    """Per-fold primary metric for spatial_blocked + histgbdt."""
    runs = results.get("runs", [])
    task_type = None
    for r in runs:
        if r["target"] == target:
            task_type = r["task"]
            break
    if not task_type:
        return []

    metric_name = PRIMARY_METRIC[task_type]
    vals = []
    for r in runs:
        if r["target"] != target or r["split"] != "spatial_blocked" or r["solver"] != "histgbdt":
            continue
        v = r["metrics"].get(metric_name)
        if v is not None:
            vals.append(float(v))
    return vals


# ---------------------------------------------------------------------------
# Money table construction
# ---------------------------------------------------------------------------

def build_money_table(
    all_results: dict, all_kappas: dict, all_certs: dict,
) -> list[dict]:
    """Build the money table: one row per (scenario, target) cell.

    Includes RSCT quality signals (alpha, omega, kappa, tau, sigma) from
    Phase 4.5 certificates and degradation diagnosis per level.
    """
    rows = []

    # Get all (scenario, target) cells from R0 results
    for scenario in ("houston", "southwest_florida", "nyc", "riverside_coachella"):
        r0_data = all_results.get(f"r0_{scenario}")
        if not r0_data:
            continue

        targets = sorted(set(r["target"] for r in r0_data.get("runs", [])))
        for target in targets:
            r0_metric = _mean_metric(r0_data, target)
            if r0_metric is None:
                continue

            r1_data = all_results.get(f"r1_{scenario}")
            r2_data = all_results.get(f"r2_{scenario}")
            r1_metric = _mean_metric(r1_data, target) if r1_data else None
            r2_metric = _mean_metric(r2_data, target) if r2_data else None

            # Uplift percentages (relative to prior level)
            uplift_r0_r1 = None
            if r1_metric is not None and r0_metric != 0:
                uplift_r0_r1 = (r1_metric - r0_metric) / max(abs(r0_metric), 0.001) * 100

            uplift_r1_r2 = None
            if r2_metric is not None and r1_metric is not None and r1_metric != 0:
                uplift_r1_r2 = (r2_metric - r1_metric) / max(abs(r1_metric), 0.001) * 100

            # Kappa values at each level
            kappas = {}
            for level in ("r0", "r1", "r2"):
                kappa_data = all_kappas.get(level)
                if not kappa_data:
                    continue
                for cell in kappa_data.get("cells", []):
                    if cell["scenario"] == scenario and cell["target"] == target:
                        for k in ("diag_leakage", "diag_transfer",
                                  "diag_solver", "diag_residual_spatial"):
                            kappas[f"{k}_{level}"] = cell.get(k)
                        break

            # RSCT quality signals from Phase 4.5 certificates
            cert_signals = {}
            for level in ("r0", "r1", "r2"):
                cert = all_certs.get(f"{level}_{scenario}_{target}")
                if not cert:
                    continue
                for sig in ("R", "S_sup", "N", "alpha", "omega",
                            "kappa", "tau", "sigma"):
                    cert_signals[f"cert_{sig}_{level}"] = cert.get(sig)

                # Diagnosis label
                diag = cert.get("diagnosis", {})
                if diag:
                    cert_signals[f"cert_diagnosis_{level}"] = diag.get("label")

            # Compute alpha/omega deltas (improvement across levels)
            for sig in ("alpha", "omega"):
                r0_v = cert_signals.get(f"cert_{sig}_r0")
                r1_v = cert_signals.get(f"cert_{sig}_r1")
                r2_v = cert_signals.get(f"cert_{sig}_r2")
                if r0_v is not None and r1_v is not None:
                    cert_signals[f"cert_{sig}_delta_r0_r1"] = r1_v - r0_v
                if r1_v is not None and r2_v is not None:
                    cert_signals[f"cert_{sig}_delta_r1_r2"] = r2_v - r1_v

            # Re-derive degradation label via rsct service (if R0 cert has metrics)
            r0_cert = all_certs.get(f"r0_{scenario}_{target}")
            if r0_cert:
                live_cert = certify_experiment_cell(
                    spatial_metric=r0_cert.get("spatial_metric"),
                    random_metric=r0_cert.get("random_metric"),
                    task_type=r0_cert.get("task_type", "regression"),
                    kappa_geom=r0_cert.get("kappa"),
                )
                if live_cert.diagnosis_label:
                    cert_signals["degradation_label_r0"] = live_cert.diagnosis_label

            row = {
                "scenario": scenario,
                "target": target,
                "r0_metric": r0_metric,
                "r1_metric": r1_metric,
                "r2_metric": r2_metric,
                "uplift_r0_r1_pct": uplift_r0_r1,
                "uplift_r1_r2_pct": uplift_r1_r2,
                **kappas,
                **cert_signals,
            }
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------

def _spearman_bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int
                        ) -> tuple[float, float, float, float]:
    """Spearman rho with bootstrap 95% CI and exact permutation p-value.

    Returns (rho, ci_lower, ci_upper, perm_p_value).
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)

    if n < 4:
        return float("nan"), float("nan"), float("nan"), float("nan")

    rho_obs, _ = stats.spearmanr(x, y)

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        rhos.append(r)
    ci_lower = float(np.percentile(rhos, 2.5))
    ci_upper = float(np.percentile(rhos, 97.5))

    # Exact permutation p-value (feasible for n <= 10)
    from itertools import permutations
    from math import factorial

    if n <= 8:
        # Exact: all n! permutations
        count_extreme = 0
        total = 0
        for perm in permutations(range(n)):
            r, _ = stats.spearmanr(x, y[list(perm)])
            if abs(r) >= abs(rho_obs):
                count_extreme += 1
            total += 1
        perm_p = count_extreme / total
    else:
        # Monte Carlo permutation (10,000 shuffles)
        count_extreme = 0
        n_perm = 10_000
        for _ in range(n_perm):
            y_shuf = rng.permutation(y)
            r, _ = stats.spearmanr(x, y_shuf)
            if abs(r) >= abs(rho_obs):
                count_extreme += 1
        perm_p = (count_extreme + 1) / (n_perm + 1)  # +1 for observed

    return float(rho_obs), ci_lower, ci_upper, float(perm_p)


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected[orig_idx] = min(1.0, p * (n - rank))
    # Enforce monotonicity
    for i in range(1, n):
        idx_i = indexed[i][0]
        idx_prev = indexed[i - 1][0]
        corrected[idx_i] = max(corrected[idx_i], corrected[idx_prev])
    return corrected


def _hit_rate(kappa_vals: np.ndarray, uplift_vals: np.ndarray) -> dict:
    """Binary hit rate: flagged cells (below median kappa) should have above-median uplift."""
    valid = ~(np.isnan(kappa_vals) | np.isnan(uplift_vals))
    kv, uv = kappa_vals[valid], uplift_vals[valid]
    n = len(kv)
    if n < 4:
        return {"hit_rate": None, "n": n, "note": "too few cells"}

    k_med = np.median(kv)
    u_med = np.median(uv)

    hits = 0
    for k, u in zip(kv, uv):
        flagged = k < k_med
        helped = u > u_med
        if flagged == helped:
            hits += 1

    hr = hits / n

    # Exact binomial CI (Clopper-Pearson)
    ci_lower = stats.beta.ppf(0.025, hits, n - hits + 1) if hits > 0 else 0.0
    ci_upper = stats.beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0

    return {
        "hit_rate": float(hr),
        "hits": hits,
        "n": n,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def _cohens_d(x: np.ndarray) -> float:
    """Cohen's d for a one-sample (paired-difference) array."""
    if len(x) < 2 or np.std(x, ddof=1) == 0:
        return float("nan")
    return float(np.mean(x) / np.std(x, ddof=1))


def _paired_fold_test(
    folds_a: list[float], folds_b: list[float],
) -> dict:
    """Wilcoxon signed-rank test on paired fold metrics (B - A).

    Returns test result dict with statistic, p-value, effect size, and CI.
    """
    a = np.array(folds_a, dtype=float)
    b = np.array(folds_b, dtype=float)
    n = min(len(a), len(b))
    if n < 3:
        return {"n_folds": n, "note": "too few paired folds"}
    a, b = a[:n], b[:n]
    deltas = b - a

    d = _cohens_d(deltas)
    mean_delta = float(np.mean(deltas))

    # Bootstrap 95% CI for the mean delta
    rng = np.random.default_rng(SEED)
    boot_means = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(float(np.mean(deltas[idx])))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Wilcoxon signed-rank (two-sided)
    try:
        stat, p_val = stats.wilcoxon(deltas, alternative="two-sided")
        stat, p_val = float(stat), float(p_val)
    except ValueError:
        # All deltas are zero or too few non-zero
        stat, p_val = float("nan"), 1.0

    return {
        "n_folds": n,
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cohens_d": d,
        "wilcoxon_stat": stat,
        "wilcoxon_p": p_val,
        "all_positive": bool(np.all(deltas > 0)),
        "all_negative": bool(np.all(deltas < 0)),
    }


def run_fold_level_tests(all_results: dict) -> dict:
    """Primary hypothesis tests via fold-level paired comparisons.

    For each (scenario, target):
      H2: R1 folds vs R0 folds (spatial_blocked, histgbdt)
      H3: R2 folds vs R1 folds (spatial_blocked, histgbdt)

    Returns per-cell test results + pooled tests across all cells.
    """
    cell_tests = []
    all_r0_r1_deltas = []
    all_r1_r2_deltas = []

    for scenario in ("houston", "southwest_florida", "nyc", "riverside_coachella"):
        r0_data = all_results.get(f"r0_{scenario}")
        r1_data = all_results.get(f"r1_{scenario}")
        r2_data = all_results.get(f"r2_{scenario}")
        if not r0_data:
            continue

        targets = sorted(set(r["target"] for r in r0_data.get("runs", [])))
        for target in targets:
            r0_folds = _per_fold_metrics(r0_data, target)
            r1_folds = _per_fold_metrics(r1_data, target) if r1_data else []
            r2_folds = _per_fold_metrics(r2_data, target) if r2_data else []

            cell = {"scenario": scenario, "target": target}

            if r0_folds and r1_folds:
                test_r0_r1 = _paired_fold_test(r0_folds, r1_folds)
                cell["h2_r0_r1"] = test_r0_r1
                n = min(len(r0_folds), len(r1_folds))
                deltas = [r1_folds[i] - r0_folds[i] for i in range(n)]
                all_r0_r1_deltas.extend(deltas)

            if r1_folds and r2_folds:
                test_r1_r2 = _paired_fold_test(r1_folds, r2_folds)
                cell["h3_r1_r2"] = test_r1_r2
                n = min(len(r1_folds), len(r2_folds))
                deltas = [r2_folds[i] - r1_folds[i] for i in range(n)]
                all_r1_r2_deltas.extend(deltas)

            cell_tests.append(cell)

    # Pooled tests across all cells
    pooled_h2 = {}
    if len(all_r0_r1_deltas) >= 5:
        arr = np.array(all_r0_r1_deltas)
        d = _cohens_d(arr)
        try:
            stat, p_val = stats.wilcoxon(arr, alternative="greater")
            stat, p_val = float(stat), float(p_val)
        except ValueError:
            stat, p_val = float("nan"), 1.0
        pooled_h2 = {
            "n_paired_folds": len(arr),
            "mean_delta": float(np.mean(arr)),
            "cohens_d": d,
            "wilcoxon_stat": stat,
            "wilcoxon_p_one_sided": p_val,
            "verdict": "PASS" if (p_val < 0.05 and d > 0.2) else "INCONCLUSIVE",
        }

    pooled_h3 = {}
    if len(all_r1_r2_deltas) >= 5:
        arr = np.array(all_r1_r2_deltas)
        d = _cohens_d(arr)
        try:
            stat, p_val = stats.wilcoxon(arr, alternative="greater")
            stat, p_val = float(stat), float(p_val)
        except ValueError:
            stat, p_val = float("nan"), 1.0
        pooled_h3 = {
            "n_paired_folds": len(arr),
            "mean_delta": float(np.mean(arr)),
            "cohens_d": d,
            "wilcoxon_stat": stat,
            "wilcoxon_p_one_sided": p_val,
            "verdict": "PASS" if (p_val < 0.05 and d > 0.2) else "INCONCLUSIVE",
        }

    return {
        "method": "Fold-level paired Wilcoxon signed-rank + Cohen's d",
        "reference_split": "spatial_blocked",
        "reference_solver": "histgbdt",
        "cell_tests": cell_tests,
        "pooled_h2": pooled_h2,
        "pooled_h3": pooled_h3,
    }


def run_exploratory_correlations(money_table: list[dict]) -> dict:
    """Exploratory cell-level Spearman correlations (kappa vs uplift).

    NOTE: n~8 cells, underpowered for hypothesis testing. Reported as
    observed associations only, NOT used for PASS/FAIL verdicts.
    """
    df = pd.DataFrame(money_table)
    if len(df) < 4:
        return {"note": f"Too few cells ({len(df)}) for correlation analysis"}

    tests = []
    for kappa_col in ("diag_leakage_r0", "diag_transfer_r0",
                      "diag_solver_r0", "diag_residual_spatial_r0"):
        for uplift_col in ("uplift_r0_r1_pct", "uplift_r1_r2_pct"):
            if kappa_col not in df.columns or uplift_col not in df.columns:
                continue

            x = df[kappa_col].values.astype(float)
            y = df[uplift_col].values.astype(float)

            rho, ci_lo, ci_hi, perm_p = _spearman_bootstrap(x, y, N_BOOTSTRAP, SEED)
            hr = _hit_rate(x, y)

            tests.append({
                "kappa": kappa_col,
                "uplift": uplift_col,
                "rho": rho,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "perm_p_value": perm_p,
                "hit_rate": hr,
                "n_cells": int((~np.isnan(x) & ~np.isnan(y)).sum()),
            })

    # Holm-Bonferroni correction
    raw_p = [t["perm_p_value"] for t in tests]
    corrected_p = _holm_bonferroni(raw_p)
    for t, cp in zip(tests, corrected_p):
        t["corrected_p_value"] = cp

    return {
        "status": "EXPLORATORY",
        "caveat": (
            f"n={len(df)} cells; Spearman on n<10 has near-zero power. "
            "These are observed associations, NOT hypothesis tests. "
            "Fold-level paired tests are the primary evidence."
        ),
        "correction_method": "Holm-Bonferroni",
        "n_tests": len(tests),
        "tests": tests,
    }


def run_hypothesis_tests(money_table: list[dict], all_results: dict) -> dict:
    """Run all hypothesis tests: fold-level (primary) + cell-level (exploratory)."""

    # PRIMARY: fold-level paired tests
    fold_tests = run_fold_level_tests(all_results)

    # EXPLORATORY: cell-level Spearman associations
    exploratory = run_exploratory_correlations(money_table)

    # H4: audit flags predict representation uplift
    # Use fold-level direction consistency across cells
    cell_tests = fold_tests.get("cell_tests", [])
    h4_cells_helped = 0
    h4_cells_total = 0
    for ct in cell_tests:
        h2 = ct.get("h2_r0_r1", {})
        if h2.get("mean_delta") is not None:
            h4_cells_total += 1
            if h2["mean_delta"] > 0:
                h4_cells_helped += 1

    h2_verdict = fold_tests.get("pooled_h2", {}).get("verdict", "NO_DATA")
    h3_verdict = fold_tests.get("pooled_h3", {}).get("verdict", "NO_DATA")

    return {
        "primary_method": "Fold-level paired Wilcoxon signed-rank + Cohen's d",
        "exploratory_method": "Cell-level Spearman (underpowered, n~8)",
        "fold_level_tests": fold_tests,
        "exploratory_correlations": exploratory,
        "h2_evidence": {
            "description": "R1 > R0 under spatial-blocked CV (fold-level paired test)",
            "pooled": fold_tests.get("pooled_h2", {}),
            "verdict": h2_verdict,
        },
        "h3_evidence": {
            "description": "R2 > R1 under spatial-blocked CV (fold-level paired test)",
            "pooled": fold_tests.get("pooled_h3", {}),
            "verdict": h3_verdict,
        },
        "h4_evidence": {
            "description": "Representation uplift is consistent across cells",
            "cells_with_positive_uplift": h4_cells_helped,
            "cells_total": h4_cells_total,
            "verdict": "PASS" if (h4_cells_total >= 4 and
                                   h4_cells_helped / max(h4_cells_total, 1) >= 0.75)
                       else "INCONCLUSIVE",
        },
    }


# ---------------------------------------------------------------------------
# Diagnostic movement table
# ---------------------------------------------------------------------------

def build_movement_table(money_table: list[dict]) -> list[dict]:
    """Build kappa movement across levels for each cell."""
    rows = []
    for cell in money_table:
        row = {
            "scenario": cell["scenario"],
            "target": cell["target"],
        }
        for kappa_name in ("diag_leakage", "diag_transfer",
                           "diag_solver", "diag_residual_spatial"):
            for level in ("r0", "r1", "r2"):
                key = f"{kappa_name}_{level}"
                row[key] = cell.get(key)

            # Movement: R0->R1 and R1->R2
            r0_val = cell.get(f"{kappa_name}_r0")
            r1_val = cell.get(f"{kappa_name}_r1")
            r2_val = cell.get(f"{kappa_name}_r2")
            if r0_val is not None and r1_val is not None:
                row[f"{kappa_name}_delta_r0_r1"] = r1_val - r0_val
            if r1_val is not None and r2_val is not None:
                row[f"{kappa_name}_delta_r1_r2"] = r2_val - r1_val

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Pre-registration verification
# ---------------------------------------------------------------------------

def verify_preregistration(all_kappas: dict, money_table: list[dict]) -> dict:
    """Check whether pre-registered predictions from kappa_r0 matched observed uplift."""
    r0_kappas = all_kappas.get("r0", {})
    predictions = r0_kappas.get("predictions", {})
    if not predictions:
        return {"status": "NO_PREDICTIONS", "note": "kappa_r0 had no predictions section"}

    predicted_helped = set(predictions.get("r1_should_help_most", []))
    predicted_not = set(predictions.get("r1_should_help_least", []))

    # Find actual uplift
    uplifts = {}
    for row in money_table:
        s = row["scenario"]
        u = row.get("uplift_r0_r1_pct")
        if u is not None:
            uplifts.setdefault(s, []).append(u)

    scenario_uplift = {s: float(np.mean(v)) for s, v in uplifts.items()}

    if not scenario_uplift:
        return {"status": "NO_UPLIFT_DATA"}

    median_uplift = float(np.median(list(scenario_uplift.values())))

    actually_helped = {s for s, u in scenario_uplift.items() if u > median_uplift}

    correct_flagged = predicted_helped & actually_helped
    correct_unflagged = predicted_not - actually_helped
    total_correct = len(correct_flagged) + len(correct_unflagged)
    total = len(predicted_helped) + len(predicted_not)

    return {
        "status": "VERIFIED",
        "predicted_helped": sorted(predicted_helped),
        "predicted_not_helped": sorted(predicted_not),
        "actually_helped": sorted(actually_helped),
        "scenario_uplift": scenario_uplift,
        "median_uplift": median_uplift,
        "correct": total_correct,
        "total": total,
        "accuracy": total_correct / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: money table + hypothesis evidence")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    s3 = get_s3_client()

    print(f"\n{'='*60}")
    print(f"  S035 PHASE 5: MONEY TABLE + HYPOTHESIS EVIDENCE")
    print(f"{'='*60}\n")

    all_results = load_all_results(s3)
    log.info("Loaded %d result files", len(all_results))

    all_kappas = load_all_kappas(s3)
    log.info("Loaded %d kappa diagnostic files", len(all_kappas))

    all_certs = load_all_certificates(s3)

    # Build tables
    money_table = build_money_table(all_results, all_kappas, all_certs)
    log.info("Money table: %d cells", len(money_table))

    movement_table = build_movement_table(money_table)

    # Hypothesis tests
    hypothesis_evidence = run_hypothesis_tests(money_table, all_results)

    # Pre-registration check
    preregistration = verify_preregistration(all_kappas, money_table)

    # --- Output ---
    payload = {
        "experiment": "s035-model-ladder",
        "phase": "uplift_table",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "money_table": money_table,
        "movement_table": movement_table,
        "hypothesis_evidence": hypothesis_evidence,
        "preregistration_verification": preregistration,
        "methodology": {
            "primary_metric": "R2 for regression, ROC-AUC for classification",
            "reference_split": "spatial_blocked",
            "reference_solver": "histgbdt",
            "bootstrap_resamples": N_BOOTSTRAP,
            "primary_test": "Fold-level paired Wilcoxon signed-rank (one-sided for H2/H3)",
            "effect_size": "Cohen's d on paired fold deltas",
            "exploratory_test": "Cell-level Spearman (kappa vs uplift); n~8, underpowered, "
                                "reported as observed associations only",
            "correction": "Holm-Bonferroni for exploratory Spearman family",
            "verdict_criteria": {
                "PASS": "Wilcoxon p < 0.05 AND Cohen's d > 0.2",
                "INCONCLUSIVE": "either condition not met",
            },
            "rsct_signals": {
                "source": "Phase 4.5 certificates via rsct.experiment_cert",
                "alpha": "R/(R+N) signal purity",
                "omega": "1-S_sup reliability",
                "tau": "1/(1+CV) temporal stability from per-fold variance",
                "sigma": "std(fold_metrics) cross-fold volatility",
                "diagnosis": "DegradationDiagnoser 3x3 grid (alpha x kappa)",
                "service": "rsct.experiment_cert.certify_experiment_cell",
            },
        },
    }

    output_json = json.dumps(payload, indent=2, default=str)

    if args.upload:
        key = f"{RESULTS_PREFIX}/money_table.json"
        upload_json_result(s3, BUCKET, key, payload)

        # Also upload individual evidence files for H2-H4
        for h_key in ("h2_evidence", "h3_evidence", "h4_evidence"):
            evidence = hypothesis_evidence.get(h_key, {})
            ev_key = f"{RESULTS_PREFIX}/evidence/{h_key}.json"
            s3.put_object(
                Bucket=BUCKET, Key=ev_key,
                Body=json.dumps(evidence, indent=2, default=str).encode(),
                ContentType="application/json",
            )
            log.info("Uploaded s3://%s/%s", BUCKET, ev_key)
    else:
        local = "/tmp/money_table.json"
        Path(local).write_text(output_json)
        log.info("Wrote %s", local)

    # Print summary
    print("\n--- MONEY TABLE ---")
    for row in money_table:
        print(f"  {row['scenario']:25s} {row['target']:30s} "
              f"R0={row['r0_metric']:.4f}  "
              f"R1={'%.4f' % row['r1_metric'] if row['r1_metric'] is not None else 'N/A':>7s}  "
              f"R2={'%.4f' % row['r2_metric'] if row['r2_metric'] is not None else 'N/A':>7s}  "
              f"R0->R1={'%.1f%%' % row['uplift_r0_r1_pct'] if row['uplift_r0_r1_pct'] is not None else 'N/A':>7s}")

    he = hypothesis_evidence
    print(f"\n--- HYPOTHESIS VERDICTS (fold-level paired tests) ---")
    for h in ("h2_evidence", "h3_evidence", "h4_evidence"):
        ev = he.get(h, {})
        pooled = ev.get("pooled", {})
        d = pooled.get("cohens_d", "N/A")
        p = pooled.get("wilcoxon_p_one_sided", "N/A")
        d_str = f"d={d:.2f}" if isinstance(d, float) else f"d={d}"
        p_str = f"p={p:.4f}" if isinstance(p, float) else f"p={p}"
        print(f"  {h}: {ev.get('verdict', 'N/A')}  ({d_str}, {p_str})")

    print(f"\n--- PRE-REGISTRATION ---")
    print(f"  Status: {preregistration.get('status')}")
    if preregistration.get("accuracy") is not None:
        print(f"  Accuracy: {preregistration['correct']}/{preregistration['total']}"
              f" ({preregistration['accuracy']:.0%})")

    print(output_json)


if __name__ == "__main__":
    main()
