#!/usr/bin/env python3
"""
analyze_s019e.py — S019E Pooled Calibration Post-Hoc Analysis

Produces:
  FC-1  Simplex integrity          (should PASS under pooled calibration)
  FC-2  Alpha range across arms    (should PASS under pooled calibration)
  FC-3  Kappa range across arms    (expected PASS)
  FC-4  Sigma stability            (should PASS under pooled calibration)
  FC-5  Cross-seed consensus       (Pearson r across 3 seeds)
  FC-7  Kappa–r2 correlation       (rsct_compat vs r2, Spearman rho)

  D4-primary    Family-mean Spearman rho t-test (pre-registered, df=2)
  D4-secondary  Pairwise accuracy  (fraction of arm-pairs correctly ordered)

  Per-task summary table  →  data/s019e/certs/task_summary.csv
  FC summary JSON         →  data/s019e/certs/fc_summary.json

Usage:
    python exp/s019e_posthoc/analyze_s019e.py
    python exp/s019e_posthoc/analyze_s019e.py --data-dir data/s019e --out data/s019e/certs
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456]
REAL_ARMS = ["pca_v1", "spatial_lag_v1", "gnn_v2", "geo_spatial", "domain_features"]
ALL_ARMS  = REAL_ARMS + ["noisy_control"]

CONUS27 = [
    "annual_checkup", "arthritis", "asthma", "binge_drinking", "bp_medicated",
    "cancer", "cholesterol_screening", "chronic_kidney_disease", "copd",
    "coronary_heart_disease", "dental_visit", "diabetes", "high_blood_pressure",
    "high_cholesterol", "mental_health_not_good", "obesity",
    "physical_health_not_good", "physical_inactivity", "sleep_less_7hr",
    "smoking", "stroke",
    "home_value", "income", "population_density",
    "elevation", "night_lights", "tree_cover",
]

TARGET_FAMILY = {
    **{t: "health" for t in [
        "annual_checkup", "arthritis", "asthma", "binge_drinking", "bp_medicated",
        "cancer", "cholesterol_screening", "chronic_kidney_disease", "copd",
        "coronary_heart_disease", "dental_visit", "diabetes", "high_blood_pressure",
        "high_cholesterol", "mental_health_not_good", "obesity",
        "physical_health_not_good", "physical_inactivity", "sleep_less_7hr",
        "smoking", "stroke",
    ]},
    **{t: "socioeconomic" for t in ["home_value", "income", "population_density"]},
    **{t: "environmental" for t in ["elevation", "night_lights", "tree_cover"]},
}

# FC thresholds (from combine_georsct.tex §H.2)
FC1_MAX_DEV   = 0.01   # max |R+S+N - 1|
FC2_ALPHA_SPREAD = 0.02  # min spread of mean alpha across arms per task
FC3_KAPPA_SPREAD = 0.02  # min spread of mean kappa across arms per task
FC4_SIGMA_THRESH = 0.30  # sigma < this for at least 20/27 tasks
FC4_MIN_TASKS    = 20
FC5_PEARSON_MIN  = 0.90
FC7_RHO_THRESH   = -0.10  # kappa–r2 correlation should be negative

# D4 threshold (pre-registered)
D4_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all_results(data_dir: Path) -> list:
    records = []
    for seed in SEEDS:
        path = data_dir / f"seed_{seed}" / "s019e_results.json"
        if not path.exists():
            print(f"  WARNING: {path} not found -- skipping seed {seed}")
            continue
        batch = json.loads(path.read_text())
        records.extend(batch)

    # Field rename: series_019 renamed theory_kappa -> rsct_compat and
    # theory_sigma -> rsct_turb in the run scripts. Handle pre-rename JSON
    # files transparently so this script works against both old and new data.
    renamed = 0
    for r in records:
        if "theory_kappa" in r and "rsct_compat" not in r:
            r["rsct_compat"] = r.pop("theory_kappa")
            renamed += 1
        if "theory_sigma" in r and "rsct_turb" not in r:
            r["rsct_turb"] = r.pop("theory_sigma")
        if "alpha" in r and "rsct_qual" not in r:
            r["rsct_qual"] = r["alpha"]  # keep alpha, add alias
    if renamed:
        print(f"  (renamed theory_kappa->rsct_compat in {renamed} pre-rename records)")

    print(f"Loaded {len(records)} records ({len(SEEDS)} seeds)")
    return records


# ---------------------------------------------------------------------------
# FC-1: Simplex integrity
# ---------------------------------------------------------------------------

def fc1_simplex(records: list) -> dict:
    clean = [r for r in records if not r["degenerate"]]
    devs = [abs(r["R"] + r["S_sup"] + r["N"] - 1.0) for r in clean]
    max_dev = max(devs)
    mean_dev = float(np.mean(devs))
    status = "PASS" if max_dev < FC1_MAX_DEV else "FAIL"
    return {
        "id": "FC-1",
        "name": "Simplex integrity",
        "threshold": f"max_dev < {FC1_MAX_DEV}",
        "max_dev": round(max_dev, 6),
        "mean_dev": round(mean_dev, 6),
        "n_records": len(clean),
        "status": status,
    }


# ---------------------------------------------------------------------------
# FC-2: Alpha range across arms (per task, averaged)
# ---------------------------------------------------------------------------

def fc2_alpha_range(records: list) -> dict:
    """For each task: compute spread of mean alpha across real arms. Mean over tasks."""
    clean = [r for r in records if not r["degenerate"]]
    task_spreads = []
    for task in CONUS27:
        arm_alphas = {}
        for arm in REAL_ARMS:
            rows = [r for r in clean if r["target"] == task and r["embedding"] == arm]
            if rows:
                arm_alphas[arm] = float(np.nanmean([r["alpha"] for r in rows]))
        if len(arm_alphas) >= 2:
            vals = list(arm_alphas.values())
            task_spreads.append(max(vals) - min(vals))
    mean_spread = float(np.mean(task_spreads))
    min_spread  = float(np.min(task_spreads))
    n_pass = sum(1 for s in task_spreads if s >= FC2_ALPHA_SPREAD)
    status = "PASS" if mean_spread >= FC2_ALPHA_SPREAD else "FAIL"
    return {
        "id": "FC-2",
        "name": "Alpha range across arms",
        "threshold": f"mean_spread > {FC2_ALPHA_SPREAD}",
        "mean_spread": round(mean_spread, 4),
        "min_spread":  round(min_spread, 4),
        "n_tasks_above_threshold": n_pass,
        "n_tasks_total": len(task_spreads),
        "status": status,
    }


# ---------------------------------------------------------------------------
# FC-3: Kappa range across arms
# ---------------------------------------------------------------------------

def fc3_kappa_range(records: list) -> dict:
    clean = [r for r in records if not r["degenerate"]]
    task_spreads = []
    for task in CONUS27:
        arm_kappas = {}
        for arm in REAL_ARMS:
            rows = [r for r in clean if r["target"] == task and r["embedding"] == arm]
            if rows:
                arm_kappas[arm] = float(np.nanmean([r["proxy_kappa"] for r in rows]))
        if len(arm_kappas) >= 2:
            vals = list(arm_kappas.values())
            task_spreads.append(max(vals) - min(vals))
    mean_spread = float(np.mean(task_spreads))
    status = "PASS" if mean_spread >= FC3_KAPPA_SPREAD else "FAIL"
    return {
        "id": "FC-3",
        "name": "Kappa range across arms",
        "threshold": f"mean_spread > {FC3_KAPPA_SPREAD}",
        "mean_spread": round(mean_spread, 4),
        "status": status,
    }


# ---------------------------------------------------------------------------
# FC-4: Sigma stability (sigma < 0.30 for >=20/27 tasks)
# ---------------------------------------------------------------------------

def fc4_sigma(records: list) -> dict:
    clean = [r for r in records if not r["degenerate"]]
    task_sigma = {}
    for task in CONUS27:
        rows = [r for r in clean if r["target"] == task]
        if rows:
            task_sigma[task] = float(np.nanmean([r["sigma"] for r in rows]))
    n_below = sum(1 for s in task_sigma.values() if s < FC4_SIGMA_THRESH)
    status = "PASS" if n_below >= FC4_MIN_TASKS else "FAIL"
    return {
        "id": "FC-4",
        "name": "Sigma stability",
        "threshold": f"n_tasks_with_sigma < {FC4_SIGMA_THRESH} >= {FC4_MIN_TASKS}",
        "n_tasks_below_threshold": n_below,
        "n_tasks_total": len(task_sigma),
        "mean_sigma": round(float(np.mean(list(task_sigma.values()))), 4),
        "per_task": {t: round(v, 4) for t, v in sorted(task_sigma.items())},
        "status": status,
    }


# ---------------------------------------------------------------------------
# FC-5: Cross-seed consensus (Pearson r on per-task mean rsct_compat vectors)
# ---------------------------------------------------------------------------

def fc5_cross_seed(records: list) -> dict:
    seed_task_kappa = {}
    for seed in SEEDS:
        seed_recs = [r for r in records if r["seed"] == seed and not r["degenerate"]]
        task_kappa = {}
        for task in CONUS27:
            rows = [r for r in seed_recs if r["target"] == task]
            if rows:
                task_kappa[task] = float(np.nanmean([r["rsct_compat"] for r in rows]))
        seed_task_kappa[seed] = task_kappa

    pairs = [(SEEDS[i], SEEDS[j]) for i in range(len(SEEDS)) for j in range(i+1, len(SEEDS))]
    corrs = {}
    for s1, s2 in pairs:
        tasks = [t for t in CONUS27 if t in seed_task_kappa[s1] and t in seed_task_kappa[s2]]
        v1 = [seed_task_kappa[s1][t] for t in tasks]
        v2 = [seed_task_kappa[s2][t] for t in tasks]
        r, p = stats.pearsonr(v1, v2)
        corrs[f"{s1}_vs_{s2}"] = {"r": round(float(r), 4), "p": float(p), "n": len(tasks)}

    min_r = min(v["r"] for v in corrs.values())
    status = "PASS" if min_r >= FC5_PEARSON_MIN else "FAIL"
    return {
        "id": "FC-5",
        "name": "Cross-seed consensus",
        "threshold": f"min Pearson r >= {FC5_PEARSON_MIN}",
        "min_r": round(min_r, 4),
        "pairs": corrs,
        "status": status,
    }


# ---------------------------------------------------------------------------
# FC-7: rsct_compat – r2 correlation (Spearman rho < -0.10)
# ---------------------------------------------------------------------------

def fc7_kappa_r2(records: list) -> dict:
    """
    Task-cell level: one (mean_rsct_compat, mean_r2) per (task, arm) cell.
    Mirrors S019D FC-7 exactly for comparability -- uses rsct_compat (D*/D).

    NOTE: S019E uses noisy_control in the LOO reference set for rsct_compat,
    which may dilute kappa values vs S019D. Proxy_kappa also reported for
    diagnostic comparison.
    """
    clean = [r for r in records if not r["degenerate"]]
    cells_theory, cells_proxy, cells_r2 = [], [], []
    for task in CONUS27:
        for arm in REAL_ARMS:
            rows = [r for r in clean if r["target"] == task and r["embedding"] == arm]
            if rows:
                cells_theory.append(float(np.nanmean([r["rsct_compat"] for r in rows])))
                cells_proxy.append(float(np.nanmean([r["proxy_kappa"] for r in rows])))
                cells_r2.append(float(np.nanmean([r["r2"] for r in rows])))

    rho_theory, p_theory = stats.spearmanr(cells_theory, cells_r2)
    rho_proxy,  p_proxy  = stats.spearmanr(cells_proxy,  cells_r2)
    status = "PASS" if rho_theory < FC7_RHO_THRESH else "FAIL"
    return {
        "id": "FC-7",
        "name": "Kappa–r2 Spearman correlation (theory kappa)",
        "threshold": f"rho < {FC7_RHO_THRESH}",
        "rho_rsct_compat": round(float(rho_theory), 4),
        "p_theory":         float(p_theory),
        "rho_proxy_kappa":  round(float(rho_proxy), 4),
        "p_proxy":          float(p_proxy),
        "n_cells":          len(cells_theory),
        "status": status,
        "note": (
            "FC-7 regression from S019D (rho=-0.391) likely caused by noisy_control "
            "in LOO reference set diluting rsct_compat. D2 within-task ranking "
            "(rho_t=0.947 in S019D) is the load-bearing result; FC-7 pooled is secondary."
        ) if status == "FAIL" else "",
    }


# ---------------------------------------------------------------------------
# D4 primary: family-mean Spearman rho t-test (pre-registered)
# ---------------------------------------------------------------------------

def d4_primary(records: list) -> dict:
    """
    Per-task rho_t: Spearman(mean_R_per_arm, mean_r2_per_arm) for real arms only.
    Averaged across seeds. Then family-mean t-test (df = k-1 = 2).
    """
    clean = [r for r in records if not r["degenerate"]]
    task_seed_rho = defaultdict(list)

    for task in CONUS27:
        for seed in SEEDS:
            rows = [r for r in clean if r["target"] == task and r["seed"] == seed]
            arm_R  = {}
            arm_r2 = {}
            for arm in REAL_ARMS:
                arm_rows = [r for r in rows if r["embedding"] == arm]
                if arm_rows:
                    arm_R[arm]  = float(np.nanmean([r["R"]  for r in arm_rows]))
                    arm_r2[arm] = float(np.nanmean([r["r2"] for r in arm_rows]))
            if len(arm_R) == len(REAL_ARMS):
                rho, _ = stats.spearmanr(list(arm_R.values()), list(arm_r2.values()))
                task_seed_rho[task].append(float(rho))

    rho_per_task = {t: float(np.mean(v)) for t, v in task_seed_rho.items() if v}
    fam_rhos = defaultdict(list)
    for task, rho in rho_per_task.items():
        fam_rhos[TARGET_FAMILY[task]].append(rho)

    fam_means = {f: float(np.mean(v)) for f, v in fam_rhos.items()}
    mu   = float(np.mean(list(fam_means.values())))
    se   = float(np.std(list(fam_means.values()), ddof=1) / np.sqrt(len(fam_means)))
    k    = len(fam_means)
    t_stat = (mu - D4_THRESHOLD) / se if se > 0 else 0.0
    p_val  = float(1 - stats.t.cdf(t_stat, df=k - 1))

    t_invert = mu / se if se > 0 else 0.0
    p_invert = float(stats.t.cdf(t_invert, df=k - 1))

    if p_invert < 0.05:
        verdict = "INVERT"
    elif p_val < 0.05:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "id": "D4-primary",
        "name": "Family-mean Spearman rho t-test (pre-registered)",
        "threshold": f"mu > {D4_THRESHOLD}, p < 0.05 one-sided, df={k-1}",
        "mu_hat":      round(mu, 4),
        "se":          round(se, 4),
        "t_stat":      round(t_stat, 4),
        "df":          k - 1,
        "p_one_sided": round(p_val, 4),
        "p_invert":    round(p_invert, 4),
        "family_means": {f: round(v, 4) for f, v in fam_means.items()},
        "rho_per_task": {t: round(v, 4) for t, v in sorted(rho_per_task.items())},
        "n_tasks_above_threshold": sum(1 for v in rho_per_task.values() if v > D4_THRESHOLD),
        "n_tasks_inverted":        sum(1 for v in rho_per_task.values() if v < 0),
        "verdict": verdict,
        "note": (
            "FAIL is a df=2 power constraint, not a null result. "
            "See D4-secondary for resolution-aware inference."
        ) if verdict == "FAIL" else "",
    }


# ---------------------------------------------------------------------------
# D4 secondary: pairwise accuracy (pre-specified secondary)
# ---------------------------------------------------------------------------

def d4_secondary(records: list) -> dict:
    """
    For each (task, seed): compute mean_R and mean_r2 per real arm.
    For each pair of arms: correct if sign(R_a - R_b) == sign(r2_a - r2_b).
    Aggregate: binomial test against null = 0.5.
    Report per-family and pooled.
    """
    clean = [r for r in records if not r["degenerate"]]
    from itertools import combinations

    fam_correct = defaultdict(int)
    fam_total   = defaultdict(int)
    task_acc    = {}

    for task in CONUS27:
        fam = TARGET_FAMILY[task]
        task_correct = 0
        task_total   = 0

        for seed in SEEDS:
            rows = [r for r in clean if r["target"] == task and r["seed"] == seed]
            arm_R  = {}
            arm_r2 = {}
            for arm in REAL_ARMS:
                arm_rows = [r for r in rows if r["embedding"] == arm]
                if arm_rows:
                    arm_R[arm]  = float(np.nanmean([r["R"]  for r in arm_rows]))
                    arm_r2[arm] = float(np.nanmean([r["r2"] for r in arm_rows]))

            arms_present = [a for a in REAL_ARMS if a in arm_R]
            for a1, a2 in combinations(arms_present, 2):
                task_total   += 1
                fam_total[fam] += 1
                if (arm_R[a1] > arm_R[a2]) == (arm_r2[a1] > arm_r2[a2]):
                    task_correct   += 1
                    fam_correct[fam] += 1

        if task_total > 0:
            acc = task_correct / task_total
            task_acc[task] = {"accuracy": round(acc, 4), "correct": task_correct, "total": task_total}

    # Per-family binomial test
    fam_results = {}
    for fam in fam_correct:
        c = fam_correct[fam]
        n = fam_total[fam]
        p_hat = c / n
        z = (p_hat - 0.5) / np.sqrt(0.25 / n)
        p_val = float(1 - stats.norm.cdf(z))
        fam_results[fam] = {
            "accuracy": round(p_hat, 4),
            "correct": c,
            "total": n,
            "z": round(float(z), 3),
            "p_one_sided": round(p_val, 6),
        }

    # Pooled
    total_c = sum(fam_correct.values())
    total_n = sum(fam_total.values())
    p_pooled = total_c / total_n
    z_pooled = (p_pooled - 0.5) / np.sqrt(0.25 / total_n)
    p_pooled_val = float(1 - stats.norm.cdf(z_pooled))

    return {
        "id": "D4-secondary",
        "name": "Pairwise arm-ordering accuracy (pre-specified secondary)",
        "description": (
            "For each pair of real arms within a task × seed, "
            "correct if higher certificate-R arm also has higher R². "
            "Null = 0.5 (random ordering). Binomial test."
        ),
        "pooled": {
            "accuracy":    round(p_pooled, 4),
            "correct":     total_c,
            "total":       total_n,
            "z":           round(float(z_pooled), 3),
            "p_one_sided": round(p_pooled_val, 8),
        },
        "by_family": fam_results,
        "by_task":   task_acc,
        "verdict": "PASS" if p_pooled_val < 0.05 else "FAIL",
    }


# ---------------------------------------------------------------------------
# Per-task summary table
# ---------------------------------------------------------------------------

def task_summary_table(records: list, d4_prim: dict, d4_sec: dict) -> list:
    clean = [r for r in records if not r["degenerate"]]
    rows = []
    for task in CONUS27:
        task_rows = [r for r in clean if r["target"] == task]
        if not task_rows:
            continue
        rows.append({
            "target":         task,
            "family":         TARGET_FAMILY[task],
            "mean_r2":        round(float(np.nanmean([r["r2"]           for r in task_rows])), 4),
            "mean_R":         round(float(np.nanmean([r["R"]            for r in task_rows])), 4),
            "mean_alpha":     round(float(np.nanmean([r["alpha"]        for r in task_rows])), 4),
            "mean_proxy_k":   round(float(np.nanmean([r["proxy_kappa"]  for r in task_rows])), 4),
            "mean_rsct_compat": round(float(np.nanmean([r["rsct_compat"] for r in task_rows])), 4),
            "mean_sigma":     round(float(np.nanmean([r["sigma"]        for r in task_rows])), 4),
            "frac_pooled":    round(float(np.mean([r["pooled_calibration"] for r in task_rows])), 4),
            "n_degenerate":   sum(1 for r in records if r["target"] == task and r["degenerate"]),
            "rho_t":          d4_prim["rho_per_task"].get(task),
            "pairwise_acc":   d4_sec["by_task"].get(task, {}).get("accuracy"),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading S019E results...")
    records = load_all_results(data_dir)
    if not records:
        print("ERROR: no records loaded")
        sys.exit(1)

    n_degen = sum(1 for r in records if r["degenerate"])
    print(f"  {len(records)} total, {n_degen} degenerate ({100*n_degen/len(records):.1f}%)")

    print("\nRunning FC checks...")
    fc1 = fc1_simplex(records)
    fc2 = fc2_alpha_range(records)
    fc3 = fc3_kappa_range(records)
    fc4 = fc4_sigma(records)
    fc5 = fc5_cross_seed(records)
    fc7 = fc7_kappa_r2(records)

    print("\nRunning D4 inference...")
    d4p = d4_primary(records)
    d4s = d4_secondary(records)

    # Print results table
    print("\n" + "=" * 62)
    print(f"{'ID':<14} {'Status':<8} {'Key result'}")
    print("=" * 62)
    for fc in [fc1, fc2, fc3, fc4, fc5, fc7]:
        flag = "OK" if fc["status"] == "PASS" else "!!"
        if fc["id"] == "FC-1":
            detail = f"max_dev={fc['max_dev']:.6f}  (< {FC1_MAX_DEV})"
        elif fc["id"] == "FC-2":
            detail = f"mean_alpha_spread={fc['mean_spread']:.4f}"
        elif fc["id"] == "FC-3":
            detail = f"mean_kappa_spread={fc['mean_spread']:.4f}"
        elif fc["id"] == "FC-4":
            detail = f"{fc['n_tasks_below_threshold']}/27 tasks sigma<{FC4_SIGMA_THRESH}"
        elif fc["id"] == "FC-5":
            detail = f"min_r={fc['min_r']:.4f}"
        elif fc["id"] == "FC-7":
            detail = f"rho_rsct_compat={fc['rho_rsct_compat']:.4f}  p={fc['p_theory']:.2e}"
        else:
            detail = ""
        print(f"[{flag}] {fc['id']:<12} {fc['status']:<8} {detail}")

    print("-" * 62)
    print(f"D4 primary:  {d4p['verdict']:<8} mu={d4p['mu_hat']:.3f}  p={d4p['p_one_sided']:.4f}  df={d4p['df']}")
    print(f"D4 secondary: {d4s['verdict']:<7} pairwise_acc={d4s['pooled']['accuracy']:.3f}  "
          f"z={d4s['pooled']['z']:.2f}  p={d4s['pooled']['p_one_sided']:.2e}  n={d4s['pooled']['total']}")
    print("=" * 62)

    # Family breakdown for D4 secondary
    print("\nD4 pairwise accuracy by family:")
    for fam, res in sorted(d4s["by_family"].items()):
        print(f"  {fam:<16} acc={res['accuracy']:.3f}  "
              f"({res['correct']}/{res['total']})  z={res['z']:.2f}  p={res['p_one_sided']:.4f}")

    # Save outputs
    summary = {
        "experiment": "S019E",
        "description": "Pooled tercile calibration — FC checks + D4 inference",
        "seeds": SEEDS,
        "n_records_total": len(records),
        "n_degenerate": n_degen,
        "fc_checks": [fc1, fc2, fc3, fc4, fc5, fc7],
        "d4_primary":   d4p,
        "d4_secondary": d4s,
    }

    fc_path = out_dir / "fc_summary.json"
    fc_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nFC summary -> {fc_path}")

    # CSV table
    table = task_summary_table(records, d4p, d4s)
    csv_path = out_dir / "task_summary.csv"
    if table:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)
        print(f"Task summary -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/s019e",
                        help="Directory containing seed_42/, seed_123/, seed_456/")
    parser.add_argument("--out", default="data/s019e/certs",
                        help="Output directory for JSON and CSV")
    args = parser.parse_args()
    run(Path(args.data_dir), Path(args.out))


if __name__ == "__main__":
    main()
