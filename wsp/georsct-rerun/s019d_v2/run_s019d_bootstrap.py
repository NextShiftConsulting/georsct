#!/usr/bin/env python3
"""
run_s019d_bootstrap.py -- S019D Bootstrap Confidence Intervals

Computes 95% CIs on N-ceiling per task via fold-level block bootstrap (B=1000).
Also reports cross-seed consistency and within-task kappa correlation CIs.

Bootstrap method:
  For each task, the 5-fold x 6-embedding R2 matrix is resampled with
  replacement over folds (preserving the fold-as-block structure).
  N-ceiling = 1 - max_k(mean_R2_k) over the resampled fold set.
  B=1000 iterations, 2.5th/97.5th percentile CIs.

Cross-seed consistency:
  Loads seed_42, seed_123, seed_456 results.
  Reports per-task N-ceiling mean and std across seeds.

Within-task kappa ρ bootstrap:
  For each task: Spearman ρ between theory_kappa and r2 across 6 embeddings.
  Bootstraps over tasks (B=1000) to get CI on mean ρ across 27 tasks.

Inputs (downloaded from S3 at runtime using IAM role):
  s3://swarm-yrsn-datasets/rsct_curriculum/series_019/results/s019d/seed_N/s019d_results.json

Output:
  s019d_bootstrap_cis.json   -- per-task N-ceiling CIs + cross-seed + kappa ρ CIs
  s019d_bootstrap_cis.csv    -- paper-ready CSV
  s019d_bootstrap_summary.json -- summary stats for paper claims
"""

import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BUCKET = "swarm-yrsn-datasets"
RESULTS_PREFIX = "rsct_curriculum/series_019_v2/results/s019d"
SEEDS = [42, 123, 456]
EMBEDDINGS = [
    "pca_v1", "spatial_lag_v1", "gnn_v2",
    "geo_spatial", "noisy_control", "domain_features",
]
N_FOLDS = 5
N_BOOT = 1000
PRIMARY_SEED = 42

HEALTH_TASKS = [
    "annual_checkup", "arthritis", "asthma", "binge_drinking", "bp_medicated",
    "cancer", "cholesterol_screening", "chronic_kidney_disease", "copd",
    "coronary_heart_disease", "dental_visit", "diabetes",
    "high_blood_pressure", "high_cholesterol", "mental_health_not_good",
    "obesity", "physical_health_not_good", "physical_inactivity",
    "sleep_less_7hr", "smoking", "stroke",
]
SOCIO_TASKS = ["income", "home_value", "population_density"]
ENV_TASKS = ["elevation", "night_lights", "tree_cover"]


def task_family(task):
    if task in HEALTH_TASKS:
        return "health"
    if task in SOCIO_TASKS:
        return "socioeconomic"
    if task in ENV_TASKS:
        return "environmental"
    return "unknown"


def load_results_json(seed, local_dir=None):
    """Load s019d_results.json for one seed from S3 or local path."""
    if local_dir:
        path = Path(local_dir) / f"seed_{seed}" / "s019d_results.json"
        with open(path) as f:
            return json.load(f)

    on_sagemaker = os.path.isdir("/opt/ml")
    profile = None if on_sagemaker else "nsc-swarm"
    session = boto3.Session(profile_name=profile, region_name="us-east-1")
    s3 = session.client("s3")
    key = f"{RESULTS_PREFIX}/seed_{seed}/s019d_results.json"
    log.info("Loading s3://%s/%s", BUCKET, key)
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.load(BytesIO(obj["Body"].read()))


def build_r2_matrix(records):
    """
    Build a dict: task -> {embedding: [r2_fold0, r2_fold1, ...]}
    from the flat list of per-(task, embedding, fold) records.
    """
    matrix = {}
    for row in records:
        task = row["target"]
        emb = row["embedding"]
        r2 = row["r2"]
        if task not in matrix:
            matrix[task] = {}
        if emb not in matrix[task]:
            matrix[task][emb] = []
        matrix[task][emb].append(r2)
    return matrix


def compute_n_ceiling(r2_matrix, task):
    """N_ceiling = 1 - max_k(mean R2_k) for one task."""
    emb_means = []
    for emb in EMBEDDINGS:
        folds = r2_matrix.get(task, {}).get(emb, [])
        if folds:
            emb_means.append(float(np.mean(folds)))
    if not emb_means:
        return np.nan
    return 1.0 - max(emb_means)


def fold_block_bootstrap_n_ceiling(r2_matrix, task, n_boot=N_BOOT, seed=0):
    """
    Fold-level block bootstrap for N_ceiling of one task.

    For each bootstrap replicate:
      - Resample 5 folds with replacement (5->5)
      - Per embedding: mean R2 over resampled folds
      - N_ceiling = 1 - max_k(mean R2_k)

    Returns (point_estimate, ci_low, ci_high, boot_samples).
    """
    rng = np.random.RandomState(seed)

    # Build [n_emb x n_folds] matrix; missing -> nan
    emb_fold_r2 = np.full((len(EMBEDDINGS), N_FOLDS), np.nan)
    for i, emb in enumerate(EMBEDDINGS):
        folds = r2_matrix.get(task, {}).get(emb, [])
        for j, r2 in enumerate(folds[:N_FOLDS]):
            emb_fold_r2[i, j] = r2

    # Point estimate
    emb_means = np.nanmean(emb_fold_r2, axis=1)
    valid_means = emb_means[~np.isnan(emb_means)]
    if len(valid_means) == 0:
        return np.nan, np.nan, np.nan, []

    point = float(1.0 - np.max(valid_means))

    # Bootstrap over folds
    boot_samples = []
    fold_indices = np.arange(N_FOLDS)
    for _ in range(n_boot):
        sampled_folds = rng.choice(fold_indices, size=N_FOLDS, replace=True)
        boot_means = np.nanmean(emb_fold_r2[:, sampled_folds], axis=1)
        valid_boot = boot_means[~np.isnan(boot_means)]
        if len(valid_boot) == 0:
            continue
        boot_samples.append(float(1.0 - np.max(valid_boot)))

    if len(boot_samples) < 100:
        return point, np.nan, np.nan, boot_samples

    boot_arr = np.array(boot_samples)
    ci_low = float(np.percentile(boot_arr, 2.5))
    ci_high = float(np.percentile(boot_arr, 97.5))
    return point, ci_low, ci_high, boot_samples


def bootstrap_kappa_rho(records, n_boot=N_BOOT, seed=0):
    """
    Bootstrap CI on mean within-task Spearman rho (kappa vs R2 across embeddings).

    For each task: compute Spearman rho between [6 embedding mean R2] and
    [6 embedding mean theory_kappa]. Then bootstrap over the 27 tasks to get
    a CI on the mean rho.

    Returns dict with point estimate, CI, and per-task rho values.
    """
    rng = np.random.RandomState(seed)

    # Per-task, per-embedding: mean R2 and mean theory_kappa across folds
    task_emb_r2 = {}
    task_emb_kappa = {}
    for row in records:
        task = row["target"]
        emb = row["embedding"]
        r2 = row["r2"]
        kappa = row.get("theory_kappa", np.nan)
        if task not in task_emb_r2:
            task_emb_r2[task] = {}
            task_emb_kappa[task] = {}
        if emb not in task_emb_r2[task]:
            task_emb_r2[task][emb] = []
            task_emb_kappa[task][emb] = []
        task_emb_r2[task][emb].append(r2)
        if kappa is not None and not (isinstance(kappa, float) and np.isnan(kappa)):
            task_emb_kappa[task][emb].append(kappa)

    tasks = sorted(task_emb_r2.keys())
    per_task_rho = {}
    for task in tasks:
        r2_vec = []
        kappa_vec = []
        for emb in EMBEDDINGS:
            r2_folds = task_emb_r2.get(task, {}).get(emb, [])
            kappa_folds = task_emb_kappa.get(task, {}).get(emb, [])
            if r2_folds and kappa_folds:
                r2_vec.append(float(np.mean(r2_folds)))
                kappa_vec.append(float(np.mean(kappa_folds)))

        if len(r2_vec) >= 3:
            rho, p = spearmanr(kappa_vec, r2_vec)
            per_task_rho[task] = {"rho": float(rho), "p": float(p), "n_emb": len(r2_vec)}
        else:
            per_task_rho[task] = {"rho": np.nan, "p": np.nan, "n_emb": len(r2_vec)}

    rho_vals = np.array([v["rho"] for v in per_task_rho.values() if np.isfinite(v["rho"])])
    if len(rho_vals) == 0:
        return {"mean_rho": np.nan, "ci_low": np.nan, "ci_high": np.nan, "per_task": per_task_rho}

    point_mean = float(np.mean(rho_vals))
    n_correct = int(np.sum(rho_vals >= 0.8))

    # Bootstrap over tasks (resample 27 tasks with replacement)
    task_indices = np.arange(len(rho_vals))
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(task_indices, size=len(rho_vals), replace=True)
        boot_means.append(float(np.mean(rho_vals[idx])))

    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    return {
        "mean_rho": round(point_mean, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "n_tasks": len(rho_vals),
        "n_correct_top_rank": n_correct,
        "per_task": per_task_rho,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="S019D bootstrap confidence intervals")
    parser.add_argument("--local-dir", default=None,
                        help="Local directory with seed_N/ subdirs (for local testing)")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output",
                        help="Output directory for results")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help="Number of bootstrap replicates (default: 1000)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Seeds to load (default: 42 123 456)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("S019D Bootstrap CI Computation")
    log.info("=" * 60)
    log.info("Seeds: %s", args.seeds)
    log.info("B: %d", args.n_boot)
    log.info("Output: %s", out)
    log.info("=" * 60)

    # --- Load all seed results ---
    seed_records = {}
    for seed in args.seeds:
        try:
            records = load_results_json(seed, local_dir=args.local_dir)
            seed_records[seed] = records
            log.info("Seed %d: %d records loaded", seed, len(records))
        except Exception as e:
            log.warning("Seed %d: failed to load -- %s", seed, e)

    if PRIMARY_SEED not in seed_records:
        log.error("Primary seed %d not loaded. Aborting.", PRIMARY_SEED)
        sys.exit(1)

    primary_records = seed_records[PRIMARY_SEED]
    primary_matrix = build_r2_matrix(primary_records)
    all_tasks = sorted(primary_matrix.keys())
    log.info("Tasks: %d, Embeddings: %d, Folds: %d",
             len(all_tasks), len(EMBEDDINGS), N_FOLDS)

    # --- Per-task N-ceiling bootstrap (primary seed) ---
    log.info("\nFold-level block bootstrap, B=%d...", args.n_boot)
    per_task_bootstrap = {}
    for task in all_tasks:
        point, ci_low, ci_high, boot_samples = fold_block_bootstrap_n_ceiling(
            primary_matrix, task, n_boot=args.n_boot, seed=PRIMARY_SEED,
        )
        ci_width = (ci_high - ci_low) if (np.isfinite(ci_low) and np.isfinite(ci_high)) else np.nan
        per_task_bootstrap[task] = {
            "n_ceiling": round(point, 4) if np.isfinite(point) else None,
            "ci_low": round(ci_low, 4) if np.isfinite(ci_low) else None,
            "ci_high": round(ci_high, 4) if np.isfinite(ci_high) else None,
            "ci_width": round(ci_width, 4) if np.isfinite(ci_width) else None,
            "n_boot_valid": len(boot_samples),
            "family": task_family(task),
        }
        log.info("  %-35s N_ceil=%.4f CI=[%.4f, %.4f] width=%.4f",
                 task, point,
                 ci_low if np.isfinite(ci_low) else -999,
                 ci_high if np.isfinite(ci_high) else -999,
                 ci_width if np.isfinite(ci_width) else -999)

    # --- Cross-seed N-ceiling consistency ---
    log.info("\nCross-seed N-ceiling consistency...")
    cross_seed_per_task = {}
    for task in all_tasks:
        seed_vals = {}
        for seed, records in seed_records.items():
            matrix = build_r2_matrix(records)
            nc = compute_n_ceiling(matrix, task)
            seed_vals[seed] = round(float(nc), 4) if np.isfinite(nc) else None
        vals = [v for v in seed_vals.values() if v is not None]
        cross_seed_per_task[task] = {
            "per_seed": seed_vals,
            "mean": round(float(np.mean(vals)), 4) if vals else None,
            "std": round(float(np.std(vals)), 4) if len(vals) > 1 else None,
            "min": round(float(np.min(vals)), 4) if vals else None,
            "max": round(float(np.max(vals)), 4) if vals else None,
        }

    cross_seed_stds = [v["std"] for v in cross_seed_per_task.values() if v["std"] is not None]
    if cross_seed_stds:
        log.info("  Cross-seed N_ceiling std: mean=%.4f max=%.4f",
                 np.mean(cross_seed_stds), np.max(cross_seed_stds))

    # --- Within-task kappa rho bootstrap ---
    log.info("\nWithin-task kappa rho bootstrap (B=%d)...", args.n_boot)
    kappa_rho = bootstrap_kappa_rho(primary_records, n_boot=args.n_boot, seed=PRIMARY_SEED)
    log.info("  Mean within-task rho=%.4f CI=[%.4f, %.4f], %d/%d tasks rho>=0.8",
             kappa_rho["mean_rho"], kappa_rho["ci_low"], kappa_rho["ci_high"],
             kappa_rho["n_correct_top_rank"], kappa_rho["n_tasks"])

    # --- CI summary stats ---
    ci_widths = [v["ci_width"] for v in per_task_bootstrap.values() if v["ci_width"] is not None]
    n_ceiling_vals = [v["n_ceiling"] for v in per_task_bootstrap.values() if v["n_ceiling"] is not None]

    log.info("\n" + "=" * 60)
    log.info("BOOTSTRAP CI SUMMARY (primary seed=%d, B=%d)", PRIMARY_SEED, args.n_boot)
    log.info("  N-ceiling range: %.4f - %.4f", min(n_ceiling_vals), max(n_ceiling_vals))
    log.info("  CI width: mean=%.4f median=%.4f max=%.4f",
             np.mean(ci_widths), np.median(ci_widths), np.max(ci_widths))
    log.info("=" * 60)

    # --- Write outputs ---

    # Main bootstrap CIs JSON
    output = {
        "experiment": "S019D",
        "description": "Fold-level block bootstrap CIs on N-ceiling (B=1000, 6-family)",
        "primary_seed": PRIMARY_SEED,
        "seeds_loaded": sorted(seed_records.keys()),
        "n_boot": args.n_boot,
        "bootstrap_method": "fold-level block (resample 5 folds with replacement per task)",
        "n_tasks": len(all_tasks),
        "n_embeddings": len(EMBEDDINGS),
        "embeddings": EMBEDDINGS,
        "n_ceiling_summary": {
            "min": round(min(n_ceiling_vals), 4),
            "max": round(max(n_ceiling_vals), 4),
            "mean": round(float(np.mean(n_ceiling_vals)), 4),
            "range_ratio": round(max(n_ceiling_vals) / min(n_ceiling_vals), 1) if min(n_ceiling_vals) > 0 else None,
        },
        "ci_width_summary": {
            "mean": round(float(np.mean(ci_widths)), 4),
            "median": round(float(np.median(ci_widths)), 4),
            "max": round(float(np.max(ci_widths)), 4),
            "min": round(float(np.min(ci_widths)), 4),
        },
        "cross_seed_std_summary": {
            "mean": round(float(np.mean(cross_seed_stds)), 4) if cross_seed_stds else None,
            "max": round(float(np.max(cross_seed_stds)), 4) if cross_seed_stds else None,
        },
        "kappa_rho": {
            "mean_within_task_rho": kappa_rho["mean_rho"],
            "ci_low": kappa_rho["ci_low"],
            "ci_high": kappa_rho["ci_high"],
            "n_tasks": kappa_rho["n_tasks"],
            "n_correct_top_rank": kappa_rho["n_correct_top_rank"],
        },
        "per_task": {
            task: {
                **per_task_bootstrap[task],
                "cross_seed": cross_seed_per_task.get(task, {}),
            }
            for task in all_tasks
        },
    }

    cis_path = out / "s019d_bootstrap_cis.json"
    with open(cis_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved %s", cis_path)

    # Paper-ready CSV
    import csv
    csv_path = out / "s019d_bootstrap_cis.csv"
    fieldnames = [
        "task", "family", "n_ceiling", "ci_low", "ci_high", "ci_width",
        "n_boot_valid", "cross_seed_mean", "cross_seed_std",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task in all_tasks:
            bt = per_task_bootstrap[task]
            cs = cross_seed_per_task.get(task, {})
            writer.writerow({
                "task": task,
                "family": bt["family"],
                "n_ceiling": bt["n_ceiling"],
                "ci_low": bt["ci_low"],
                "ci_high": bt["ci_high"],
                "ci_width": bt["ci_width"],
                "n_boot_valid": bt["n_boot_valid"],
                "cross_seed_mean": cs.get("mean"),
                "cross_seed_std": cs.get("std"),
            })
    log.info("Saved %s", csv_path)

    # Summary JSON (compact -- paper-claim level)
    summary = {
        "experiment": "S019D",
        "primary_seed": PRIMARY_SEED,
        "n_boot": args.n_boot,
        "n_ceiling_range": [
            round(min(n_ceiling_vals), 4),
            round(max(n_ceiling_vals), 4),
        ],
        "n_ceiling_range_ratio": output["n_ceiling_summary"]["range_ratio"],
        "ci_width_mean": output["ci_width_summary"]["mean"],
        "ci_width_max": output["ci_width_summary"]["max"],
        "cross_seed_std_mean": output["cross_seed_std_summary"]["mean"],
        "cross_seed_std_max": output["cross_seed_std_summary"]["max"],
        "kappa_rho_mean": kappa_rho["mean_rho"],
        "kappa_rho_ci": [kappa_rho["ci_low"], kappa_rho["ci_high"]],
        "kappa_rho_n_correct_top_rank": kappa_rho["n_correct_top_rank"],
    }
    summary_path = out / "s019d_bootstrap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved %s", summary_path)

    log.info("\n" + "=" * 60)
    log.info("DONE. S019D bootstrap CIs ready for camera-ready.")
    log.info("N-ceiling: %.4f - %.4f (%.1fx range)",
             min(n_ceiling_vals), max(n_ceiling_vals),
             output["n_ceiling_summary"]["range_ratio"])
    log.info("CI widths: mean=%.4f max=%.4f", np.mean(ci_widths), np.max(ci_widths))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
