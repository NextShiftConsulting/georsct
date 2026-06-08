#!/usr/bin/env python3
"""
run_s018b_extrapolation.py -- S018B V2 State-Holdout Extrapolation Benchmark

Replicates S018B with all MMAR fixes applied:
  Fix 1: GNN zcta_id alignment (shared/representations.py)
  Fix 2: kappa_gate -> kappa_compat (domain/certificate.py)
  Fix 3: shared_boundaries=True (pooled tercile calibration)
  Fix 4: S3 prefixes -> series_019_v2/
  Fix 6: State-level holdout for geographic extrapolation

Protocol (PDFM arXiv 2411.07207v6 sec 3.2):
  Hold out ~20% of US states (seed-dependent).
  For each holdout state set:
    - Train on all ZCTAs NOT in holdout states
    - Evaluate on held-out ZCTAs
    - Run full RSCT certification with theory kappa + gatekeepers

6 embeddings x 27 targets x N holdout folds x 1 solver = ~810+ fits.
Each certificate evaluated under flat + oobleck gatekeepers.

Data: geocert v23.0.2 (georsct_table.parquet, 106 columns, ~32k ZCTAs)

Compute profile:
  - Instance: ml.m5.4xlarge (16 vCPU, 64 GB RAM)
  - Image: pytorch-training:2.9.0-cpu-py312 (no GPU needed)
  - Timeout: 12 hours (43200s)
  - Checkpointing: per-target S3 checkpoint for crash recovery

Paper reference: GeoRSCT V3, Table 2 (S018B extrapolation).
"""

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Shared imports -- handle both SageMaker and local layouts
_here = Path(__file__).parent
for _base in [_here, _here.parent]:
    _s019a = _base / "s019a_certificate_invariance_gradient"
    if _s019a.is_dir():
        sys.path.insert(0, str(_s019a))
    _p = str(_base)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.constants import CONUS27_TASKS, N_FOLDS

# S018B V2 uses same 6 embeddings as S019D
EMBEDDINGS = [
    "pca_v1", "spatial_lag_v1", "gnn_v2",
    "geo_spatial", "noisy_control", "domain_features",
]
from shared.theory_certifier import certify_group
from run_s019a import _build_embeddings

# yrsn imports
from yrsn.core.decomposition.instability_computation import compute_sigma_request
from yrsn.core.certificates.estimate import CPGatekeeperInput
from yrsn_controlplane import SequentialGatekeeper, GatekeeperConfig, get_preset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = int(os.environ.get("S018B_SEED", "42"))

# PDFM extrapolation baselines (arXiv 2411.07207v6 sec 3.2, Table 2)
PDFM_EXTRAP_ALL_MEAN = 0.70
PDFM_EXTRAP_HEALTH_MEAN = 0.58

HEALTH_TASKS = {
    "annual_checkup", "arthritis", "asthma", "binge_drinking", "cancer",
    "chronic_kidney_disease", "copd", "coronary_heart_disease", "diabetes",
    "high_blood_pressure", "bp_medicated", "high_cholesterol",
    "mental_health_not_good", "obesity", "physical_health_not_good",
    "physical_inactivity", "cholesterol_screening", "dental_visit",
    "sleep_less_7hr", "smoking", "stroke",
}

TARGET_FAMILY = {
    "annual_checkup": "health", "arthritis": "health", "asthma": "health",
    "binge_drinking": "health", "bp_medicated": "health", "cancer": "health",
    "cholesterol_screening": "health", "chronic_kidney_disease": "health",
    "copd": "health", "coronary_heart_disease": "health", "dental_visit": "health",
    "diabetes": "health", "high_blood_pressure": "health",
    "high_cholesterol": "health", "mental_health_not_good": "health",
    "obesity": "health", "physical_health_not_good": "health",
    "physical_inactivity": "health", "sleep_less_7hr": "health",
    "smoking": "health", "stroke": "health",
    "home_value": "socioeconomic", "income": "socioeconomic",
    "population_density": "socioeconomic",
    "elevation": "environmental", "night_lights": "environmental",
    "tree_cover": "environmental",
}


# ---------------------------------------------------------------------------
# Gatekeeper setup (identical to S019D)
# ---------------------------------------------------------------------------

def build_gatekeepers():
    """Build flat and oobleck gatekeepers."""
    flat_config = get_preset("geospatial-conus27")
    oobleck_config = GatekeeperConfig(
        N_thr=flat_config.N_thr,
        alpha_min=flat_config.alpha_min,
        c_min=flat_config.c_min,
        gate_2_require_coherence=flat_config.gate_2_require_coherence,
        sigma_thr=flat_config.sigma_thr,
        kappa_base=flat_config.kappa_base,
        lambda_turbulence=0.4,
        steepness=10.0,
        sigma_c=0.35,
        epsilon_L=flat_config.epsilon_L,
        enable_gate_3b=flat_config.enable_gate_3b,
        r_bar_min=flat_config.r_bar_min,
        gate_3b_action=flat_config.gate_3b_action,
        kappa_L_min=flat_config.kappa_L_min,
    )
    return {
        "flat": SequentialGatekeeper(flat_config),
        "oobleck": SequentialGatekeeper(oobleck_config),
    }


# ---------------------------------------------------------------------------
# Gate evaluation helper (identical to S019D)
# ---------------------------------------------------------------------------

def evaluate_gates(cert: dict, gatekeepers: dict) -> dict:
    """Evaluate one certificate under both gatekeepers."""
    sigma = float(compute_sigma_request(cert["N"]))
    cert_input = CPGatekeeperInput(
        alpha=cert["alpha"],
        kappa_compat=cert["theory_kappa"],
        sigma=sigma,
        source_mode="direct",
        evidence={
            "N": cert["N"],
            "R": cert["R"],
            "S": cert["S_sup"],
            "noise_admissibility": cert["N"],
            "omega": cert["omega"],
            "entropy": cert["entropy"],
            "collapse_risk": cert["collapse_risk"],
            "kappa_mean": cert["theory_kappa_mean"],
            "kappa_std": cert.get("theory_sigma", 0.0),
            "n_samples": cert["n_test"],
        },
    )

    gate_results = {}
    for gk_name, gk in gatekeepers.items():
        result = gk.evaluate(cert_input)
        g3_ev = result.gate_evidence.get("gate_3_admissibility", {})
        kappa_req = float(g3_ev["kappa_req"]) if "kappa_req" in g3_ev else None
        margin = float(cert["theory_kappa"] - kappa_req) if kappa_req is not None else None

        gate_results[f"gate_{gk_name}"] = {
            "gate_decision": str(result.decision),
            "kappa_req": kappa_req,
            "margin": margin,
            "gate_reached": str(result.gate_reached),
            "failure_reason": result.failure_reason,
        }

    return gate_results


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CHECKPOINT_BUCKET = os.environ.get("CHECKPOINT_BUCKET", "swarm-yrsn-datasets")
CHECKPOINT_PREFIX = os.environ.get(
    "CHECKPOINT_PREFIX",
    f"rsct_curriculum/series_019_v2/results/s018b_v2/seed_{SEED}/checkpoints",
)


def _s3_client():
    """Lazy S3 client (IAM role in container)."""
    import boto3
    return boto3.client("s3", region_name="us-east-1")


def _checkpoint_key(target: str) -> str:
    return f"{CHECKPOINT_PREFIX}/{target}.json"


def _save_checkpoint(target: str, data: list):
    """Write one target's results directly to S3."""
    try:
        key = _checkpoint_key(target)
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        _s3_client().put_object(Bucket=CHECKPOINT_BUCKET, Key=key, Body=body)
        log.info("  checkpoint -> s3://%s/%s", CHECKPOINT_BUCKET, key)
    except Exception as e:
        log.warning("  checkpoint FAILED for %s: %s", target, e)


def _load_completed_targets() -> set:
    """Check S3 for targets already checkpointed."""
    try:
        prefix = f"{CHECKPOINT_PREFIX}/"
        resp = _s3_client().list_objects_v2(Bucket=CHECKPOINT_BUCKET, Prefix=prefix)
        completed = set()
        for obj in resp.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if name.endswith(".json"):
                completed.add(name[:-5])
        if completed:
            log.info("Resuming: found %d completed checkpoints", len(completed))
        return completed
    except Exception as e:
        log.warning("Could not check for existing checkpoints: %s", e)
        return set()


def _load_checkpoint(target: str) -> list:
    """Load a previously checkpointed target's results from S3."""
    try:
        key = _checkpoint_key(target)
        resp = _s3_client().get_object(Bucket=CHECKPOINT_BUCKET, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# State holdout fold generation
# ---------------------------------------------------------------------------

def generate_state_holdout_folds(
    df, seed: int, n_holdout_frac: float = 0.20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate state-holdout folds following PDFM protocol.

    Randomly selects ~20% of states as holdout. Each holdout state
    becomes one fold: train on all non-holdout ZCTAs, test on that
    state's ZCTAs.

    Returns list of (train_indices, test_indices) tuples.
    """
    all_states = sorted(df["state"].unique())
    rng = np.random.default_rng(seed)
    n_holdout = max(1, round(len(all_states) * n_holdout_frac))
    holdout_states = sorted(rng.choice(all_states, size=n_holdout, replace=False))

    log.info("State holdout: %d/%d states held out", len(holdout_states), len(all_states))
    log.info("Holdout states: %s", holdout_states)

    folds = []
    for state in holdout_states:
        mask = df["state"].values == state
        test_idx = np.where(mask)[0]
        train_idx = np.where(~mask)[0]
        folds.append((train_idx, test_idx))

    return folds, holdout_states


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(data_dir: str, repr_dir: str, output_dir: str):
    """Run full S018B V2 state-holdout extrapolation experiment."""
    import pandas as pd
    from joblib import Parallel, delayed

    log.info("S018B V2: State-Holdout Extrapolation Benchmark (seed=%d)", SEED)
    log.info("  Protocol: PDFM sec 3.2 -- random 20%% state holdout")
    log.info("  Targets: %d CONUS-27 tasks", len(CONUS27_TASKS))
    log.info("  Embeddings: %s", EMBEDDINGS)
    log.info("  Solver: HistGBDT")
    log.info("  Gatekeepers: flat + oobleck (sigmoidal)")
    log.info("  PDFM baselines: all=%.2f, health=%.2f",
             PDFM_EXTRAP_ALL_MEAN, PDFM_EXTRAP_HEALTH_MEAN)

    t0 = time.time()

    # Load data
    data_path = Path(data_dir)
    feat_path = data_path / "georsct_table.parquet"
    if not feat_path.exists():
        feat_path = data_path / "zcta_features_labels.parquet"
    df = pd.read_parquet(feat_path)
    log.info("Loaded %d ZCTAs, %d states", len(df), df["state"].nunique())

    if "state" not in df.columns:
        raise KeyError("DataFrame missing 'state' column for state-holdout protocol")

    acs_cols = sorted(
        c for c in df.columns
        if c.startswith("acs_") and pd.api.types.is_numeric_dtype(df[c])
    )

    # Build embeddings (reuse S019A builder)
    repr_path = Path(repr_dir) if repr_dir else None
    embeddings = _build_embeddings(df, acs_cols, repr_path)

    # Generate state-holdout folds
    folds, holdout_states = generate_state_holdout_folds(df, SEED)
    n_folds = len(folds)
    log.info("Generated %d state-holdout folds", n_folds)

    # Gatekeepers
    gatekeepers = build_gatekeepers()
    n_cpus = os.cpu_count() or 1
    log.info("Using %d CPUs for parallel folds", n_cpus)

    # --------------- Pre-compute per-target data ---------------
    target_y = {}
    target_mask = {}
    for target in CONUS27_TASKS:
        target_col = f"target_{target}" if f"target_{target}" in df.columns else target
        y = df[target_col].values.astype(np.float64)
        nan_mask = ~np.isnan(y)
        target_y[target] = y
        target_mask[target] = nan_mask

    # --------------- Fold-level worker ---------------
    def _fit_one_fold(emb_dict_fold, y_train, y_test, fold_idx, seed):
        """Certify one fold across all embeddings."""
        t_start = time.time()
        results = certify_group(
            emb_dict_fold, y_train, y_test,
            solver_name="histgbdt", seed=seed + fold_idx,
            shared_boundaries=True,  # Fix 3: pooled tercile calibration
        )
        wall = round(time.time() - t_start, 2)
        for r in results:
            r["wall_clock_s"] = wall
        return results

    # --------------- Primary run with checkpoint/resume ---------------
    all_results = []
    failed_targets = []
    completed = _load_completed_targets()

    for target in completed & set(CONUS27_TASKS):
        restored = _load_checkpoint(target)
        if restored:
            all_results.extend(restored)
            log.info("  Restored %d rows for %s from checkpoint", len(restored), target)

    for task_idx, target in enumerate(CONUS27_TASKS):
        if target in completed:
            log.info("  [%d/%d] %s: SKIP (already checkpointed)",
                     task_idx + 1, len(CONUS27_TASKS), target)
            continue

        nan_mask = target_mask[target]
        y_full = target_y[target]

        try:
            # Build per-fold embedding dicts using state-holdout splits
            fold_jobs = []
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                # Intersect with non-NaN mask for this target
                train_valid = train_idx[nan_mask[train_idx]]
                test_valid = test_idx[nan_mask[test_idx]]

                if len(test_valid) < 20:
                    log.warning("  %s fold %d (%s): only %d test samples, skipping",
                                target, fold_idx, holdout_states[fold_idx], len(test_valid))
                    continue

                emb_dict_fold = {
                    emb_name: {
                        "train": embeddings[emb_name][train_valid],
                        "test": embeddings[emb_name][test_valid],
                    }
                    for emb_name in EMBEDDINGS
                }
                fold_jobs.append((
                    fold_idx,
                    delayed(_fit_one_fold)(
                        emb_dict_fold,
                        y_full[train_valid],
                        y_full[test_valid],
                        fold_idx, SEED,
                    ),
                ))

            if not fold_jobs:
                log.warning("  %s: no valid folds, skipping", target)
                continue

            # Parallelize over folds
            fold_indices = [fj[0] for fj in fold_jobs]
            fold_delayed = [fj[1] for fj in fold_jobs]
            fold_results = Parallel(n_jobs=n_cpus, prefer="processes")(fold_delayed)

            target_rows = []
            for fold_idx, fold_certs in zip(fold_indices, fold_results):
                for cert in fold_certs:
                    gate_evals = evaluate_gates(cert, gatekeepers)

                    row = {k: v for k, v in cert.items() if not k.startswith("_")}
                    row["fold"] = fold_idx
                    row["holdout_state"] = holdout_states[fold_idx]
                    row["target"] = target
                    row["solver"] = "histgbdt"
                    row["seed"] = SEED
                    row["target_family"] = TARGET_FAMILY.get(target, "unknown")
                    row["protocol"] = "state_holdout"
                    row.update(gate_evals)
                    target_rows.append(row)

            all_results.extend(target_rows)
            _save_checkpoint(target, target_rows)

            # Per-task log summary
            mean_r2 = np.mean([r["r2"] for r in target_rows])
            mean_tk = np.mean([r["theory_kappa"] for r in target_rows])
            flat_pass = sum(1 for r in target_rows
                           if r["gate_flat"]["gate_decision"] == "EnforcementDecision.EXECUTE")
            oobleck_pass = sum(1 for r in target_rows
                               if r["gate_oobleck"]["gate_decision"] == "EnforcementDecision.EXECUTE")
            n_rows = len(target_rows)
            log.info(
                "  [%d/%d] %s: R2=%.3f  theory_k=%.3f  "
                "flat=%d/%d  oobleck=%d/%d",
                task_idx + 1, len(CONUS27_TASKS), target,
                mean_r2, mean_tk,
                flat_pass, n_rows, oobleck_pass, n_rows,
            )

        except Exception as e:
            log.error("  FAILED %s: %s", target, e)
            log.error("  %s", traceback.format_exc())
            failed_targets.append(target)
            continue

    if failed_targets:
        log.warning("FAILED TARGETS (%d): %s", len(failed_targets), failed_targets)

    elapsed = time.time() - t0
    log.info("S018B V2 complete: %d results in %.1f seconds", len(all_results), elapsed)

    # --------------- Aggregate R2 for PDFM comparison ---------------
    task_mean_r2 = {}
    for target in CONUS27_TASKS:
        task_rows = [r for r in all_results if r["target"] == target]
        if task_rows:
            task_mean_r2[target] = round(float(np.mean([r["r2"] for r in task_rows])), 4)

    health_r2 = [v for t, v in task_mean_r2.items() if t in HEALTH_TASKS]
    all_r2 = list(task_mean_r2.values())
    mean_health = round(float(np.mean(health_r2)), 4) if health_r2 else float("nan")
    mean_all = round(float(np.mean(all_r2)), 4) if all_r2 else float("nan")

    log.info("=" * 60)
    log.info("PDFM comparison (state-holdout extrapolation):")
    log.info("  Mean R2 (all %d tasks):    %.3f  [PDFM: %.2f, delta: %+.3f]",
             len(all_r2), mean_all, PDFM_EXTRAP_ALL_MEAN, mean_all - PDFM_EXTRAP_ALL_MEAN)
    log.info("  Mean R2 (health %d tasks): %.3f  [PDFM: %.2f, delta: %+.3f]",
             len(health_r2), mean_health, PDFM_EXTRAP_HEALTH_MEAN, mean_health - PDFM_EXTRAP_HEALTH_MEAN)
    log.info("=" * 60)

    # --------------- Gate summary ---------------
    n_total = len(all_results)
    n_flat_pass = sum(1 for r in all_results
                      if r.get("gate_flat", {}).get("gate_decision") == "EnforcementDecision.EXECUTE")
    n_oobleck_pass = sum(1 for r in all_results
                         if r.get("gate_oobleck", {}).get("gate_decision") == "EnforcementDecision.EXECUTE")
    n_flip = sum(
        1 for r in all_results
        if r.get("gate_flat", {}).get("gate_decision") != r.get("gate_oobleck", {}).get("gate_decision")
    )
    log.info("Gate summary: flat=%d/%d pass, oobleck=%d/%d pass, flips=%d",
             n_flat_pass, n_total, n_oobleck_pass, n_total, n_flip)

    # --------------- Save outputs ---------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "s018b_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Per-task summary
    task_summaries = []
    for target in CONUS27_TASKS:
        task_rows = [r for r in all_results if r["target"] == target]
        if not task_rows:
            continue
        task_summaries.append({
            "target": target,
            "target_family": TARGET_FAMILY.get(target, "unknown"),
            "n_cells": len(task_rows),
            "mean_r2": round(float(np.mean([r["r2"] for r in task_rows])), 4),
            "std_r2": round(float(np.std([r["r2"] for r in task_rows])), 4),
            "mean_theory_kappa": round(float(np.mean([r["theory_kappa"] for r in task_rows])), 4),
            "std_theory_kappa": round(float(np.std([r["theory_kappa"] for r in task_rows])), 4),
            "flat_pass_rate": round(
                sum(1 for r in task_rows
                    if r["gate_flat"]["gate_decision"] == "EnforcementDecision.EXECUTE") / len(task_rows), 4
            ),
            "oobleck_pass_rate": round(
                sum(1 for r in task_rows
                    if r["gate_oobleck"]["gate_decision"] == "EnforcementDecision.EXECUTE") / len(task_rows), 4
            ),
        })

    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    summary = {
        "experiment": "S018B_V2",
        "description": "State-holdout extrapolation benchmark with MMAR fixes (6 emb x 27 tasks)",
        "protocol": "PDFM sec 3.2 -- random 20% state holdout",
        "seed": SEED,
        "git_hash": git_hash,
        "holdout_states": holdout_states,
        "n_folds": n_folds,
        "n_results": len(all_results),
        "n_failed": len(failed_targets),
        "failed_targets": failed_targets,
        "elapsed_seconds": round(elapsed, 1),
        "pdfm_comparison": {
            "mean_r2_all": mean_all,
            "mean_r2_health": mean_health,
            "pdfm_all": PDFM_EXTRAP_ALL_MEAN,
            "pdfm_health": PDFM_EXTRAP_HEALTH_MEAN,
            "delta_all": round(mean_all - PDFM_EXTRAP_ALL_MEAN, 4),
            "delta_health": round(mean_health - PDFM_EXTRAP_HEALTH_MEAN, 4),
        },
        "gate_summary": {
            "flat_pass": n_flat_pass,
            "oobleck_pass": n_oobleck_pass,
            "total": n_total,
            "flips": n_flip,
        },
        "per_task": task_summaries,
    }

    with open(out_path / "s018b_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("Results saved to %s", out_path)
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S018B V2 State-Holdout Extrapolation")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--repr-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: config valid")
        log.info("  data_dir: %s", args.data_dir)
        log.info("  repr_dir: %s", args.repr_dir)
        log.info("  output_dir: %s", args.output_dir)
        log.info("  embeddings: %d", len(EMBEDDINGS))
        log.info("  targets: %d", len(CONUS27_TASKS))
        log.info("  protocol: state-holdout (PDFM sec 3.2)")
        log.info("  gatekeepers: flat + oobleck")
        sys.exit(0)

    run_experiment(args.data_dir, args.repr_dir, args.output_dir)
