"""
run_s018d_solver_diversity.py -- Solver-Diversity Stress Test for RSCT.

S018C showed NO_CHANGE when expanding the feature substrate: the
near-uniform R/S/N simplex persisted.  S018D holds the representation
and certificate pipeline fixed and instead expands the SOLVER portfolio
to 12 solvers across 10 families.

If the certificate separates when solver diversity is introduced, the
near-uniform simplex was caused by near-interchangeable solvers, not
a weak certificate.  If it stays uniform even with a noisy/bad solver,
the certificate method itself needs revision.

Solvers (10 families):
  trivial:       mean_baseline, noisy_solver
  linear:        linear_ridge
  kernel:        svr_rbf
  instance:      knn
  tree:          lightgbm, random_forest
  neural:        mlp_regressor
  hierarchical:  hrm_regressor
  spectral+tree: pca_v1       (from OOF)
  spatial:       spatial_lag_v1 (from OOF)
  graph:         gnn_v2        (from OOF)

Pre-registered pass conditions:
  1. N_noisy - N_portfolio_mean >= 0.05
  2. alpha_serious_best - alpha_bad >= 0.03
  3. delta_R_range >= 0.03 or delta_N_range >= 0.05 vs S018C
  4. At least one solver shows target-family-specific alpha lift
  5. Simplex no longer uniformly ~1/3 across all solvers

Usage:
    python run_s018d_solver_diversity.py \\
        --oof-dir /opt/ml/processing/input/oof \\
        --zcta-features /opt/ml/processing/input/data/zcta_features_labels_with_lags.parquet \\
        --out /opt/ml/processing/output \\
        --mode full
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Import solvers
sys.path.insert(0, str(Path(__file__).parent))
from solvers import (
    SOLVER_REGISTRY, OOF_SOLVERS,
    MeanBaseline, NoisySolver, LinearRidge, SVRSolver, KNNSolver,
    LightGBMSolver, RandomForestSolver, MLPRegressor, HRMRegressor,
)

# Import shared certificate pipeline (same as S018C -- do not change)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.canonical_certifier import compute_per_sample_scores, certify_task
from shared.constants import CONUS27_TASKS

log = logging.getLogger("s018d")

# S018C baseline for comparison
S018C_BASELINE = {
    "R_mean": 0.3253, "S_mean": 0.3357, "N_mean": 0.3141,
    "R_range": 0.1325, "N_range": 0.1896,
    "alpha_mean": 0.5104, "kappa_mean": 0.2222,
}

SOLVER_FAMILIES = {
    "mean_baseline": "trivial",
    "noisy_solver": "trivial",
    "linear_ridge": "linear",
    "svr_rbf": "kernel",
    "knn": "instance",
    "lightgbm": "tree",
    "random_forest": "tree",
    "mlp_regressor": "neural",
    "hrm_regressor": "hierarchical",
    "pca_v1": "spectral+tree",
    "spatial_lag_v1": "spatial",
    "gnn_v2": "graph",
}

TARGET_FAMILIES = {
    "arthritis": "health", "asthma": "health", "binge_drinking": "health",
    "bp_medicated": "health", "cancer": "health", "cholesterol_screening": "health",
    "chronic_kidney_disease": "health", "copd": "health",
    "coronary_heart_disease": "health", "dental_visit": "health",
    "diabetes": "health", "high_blood_pressure": "health",
    "high_cholesterol": "health", "mental_health_not_good": "health",
    "obesity": "health", "physical_health_not_good": "health",
    "physical_inactivity": "health", "sleep_less_7hr": "health",
    "smoking": "health", "stroke": "health",
    "annual_checkup": "health",
    "income": "economic", "home_value": "economic",
    "night_lights": "environment", "population_density": "environment",
    "tree_cover": "environment", "elevation": "environment",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(path: str) -> tuple:
    """Load ZCTA features, return df + full numeric cols."""
    df = pd.read_parquet(path)
    skip = {"zcta_id", "county_name", "state", "split_imputation",
            "split_extrap", "split_superres"}
    full_cols = sorted(
        c for c in df.columns
        if c not in skip
        and not c.startswith("target_")
        and pd.api.types.is_numeric_dtype(df[c])
    )
    return df, full_cols


def build_embedding(df, feat_cols, train_mask, n_components=32, seed=42):
    """StandardScaler + PCA-32. Same as S018C."""
    X = df[feat_cols].values.astype(np.float64)
    medians = np.nanmedian(X[train_mask], axis=0)
    for j in range(X.shape[1]):
        nans = np.isnan(X[:, j])
        if nans.any():
            X[nans, j] = medians[j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[train_mask])
    X_all_scaled = scaler.transform(X)

    nc = min(n_components, X.shape[1])
    pca = PCA(n_components=nc, random_state=seed)
    pca.fit(X_scaled)
    Z = pca.transform(X_all_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    return Z, var_explained


def get_target_col(df, task):
    """Find the target column for a task name."""
    candidates = [
        f"target_{task}", task,
        # CONUS27 long names
        f"Percent_Person_With{task.title().replace('_', '')}",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Brute force: look for partial match
    for c in df.columns:
        if task.replace("_", "") in c.lower().replace("_", ""):
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    return None


# ---------------------------------------------------------------------------
# OOF solver loading
# ---------------------------------------------------------------------------

def load_oof_residuals(oof_dir: str, solver: str, tasks: list,
                       zcta_ids: np.ndarray) -> Dict[str, np.ndarray]:
    """Load OOF residuals for an existing solver from parquet."""
    path = Path(oof_dir) / f"oof_{solver}.parquet"
    if not path.exists():
        log.warning("OOF file not found: %s", path)
        return {}

    df_oof = pd.read_parquet(path)
    df_oof["abs_residual"] = df_oof["residual"].abs()

    result = {}
    for task in tasks:
        task_df = df_oof[df_oof["task"] == task]
        if len(task_df) == 0:
            continue
        zcta_res = task_df.groupby("zcta")["abs_residual"].mean()
        res_all = np.array([zcta_res.get(z, np.nan) for z in zcta_ids])
        result[task] = res_all
    return result


# ---------------------------------------------------------------------------
# Train new solvers and get residuals
# ---------------------------------------------------------------------------

def _make_solver(factory, embed_dim):
    """Instantiate a solver, passing input_dim if needed."""
    try:
        return factory(embed_dim)
    except TypeError:
        return factory()


def train_and_get_residuals(
    solver_name: str, Z: np.ndarray, df: pd.DataFrame,
    tasks: list, train_mask: np.ndarray, test_mask: np.ndarray,
    seed: int = 42, n_cv_folds: int = 3,
) -> Dict[str, np.ndarray]:
    """Train a solver on each task, return abs residuals on train + test.

    Training residuals are computed via K-fold CV to avoid degenerate
    tercile labels from in-sample memorization (e.g. KNN).
    """
    embed_dim = Z.shape[1]
    result = {}

    for task in tasks:
        target_col = get_target_col(df, task)
        if target_col is None:
            continue

        y = pd.to_numeric(df[target_col], errors="coerce").values
        valid = np.isfinite(y)
        v_train = valid & train_mask
        v_test = valid & test_mask

        if v_train.sum() < 100 or v_test.sum() < 20:
            continue

        Z_tr = Z[v_train]
        Z_te = Z[v_test]
        y_tr = y[v_train]
        y_te = y[v_test]

        factory = SOLVER_REGISTRY[solver_name]

        # -- CV predictions on train for non-degenerate tercile labels --
        train_preds = np.zeros(len(y_tr))
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
        for fold_tr, fold_val in kf.split(Z_tr):
            solver_cv = _make_solver(factory, embed_dim)
            try:
                solver_cv.fit(Z_tr[fold_tr], y_tr[fold_tr])
                train_preds[fold_val] = solver_cv.predict(Z_tr[fold_val])
            except Exception:
                train_preds[fold_val] = np.mean(y_tr[fold_tr])
        abs_res_tr = np.abs(y_tr - train_preds)

        # -- Final model on full train for test predictions --
        solver = _make_solver(factory, embed_dim)
        try:
            solver.fit(Z_tr, y_tr)
            y_pred = solver.predict(Z_te)
        except Exception as e:
            log.warning("  %s/%s failed: %s", solver_name, task, e)
            continue
        abs_res_te = np.abs(y_te - y_pred)

        # Build full-length residual array with both train + test
        full_res = np.full(len(df), np.nan)
        full_res[v_train] = abs_res_tr
        full_res[v_test] = abs_res_te
        result[task] = full_res

    return result


# ---------------------------------------------------------------------------
# Certificate pipeline (same as S018C -- DO NOT CHANGE)
# ---------------------------------------------------------------------------

def build_tercile_labels(residuals, train_mask):
    """Assign residual tercile labels: 0=R, 1=S, 2=N."""
    train_res = residuals[train_mask]
    valid_train = train_res[np.isfinite(train_res)]
    if len(valid_train) < 30:
        return None
    q33 = np.quantile(valid_train, 1/3)
    q66 = np.quantile(valid_train, 2/3)
    labels = np.ones(len(residuals), dtype=np.int64)
    labels[residuals < q33] = 0
    labels[residuals >= q66] = 2
    return labels


def certify_solver_task(
    Z: np.ndarray, residuals: np.ndarray,
    train_mask: np.ndarray, test_mask: np.ndarray,
    solver_name: str, task: str, seed: int = 42,
) -> Optional[dict]:
    """Run the R/S/N certificate pipeline for one solver x task.

    Same method as S018C: residual tercile -> MLP -> softmax -> R/S/N.
    Extended to include sigma, omega, alpha_omega, tau, entropy, gate.
    """
    valid = np.isfinite(residuals)
    v_train = valid & train_mask
    v_test = valid & test_mask

    if v_train.sum() < 50 or v_test.sum() < 20:
        return None

    labels = build_tercile_labels(residuals, v_train)
    if labels is None:
        return None

    # Extract valid subsets
    Z_tr = Z[v_train]
    Z_te = Z[v_test]
    lab_tr = labels[v_train]
    lab_te = labels[v_test]

    # Scale and train MLP
    sc = StandardScaler()
    Z_tr_s = sc.fit_transform(Z_tr)
    Z_te_s = sc.transform(Z_te)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500,
        random_state=seed, early_stopping=True,
        validation_fraction=0.15,
    )
    clf.fit(Z_tr_s, lab_tr)
    probs_test = clf.predict_proba(Z_te_s)

    # Ensure 3 classes
    if probs_test.shape[1] != 3:
        return None

    # Per-sample scores
    scores = compute_per_sample_scores(probs_test)
    R_arr = scores["R"]
    S_arr = scores["S"]
    N_arr = scores["N"]
    kappa_arr = scores["kappa"]
    alpha_arr = scores["alpha"]

    # Aggregates
    R_med = float(np.median(R_arr))
    S_med = float(np.median(S_arr))
    N_med = float(np.median(N_arr))
    kappa_med = float(np.median(kappa_arr))
    alpha_med = float(np.median(alpha_arr))

    # sigma = std of per-sample kappa (canonical DEF-14)
    sigma = float(np.std(kappa_arr))

    # omega = median(1 - S_i) (proxy, not ADR-008 chi-squared)
    omega = float(np.median(1.0 - S_arr))

    # alpha_omega = alpha * omega + prior * (1 - omega), P16 blended quality
    prior = 0.5
    alpha_omega = alpha_med * omega + prior * (1.0 - omega)

    # tau = 1 / alpha_omega (temperature)
    tau = 1.0 / alpha_omega if alpha_omega > 1e-12 else float("inf")

    # simplex entropy: -sum(p * log(p)) for median simplex point
    simplex = np.array([R_med, S_med, N_med])
    simplex = np.clip(simplex, 1e-12, 1.0)
    entropy = float(-np.sum(simplex * np.log(simplex)))
    max_entropy = float(np.log(3))  # ~1.099 for uniform

    # Collapse risk: fraction of samples where N > 0.5
    collapse_risk = float(np.mean(N_arr > 0.5))

    # Gate decision via canonical certifier
    gate_decision = "UNKNOWN"
    gate_reached = 0
    try:
        cert = certify_task(
            probs_test=probs_test,
            labels_test=lab_te,
            task=task,
            model_version=solver_name,
            classifier_type="mlp",
        )
        gd_raw = cert.gate_decision
        gate_decision = gd_raw.name if hasattr(gd_raw, "name") else str(gd_raw)
        # Extract gate_reached from gate_details if available
        gd = cert.gate_details
        if hasattr(gd, "get"):
            gate_reached = gd.get("gate_reached", 0)
        elif hasattr(gd, "gate_reached"):
            gate_reached = gd.gate_reached
    except Exception as e:
        log.warning("  gate eval failed for %s/%s: %s", solver_name, task, e)

    return {
        "solver": solver_name,
        "family": SOLVER_FAMILIES.get(solver_name, "unknown"),
        "task": task,
        "target_family": TARGET_FAMILIES.get(task, "unknown"),
        "R": R_med, "S": S_med, "N": N_med,
        "alpha": alpha_med, "kappa": kappa_med,
        "sigma": sigma, "omega": omega,
        "alpha_omega": alpha_omega, "tau": tau,
        "entropy": entropy, "entropy_ratio": entropy / max_entropy,
        "collapse_risk": collapse_risk,
        "gate_decision": gate_decision,
        "gate_reached": gate_reached,
        "n_test": int(v_test.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--oof-dir", required=True,
                        help="Directory with oof_*.parquet for existing solvers")
    parser.add_argument("--zcta-features", required=True,
                        help="Path to zcta_features_labels_with_lags.parquet")
    parser.add_argument("--out", default="evidence/",
                        help="Output directory")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full",
                        help="smoke = 2 targets + 5 solvers; full = all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    log.info("Loading features from %s", args.zcta_features)
    df, full_cols = load_features(args.zcta_features)
    zcta_ids = df["zcta_id"].values

    if "split_imputation" in df.columns:
        train_mask = (df["split_imputation"] != "test").values
    else:
        rng = np.random.RandomState(args.seed)
        train_mask = rng.rand(len(df)) > 0.2
    test_mask = ~train_mask

    log.info("  %d ZCTAs, train=%d, test=%d, features=%d",
             len(df), train_mask.sum(), test_mask.sum(), len(full_cols))

    # ---- Build embedding (same as S018C Arm C1) ---------------------------
    log.info("Building PCA-32 embedding (full substrate, %d features)", len(full_cols))
    Z, var_explained = build_embedding(df, full_cols, train_mask,
                                       n_components=32, seed=args.seed)
    log.info("  PCA-32 variance explained: %.3f", var_explained)

    # ---- Select solvers and targets ---------------------------------------
    if args.mode == "smoke":
        new_solvers = ["mean_baseline", "noisy_solver", "linear_ridge",
                       "lightgbm", "knn"]
        oof_solvers = []
        tasks = ["diabetes", "income"]
    else:
        new_solvers = list(SOLVER_REGISTRY.keys())
        oof_solvers = OOF_SOLVERS
        tasks = sorted(CONUS27_TASKS)

    all_solvers = new_solvers + oof_solvers
    log.info("Solvers (%d): %s", len(all_solvers), ", ".join(all_solvers))
    log.info("Tasks (%d): %s", len(tasks), ", ".join(tasks))

    # ---- Collect residuals per solver x task ------------------------------
    # residuals[solver][task] = np.ndarray of shape (n_zcta,)
    residuals: Dict[str, Dict[str, np.ndarray]] = {}

    # Train new solvers
    for solver_name in new_solvers:
        log.info("Training solver: %s", solver_name)
        st = time.time()
        residuals[solver_name] = train_and_get_residuals(
            solver_name, Z, df, tasks, train_mask, test_mask, seed=args.seed,
        )
        elapsed = time.time() - st
        n_tasks = len(residuals[solver_name])
        log.info("  %s: %d tasks in %.1fs", solver_name, n_tasks, elapsed)

    # Load existing OOF solvers
    for solver_name in oof_solvers:
        log.info("Loading OOF solver: %s", solver_name)
        residuals[solver_name] = load_oof_residuals(
            args.oof_dir, solver_name, tasks, zcta_ids,
        )
        log.info("  %s: %d tasks loaded", solver_name,
                 len(residuals[solver_name]))

    # ---- Certificate pipeline ---------------------------------------------
    log.info("Running certificate pipeline...")
    all_certs = []

    for solver_name in all_solvers:
        solver_res = residuals.get(solver_name, {})
        for task in tasks:
            if task not in solver_res:
                continue
            res = solver_res[task]
            cert = certify_solver_task(
                Z, res, train_mask, test_mask,
                solver_name, task, seed=args.seed,
            )
            if cert is not None:
                all_certs.append(cert)

    log.info("  %d certificates computed", len(all_certs))

    if len(all_certs) == 0:
        log.error("No certificates produced. Check data alignment.")
        return 1

    cert_df = pd.DataFrame(all_certs)

    # ---- Summary tables ---------------------------------------------------
    # Per-solver aggregates
    solver_agg = cert_df.groupby("solver").agg({
        "R": ["mean", "std", lambda x: x.max() - x.min()],
        "S": ["mean"],
        "N": ["mean", "std", lambda x: x.max() - x.min()],
        "alpha": ["mean", "std"],
        "kappa": ["mean", "std"],
        "sigma": ["mean"],
        "omega": ["mean"],
        "alpha_omega": ["mean"],
        "tau": ["mean"],
        "entropy": ["mean"],
        "collapse_risk": ["mean"],
    })
    solver_agg.columns = ["_".join(c).strip("_") for c in solver_agg.columns]

    # Add family
    solver_agg["family"] = solver_agg.index.map(
        lambda s: SOLVER_FAMILIES.get(s, "unknown")
    )

    # Gate decision mode per solver
    gate_mode = cert_df.groupby("solver")["gate_decision"].agg(
        lambda x: x.value_counts().index[0]
    )
    solver_agg["gate_mode"] = gate_mode

    # Gate decision distribution per solver
    gate_dist = cert_df.groupby("solver")["gate_decision"].value_counts().unstack(fill_value=0)

    # ---- Print readout ----------------------------------------------------
    print("\n" + "=" * 130)
    print("S018D SOLVER-DIVERSITY STRESS TEST: CERTIFICATE COMPARISON")
    print("=" * 130)
    print(f"{'Solver':<18} {'Family':<14} {'R':>6} {'S':>6} {'N':>6} "
          f"{'alpha':>6} {'kappa':>6} {'sigma':>6} {'omega':>6} "
          f"{'a_omega':>7} {'tau':>6} {'entropy':>7} {'gate':>10}")
    print("-" * 130)

    for solver_name in all_solvers:
        if solver_name not in solver_agg.index:
            continue
        row = solver_agg.loc[solver_name]
        family = SOLVER_FAMILIES.get(solver_name, "?")
        gm = gate_mode.get(solver_name, "?")
        print(f"  {solver_name:<16} {family:<14} "
              f"{row['R_mean']:6.4f} {row['S_mean']:6.4f} {row['N_mean']:6.4f} "
              f"{row['alpha_mean']:6.4f} {row['kappa_mean']:6.4f} "
              f"{row['sigma_mean']:6.4f} {row['omega_mean']:6.4f} "
              f"{row['alpha_omega_mean']:7.4f} {row['tau_mean']:6.3f} "
              f"{row['entropy_mean']:7.4f} {gm:>10}")

    # ---- Pass condition evaluation ----------------------------------------
    print("\n" + "=" * 130)
    print("PASS CONDITION EVALUATION")
    print("=" * 130)

    # Compute portfolio means (excluding trivial controls)
    serious = cert_df[~cert_df["solver"].isin(["mean_baseline", "noisy_solver"])]
    bad = cert_df[cert_df["solver"].isin(["mean_baseline", "noisy_solver"])]

    portfolio_N_mean = float(cert_df["N"].mean())
    noisy_N_mean = float(cert_df[cert_df["solver"] == "noisy_solver"]["N"].mean()) if "noisy_solver" in cert_df["solver"].values else np.nan
    mean_N_mean = float(cert_df[cert_df["solver"] == "mean_baseline"]["N"].mean()) if "mean_baseline" in cert_df["solver"].values else np.nan

    serious_alpha_best = float(serious["alpha"].max()) if len(serious) > 0 else 0
    bad_alpha_mean = float(bad["alpha"].mean()) if len(bad) > 0 else 0

    # Per-solver R_range and N_range
    R_range_all = float(cert_df["R"].max() - cert_df["R"].min())
    N_range_all = float(cert_df["N"].max() - cert_df["N"].min())
    delta_R_range = R_range_all - S018C_BASELINE["R_range"]
    delta_N_range = N_range_all - S018C_BASELINE["N_range"]

    # Check each condition
    cond1 = noisy_N_mean - portfolio_N_mean >= 0.05 if np.isfinite(noisy_N_mean) else False
    cond2 = serious_alpha_best - bad_alpha_mean >= 0.03
    cond3 = delta_R_range >= 0.03 or delta_N_range >= 0.05
    # cond4: target-family-specific alpha lift
    tf_alpha = cert_df.groupby(["target_family", "solver"])["alpha"].mean().unstack()
    cond4 = False
    if len(tf_alpha) > 0:
        for solver_name in serious["solver"].unique():
            if solver_name in tf_alpha.columns:
                solver_alpha = tf_alpha[solver_name]
                if solver_alpha.max() - solver_alpha.min() > 0.03:
                    cond4 = True
                    break
    # cond5: simplex no longer uniformly ~1/3
    solver_R_means = cert_df.groupby("solver")["R"].mean()
    cond5 = float(solver_R_means.max() - solver_R_means.min()) > 0.02

    conditions = [
        ("1. noisy_solver N lift >= 0.05",
         cond1,
         f"N_noisy={noisy_N_mean:.4f}, N_portfolio={portfolio_N_mean:.4f}, "
         f"lift={noisy_N_mean - portfolio_N_mean:+.4f}" if np.isfinite(noisy_N_mean) else "noisy_solver not run"),
        ("2. alpha_serious_best - alpha_bad >= 0.03",
         cond2,
         f"alpha_best={serious_alpha_best:.4f}, alpha_bad={bad_alpha_mean:.4f}, "
         f"gap={serious_alpha_best - bad_alpha_mean:+.4f}"),
        ("3. R_range or N_range increases vs S018C",
         cond3,
         f"delta_R_range={delta_R_range:+.4f}, delta_N_range={delta_N_range:+.4f}"),
        ("4. target-family-specific alpha lift",
         cond4,
         "at least one solver shows target-family separation" if cond4 else "no solver shows target-family separation"),
        ("5. simplex not uniformly ~1/3 across solvers",
         cond5,
         f"R_mean range across solvers: {float(solver_R_means.max() - solver_R_means.min()):.4f}"),
    ]

    pass_count = 0
    for name, passed, detail in conditions:
        status = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        print(f"  [{status}] {name}")
        print(f"         {detail}")

    # Fail condition: noisy solver still near-uniform
    fail_check = False
    if np.isfinite(noisy_N_mean):
        noisy_R = float(cert_df[cert_df["solver"] == "noisy_solver"]["R"].mean())
        noisy_S = float(cert_df[cert_df["solver"] == "noisy_solver"]["S"].mean())
        fail_check = (abs(noisy_R - 1/3) < 0.02 and
                      abs(noisy_N_mean - 1/3) < 0.02 and
                      abs(noisy_S - 1/3) < 0.02)
        if fail_check:
            print("\n  [CRITICAL] noisy_solver still near-uniform R/S/N")
            print("  -> Certificate method may not be sensitive enough")

    # Verdict
    if pass_count >= 4:
        verdict = "SEPARATION_CONFIRMED"
    elif pass_count >= 2:
        verdict = "PARTIAL_SEPARATION"
    elif fail_check:
        verdict = "CERTIFICATE_INSENSITIVE"
    else:
        verdict = "NO_SEPARATION"

    print(f"\n  VERDICT: {verdict} ({pass_count}/5 conditions passed)")
    print("=" * 130)

    # ---- Per-solver gate distribution -------------------------------------
    print("\n" + "=" * 130)
    print("GATE DECISION DISTRIBUTION BY SOLVER")
    print("=" * 130)
    for solver_name in all_solvers:
        if solver_name not in gate_dist.index:
            continue
        row = gate_dist.loc[solver_name]
        counts = {col: int(row[col]) for col in gate_dist.columns if row[col] > 0}
        total = sum(counts.values())
        pcts = {k: f"{v/total*100:.0f}%" for k, v in counts.items()}
        print(f"  {solver_name:<18} {pcts}")

    # ---- Per target-family x solver alpha table ---------------------------
    print("\n" + "=" * 130)
    print("ALPHA BY TARGET FAMILY x SOLVER")
    print("=" * 130)
    if len(tf_alpha) > 0:
        # Print header
        solver_names = [s for s in all_solvers if s in tf_alpha.columns]
        header = f"{'target_family':<14}"
        for s in solver_names:
            header += f" {s[:10]:>10}"
        print(header)
        print("-" * len(header))
        for tf in sorted(tf_alpha.index):
            row_str = f"  {tf:<12}"
            for s in solver_names:
                val = tf_alpha.loc[tf, s] if s in tf_alpha.columns else np.nan
                if np.isfinite(val):
                    row_str += f" {val:10.4f}"
                else:
                    row_str += f" {'--':>10}"
            print(row_str)

    # ---- Save artifacts ---------------------------------------------------
    elapsed = time.time() - t0

    # certificate_rsn.parquet
    cert_df.to_parquet(out / "certificate_rsn.parquet", index=False)

    # solver_metrics.parquet
    solver_agg.to_parquet(out / "solver_metrics.parquet")

    # target_family_metrics
    if len(tf_alpha) > 0:
        tf_alpha.to_parquet(out / "target_family_metrics.parquet")

    # summary.json
    summary = {
        "experiment": "s018d_solver_diversity",
        "solver_count": len(all_solvers),
        "target_count": len(tasks),
        "zcta_count": len(df),
        "family_count": len(set(SOLVER_FAMILIES[s] for s in all_solvers)),
        "certificate_count": len(all_certs),
        "var_explained": round(var_explained, 4),
        "elapsed_seconds": round(elapsed, 1),
        "verdict": verdict,
        "pass_count": pass_count,
        "conditions": {name: passed for name, passed, _ in conditions},
        "s018c_baseline": S018C_BASELINE,
        "delta_vs_s018c": {
            "delta_R_range": round(delta_R_range, 4),
            "delta_N_range": round(delta_N_range, 4),
        },
        "per_solver": {},
    }

    for solver_name in all_solvers:
        if solver_name not in solver_agg.index:
            continue
        row = solver_agg.loc[solver_name]
        gm = gate_mode.get(solver_name, "UNKNOWN")
        summary["per_solver"][solver_name] = {
            "family": SOLVER_FAMILIES.get(solver_name, "unknown"),
            "R_mean": round(float(row["R_mean"]), 4),
            "S_mean": round(float(row["S_mean"]), 4),
            "N_mean": round(float(row["N_mean"]), 4),
            "alpha_mean": round(float(row["alpha_mean"]), 4),
            "kappa_mean": round(float(row["kappa_mean"]), 4),
            "sigma_mean": round(float(row["sigma_mean"]), 4),
            "omega_mean": round(float(row["omega_mean"]), 4),
            "alpha_omega_mean": round(float(row["alpha_omega_mean"]), 4),
            "tau_mean": round(float(row["tau_mean"]), 3),
            "entropy_mean": round(float(row["entropy_mean"]), 4),
            "collapse_risk_mean": round(float(row["collapse_risk_mean"]), 4),
            "gate_mode": gm,
        }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("Wrote %d artifacts to %s in %.1fs", 4, out, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
