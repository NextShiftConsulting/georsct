"""
theory_certifier.py -- Shared certification with theory kappa (D*/D).

Computes proxy kappa R*(1-N) and theory kappa D*/D for the same
(target, solver, fold) triple across all embeddings. Theory kappa uses
leave-one-out references: D* for embedding e is computed from all
embeddings EXCEPT e, preventing self-comparison inflation.

Usage:
    from shared.theory_certifier import certify_group

    results = certify_group(
        embeddings_dict={"pca_v1": Z_pca, "spatial_lag_v1": Z_lag, "gnn_v2": Z_gnn},
        y_train=y_train, y_test=y_test,
        solver_name="histgbdt", seed=42,
    )
    # Returns list of dicts, one per embedding, each with both kappas.
"""

import logging
from typing import Dict, List

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, balanced_accuracy_score

log = logging.getLogger(__name__)

SEED = 42


def make_solver(solver_name: str, seed: int = SEED):
    """Create a solver instance by name."""
    if solver_name == "histgbdt":
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=seed
        )
    elif solver_name == "ridge":
        return Ridge(alpha=1.0)
    elif solver_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed,
            early_stopping=True, validation_fraction=0.1,
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def certify_group(
    embeddings_dict: Dict[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    solver_name: str,
    seed: int = SEED,
    shared_boundaries: bool = False,
) -> List[dict]:
    """Certify one (target, solver, fold) across ALL embeddings.

    Fits the same solver on each embedding independently, then computes:
      1. Proxy kappa via R*(1-N) from MLP tercile classifier
      2. Theory kappa via D*/D using RegressionKappaEvaluator

    D* = min over all embeddings of per-sample squared residual.
    This is the tightest achievable loss for each sample.

    Args:
        embeddings_dict: {emb_name: Z_array} where Z is (n_samples, dim).
            Train/test split already applied (pass masked arrays).
        y_train, y_test: Target values for train/test split.
        solver_name: Which solver to fit.
        seed: Random seed.
        shared_boundaries: If True, compute tercile boundaries from pooled
            residuals across ALL arms before classifying. This removes the
            per-arm 1/3 normalization that makes R/S/N identical across arms
            and allows the simplex to discriminate embedding quality.
            If False (default), each arm uses its own residual distribution
            (original behaviour, preserves S019A/C compatibility).

    Returns:
        List of dicts, one per embedding, each containing:
          - r2, proxy_kappa, theory_kappa, theory_sigma, proxy vs theory comparison
          - Full simplex (R, S, N), alpha, sigma=N
          - Per-sample arrays prefixed with _
    """
    from yrsn.core.decomposition.score import aggregate_scores_from_probs
    from yrsn.core.decomposition.instability_computation import compute_sigma_request
    from yrsn.core.kappa.difficulty_theory.evaluator import RegressionKappaEvaluator

    emb_names = sorted(embeddings_dict.keys())

    # Step 1: Fit solver on each embedding, collect predictions
    solver_preds_test = {}
    solver_preds_train = {}
    solvers = {}
    r2_scores = {}

    for emb_name in emb_names:
        Z_train = embeddings_dict[emb_name]["train"]
        Z_test = embeddings_dict[emb_name]["test"]

        solver = make_solver(solver_name, seed=seed)
        solver.fit(Z_train, y_train)
        solvers[emb_name] = solver

        preds_test = solver.predict(Z_test)
        preds_train = solver.predict(Z_train)
        solver_preds_test[emb_name] = preds_test
        solver_preds_train[emb_name] = preds_train
        r2_scores[emb_name] = float(r2_score(y_test, preds_test))

    # Shared tercile boundaries (optional): pool residuals across all arms so
    # that the MLP classifier uses the same q33/q66 for every embedding.
    # Without this, each arm normalises its own residuals to 1/3 buckets and
    # R≈S≈N≈0.33 for all arms by construction.
    if shared_boundaries:
        pooled_residuals = np.concatenate([
            np.abs(y_train - solver_preds_train[e]) for e in emb_names
        ])
        shared_q33 = float(np.quantile(pooled_residuals, 1 / 3))
        shared_q66 = float(np.quantile(pooled_residuals, 2 / 3))
    else:
        shared_q33 = None
        shared_q66 = None

    # Step 2 + 3: Theory kappa via leave-one-out RegressionKappaEvaluator
    # For each embedding e, D* = min residual across all embeddings EXCEPT e.
    # Prevents self-comparison (GNN comparing to itself -> kappa ~ 1.0).
    results = []
    for emb_name in emb_names:
        Z_train = embeddings_dict[emb_name]["train"]
        Z_test = embeddings_dict[emb_name]["test"]
        y_pred_test = solver_preds_test[emb_name]
        y_pred_train = solver_preds_train[emb_name]

        # LOO references: all embeddings except the one being evaluated
        loo_refs = [e for e in emb_names if e != emb_name]
        evaluator = RegressionKappaEvaluator(
            predictions=solver_preds_test,
            y_true=y_test,
            reference_reps=loo_refs,
        )

        # Theory kappa
        evidence = evaluator.evidence(emb_name)
        theory_kappa = float(evidence.get("kappa_compat", evidence.get("kappa")))
        theory_sigma = float(evidence["sigma"])
        kappa_per_sample_theory = evidence["batch_result"].kappa_per_sample

        # Proxy kappa via MLP tercile classifier
        residuals_train = np.abs(y_train - y_pred_train)
        if shared_boundaries:
            q33 = shared_q33
            q66 = shared_q66
        else:
            q33 = float(np.quantile(residuals_train, 1 / 3))
            q66 = float(np.quantile(residuals_train, 2 / 3))

        labels_train = np.zeros(len(y_train), dtype=int)
        labels_train[residuals_train <= q33] = 0
        labels_train[residuals_train > q66] = 2
        labels_train[(residuals_train > q33) & (residuals_train <= q66)] = 1

        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed,
            early_stopping=True, validation_fraction=0.1,
        )
        clf.fit(Z_train, labels_train)
        probs_test = clf.predict_proba(Z_test)

        if probs_test.shape[1] < 3:
            full_probs = np.zeros((len(Z_test), 3))
            for i, c in enumerate(clf.classes_):
                full_probs[:, c] = probs_test[:, i]
            probs_test = full_probs

        ba_test = float(balanced_accuracy_score(
            np.digitize(np.abs(y_test - y_pred_test), [q33, q66]),
            clf.predict(Z_test),
        ))

        agg = aggregate_scores_from_probs(probs_test, labels=None)
        R_med = float(agg["R_median"])
        S_med = float(agg["S_median"])
        N_med = float(agg["N_median"])
        proxy_kappa = float(agg.get("kappa_compat", agg.get("kappa")))
        alpha_agg = float(agg["alpha"])
        sigma_n = float(compute_sigma_request(N_med))

        results.append({
            "embedding": emb_name,
            "r2": r2_scores[emb_name],
            "ba_test": ba_test,
            "n_test": len(y_test),
            "n_train": len(y_train),
            # Simplex
            "R": R_med,
            "S_sup": S_med,
            "N": N_med,
            "alpha": alpha_agg,
            "simplex_sum": R_med + S_med + N_med,
            # Proxy kappa (R*(1-N))
            "proxy_kappa": proxy_kappa,
            "proxy_kappa_mean": float(agg["kappa_mean"]),
            "proxy_kappa_std": float(agg["kappa_std"]),
            # Theory kappa (D*/D)
            "theory_kappa": theory_kappa,
            "theory_kappa_mean": float(evidence["kappa_mean"]),
            "theory_kappa_min": float(evidence["batch_result"].kappa_min),
            "theory_sigma": theory_sigma,
            # Sigma = N (per-request turbulence)
            "sigma": sigma_n,
            # Diagnostics
            "omega": float(agg["omega"]),
            "entropy": float(agg["entropy"]),
            "collapse_risk": float(agg["collapse_risk"]),
            # Per-sample (not serialized)
            "_kappa_per_proxy": agg["kappa_per"],
            "_kappa_per_theory": kappa_per_sample_theory,
            "_probs_test": probs_test,
        })

    return results
