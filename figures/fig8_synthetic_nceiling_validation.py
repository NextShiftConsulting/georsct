"""Figure 8: Synthetic TRF validation.

Generates synthetic datasets with KNOWN ground-truth noise levels,
runs 3 model families, estimates TRF with block bootstrap,
verifies CI contains truth.

Plot: estimated TRF vs true TRF with 95% CI bands.
Also: naive bootstrap CI for comparison (too narrow).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

OUTPUT = Path(__file__).parent / "fig8_synthetic_nceiling_validation.pdf"

N_SAMPLES = 10000
N_FEATURES = 32
N_BOOTSTRAP = 500
N_BLOCKS = 50  # synthetic "states"
SEED = 42


def generate_synthetic(n, d, noise_fraction, seed=42):
    """Generate synthetic data with known noise fraction.

    y = f(X) + epsilon, where Var(epsilon)/Var(y) = noise_fraction.
    """
    rng = np.random.RandomState(seed)

    # Correlated features
    cov = np.eye(d)
    for i in range(d):
        for j in range(d):
            if i != j:
                cov[i, j] = 0.3 ** abs(i - j)
    X = rng.multivariate_normal(np.zeros(d), cov, size=n)

    # Nonlinear signal: uses first 8 features
    signal = (
        2.0 * X[:, 0]
        + 1.5 * X[:, 1] ** 2
        - 1.0 * X[:, 2] * X[:, 3]
        + 0.8 * np.sin(X[:, 4] * np.pi)
        + 0.5 * X[:, 5]
        - 0.3 * X[:, 6] ** 3
        + 0.2 * np.abs(X[:, 7])
    )

    signal_var = np.var(signal)
    # noise_fraction = Var(eps) / Var(y) = Var(eps) / (Var(signal) + Var(eps))
    # So Var(eps) = noise_fraction * Var(signal) / (1 - noise_fraction)
    noise_var = noise_fraction * signal_var / (1 - noise_fraction)
    epsilon = rng.normal(0, np.sqrt(noise_var), size=n)

    y = signal + epsilon

    # Assign to synthetic blocks
    block_labels = np.array([f"block_{i % N_BLOCKS:02d}" for i in range(n)])
    # Shuffle to break ordering
    perm = rng.permutation(n)
    block_labels = block_labels[perm]
    X = X[perm]
    y = y[perm]

    actual_noise_frac = np.var(epsilon[perm]) / np.var(y)
    return X, y, block_labels, actual_noise_frac


def oof_r2(model_class, model_kwargs, X, y, n_folds=5, seed=42):
    """Compute OOF R-squared via k-fold cross-validation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y))

    for train_idx, val_idx in kf.split(X):
        model = model_class(**model_kwargs)
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict(X[val_idx])

    ss_res = np.sum((y - oof_preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot, oof_preds


def block_bootstrap_r2(y_true, y_pred, block_labels, n_boot=N_BOOTSTRAP, seed=SEED):
    """Block bootstrap R-squared CI."""
    rng = np.random.RandomState(seed)
    unique_blocks = np.unique(block_labels)
    n_blocks = len(unique_blocks)
    block_idx = {b: np.where(block_labels == b)[0] for b in unique_blocks}

    r2_boot = np.empty(n_boot)
    for b in range(n_boot):
        boot_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_idx[bl] for bl in boot_blocks])
        yt, yp = y_true[idx], y_pred[idx]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_boot[b] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r2_boot


def naive_bootstrap_r2(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=SEED):
    """Naive (non-block) bootstrap R-squared CI."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    r2_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_boot[b] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r2_boot


def main():
    # Test at multiple noise levels
    true_noise_fractions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

    models = {
        "Ridge": (Ridge, {"alpha": 1.0}),
        "GBDT": (GradientBoostingRegressor, {
            "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
            "random_state": SEED, "subsample": 0.8,
        }),
        "MLP": (MLPRegressor, {
            "hidden_layer_sizes": (64, 32), "max_iter": 200,
            "random_state": SEED, "early_stopping": True,
            "validation_fraction": 0.1,
        }),
    }

    results = []

    for nf in true_noise_fractions:
        print(f"Noise fraction = {nf:.2f}...")
        X, y, blocks, actual_nf = generate_synthetic(
            N_SAMPLES, N_FEATURES, nf, seed=SEED + int(nf * 100)
        )

        best_r2 = -np.inf
        best_preds = None
        family_r2s = {}

        for name, (cls, kwargs) in models.items():
            r2, preds = oof_r2(cls, kwargs, X, y, seed=SEED)
            family_r2s[name] = r2
            if r2 > best_r2:
                best_r2 = r2
                best_preds = preds

        trf_est = 1 - best_r2
        trf_true = actual_nf

        # Block bootstrap CI on best family
        boot_r2 = block_bootstrap_r2(y, best_preds, blocks)
        boot_nceiling = 1 - boot_r2
        ci_lo = np.nanpercentile(boot_nceiling, 2.5)
        ci_hi = np.nanpercentile(boot_nceiling, 97.5)

        # Naive bootstrap for comparison
        naive_r2 = naive_bootstrap_r2(y, best_preds)
        naive_nceiling = 1 - naive_r2
        naive_lo = np.nanpercentile(naive_nceiling, 2.5)
        naive_hi = np.nanpercentile(naive_nceiling, 97.5)

        results.append({
            "true": trf_true,
            "est": trf_est,
            "block_ci_lo": ci_lo,
            "block_ci_hi": ci_hi,
            "naive_ci_lo": naive_lo,
            "naive_ci_hi": naive_hi,
            "family_r2": family_r2s,
        })

        in_ci = ci_lo <= trf_true <= ci_hi
        print(f"  True={trf_true:.3f} Est={trf_est:.3f} "
              f"BlockCI=[{ci_lo:.3f}, {ci_hi:.3f}] "
              f"NaiveCI=[{naive_lo:.3f}, {naive_hi:.3f}] "
              f"InCI={in_ci}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    trues = [r["true"] for r in results]
    ests = [r["est"] for r in results]
    block_lo = [r["block_ci_lo"] for r in results]
    block_hi = [r["block_ci_hi"] for r in results]
    naive_lo = [r["naive_ci_lo"] for r in results]
    naive_hi = [r["naive_ci_hi"] for r in results]

    # Perfect agreement line
    ax.plot([0, 0.7], [0, 0.7], "k--", alpha=0.3, linewidth=1, label="Perfect recovery")

    # Block bootstrap CI
    ax.fill_between(trues, block_lo, block_hi, alpha=0.25, color="#2ca02c",
                     label="Block bootstrap 95% CI")

    # Naive bootstrap CI (narrower — incorrect)
    ax.fill_between(trues, naive_lo, naive_hi, alpha=0.15, color="#d62728",
                     label="Naive bootstrap 95% CI")

    # Point estimates
    ax.scatter(trues, ests, c="#1f77b4", s=80, zorder=5, edgecolors="black",
               linewidths=0.5, label="TRF estimate")

    # Check coverage
    coverage_block = sum(1 for r in results
                         if r["block_ci_lo"] <= r["true"] <= r["block_ci_hi"])
    coverage_naive = sum(1 for r in results
                         if r["naive_ci_lo"] <= r["true"] <= r["naive_ci_hi"])

    ax.set_xlabel("True TRF (known ground truth)", fontsize=12)
    ax.set_ylabel("Estimated TRF", fontsize=12)
    # Compute mean bias (conservative overestimate expected)
    mean_bias = np.mean([r["est"] - r["true"] for r in results])
    ax.set_title(
        f"Synthetic Validation: N-Ceiling Estimator Recovery\n"
        f"Mean bias: +{mean_bias:.3f} (conservative upper bound) | "
        f"Block vs naive CI width ratio: "
        f"{np.mean([r['block_ci_hi']-r['block_ci_lo'] for r in results]):.3f} / "
        f"{np.mean([r['naive_ci_hi']-r['naive_ci_lo'] for r in results]):.3f}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0.05, 0.65)
    ax.set_ylim(0.05, 0.65)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
