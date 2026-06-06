"""
E7: Non-Redundancy Matrix
===========================

The KILL SWITCH experiment. Computes pairwise Spearman correlation between
all 6 geometry kappa proxies + kappa_compat + theory_sigma.

If any pair has |rho| > 0.9, the weaker proxy should be dropped.

Pass criteria:
  - NO pair of geometry proxies has |Spearman| > 0.9
  - NO geometry proxy correlates > 0.9 with kappa_compat or theory_sigma
  - Average pairwise |correlation| < 0.6

Usage:
  python run_E7_nonredundancy.py [--output-dir outputs/E7]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

YRSN_SRC = Path(__file__).resolve().parents[3] / "yrsn" / "src"
if str(YRSN_SRC) not in sys.path:
    sys.path.insert(0, str(YRSN_SRC))


def load_evidence(output_base: Path) -> dict:
    """Load per-task kappa values from E1-E6 evidence files."""
    kappas = {}

    # E1: kappa_smooth
    e1_path = output_base / "E1" / "evidence_E1.json"
    if e1_path.exists():
        with open(e1_path) as f:
            e1 = json.load(f)
        kappas["kappa_smooth"] = {t: v["kappa_smooth"] for t, v in e1["per_task"].items()}
        kappas["kappa_compat"] = {t: v["kappa_compat"] for t, v in e1["per_task"].items()}

    # E4: kappa_hierarchical
    e4_path = output_base / "E4" / "evidence_E4.json"
    if e4_path.exists():
        with open(e4_path) as f:
            e4 = json.load(f)
        kappas["kappa_hierarchical"] = {t: v["county_kappa"] for t, v in e4["per_task"].items()}

    # E5: kappa_logical
    e5_path = output_base / "E5" / "evidence_E5.json"
    if e5_path.exists():
        with open(e5_path) as f:
            e5 = json.load(f)
        kappas["kappa_logical"] = {t: v["kappa"] for t, v in e5["per_task"].items()}

    # Load theory_sigma from s019d directly
    s019d_path = Path(__file__).resolve().parents[2] / "data" / "s019d" / "seed_42" / "s019d_results.json"
    if s019d_path.exists():
        with open(s019d_path) as f:
            rows = json.load(f)
        from collections import defaultdict
        sigma_by_target = defaultdict(list)
        for r in rows:
            sigma_by_target[r["target"]].append(r["theory_sigma"])
        kappas["theory_sigma"] = {t: float(np.median(v)) for t, v in sigma_by_target.items()}

    return kappas


def compute_correlation_matrix(kappas: dict) -> dict:
    """Compute pairwise Spearman correlations between all kappa variants."""
    proxy_names = sorted(kappas.keys())
    # Find common targets
    common_targets = None
    for name in proxy_names:
        targets = set(kappas[name].keys())
        common_targets = targets if common_targets is None else common_targets & targets
    common_targets = sorted(common_targets)

    n = len(proxy_names)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i, name_i in enumerate(proxy_names):
        for j, name_j in enumerate(proxy_names):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            elif i < j:
                vals_i = [kappas[name_i][t] for t in common_targets]
                vals_j = [kappas[name_j][t] for t in common_targets]
                rho, p = spearmanr(vals_i, vals_j)
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho
                p_matrix[i, j] = p
                p_matrix[j, i] = p

    return {
        "proxy_names": proxy_names,
        "corr_matrix": corr_matrix,
        "p_matrix": p_matrix,
        "n_targets": len(common_targets),
    }


def render_figure(corr_data: dict, output_path: Path):
    """Render the non-redundancy heatmap."""
    names = corr_data["proxy_names"]
    matrix = corr_data["corr_matrix"]
    n = len(names)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap: green (independent) -> yellow -> orange -> red (redundant)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ["#2E7D32", "#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]
    cmap = LinearSegmentedColormap.from_list("redundancy", colors_list)

    im = ax.imshow(np.abs(matrix), cmap=cmap, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [name.replace("kappa_", "k_").replace("theory_", "t_") for name in names]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            abs_val = abs(val)
            color = "white" if abs_val > 0.7 else "black"
            weight = "bold" if abs_val > 0.9 else "normal"
            marker = " ***" if abs_val > 0.9 else (" **" if abs_val > 0.7 else "")
            ax.text(j, i, f"{val:.2f}{marker}", ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)

    plt.colorbar(im, ax=ax, label="|Spearman rho|", shrink=0.8)

    # Add border highlighting for redundant pairs
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i, j]) > 0.9:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=3, edgecolor="red", facecolor="none")
                ax.add_patch(rect)

    ax.set_title(f"E7: Non-Redundancy Matrix ({n} proxies, {corr_data['n_targets']} tasks)\n"
                 f"Green = independent, Red = redundant (DROP if |rho| > 0.9)",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E7: Non-Redundancy Matrix")
    parser.add_argument("--output-dir", type=str, default="outputs/E7")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_base = Path(args.output_dir).parent

    print("Loading evidence from E1-E6...")
    kappas = load_evidence(output_base)
    print(f"  Loaded proxies: {sorted(kappas.keys())}")
    for name, vals in kappas.items():
        print(f"    {name}: {len(vals)} tasks, range [{min(vals.values()):.4f}, {max(vals.values()):.4f}]")

    if len(kappas) < 3:
        print("ERROR: Need at least 3 proxies to compute meaningful correlations.")
        print("Run E1, E4, E5 first.")
        return 1

    print("\nComputing pairwise Spearman correlations...")
    corr_data = compute_correlation_matrix(kappas)

    names = corr_data["proxy_names"]
    matrix = corr_data["corr_matrix"]
    n = len(names)

    # Summary
    print(f"\n{'Proxy A':<20} {'Proxy B':<20} {'Spearman':>10} {'Redundant?':>12}")
    print("-" * 65)
    redundant_pairs = []
    all_abs_corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            rho = matrix[i, j]
            is_red = abs(rho) > 0.9
            all_abs_corrs.append(abs(rho))
            marker = " *** KILL ***" if is_red else ""
            print(f"{names[i]:<20} {names[j]:<20} {rho:>10.4f} {marker:>12}")
            if is_red:
                redundant_pairs.append((names[i], names[j], float(rho)))

    mean_abs_corr = np.mean(all_abs_corrs)
    print(f"\nMean |correlation|: {mean_abs_corr:.4f}")
    print(f"Redundant pairs (|rho| > 0.9): {len(redundant_pairs)}")
    for a, b, rho in redundant_pairs:
        print(f"  KILL: {a} vs {b} (rho={rho:.4f})")

    # Render
    print("\nRendering figure...")
    render_figure(corr_data, output_dir / "fig_E7_nonredundancy_matrix.png")

    # Evidence
    evidence = {
        "experiment": "E7_nonredundancy",
        "n_proxies": n,
        "n_targets": corr_data["n_targets"],
        "proxy_names": names,
        "correlation_matrix": matrix.tolist(),
        "mean_abs_correlation": float(mean_abs_corr),
        "redundant_pairs": redundant_pairs,
        "pass_criteria": {
            "no_redundant_pairs": len(redundant_pairs) == 0,
            "mean_abs_corr_lt_0.6": mean_abs_corr < 0.6,
        },
    }
    with open(output_dir / "evidence_E7.json", "w") as f:
        json.dump(evidence, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.__class__) and isinstance(o, (np.bool_,)) else float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))

    all_pass = all(evidence["pass_criteria"].values())
    print(f"\n{'='*50}")
    print(f"E7 VERDICT: {'PASS' if all_pass else 'FAIL'}")
    for k, v in evidence["pass_criteria"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if redundant_pairs:
        print(f"\n  RECOMMENDED: Drop one proxy from each redundant pair")
    print(f"{'='*50}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
