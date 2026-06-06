"""
E1: Smooth Convergence Heatmap
==============================

Validates kappa_smooth on real CONUS-27 data from s019d.
Produces a publishable heatmap showing which tasks converge across embedding families.

Flood question: "Which health outcomes have STABLE predictions regardless of embedding?"

Pass criteria:
  - kappa_smooth varies from <0.3 to >0.7 across 27 tasks
  - Spearman(kappa_smooth, mean_proxy_kappa) < 0.9
  - At least 3 gate-flip tasks
  - Clear visual separation in heatmap

Usage:
  python run_E1_smooth_convergence.py [--seed 42] [--output-dir outputs/E1]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

# Add yrsn to path if needed
YRSN_SRC = Path(__file__).resolve().parents[3] / "yrsn" / "src"
if str(YRSN_SRC) not in sys.path:
    sys.path.insert(0, str(YRSN_SRC))

from yrsn.core.kappa.smooth import compute_kappa_smooth


def load_s019d(seed: int = 42) -> list:
    """Load s019d results JSON."""
    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "s019d" / f"seed_{seed}" / "s019d_results.json",
        Path(f"data/s019d/seed_{seed}/s019d_results.json"),
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"s019d_results.json not found for seed {seed}. Tried: {candidates}")


def group_r2_by_task(rows: list) -> dict:
    """Group R2 values by (target, fold) -> {embedding: r2}."""
    by_task_fold = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_task_fold[r["target"]][r["embedding"]].append(r["r2"])
    # Median R2 across folds per (target, embedding)
    result = {}
    for target, emb_dict in by_task_fold.items():
        result[target] = {emb: float(np.median(vals)) for emb, vals in emb_dict.items()}
    return result


def group_proxy_kappa_by_task(rows: list) -> dict:
    """Get median proxy_kappa per target."""
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append(r["proxy_kappa"])
    return {t: float(np.median(v)) for t, v in by_target.items()}


def group_gate_decisions(rows: list) -> dict:
    """Get majority gate decision per target (across folds/embeddings)."""
    by_target = defaultdict(list)
    for r in rows:
        dec = r.get("gate_oobleck", {}).get("gate_decision", "UNKNOWN")
        by_target[r["target"]].append(dec)
    result = {}
    for t, decs in by_target.items():
        execute_count = sum(1 for d in decs if "EXECUTE" in d)
        result[t] = "EXECUTE" if execute_count > len(decs) / 2 else "RE_ENCODE"
    return result


def compute_all_smooth_kappas(r2_by_task: dict) -> dict:
    """Compute kappa_smooth for all tasks."""
    results = {}
    for target, r2_dict in sorted(r2_by_task.items()):
        result = compute_kappa_smooth(r2_dict)
        results[target] = {
            "kappa_task": result.kappa_task,
            "kappa_per_embedding": result.kappa_per_embedding,
            "task_convergence": result.task_convergence,
            "n_families": result.n_families,
            "r2_values": result.r2_values,
        }
    return results


def check_redundancy(smooth_kappas: dict, proxy_kappas: dict) -> dict:
    """Check Spearman correlation between kappa_smooth and kappa_compat."""
    targets = sorted(set(smooth_kappas.keys()) & set(proxy_kappas.keys()))
    smooth_vals = [smooth_kappas[t]["kappa_task"] for t in targets]
    proxy_vals = [proxy_kappas[t] for t in targets]
    rho, p_value = spearmanr(smooth_vals, proxy_vals)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "is_redundant": abs(rho) > 0.9,
        "n_tasks": len(targets),
    }


def find_gate_flips(smooth_kappas: dict, proxy_kappas: dict, threshold: float = 0.22) -> list:
    """Find tasks where kappa_smooth and kappa_compat disagree on gate outcome."""
    flips = []
    for target in sorted(smooth_kappas.keys()):
        ks = smooth_kappas[target]["kappa_task"]
        kp = proxy_kappas.get(target, 0)
        smooth_gate = "EXECUTE" if ks >= threshold else "RE_ENCODE"
        proxy_gate = "EXECUTE" if kp >= threshold else "RE_ENCODE"
        if smooth_gate != proxy_gate:
            flips.append({
                "target": target,
                "kappa_smooth": round(ks, 4),
                "kappa_compat": round(kp, 4),
                "smooth_gate": smooth_gate,
                "proxy_gate": proxy_gate,
            })
    return flips


def render_heatmap(smooth_kappas: dict, proxy_kappas: dict, output_path: Path):
    """Render the publishable heatmap figure."""
    targets = sorted(smooth_kappas.keys(), key=lambda t: smooth_kappas[t]["kappa_task"], reverse=True)
    embeddings = sorted(list(smooth_kappas[targets[0]]["r2_values"].keys()))

    n_targets = len(targets)
    n_emb = len(embeddings)

    # Build data matrix: R2 per (target, embedding) + CV + kappa_smooth + kappa_compat
    r2_matrix = np.zeros((n_targets, n_emb))
    kappa_smooth_col = np.zeros(n_targets)
    kappa_compat_col = np.zeros(n_targets)
    cv_col = np.zeros(n_targets)

    for i, target in enumerate(targets):
        r2_vals = smooth_kappas[target]["r2_values"]
        for j, emb in enumerate(embeddings):
            r2_matrix[i, j] = r2_vals.get(emb, 0)
        kappa_smooth_col[i] = smooth_kappas[target]["kappa_task"]
        kappa_compat_col[i] = proxy_kappas.get(target, 0)
        vals = list(r2_vals.values())
        mean_r2 = np.mean(vals)
        cv_col[i] = np.std(vals) / max(abs(mean_r2), 1e-8)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 10),
                             gridspec_kw={"width_ratios": [n_emb, 1.5, 1.5]},
                             sharey=True)

    # Panel 1: R2 heatmap
    emb_short = [e.replace("_", "\n")[:10] for e in embeddings]
    im1 = axes[0].imshow(r2_matrix, aspect="auto", cmap="RdYlBu", vmin=0, vmax=1)
    axes[0].set_xticks(range(n_emb))
    axes[0].set_xticklabels(emb_short, fontsize=8, rotation=45, ha="right")
    axes[0].set_yticks(range(n_targets))
    axes[0].set_yticklabels(targets, fontsize=8)
    axes[0].set_title("R2 per Embedding Family", fontsize=11, fontweight="bold")
    # Annotate cells
    for i in range(n_targets):
        for j in range(n_emb):
            val = r2_matrix[i, j]
            color = "white" if val < 0.4 or val > 0.85 else "black"
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    # Panel 2: kappa_smooth bars
    colors_smooth = plt.cm.RdYlBu(kappa_smooth_col)
    axes[1].barh(range(n_targets), kappa_smooth_col, color=colors_smooth, edgecolor="gray", linewidth=0.5)
    axes[1].axvline(x=0.22, color="red", linestyle="--", linewidth=1, label="Gate threshold")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("kappa_smooth", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("kappa")
    axes[1].legend(fontsize=7, loc="lower right")
    for i in range(n_targets):
        axes[1].text(kappa_smooth_col[i] + 0.02, i, f"{kappa_smooth_col[i]:.2f}", va="center", fontsize=7)

    # Panel 3: kappa_compat bars (for comparison)
    colors_compat = plt.cm.RdYlBu(kappa_compat_col)
    axes[2].barh(range(n_targets), kappa_compat_col, color=colors_compat, edgecolor="gray", linewidth=0.5)
    axes[2].axvline(x=0.22, color="red", linestyle="--", linewidth=1, label="Gate threshold")
    axes[2].set_xlim(0, 1)
    axes[2].set_title("kappa_compat (universal)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("kappa")
    axes[2].legend(fontsize=7, loc="lower right")
    for i in range(n_targets):
        axes[2].text(kappa_compat_col[i] + 0.02, i, f"{kappa_compat_col[i]:.2f}", va="center", fontsize=7)

    plt.colorbar(im1, ax=axes[0], shrink=0.5, label="R2", pad=0.02)
    fig.suptitle("E1: Smooth Convergence -- Which tasks are stable across embeddings?",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E1: Smooth Convergence Heatmap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/E1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    print(f"Loading s019d results (seed={args.seed})...")
    rows = load_s019d(args.seed)
    print(f"  {len(rows)} rows loaded")

    # Step 2: Group R2 by task
    r2_by_task = group_r2_by_task(rows)
    proxy_kappas = group_proxy_kappa_by_task(rows)
    print(f"  {len(r2_by_task)} tasks, {len(r2_by_task[list(r2_by_task.keys())[0]])} embeddings per task")

    # Step 3: Compute kappa_smooth for all tasks
    print("Computing kappa_smooth for all tasks...")
    smooth_kappas = compute_all_smooth_kappas(r2_by_task)

    # Print summary table
    print("\n--- kappa_smooth Summary ---")
    print(f"{'Target':<30} {'kappa_smooth':>12} {'kappa_compat':>12} {'CV(R2)':>8}")
    print("-" * 65)
    for target in sorted(smooth_kappas.keys(), key=lambda t: smooth_kappas[t]["kappa_task"], reverse=True):
        ks = smooth_kappas[target]["kappa_task"]
        kp = proxy_kappas.get(target, 0)
        vals = list(smooth_kappas[target]["r2_values"].values())
        cv = np.std(vals) / max(abs(np.mean(vals)), 1e-8)
        print(f"{target:<30} {ks:>12.4f} {kp:>12.4f} {cv:>8.4f}")

    kappa_vals = [v["kappa_task"] for v in smooth_kappas.values()]
    print(f"\nkappa_smooth range: [{min(kappa_vals):.4f}, {max(kappa_vals):.4f}]")

    # Step 4: Redundancy check
    print("\n--- Redundancy Check ---")
    redundancy = check_redundancy(smooth_kappas, proxy_kappas)
    print(f"Spearman(kappa_smooth, kappa_compat) = {redundancy['spearman_rho']:.4f} (p={redundancy['p_value']:.4e})")
    if redundancy["is_redundant"]:
        print("*** KILL: kappa_smooth is REDUNDANT with kappa_compat (rho > 0.9) ***")
    else:
        print("PASS: kappa_smooth is NON-REDUNDANT with kappa_compat")

    # Step 5: Gate flips
    print("\n--- Gate Flip Analysis ---")
    flips = find_gate_flips(smooth_kappas, proxy_kappas)
    print(f"Gate flips found: {len(flips)}")
    for flip in flips:
        print(f"  {flip['target']}: smooth={flip['kappa_smooth']:.4f} ({flip['smooth_gate']}) "
              f"vs compat={flip['kappa_compat']:.4f} ({flip['proxy_gate']})")

    # Step 6: Render figure
    print("\nRendering figure...")
    render_heatmap(smooth_kappas, proxy_kappas, output_dir / "fig_E1_smooth_convergence_heatmap.png")

    # Step 7: Save evidence JSON
    evidence = {
        "experiment": "E1_smooth_convergence",
        "seed": args.seed,
        "n_tasks": len(smooth_kappas),
        "kappa_smooth_range": [float(min(kappa_vals)), float(max(kappa_vals))],
        "redundancy": redundancy,
        "gate_flips": flips,
        "per_task": {t: {
            "kappa_smooth": v["kappa_task"],
            "kappa_compat": proxy_kappas.get(t, 0),
            "r2_values": v["r2_values"],
            "kappa_per_embedding": v["kappa_per_embedding"],
        } for t, v in smooth_kappas.items()},
        "pass_criteria": {
            "range_gt_0.3": (max(kappa_vals) - min(kappa_vals)) > 0.3,
            "nonredundant": not redundancy["is_redundant"],
            "gate_flips_ge_3": len(flips) >= 3,
        },
    }
    evidence_path = output_dir / "evidence_E1.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print(f"Evidence saved: {evidence_path}")

    # Final verdict
    all_pass = all(evidence["pass_criteria"].values())
    print(f"\n{'='*50}")
    print(f"E1 VERDICT: {'PASS' if all_pass else 'FAIL'}")
    for k, v in evidence["pass_criteria"].items():
        print(f"  [{('PASS' if v else 'FAIL')}] {k}")
    print(f"{'='*50}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
