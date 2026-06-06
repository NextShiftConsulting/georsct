"""
E9: Gate Decision Flip Analysis
=================================

Compares gate decisions under geometry-specific kappa vs universal kappa_compat.
Shows where geometry proxies catch failures the universal proxy misses (and vice versa).

Pass criteria:
  - At least 5 total gate flips across all proxies
  - At least 1 "false accept rescued" and 1 "false reject rescued"
  - Flips are interpretable

Usage:
  python run_E9_gate_flips.py [--output-dir outputs/E9]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YRSN_SRC = Path(__file__).resolve().parents[3] / "yrsn" / "src"
if str(YRSN_SRC) not in sys.path:
    sys.path.insert(0, str(YRSN_SRC))


def load_evidence(output_base: Path) -> dict:
    """Load per-task kappa values from E1-E6 evidence."""
    kappas = {}

    e1_path = output_base / "E1" / "evidence_E1.json"
    if e1_path.exists():
        with open(e1_path) as f:
            e1 = json.load(f)
        kappas["kappa_smooth"] = {t: v["kappa_smooth"] for t, v in e1["per_task"].items()}
        kappas["kappa_compat"] = {t: v["kappa_compat"] for t, v in e1["per_task"].items()}

    e4_path = output_base / "E4" / "evidence_E4.json"
    if e4_path.exists():
        with open(e4_path) as f:
            e4 = json.load(f)
        kappas["kappa_hierarchical"] = {t: v["county_kappa"] for t, v in e4["per_task"].items()}

    e5_path = output_base / "E5" / "evidence_E5.json"
    if e5_path.exists():
        with open(e5_path) as f:
            e5 = json.load(f)
        kappas["kappa_logical"] = {t: v["kappa"] for t, v in e5["per_task"].items()}

    return kappas


def compute_gate_flips(kappas: dict, threshold: float = 0.22) -> dict:
    """Compute gate flips between each geometry proxy and kappa_compat."""
    if "kappa_compat" not in kappas:
        return {}

    compat = kappas["kappa_compat"]
    geometry_proxies = {k: v for k, v in kappas.items() if k != "kappa_compat"}

    results = {}
    for proxy_name, proxy_vals in geometry_proxies.items():
        common_targets = sorted(set(compat.keys()) & set(proxy_vals.keys()))
        flips = []
        confusion = {"TP": 0, "TN": 0, "FP_rescued": 0, "FN_rescued": 0}

        for t in common_targets:
            kc = compat[t]
            kp = proxy_vals[t]
            compat_gate = "EXECUTE" if kc >= threshold else "RE_ENCODE"
            proxy_gate = "EXECUTE" if kp >= threshold else "RE_ENCODE"

            if compat_gate == proxy_gate:
                if compat_gate == "EXECUTE":
                    confusion["TP"] += 1
                else:
                    confusion["TN"] += 1
            else:
                if compat_gate == "EXECUTE" and proxy_gate == "RE_ENCODE":
                    confusion["FP_rescued"] += 1  # geometry catches a false accept
                else:
                    confusion["FN_rescued"] += 1  # geometry rescues a false reject
                flips.append({
                    "target": t,
                    "kappa_compat": round(kc, 4),
                    f"{proxy_name}": round(kp, 4),
                    "compat_gate": compat_gate,
                    "geometry_gate": proxy_gate,
                    "flip_type": "FP_rescued" if compat_gate == "EXECUTE" else "FN_rescued",
                })

        results[proxy_name] = {
            "n_targets": len(common_targets),
            "n_flips": len(flips),
            "confusion": confusion,
            "flips": flips,
        }

    return results


def render_figure(kappas: dict, flip_results: dict, output_path: Path, threshold: float = 0.22):
    """Render the gate flip scatter plots."""
    geometry_proxies = [k for k in sorted(kappas.keys()) if k != "kappa_compat"]
    n_proxies = len(geometry_proxies)

    if n_proxies == 0:
        print("No geometry proxies to plot.")
        return

    fig, axes = plt.subplots(1, min(n_proxies, 3), figsize=(6 * min(n_proxies, 3), 6), squeeze=False)
    axes = axes[0]

    compat = kappas["kappa_compat"]

    for idx, proxy_name in enumerate(geometry_proxies[:3]):
        ax = axes[idx]
        proxy_vals = kappas[proxy_name]
        common = sorted(set(compat.keys()) & set(proxy_vals.keys()))

        x = [compat[t] for t in common]
        y = [proxy_vals[t] for t in common]

        # Color by flip type
        colors = []
        markers = []
        for t in common:
            kc = compat[t]
            kp = proxy_vals[t]
            compat_gate = kc >= threshold
            proxy_gate = kp >= threshold
            if compat_gate == proxy_gate:
                colors.append("#4C72B0" if compat_gate else "#BBBBBB")
                markers.append("o")
            elif compat_gate and not proxy_gate:
                colors.append("#F44336")  # FP rescued (red)
                markers.append("s")
            else:
                colors.append("#4CAF50")  # FN rescued (green)
                markers.append("^")

        for i, t in enumerate(common):
            ax.scatter(x[i], y[i], c=colors[i], s=60, marker=markers[i],
                      edgecolors="black", linewidth=0.5, zorder=3)

        # Gate threshold lines
        ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5)
        ax.axvline(x=threshold, color="red", linestyle="--", alpha=0.5)

        # Fill flip zones
        ax.fill_between([threshold, 1], threshold, 0, alpha=0.05, color="red",
                       label="FP zone (compat=EX, geo=RE)")
        ax.fill_between([0, threshold], 1, threshold, alpha=0.05, color="green",
                       label="FN zone (compat=RE, geo=EX)")

        ax.set_xlabel("kappa_compat (universal)")
        ax.set_ylabel(proxy_name)
        ax.set_title(f"{proxy_name}\n{flip_results.get(proxy_name, {}).get('n_flips', 0)} flips",
                    fontweight="bold", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], "k:", alpha=0.2)
        ax.legend(fontsize=6, loc="lower right")

        # Annotate flip targets
        flips = flip_results.get(proxy_name, {}).get("flips", [])
        for flip in flips:
            t = flip["target"]
            kc = flip["kappa_compat"]
            kp = flip.get(proxy_name, 0)
            ax.annotate(t[:10], (kc, kp), fontsize=6, alpha=0.8,
                       xytext=(5, 5), textcoords="offset points")

    fig.suptitle("E9: Gate Decision Flips -- Where geometry kappa disagrees with universal",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E9: Gate Decision Flip Analysis")
    parser.add_argument("--output-dir", type=str, default="outputs/E9")
    parser.add_argument("--threshold", type=float, default=0.22)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_base = Path(args.output_dir).parent

    print("Loading evidence from E1-E6...")
    kappas = load_evidence(output_base)
    print(f"  Loaded: {sorted(kappas.keys())}")

    if "kappa_compat" not in kappas:
        print("ERROR: kappa_compat not found. Run E1 first.")
        return 1

    print(f"\nComputing gate flips (threshold={args.threshold})...")
    flip_results = compute_gate_flips(kappas, args.threshold)

    total_flips = 0
    total_fp_rescued = 0
    total_fn_rescued = 0

    for proxy_name, result in sorted(flip_results.items()):
        print(f"\n--- {proxy_name} ---")
        c = result["confusion"]
        print(f"  Agree: {c['TP']} EXECUTE, {c['TN']} RE_ENCODE")
        print(f"  Flips: {c['FP_rescued']} FP-rescued, {c['FN_rescued']} FN-rescued")
        total_flips += result["n_flips"]
        total_fp_rescued += c["FP_rescued"]
        total_fn_rescued += c["FN_rescued"]
        for flip in result["flips"][:5]:
            print(f"    {flip['target']}: compat={flip['kappa_compat']:.4f} ({flip['compat_gate']}) "
                  f"-> geometry ({flip['flip_type']})")

    print(f"\n--- TOTALS ---")
    print(f"Total flips: {total_flips}")
    print(f"FP rescued (compat wrong EXECUTE): {total_fp_rescued}")
    print(f"FN rescued (compat wrong RE_ENCODE): {total_fn_rescued}")

    # Render
    print("\nRendering figure...")
    render_figure(kappas, flip_results, output_dir / "fig_E9_gate_flips_scatter.png", args.threshold)

    # Evidence
    evidence = {
        "experiment": "E9_gate_flips",
        "threshold": args.threshold,
        "total_flips": total_flips,
        "total_fp_rescued": total_fp_rescued,
        "total_fn_rescued": total_fn_rescued,
        "per_proxy": {k: {
            "n_flips": v["n_flips"],
            "confusion": v["confusion"],
            "flips": v["flips"],
        } for k, v in flip_results.items()},
        "pass_criteria": {
            "total_flips_ge_5": total_flips >= 5,
            "has_fp_rescued": total_fp_rescued > 0,
            "has_fn_rescued": total_fn_rescued > 0,
        },
    }
    with open(output_dir / "evidence_E9.json", "w") as f:
        json.dump(evidence, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.__class__) and isinstance(o, (np.bool_,)) else float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))

    all_pass = all(evidence["pass_criteria"].values())
    print(f"\n{'='*50}")
    print(f"E9 VERDICT: {'PASS' if all_pass else 'FAIL'}")
    for k, v in evidence["pass_criteria"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"{'='*50}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
