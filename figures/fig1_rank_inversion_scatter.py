"""Figure 1: Accuracy Rank vs Alpha Rank scatter plot.

Shows the certification gap — 56% of models have >= 3-position rank inversion.
Spearman rho = 0.312.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

LEADERBOARD = Path(__file__).parent.parent.parent / "yrsn-train" / "apps" / "train_rsn_head" / "paper_data" / "leaderboard.json"
OUTPUT = Path(__file__).parent / "fig1_rank_inversion_scatter.pdf"


def main():
    with open(LEADERBOARD) as f:
        lb = json.load(f)

    models = lb["models"]
    names, acc_ranks, alpha_ranks = [], [], []

    for name, m in models.items():
        ar = m.get("balanced_acc_rank")
        alr = m.get("alpha_rank")
        if ar is not None and alr is not None:
            names.append(name)
            acc_ranks.append(ar)
            alpha_ranks.append(alr)

    acc_ranks = np.array(acc_ranks)
    alpha_ranks = np.array(alpha_ranks)
    inversions = np.abs(acc_ranks - alpha_ranks)

    # Spearman rho
    n = len(acc_ranks)
    d_sq = np.sum((acc_ranks - alpha_ranks) ** 2)
    rho = 1 - (6 * d_sq) / (n * (n**2 - 1))

    fig, ax = plt.subplots(figsize=(7, 6))

    # Diagonal (perfect agreement)
    ax.plot([0.5, 16.5], [0.5, 16.5], "k--", alpha=0.3, linewidth=1, label="Perfect agreement")

    # Color by inversion magnitude
    colors = []
    for inv in inversions:
        if inv >= 10:
            colors.append("#d62728")  # red — extreme
        elif inv >= 5:
            colors.append("#ff7f0e")  # orange — large
        elif inv >= 3:
            colors.append("#bcbd22")  # olive — moderate
        else:
            colors.append("#2ca02c")  # green — small

    ax.scatter(acc_ranks, alpha_ranks, c=colors, s=120, zorder=5, edgecolors="black", linewidths=0.5)

    # Label key models
    highlight = {"nova_embed", "openai_3_small", "titan_v2", "voyage4_large", "minilm"}
    for name, ar, alr in zip(names, acc_ranks, alpha_ranks):
        if name in highlight:
            label = name.replace("_", " ")
            # Offset labels to avoid overlap
            offsets = {
                "nova_embed": (0.4, -0.6),
                "openai_3_small": (-4.5, 0.3),
                "titan_v2": (0.4, -0.6),
                "voyage4_large": (0.4, 0.4),
                "minilm": (0.4, -0.6),
            }
            dx, dy = offsets.get(name, (0.3, 0.3))
            ax.annotate(
                label,
                (ar, alr),
                (ar + dx, alr + dy),
                fontsize=8,
                fontstyle="italic",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5) if abs(dx) > 1 or abs(dy) > 1 else None,
            )

    ax.set_xlabel("Accuracy Rank", fontsize=12)
    ax.set_ylabel("Certificate Rank (alpha)", fontsize=12)
    ax.set_title(
        f"The Certification Gap: Accuracy vs Certificate Rank\n"
        f"Spearman $\\rho$ = {rho:.2f} | {np.sum(inversions >= 3)}/{n} models inverted $\\geq$3 positions",
        fontsize=11,
    )

    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0.5, 16.5)
    ax.set_xticks(range(1, 17))
    ax.set_yticks(range(1, 17))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#2ca02c", label="< 3 positions"),
        mpatches.Patch(color="#bcbd22", label="3-4 positions"),
        mpatches.Patch(color="#ff7f0e", label="5-9 positions"),
        mpatches.Patch(color="#d62728", label=">= 10 positions"),
    ]
    ax.legend(handles=legend_elements, title="Rank inversion", loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    print(f"Saved: {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
