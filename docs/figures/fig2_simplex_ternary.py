"""Figure 2: R/S/N simplex ternary plot with 16 models.

Models plotted by their mean R, S, N coordinates on the unit simplex.
Color encodes accuracy rank to show that simplex position != accuracy.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from pathlib import Path

LEADERBOARD = Path(__file__).parent.parent.parent / "yrsn-train" / "apps" / "train_rsn_head" / "paper_data" / "leaderboard.json"
OUTPUT = Path(__file__).parent / "fig2_simplex_ternary.pdf"


def ternary_to_cartesian(r, s, n):
    """Convert R/S/N simplex coordinates to 2D Cartesian for plotting."""
    x = 0.5 * (2 * s + n)
    y = (np.sqrt(3) / 2) * n
    return x, y


def draw_triangle(ax):
    """Draw the simplex triangle with axis labels."""
    # Vertices: R at bottom-left, S at bottom-right, N at top
    vertices = np.array([
        [0.0, 0.0],          # R = 1 (bottom-left)
        [1.0, 0.0],          # S = 1 (bottom-right)
        [0.5, np.sqrt(3)/2], # N = 1 (top)
        [0.0, 0.0],          # close triangle
    ])
    ax.plot(vertices[:, 0], vertices[:, 1], "k-", linewidth=1.5)

    # Grid lines
    for i in range(1, 10):
        f = i / 10
        # Lines of constant R
        p1 = ternary_to_cartesian(f, 1-f, 0)
        p2 = ternary_to_cartesian(f, 0, 1-f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", alpha=0.08, linewidth=0.5)
        # Lines of constant S
        p1 = ternary_to_cartesian(0, f, 1-f)
        p2 = ternary_to_cartesian(1-f, f, 0)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", alpha=0.08, linewidth=0.5)
        # Lines of constant N
        p1 = ternary_to_cartesian(1-f, 0, f)
        p2 = ternary_to_cartesian(0, 1-f, f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", alpha=0.08, linewidth=0.5)

    # Vertex labels
    offset = 0.04
    ax.text(-offset, -offset, "R = 1", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(1 + offset, -offset, "S = 1", ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(0.5, np.sqrt(3)/2 + offset, "N = 1", ha="center", va="bottom", fontsize=11, fontweight="bold")


def main():
    with open(LEADERBOARD) as f:
        lb = json.load(f)

    models = lb["models"]

    fig, ax = plt.subplots(figsize=(8, 7))
    draw_triangle(ax)

    names, xs, ys, acc_ranks, alphas = [], [], [], [], []

    for name, m in models.items():
        r = m.get("R_mean")
        s = m.get("S_mean")
        n = m.get("N_mean")
        ar = m.get("balanced_acc_rank")
        if r is not None and s is not None and n is not None:
            x, y = ternary_to_cartesian(r, s, n)
            names.append(name)
            xs.append(x)
            ys.append(y)
            acc_ranks.append(ar)
            alphas.append(m.get("alpha", 0))

    xs = np.array(xs)
    ys = np.array(ys)
    acc_ranks = np.array(acc_ranks)

    # Color by accuracy rank (1=best=dark, 16=worst=light)
    scatter = ax.scatter(
        xs, ys,
        c=acc_ranks,
        cmap="RdYlGn_r",
        s=150,
        zorder=5,
        edgecolors="black",
        linewidths=0.7,
        vmin=1, vmax=16,
    )

    # Label all models
    for name, x, y, ar in zip(names, xs, ys, acc_ranks):
        label = name.replace("_", "\n")
        ax.annotate(
            label,
            (x, y),
            (x + 0.01, y + 0.01),
            fontsize=5.5,
            alpha=0.8,
        )

    # Center of simplex (equal R=S=N=1/3)
    cx, cy = ternary_to_cartesian(1/3, 1/3, 1/3)
    ax.plot(cx, cy, "k+", markersize=15, markeredgewidth=2, zorder=6, alpha=0.3)
    ax.annotate("R=S=N", (cx, cy), (cx - 0.06, cy - 0.03), fontsize=7, alpha=0.4)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Accuracy Rank (1 = best)", fontsize=10)

    ax.set_title(
        "16 Embedding Models in R/S/N Simplex Space\n"
        "Color = accuracy rank (does not predict simplex position)",
        fontsize=11,
    )
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
