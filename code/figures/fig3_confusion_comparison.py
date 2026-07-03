"""Figure 3: Confusion matrix comparison — nova_embed vs openai_3_small.

nova_embed: #1 accuracy, #15 alpha (worst certificate)
openai_3_small: #6 accuracy, #1 alpha (best certificate)

Shows WHY they invert: nova_embed has high N recall but low R recall.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LEADERBOARD = Path(__file__).parent.parent.parent / "yrsn-train" / "apps" / "train_rsn_head" / "paper_data" / "leaderboard.json"
OUTPUT = Path(__file__).parent / "fig3_confusion_comparison.pdf"

CLASSES = ["R", "S", "N"]


def extract_cm(model_data):
    """Extract 3x3 confusion matrix from model data."""
    cm = model_data["confusion_matrix"]
    return np.array([
        [cm["R_R"], cm["R_S"], cm["R_N"]],
        [cm["S_R"], cm["S_S"], cm["S_N"]],
        [cm["N_R"], cm["N_S"], cm["N_N"]],
    ])


def plot_cm(ax, cm, title, subtitle):
    """Plot a single confusion matrix."""
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_yticklabels(CLASSES, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    return im


def main():
    with open(LEADERBOARD) as f:
        lb = json.load(f)

    nova = lb["models"]["nova_embed"]
    openai = lb["models"]["openai_3_small"]

    cm_nova = extract_cm(nova)
    cm_openai = extract_cm(openai)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    plot_cm(
        ax1, cm_nova,
        "nova_embed",
        f"Accuracy #1 (77.1%) | Alpha #15 ({nova['alpha']:.3f})",
    )
    plot_cm(
        ax2, cm_openai,
        "openai_3_small",
        f"Accuracy #6 (74.5%) | Alpha #1 ({openai['alpha']:.3f})",
    )

    fig.suptitle(
        "Why the #1 accuracy model has the worst certificate",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Annotation: what to notice
    fig.text(
        0.5, -0.06,
        "nova_embed: N recall = 0.956 (memorizes noise) | R recall = 0.661 (misses structure)\n"
        "openai_3_small: N recall = 0.940 (comparable) | R recall = 0.654 (comparable) but R precision = 0.732 vs 0.650",
        ha="center", fontsize=9, style="italic", color="gray",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
