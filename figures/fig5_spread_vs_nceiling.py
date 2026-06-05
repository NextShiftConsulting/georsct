"""Figure 5: Cross-family R-squared spread vs TRF for 27 CONUS tasks.

Key insight: task noise (TRF) varies 4x while cross-family spread
is tiny (~0.02). Architecture choice is nearly invisible against task difficulty.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT = Path(__file__).parent / "fig5_spread_vs_nceiling.pdf"

# Data from S3 geo_cert artifacts (2026-04-20)
# Format: task -> [PCA32, GNN, SpatialLag]
TASK_R2 = {
    "night_lights":           [0.8445, 0.8335, 0.8445],
    "smoking":                [0.8084, 0.7928, 0.8084],
    "physical_health":        [0.8055, 0.7841, 0.8055],
    "physical_inactivity":    [0.8042, 0.7726, 0.8042],
    "population_density":     [0.7996, 0.7911, 0.7996],
    "dental_visit":           [0.7946, 0.7791, 0.7946],
    "copd":                   [0.7634, 0.7629, 0.7634],
    "obesity":                [0.7536, 0.7348, 0.7536],
    "diabetes":               [0.7393, 0.7150, 0.7393],
    "high_blood_pressure":    [0.7146, 0.7013, 0.7146],
    "home_value":             [0.7094, 0.6697, 0.7094],
    "mental_health":          [0.7070, 0.6653, 0.7070],
    "arthritis":              [0.7049, 0.6859, 0.7002],
    "stroke":                 [0.6969, 0.6760, 0.6969],
    "sleep_less_7hr":         [0.6840, 0.6787, 0.6840],
    "income":                 [0.6770, 0.6346, 0.6770],
    "coronary_heart_disease": [0.6560, 0.6436, 0.6560],
    "chronic_kidney_disease": [0.6219, 0.5978, 0.6219],
    "asthma":                 [0.6145, 0.5701, 0.6193],
    "cancer":                 [0.6078, 0.5654, 0.5999],
    "bp_medicated":           [0.5913, 0.5299, 0.5664],
    "annual_checkup":         [0.5754, 0.5628, 0.5660],
    "tree_cover":             [0.5402, 0.5579, 0.5402],
    "high_cholesterol":       [0.5140, 0.5041, 0.5140],
    "elevation":              [0.4608, 0.5165, 0.4608],
    "binge_drinking":         [0.4489, 0.4445, 0.4634],
    "cholesterol_screening":  [0.4066, 0.4066, 0.3650],
}


def main():
    tasks = list(TASK_R2.keys())
    trf_vals = []
    spreads = []
    labels = []

    for task in tasks:
        vals = TASK_R2[task]
        trf = 1.0 - max(vals)
        spread = max(vals) - min(vals)
        trf_vals.append(trf)
        spreads.append(spread)
        labels.append(task)

    trf_vals = np.array(trf_vals)
    spreads = np.array(spreads)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by TRF magnitude
    scatter = ax.scatter(
        trf_vals, spreads,
        c=trf_vals,
        cmap="RdYlGn_r",
        s=80,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )

    # Label extreme points
    for i, label in enumerate(labels):
        if spreads[i] > 0.04 or trf_vals[i] > 0.5 or trf_vals[i] < 0.17:
            ax.annotate(
                label.replace("_", " "),
                (trf_vals[i], spreads[i]),
                (trf_vals[i] + 0.01, spreads[i] + 0.002),
                fontsize=7,
                alpha=0.8,
            )

    # Reference lines
    ax.axhline(y=np.mean(spreads), color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(0.55, np.mean(spreads) + 0.001, f"mean spread = {np.mean(spreads):.3f}",
            fontsize=8, color="gray")

    ax.set_xlabel("TRF (irreducible noise)", fontsize=12)
    ax.set_ylabel("Cross-family R-squared spread", fontsize=12)
    ax.set_title(
        "Architecture Choice vs Task Difficulty: 27 CONUS Tasks\n"
        "Spread is tiny (~0.02) while TRF varies 4x",
        fontsize=11,
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label("TRF", fontsize=10)

    ax.set_xlim(0.1, 0.65)
    ax.set_ylim(-0.005, 0.07)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
