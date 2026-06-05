"""Figure 4: CONUS-27 TRF heatmap.

Shows R-squared across 3 model families and 27 geospatial tasks,
with TRF column. Sorted by TRF to reveal task difficulty spectrum.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT = Path(__file__).parent / "fig4_nceiling_heatmap.pdf"

# Data from S3 geo_cert artifacts (2026-04-20)
FAMILIES = ["PCA32", "GNN", "Spatial Lag"]
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
    # Sort by TRF (ascending = easiest tasks first)
    tasks_sorted = sorted(TASK_R2.keys(), key=lambda t: -max(TASK_R2[t]))
    n_tasks = len(tasks_sorted)

    # Build matrix
    r2_matrix = np.array([TASK_R2[t] for t in tasks_sorted])
    trf_vals = 1.0 - np.max(r2_matrix, axis=1)

    # Add TRF as a 4th column
    display_matrix = np.column_stack([r2_matrix, trf_vals])

    fig, ax = plt.subplots(figsize=(8, 10))

    # Custom colormap: R2 columns in blues, TRF column in reds
    im = ax.imshow(
        r2_matrix,
        cmap="YlGnBu",
        aspect="auto",
        vmin=0.3, vmax=0.85,
    )

    # Overlay TRF as separate colored cells
    for i, nc in enumerate(trf_vals):
        # TRF color: red intensity proportional to noise floor
        red_val = nc / 0.6  # normalize to [0, 1] roughly
        color = plt.cm.Reds(min(red_val, 1.0))
        ax.add_patch(plt.Rectangle((2.5, i - 0.5), 1, 1, facecolor=color, edgecolor="white", linewidth=0.5))

    # Text annotations
    for i in range(n_tasks):
        for j in range(3):
            val = r2_matrix[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
        # TRF column
        nc = trf_vals[i]
        color = "white" if nc > 0.4 else "black"
        ax.text(3, i, f"{nc:.2f}", ha="center", va="center", fontsize=7, fontweight="bold", color=color)

    ax.set_xticks(range(4))
    ax.set_xticklabels(FAMILIES + ["TRF"], fontsize=10)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([t.replace("_", " ") for t in tasks_sorted], fontsize=8)

    ax.set_title(
        "CONUS-27: R-squared by Model Family and N-Ceiling per Task\n"
        "TRF = irreducible noise floor (higher = harder task)",
        fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("R-squared", fontsize=10)

    # Divider line before TRF column
    ax.axvline(x=2.5, color="black", linewidth=2)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
