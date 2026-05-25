r"""
figures.py — Generate Figures 1, 4, 5 for the GeoCert paper.

Three figures, hard-coded data from canonical OOF artifacts so the paper
text and the figures cannot drift apart:

    Figure 1: fig1_rank_scatter.pdf
        Accuracy rank vs alpha rank for 16 text embedding models on MIRACL.
        Source: table1_leaderboard.tex (canonical text-side OOF artifacts).

    Figure 4: fig4_conus27_heatmap.pdf
        CONUS-27 holdout R^2 by family + N-ceiling, 27 tasks x 4 columns.
        Source: table2_conus27.tex (canonical OOF test-fold values per
        EVIDENCE_AUDIT 2026-04-29).

    Figure 5: fig5_spread_vs_nceil.pdf
        Cross-family R^2 spread vs N-ceiling for 27 CONUS tasks.
        Source: derived from table2_conus27.tex.

Usage:
    python figures.py             # writes all three to ./figures/
    python figures.py --hf        # alternative path: load from HF dataset
                                  #    rudymartin/geocert (verify schema first)

Style notes:
    NeurIPS column width is ~3.3 inches. Figure 1 and Figure 5 use 3.3" wide.
    Figure 4 needs more vertical real estate (27 rows of tasks) and renders
    at 6.5" wide x 7.5" tall; in the LaTeX, prefer width=0.95\textwidth or
    width=\linewidth in a single-column figure environment for readability.
    Update section 8 of paper_geocert.tex accordingly if you keep the
    \includegraphics[width=0.48\textwidth]{...} call too narrow for the
    heatmap.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Style — uniform across all three figures.
# ---------------------------------------------------------------------------

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ---------------------------------------------------------------------------
# Data — extracted from paper tables. Update only if the canonical OOF
# artifacts change (e.g., another EVIDENCE_AUDIT correction).
# ---------------------------------------------------------------------------

# Table 1 leaderboard — 16 text models on MIRACL.
# Columns: name, accuracy, accuracy_rank, alpha, alpha_rank
TEXT_MODELS = [
    ("nova_embed",       0.771,  1, 0.483, 15),
    ("voyage4_large",    0.768,  2, 0.509,  3),
    ("gemini",           0.766,  3, 0.505,  7),
    ("bge_m3",           0.762,  4, 0.509,  2),
    ("qwen3_8b",         0.749,  5, 0.507,  5),
    ("openai_3_small",   0.745,  6, 0.514,  1),
    ("jina",             0.737,  7, 0.501,  9),
    ("jina_v3_mrl384",   0.737,  8, 0.485, 13),
    ("cohere_v4",        0.733,  9, 0.487, 12),
    ("jina_v3_pca384",   0.730, 10, 0.505,  8),
    ("voyage4_nano",     0.719, 11, 0.506,  6),
    ("nemotron_native",  0.712, 12, 0.496, 10),
    ("nemotron_mrl1024", 0.710, 13, 0.482, 16),
    ("nemotron",         0.694, 14, 0.484, 14),
    ("titan_v2",         0.675, 15, 0.508,  4),
    ("minilm",           0.658, 16, 0.495, 11),
]

# Table 2 CONUS-27 — post EVIDENCE_AUDIT 2026-04-29 (canonical OOF test-fold).
# Columns: category, task, pca32_r2, gnn_r2, slag_r2, n_ceiling, spread
CONUS_27 = [
    ("Environ.", "Night lights",           0.799, 0.833, 0.845, 0.155, 0.045),
    ("Health",   "Smoking",                0.803, 0.793, 0.808, 0.192, 0.016),
    ("Health",   "Physical health",        0.800, 0.784, 0.806, 0.194, 0.021),
    ("Health",   "Physical inactivity",    0.799, 0.773, 0.804, 0.196, 0.032),
    ("Environ.", "Population density",     0.754, 0.791, 0.800, 0.200, 0.045),
    ("Health",   "Dental visit",           0.794, 0.779, 0.795, 0.205, 0.016),
    ("Health",   "COPD",                   0.759, 0.763, 0.763, 0.237, 0.005),
    ("Health",   "Obesity",                0.747, 0.735, 0.754, 0.246, 0.019),
    ("Health",   "Diabetes",               0.746, 0.715, 0.739, 0.254, 0.030),
    ("Health",   "High blood pressure",    0.714, 0.701, 0.715, 0.285, 0.013),
    ("Socio.",   "Home value",             0.676, 0.670, 0.709, 0.291, 0.040),
    ("Health",   "Stroke",                 0.708, 0.676, 0.697, 0.292, 0.032),
    ("Health",   "Mental health",          0.707, 0.665, 0.707, 0.293, 0.042),
    ("Health",   "Arthritis",              0.705, 0.686, 0.700, 0.295, 0.019),
    ("Health",   "Sleep <7hr",             0.662, 0.679, 0.684, 0.316, 0.022),
    ("Socio.",   "Income",                 0.684, 0.635, 0.677, 0.316, 0.049),
    ("Health",   "Coronary heart disease", 0.660, 0.644, 0.656, 0.340, 0.016),
    ("Health",   "Chronic kidney disease", 0.625, 0.598, 0.622, 0.375, 0.028),
    ("Health",   "Asthma",                 0.615, 0.570, 0.619, 0.381, 0.049),
    ("Health",   "Cancer",                 0.608, 0.565, 0.600, 0.392, 0.042),
    ("Health",   "BP medicated",           0.591, 0.530, 0.566, 0.409, 0.061),
    ("Health",   "Annual checkup",         0.575, 0.563, 0.566, 0.425, 0.013),
    ("Environ.", "Tree cover",             0.474, 0.558, 0.540, 0.442, 0.084),
    ("Health",   "High cholesterol",       0.531, 0.504, 0.514, 0.469, 0.027),
    ("Environ.", "Elevation",              0.350, 0.517, 0.461, 0.483, 0.166),
    ("Health",   "Binge drinking",         0.449, 0.445, 0.463, 0.537, 0.019),
    ("Health",   "Cholesterol screening",  0.349, 0.407, 0.365, 0.593, 0.057),
]

# Category styling for Figure 5 (and Figure 4 row markers).
CATEGORY_STYLE = {
    "Health":   dict(color="#2c7fb8", marker="o", label="Health"),
    "Socio.":   dict(color="#f5a623", marker="s", label="Socioeconomic"),
    "Environ.": dict(color="#2ca25f", marker="D", label="Environmental"),
}


# ---------------------------------------------------------------------------
# Figure 1 — Accuracy rank vs alpha rank scatter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ScatterPoint:
    name: str
    acc_rank: int
    alpha_rank: int
    inversion: int


def _prepare_rank_scatter():
    return [
        _ScatterPoint(
            name=m[0],
            acc_rank=m[2],
            alpha_rank=m[4],
            inversion=abs(m[2] - m[4]),
        )
        for m in TEXT_MODELS
    ]


def figure_1_rank_scatter(out_path: str) -> None:
    points = _prepare_rank_scatter()
    fig, ax = plt.subplots(figsize=(3.3, 3.0))

    # Diagonal reference (perfect agreement).
    ax.plot([0, 17], [0, 17], color="0.7", lw=0.8, ls="--", zorder=1)

    # Points colored by inversion magnitude.
    inversions = np.array([p.inversion for p in points])
    sc = ax.scatter(
        [p.acc_rank for p in points],
        [p.alpha_rank for p in points],
        c=inversions,
        cmap="viridis_r",
        s=42,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

    # Label the three extreme inversions called out in the paper.
    label_targets = {"nova_embed", "titan_v2", "openai_3_small"}
    offsets = {
        "nova_embed":     (-0.5, -0.8),
        "titan_v2":       (-1.1,  0.7),
        "openai_3_small": (-1.0,  0.5),
    }
    for p in points:
        if p.name in label_targets:
            dx, dy = offsets[p.name]
            ax.annotate(
                p.name,
                xy=(p.acc_rank, p.alpha_rank),
                xytext=(p.acc_rank + dx, p.alpha_rank + dy),
                fontsize=6.5,
                ha="left" if dx >= 0 else "right",
                va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4"),
            )

    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0.5, 16.5)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_yticks([1, 4, 8, 12, 16])
    ax.invert_yaxis()
    ax.invert_xaxis()  # rank 1 in upper-left for readability
    ax.set_xlabel("Accuracy rank (1 = best)")
    ax.set_ylabel(r"$\alpha$ rank (1 = best)")
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("|rank inversion|", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — CONUS-27 heatmap
# ---------------------------------------------------------------------------

def figure_4_conus27_heatmap(out_path: str) -> None:
    n_tasks = len(CONUS_27)
    pca = np.array([row[2] for row in CONUS_27])
    gnn = np.array([row[3] for row in CONUS_27])
    slg = np.array([row[4] for row in CONUS_27])
    nce = np.array([row[5] for row in CONUS_27])
    task_names = [row[1] for row in CONUS_27]
    categories = [row[0] for row in CONUS_27]

    # Sort by N-ceiling ascending (rows already sorted in source).
    fig = plt.figure(figsize=(6.5, 7.5))

    # Two side-by-side subplots: R^2 columns (3) and N-ceiling column (1).
    # Use gridspec for variable width allocation.
    gs = fig.add_gridspec(
        nrows=1, ncols=4,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        wspace=0.05,
    )

    r2_cmap = LinearSegmentedColormap.from_list(
        "r2_blue", ["#f7fbff", "#08306b"]
    )
    nce_cmap = LinearSegmentedColormap.from_list(
        "nce_red", ["#fff5f0", "#67000d"]
    )
    r2_norm = mpl.colors.Normalize(vmin=0.30, vmax=0.85)
    nce_norm = mpl.colors.Normalize(vmin=0.10, vmax=0.65)

    column_data = [pca, gnn, slg, nce]
    column_labels = ["PCA32", "GNN", "Spat. Lag", r"$N_{\rm ceil}$"]
    column_cmaps = [r2_cmap, r2_cmap, r2_cmap, nce_cmap]
    column_norms = [r2_norm, r2_norm, r2_norm, nce_norm]

    axes = []
    for col_idx, (data, lab, cmap, norm) in enumerate(
        zip(column_data, column_labels, column_cmaps, column_norms)
    ):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.imshow(
            data.reshape(-1, 1),
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="none",
        )
        # Annotate each cell with its value.
        for i, v in enumerate(data):
            text_color = "white" if norm(v) > 0.55 else "black"
            ax.text(
                0, i, f"{v:.3f}",
                ha="center", va="center",
                fontsize=6.5,
                color=text_color,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lab, fontsize=7.5, pad=4)
        # Left-most column: task names + category color stripe.
        if col_idx == 0:
            ax.set_yticks(range(n_tasks))
            ax.set_yticklabels(task_names, fontsize=6.5)
            # Category indicator strip via tick label color.
            for tick, cat in zip(ax.get_yticklabels(), categories):
                tick.set_color(CATEGORY_STYLE[cat]["color"])
        axes.append(ax)

    # Custom legend for category colors.
    handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=CATEGORY_STYLE[c]["color"],
                   markersize=7, label=CATEGORY_STYLE[c]["label"])
        for c in ["Health", "Socio.", "Environ."]
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Cross-family R^2 spread vs N-ceiling
# ---------------------------------------------------------------------------

def figure_5_spread_vs_nceil(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.7))

    # Mean spread reference line.
    mean_spread = np.mean([row[6] for row in CONUS_27])
    ax.axhline(
        mean_spread,
        color="0.5",
        lw=0.8,
        ls="--",
        zorder=1,
        label=f"mean = {mean_spread:.3f}",
    )

    # Plot points by category.
    plotted_categories = set()
    for cat in ["Health", "Socio.", "Environ."]:
        rows = [row for row in CONUS_27 if row[0] == cat]
        if not rows:
            continue
        x = [row[5] for row in rows]
        y = [row[6] for row in rows]
        st = CATEGORY_STYLE[cat]
        ax.scatter(
            x, y,
            color=st["color"],
            marker=st["marker"],
            s=36,
            edgecolors="black",
            linewidths=0.4,
            label=st["label"],
            zorder=3,
            alpha=0.9,
        )
        plotted_categories.add(cat)

    # Label the four outliers above 0.05 spread.
    outliers = [row for row in CONUS_27 if row[6] > 0.05]
    label_offsets = {
        "Elevation":             (0.012,  0.002),
        "Tree cover":            (0.012,  0.000),
        "BP medicated":          (-0.020, 0.012),
        "Cholesterol screening": (0.018, -0.010),
    }
    for row in outliers:
        cat, task, _, _, _, nce, sp = row
        dx, dy = label_offsets.get(task, (0.010, 0.005))
        ha = "left" if dx > 0 else "right"
        ax.annotate(
            task,
            xy=(nce, sp),
            xytext=(nce + dx, sp + dy),
            fontsize=6.5,
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4"),
        )

    ax.set_xlim(0.10, 0.65)
    ax.set_ylim(0.0, 0.18)
    ax.set_xlabel(r"$N_{\rm ceiling}$")
    ax.set_ylabel(r"Cross-family $R^2$ spread")
    ax.legend(loc="upper left", frameon=False)

    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional HuggingFace path — for when the user wants to verify
# numerical equivalence between hard-coded values and the published dataset.
# ---------------------------------------------------------------------------

def _load_from_hf():
    """Load CONUS-27 values from the published HF dataset.

    Returns a list shaped like CONUS_27 (category, task, pca, gnn, slag,
    n_ceiling, spread) drawn from the canonical OOF artifacts on HF.

    This is a verification path; the figures ship with hard-coded values
    so that paper text and figures cannot drift apart.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "datasets library not installed. "
            "Install with: pip install datasets"
        ) from e

    ds = load_dataset("rudymartin/geocert", "conus27_oof_test", split="train")
    rows = []
    for r in ds:
        # Schema must match: category, task, pca32_r2, gnn_r2, slag_r2, n_ceiling, spread
        rows.append(
            (r["category"], r["task"],
             r["pca32_r2"], r["gnn_r2"], r["slag_r2"],
             r["n_ceiling"], r["spread"])
        )
    return rows


def _verify_against_hf(tolerance: float = 1e-3) -> None:
    """Compare hard-coded CONUS_27 to HF dataset row-by-row."""
    hf_rows = _load_from_hf()
    if len(hf_rows) != len(CONUS_27):
        print(f"row count differs: hf={len(hf_rows)} local={len(CONUS_27)}")
        return
    by_task_hf = {r[1]: r for r in hf_rows}
    diffs = 0
    for local in CONUS_27:
        task = local[1]
        if task not in by_task_hf:
            print(f"task missing on HF: {task}")
            diffs += 1
            continue
        hf = by_task_hf[task]
        for col_idx, col_name in enumerate(
            ["pca", "gnn", "slag", "n_ceiling", "spread"], start=2
        ):
            if abs(local[col_idx] - hf[col_idx]) > tolerance:
                print(
                    f"{task}.{col_name}: local={local[col_idx]:.4f} "
                    f"hf={hf[col_idx]:.4f}"
                )
                diffs += 1
    print(f"verification: {diffs} discrepancies (tolerance {tolerance})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", default="figures",
        help="Output directory for PDF figures.",
    )
    p.add_argument(
        "--hf", action="store_true",
        help="Verify hard-coded values against HF dataset before plotting.",
    )
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.hf:
        _verify_against_hf()

    figure_1_rank_scatter(os.path.join(args.out, "fig1_rank_scatter.pdf"))
    figure_4_conus27_heatmap(os.path.join(args.out, "fig4_conus27_heatmap.pdf"))
    figure_5_spread_vs_nceil(os.path.join(args.out, "fig5_spread_vs_nceil.pdf"))

    print(f"wrote 3 figures to {args.out}/")


if __name__ == "__main__":
    main()
