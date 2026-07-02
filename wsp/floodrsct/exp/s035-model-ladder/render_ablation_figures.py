#!/usr/bin/env python3
"""render_ablation_figures.py -- Publication-quality ablation charts for s035.

Reads ablation results from S3 and generates:
  - Figure A: Grouped bar chart per target (R0 -> R1 variants -> R2 variants)
  - Figure B: Feature-group contribution heatmap
  - Figure C: Waterfall chart showing incremental feature-group contribution
  - Table 1: Full ablation results (LaTeX)

Usage:
    python render_ablation_figures.py          # save to figures/
    python render_ablation_figures.py --show   # also display interactively
"""

import argparse
import json
import statistics
from pathlib import Path

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BUCKET = "swarm-floodrsct-data"
PREFIX = "results/s035"
OUT_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida",
]
SCENARIO_LABELS = {
    "houston": "Houston",
    "new_orleans": "New Orleans",
    "nyc": "NYC",
    "riverside_coachella": "Riverside",
    "southwest_florida": "SW Florida",
}

TARGETS = [
    ("obs_has_311", "classification", "roc_auc", "311 Reports (ROC-AUC)"),
    ("obs_has_hwm", "classification", "roc_auc", "High Water Marks (ROC-AUC)"),
    ("obs_nfip_event_claims", "regression", "r2", "NFIP Claims (R$^2$)"),
]

# Ordered variants for the bar chart
VARIANTS = {
    "R0":                "r0_{s}.json",
    "R1-full":           "r1_hydrology_{s}.json",
    "R1-no-wlag":        "r1_no_wlag_{s}.json",
    "R1-wlag-only":      "r1_wlag_only_{s}.json",
    "R1-no-target-lag":  "r1_no_target_lag_{s}.json",
    "R2-full":           "r2_{s}.json",
    "R2-no-storm":       "r2_no_storm_track_{s}.json",
    "R2-no-rain":        "r2_no_rainfall_{s}.json",
    "R2-temp-only":      "r2_temporal_only_{s}.json",
}

# Color scheme: R0=grey, R1=blue family, R2=red family
COLORS = {
    "R0":               "#8c8c8c",
    "R1-full":          "#1f77b4",
    "R1-no-wlag":       "#6baed6",
    "R1-wlag-only":     "#9ecae1",
    "R1-no-target-lag": "#c6dbef",
    "R2-full":          "#d62728",
    "R2-no-storm":      "#e6550d",
    "R2-no-rain":       "#fdae6b",
    "R2-temp-only":     "#fdd0a2",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(s3_client):
    """Load all variant x scenario results from S3."""
    data = {}
    for variant, pattern in VARIANTS.items():
        data[variant] = {}
        for s in SCENARIOS:
            key = f"{PREFIX}/{pattern.format(s=s)}"
            try:
                resp = s3_client.get_object(Bucket=BUCKET, Key=key)
                data[variant][s] = json.loads(resp["Body"].read())
            except Exception:
                data[variant][s] = None
    return data


def extract_primary(result_data, target, metric_name):
    """Extract mean primary metric across spatial_blocked folds, histgbdt solver."""
    if result_data is None:
        return None
    runs = result_data.get("runs", [])
    vals = []
    for r in runs:
        if (r["target"] != target or r["solver"] != "histgbdt"
                or r["split"] != "spatial_blocked"):
            continue
        v = r["metrics"].get(metric_name)
        if v is None:
            continue
        if isinstance(v, dict):
            if v.get("status") != "MEASURED" or v.get("value") is None:
                continue
            vals.append(float(v["value"]))
        else:
            vals.append(float(v))
    return statistics.mean(vals) if vals else None


def build_matrix(data):
    """Build {variant: {(scenario, target): value}} matrix."""
    matrix = {}
    for variant in VARIANTS:
        matrix[variant] = {}
        for s in SCENARIOS:
            for target, task, metric, label in TARGETS:
                v = extract_primary(data[variant][s], target, metric)
                matrix[variant][(s, target)] = v
    return matrix


# ---------------------------------------------------------------------------
# Figure A: Grouped bar chart per target
# ---------------------------------------------------------------------------

def render_grouped_bars(matrix, show=False):
    """One subplot per target, bars grouped by scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    fig.subplots_adjust(wspace=0.28, bottom=0.22)

    variant_names = list(VARIANTS.keys())
    n_variants = len(variant_names)

    for ax_idx, (target, task, metric, title) in enumerate(TARGETS):
        ax = axes[ax_idx]

        # Collect scenarios that have data for this target
        active_scenarios = []
        for s in SCENARIOS:
            has_data = any(
                matrix[v].get((s, target)) is not None for v in variant_names
            )
            if has_data:
                active_scenarios.append(s)

        n_scenarios = len(active_scenarios)
        if n_scenarios == 0:
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        bar_width = 0.7 / n_variants
        x = np.arange(n_scenarios)

        for i, variant in enumerate(variant_names):
            vals = []
            for s in active_scenarios:
                v = matrix[variant].get((s, target))
                vals.append(v if v is not None else 0)
            offset = (i - n_variants / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, vals, bar_width,
                label=variant if ax_idx == 0 else "",
                color=COLORS[variant],
                edgecolor="white", linewidth=0.3,
            )
            # Add value labels on tall bars
            for bar, val in zip(bars, vals):
                if val > 0.05:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=5, rotation=90,
                    )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SCENARIO_LABELS[s] for s in active_scenarios],
            fontsize=9, rotation=30, ha="right",
        )
        ax.set_ylabel(metric.upper() if ax_idx == 0 else "")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        ncol=5, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        "S035 Ablation Study: Feature Sub-Group Contributions",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for fmt in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"fig_ablation_bars.{fmt}",
            dpi=300, bbox_inches="tight",
        )
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved fig_ablation_bars.pdf/png")


# ---------------------------------------------------------------------------
# Figure B: Feature-group contribution heatmap
# ---------------------------------------------------------------------------

def render_contribution_heatmap(matrix, show=False):
    """Heatmap showing delta from R0 baseline for each variant x target."""
    fig, ax = plt.subplots(figsize=(10, 5))

    variant_names = [v for v in VARIANTS if v != "R0"]

    # Average across scenarios for each variant x target
    grid = np.full((len(variant_names), len(TARGETS)), np.nan)
    for i, variant in enumerate(variant_names):
        for j, (target, task, metric, label) in enumerate(TARGETS):
            v_vals = []
            b_vals = []
            for s in SCENARIOS:
                v = matrix[variant].get((s, target))
                b = matrix["R0"].get((s, target))
                if v is not None and b is not None:
                    v_vals.append(v)
                    b_vals.append(b)
            if v_vals:
                grid[i, j] = statistics.mean(v_vals) - statistics.mean(b_vals)

    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.25)
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([t[3] for t in TARGETS], fontsize=10)
    ax.set_yticks(range(len(variant_names)))
    ax.set_yticklabels(variant_names, fontsize=9)

    # Annotate cells
    for i in range(len(variant_names)):
        for j in range(len(TARGETS)):
            val = grid[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center", fontsize=9,
                        color="gray")
            else:
                sign = "+" if val >= 0 else ""
                color = "white" if abs(val) > 0.12 else "black"
                ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    # Horizontal lines separating R1 and R2 groups
    ax.axhline(y=3.5, color="black", linewidth=1.5)

    ax.set_title(
        "Feature Sub-Group Contribution (delta from R0 baseline)",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta from R0")

    for fmt in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"fig_ablation_heatmap.{fmt}",
            dpi=300, bbox_inches="tight",
        )
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved fig_ablation_heatmap.pdf/png")


# ---------------------------------------------------------------------------
# Figure C: Waterfall -- incremental contribution of each feature group
# ---------------------------------------------------------------------------

def render_waterfall(matrix, show=False):
    """Waterfall chart: R0 -> +hydro -> +wlag -> +rainfall -> +storm-track."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.subplots_adjust(wspace=0.30, bottom=0.18)

    # Steps: R0, +hydro (R1-no-wlag - R0), +wlag (R1-full - R1-no-wlag),
    #         +rainfall (R2-no-storm - R1-full), +storm-track (R2-full - R2-no-storm)
    step_defs = [
        ("R0 Baseline",     "R0",           None),
        ("+Hydrology",      "R1-no-wlag",   "R0"),
        ("+W-Matrix",       "R1-full",      "R1-no-wlag"),
        ("+Rainfall",       "R2-no-storm",  "R1-full"),
        ("+Storm Track",    "R2-full",      "R2-no-storm"),
    ]
    step_colors = ["#8c8c8c", "#9ecae1", "#1f77b4", "#fdae6b", "#d62728"]

    for ax_idx, (target, task, metric, title) in enumerate(TARGETS):
        ax = axes[ax_idx]
        steps = []
        for label, variant, base_variant in step_defs:
            v_vals = []
            b_vals = []
            for s in SCENARIOS:
                v = matrix[variant].get((s, target))
                if base_variant is None:
                    if v is not None:
                        v_vals.append(v)
                else:
                    b = matrix[base_variant].get((s, target))
                    if v is not None and b is not None:
                        v_vals.append(v)
                        b_vals.append(b)

            if base_variant is None:
                steps.append(statistics.mean(v_vals) if v_vals else 0)
            else:
                delta = (statistics.mean(v_vals) - statistics.mean(b_vals)
                         if v_vals else 0)
                steps.append(delta)

        # Draw waterfall
        cumulative = 0
        x_pos = range(len(steps))
        for i, (val, color) in enumerate(zip(steps, step_colors)):
            if i == 0:
                ax.bar(i, val, color=color, edgecolor="white", linewidth=0.5)
                ax.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
                cumulative = val
            else:
                ax.bar(i, val, bottom=cumulative, color=color,
                       edgecolor="white", linewidth=0.5)
                mid = cumulative + val / 2
                sign = "+" if val >= 0 else ""
                ax.text(i, mid, f"{sign}{val:.3f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(val) > 0.03 else "black")
                # Connector line
                if i < len(steps) - 1:
                    ax.plot([i - 0.4, i + 0.4], [cumulative + val] * 2,
                            color="gray", linewidth=0.5, linestyle="--")
                cumulative += val

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [s[0] for s in step_defs],
            fontsize=8, rotation=35, ha="right",
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(metric.upper() if ax_idx == 0 else "")
        ax.set_ylim(0, max(1.0, cumulative + 0.1))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Incremental Feature-Group Contribution (Waterfall)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for fmt in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"fig_ablation_waterfall.{fmt}",
            dpi=300, bbox_inches="tight",
        )
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved fig_ablation_waterfall.pdf/png")


# ---------------------------------------------------------------------------
# Table 1: Full results (LaTeX)
# ---------------------------------------------------------------------------

def render_latex_table(matrix):
    """Generate LaTeX table of ablation results."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study results across representation levels "
                 r"and feature sub-groups. Primary metric: ROC-AUC for "
                 r"classification targets, $R^2$ for regression. "
                 r"Values are mean across spatial-blocked folds, HistGBDT solver. "
                 r"Best per-target highlighted in \textbf{bold}.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "r" * 3 + "}")
    lines.append(r"\toprule")
    lines.append(r"Variant & 311 Reports & High Water Marks & NFIP Claims \\")
    lines.append(r" & (ROC-AUC) & (ROC-AUC) & ($R^2$) \\")
    lines.append(r"\midrule")

    # Compute means across scenarios per variant x target
    means = {}
    for variant in VARIANTS:
        means[variant] = {}
        for target, task, metric, label in TARGETS:
            vals = []
            for s in SCENARIOS:
                v = matrix[variant].get((s, target))
                if v is not None:
                    vals.append(v)
            means[variant][target] = statistics.mean(vals) if vals else None

    # Find best per target
    best = {}
    for target, _, _, _ in TARGETS:
        best_val = -999
        for variant in VARIANTS:
            v = means[variant].get(target)
            if v is not None and v > best_val:
                best_val = v
                best[target] = variant

    prev_level = ""
    for variant in VARIANTS:
        level = variant.split("-")[0]
        if level != prev_level and prev_level:
            lines.append(r"\addlinespace")
        prev_level = level

        cells = []
        for target, _, _, _ in TARGETS:
            v = means[variant].get(target)
            if v is None:
                cells.append("--")
            elif best.get(target) == variant:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        lines.append(f"{variant} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    out_path = RESULTS_DIR / "table_ablation.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    print(f"  Saved {out_path}")
    return tex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results from S3...")
    s3 = boto3.Session(profile_name="nsc-swarm").client("s3", region_name="us-east-1")
    data = load_all_results(s3)
    matrix = build_matrix(data)

    print("Rendering figures...")
    render_grouped_bars(matrix, show=args.show)
    render_contribution_heatmap(matrix, show=args.show)
    render_waterfall(matrix, show=args.show)

    print("Rendering tables...")
    render_latex_table(matrix)

    print("Done.")


if __name__ == "__main__":
    main()
