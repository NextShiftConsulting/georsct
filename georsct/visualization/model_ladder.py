"""Render model-ladder metric trajectories (R0 -> R1 -> R2).

Two-panel figure showing per-cell metric evolution across representation
levels, split by target type:
  Left:  classification targets (ROC-AUC)
  Right: regression targets (R^2, spatial-blocked)

Sized for two-column TeX figure* at textwidth (~7in).

Usage:
    from georsct.visualization.model_ladder import render_model_ladder
    render_model_ladder(cells, Path("fig_model_ladder.pdf"))
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from georsct.visualization.palette import (
    PAPER_RCPARAMS,
    SCENARIO_COLORS,
    SCENARIO_LABELS,
    TARGET_LABELS,
    TARGET_MARKERS,
)


def _plot_cells(
    ax: plt.Axes,
    cells: list[dict],
    title: str,
    ylabel: str,
    show_target_in_legend: bool = True,
) -> None:
    """Plot metric trajectories for a list of cells on one axes."""
    x = np.arange(3)
    x_labels = ["R0\n(static)", "R1\n(+hydro)", "R2\n(+temporal)"]

    for c in cells:
        raw = [c["metric_r0"], c["metric_r1"], c["metric_r2"]]
        valid_x = [x[i] for i, v in enumerate(raw) if v is not None]
        valid_v = [v for v in raw if v is not None]
        if not valid_v:
            continue

        sc = c["scenario"]
        tg = c["target"]
        color = SCENARIO_COLORS.get(sc, "#333333")
        marker = TARGET_MARKERS.get(tg, "o")
        if show_target_in_legend:
            label = f"{SCENARIO_LABELS.get(sc, sc)} ({TARGET_LABELS.get(tg, tg)})"
        else:
            label = SCENARIO_LABELS.get(sc, sc)

        ax.plot(
            valid_x, valid_v,
            color=color, lw=1.8, marker=marker, markersize=5,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.4, label=label, zorder=3,
        )

    ax.set_title(title, fontsize=8.5, pad=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.15)
    ax.legend(fontsize=5.5, loc="lower right", frameon=False)


def render_model_ladder(
    cells: list[dict],
    out_path: str | Path,
    also_png: bool = True,
) -> Path:
    """Render the two-panel model-ladder figure.

    Args:
        cells: List of per-cell dicts, each with keys: scenario, target,
            target_type ("observation"|"claims"), metric_r0, metric_r1,
            metric_r2. Typically from per_target_h2_breakdown.json
            ``per_cell_table``.
        out_path: PDF output path.
        also_png: Also write a same-stem .png at 200 dpi.

    Returns:
        Path to the written PDF.
    """
    plt.rcParams.update(PAPER_RCPARAMS)

    obs = [c for c in cells if c.get("target_type") == "observation"]
    reg = [c for c in cells if c.get("target_type") == "claims"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    _plot_cells(ax1, obs, "Classification (AUC)", "ROC-AUC",
                show_target_in_legend=True)
    _plot_cells(ax2, reg, "Regression ($R^2$)", "$R^2$ (spatial-blocked)",
                show_target_in_legend=False)
    ax2.axhline(0, color="#999999", ls=":", lw=0.7, zorder=1)

    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path)
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out_path
