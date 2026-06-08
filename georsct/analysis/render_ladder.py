"""Render the variance-control stack ladder as a publication small-multiple panel.

The decompose_stack() ladder IS the paper's central illustration. This module
turns one or more per-cell ladders into a single figure that makes the thesis
visible:

    A delta_rmse only counts as a real improvement when the residual margin gap
    dropped alongside it. If the gap held or rose, the RMSE move was a
    regularization / extreme-weight artifact -- and the verdict says so.

Each panel encodes:
  * RMSE trajectory across the 5 ladder steps (markers, status-colored).
  * residual_margin_gap as faint bars on a secondary axis, with the
    max_residual_gap threshold drawn as a dashed line.
  * The connecting RMSE segments colored by VERDICT:
      green  = alignment improved this step (gap dropped)        -> credible move
      amber  = estimate moved but gap did not drop               -> artifact / reg.
      grey   = no-op (e.g. the coverage gate does not reweight)
  * Per-step marker color = stack_status (PASS / WARN_* / FAIL_*).
  * Optional dashed horizontal line = true deployment RMSE (DEMO ONLY; on real
    data the deployment risk is unknown -- that is what TWCV estimates).

Usage (production):
    from georsct.analysis.variance_stack import decompose_stack
    ladders = {label: decompose_stack(val_b, tgt_b, bin_cols, y_true, y_pred)
               for label, ... in cells}
    render_ladder_panel(ladders, "fig_stack_ladder.pdf")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Palette (Okabe-Ito, colorblind-safe)
# ---------------------------------------------------------------------------
STATUS_COLORS = {
    "PASS": "#009E73",
    "WARN": "#E69F00",
    "FAIL": "#D55E00",
}
VERDICT_COLORS = {
    "align": "#009E73",   # gap dropped -> credible
    "artifact": "#E69F00",  # moved without aligning
    "noop": "#9A9A9A",    # no estimate change
}
GAP_BAR = "#BBBBBB"
STEP_LABELS = {
    "0_unweighted": "unwtd",
    "1_coverage_gate": "+cov",
    "2_marginal_product": "+prod",
    "3_marginal_raking": "+rake",
    "4_raking_shrunk": "+shrnk",
}
_EPS = 1e-4


def _status_color(status: str) -> str:
    if isinstance(status, str):
        if status.startswith("FAIL"):
            return STATUS_COLORS["FAIL"]
        if status.startswith("WARN"):
            return STATUS_COLORS["WARN"]
        if status.startswith("PASS"):
            return STATUS_COLORS["PASS"]
    return VERDICT_COLORS["noop"]


def _segment_verdict(config: str, d_rmse: float, d_gap: float) -> str:
    """Color rule for the segment LEADING INTO this step.

    Keyed on alignment (gap), not on the sign of delta_rmse: an RMSE move is
    only credible if the weights actually matched the target margins better.
    """
    if config == "1_coverage_gate":
        return "noop"
    if not np.isfinite(d_gap):
        return "noop"
    if d_gap < -_EPS:
        return "align"
    if np.isfinite(d_rmse) and abs(d_rmse) > _EPS:
        return "artifact"
    return "noop"


def _recommended_point(ladder: pd.DataFrame) -> tuple[int, str, str]:
    """Pick the operating point a practitioner should report.

    Restricts to gated steps (layer 0 on), then selects the step with
    minimum residual margin gap among admissible steps (PASS first,
    then WARN if no PASS exists). If all steps FAIL, returns the first
    FAIL (the domain cannot be certified at all).

    Returns (row_index, status, short_step_label).
    """
    lad = ladder.reset_index(drop=True)
    # Gated steps have coverage gate active (all except 0_unweighted).
    # Support both the raw certificate field and averaged DataFrames
    # where the boolean was dropped during fold averaging.
    if "layer0_coverage_gate" in lad.columns:
        cand = [i for i in range(len(lad)) if bool(lad.loc[i, "layer0_coverage_gate"])]
    else:
        cand = [i for i in range(len(lad)) if lad.loc[i, "config"] != "0_unweighted"]
    if not cand:
        cand = list(range(len(lad)))

    # Prefer PASS steps; pick the one with minimum margin gap
    passes = [i for i in cand if str(lad.loc[i, "stack_status"]).startswith("PASS")]
    if passes:
        idx = min(passes, key=lambda i: lad.loc[i, "residual_margin_gap"])
    else:
        # Among non-FAIL (WARN) steps, pick minimum margin gap
        warns = [i for i in cand if not str(lad.loc[i, "stack_status"]).startswith("FAIL")]
        if warns:
            idx = min(warns, key=lambda i: lad.loc[i, "residual_margin_gap"])
        else:
            # All FAIL: return first FAIL (fail closed)
            idx = cand[0]

    cfg = lad.loc[idx, "config"]
    return idx, str(lad.loc[idx, "stack_status"]), STEP_LABELS.get(cfg, cfg)


def _draw_one_panel(
    ax: plt.Axes,
    label: str,
    lad: pd.DataFrame,
    truth_val: float | None = None,
    show_ylabel: bool = False,
) -> None:
    """Draw a single ladder panel on *ax*."""
    lad = lad.reset_index(drop=True)
    x = np.arange(len(lad))
    rmse = lad["rmse"].to_numpy(dtype=float)
    gap = lad["residual_margin_gap"].to_numpy(dtype=float)
    status = lad["stack_status"].tolist()
    cfgs = lad["config"].tolist()
    d_rmse = lad["delta_rmse"].to_numpy(dtype=float)
    d_gap = lad["delta_margin_gap"].to_numpy(dtype=float)

    # secondary axis: margin gap as faint bars (behind)
    ax2 = ax.twinx()
    ax2.bar(x, gap, width=0.55, color=GAP_BAR, alpha=0.45, zorder=1)
    ax2.axhline(0.05, color=GAP_BAR, ls=":", lw=0.8, zorder=1)
    gmax = np.nanmax(gap) if np.isfinite(np.nanmax(gap)) else 0.1
    ax2.set_ylim(0, max(gmax * 1.25, 0.06))
    ax2.tick_params(axis="y", labelsize=6, colors="#777777", length=2)
    ax2.set_ylabel("margin gap", fontsize=6.5, color="#777777")

    # primary axis: rmse trajectory with verdict-colored segments
    for i in range(1, len(x)):
        v = _segment_verdict(cfgs[i], d_rmse[i], d_gap[i])
        ax.plot(
            [x[i - 1], x[i]], [rmse[i - 1], rmse[i]],
            color=VERDICT_COLORS[v], lw=2.2, solid_capstyle="round", zorder=3,
        )
    for i in range(len(x)):
        ax.scatter(
            x[i], rmse[i], s=40, zorder=4,
            color="white", edgecolors=_status_color(status[i]), linewidths=1.6,
        )
    rec_idx, rec_status, rec_label = _recommended_point(lad)
    ax.scatter(
        x[rec_idx], rmse[rec_idx], s=40, zorder=5,
        color=_status_color(rec_status), edgecolors=_status_color(rec_status),
        linewidths=1.6,
    )

    # optional true-deployment reference (demo only)
    if truth_val is not None:
        ax.axhline(truth_val, color="#444444", ls="--", lw=1.0, zorder=2)
        ax.text(
            -0.45, truth_val, "true ", va="center", ha="right",
            fontsize=6, color="#444444",
        )

    # cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels([STEP_LABELS.get(c, c) for c in cfgs],
                       rotation=25, ha="right", fontsize=6)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    ax.margins(x=0.10)
    ax.set_title(label, fontsize=8, pad=10)
    ax.text(0.5, 1.005, f"{rec_status} @ {rec_label}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5,
            color=_status_color(rec_status), fontweight="bold")
    for s in ("top",):
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)
    if show_ylabel:
        ax.set_ylabel("estimated RMSE", fontsize=7.5)


def render_ladder_panel(
    ladders: dict[str, pd.DataFrame],
    out_path: str | Path,
    truth: dict[str, float] | None = None,
    suptitle: str | None = None,
    also_png: bool = True,
    ncols: int | None = None,
) -> Path:
    """Render a small-multiple panel, one cell per ladder.

    Automatically wraps to multiple rows when there are more than 6 panels.

    Args:
        ladders: {cell_label: decompose_stack output DataFrame}.
        out_path: PDF output path.
        truth: optional {cell_label: true_deployment_rmse} for a reference line.
        suptitle: optional figure title.
        also_png: also write a same-stem .png for previewing.
        ncols: panels per row (default: min(n, 6)).

    Returns:
        Path to the written PDF.
    """
    truth = truth or {}
    labels = list(ladders.keys())
    n = len(labels)
    if ncols is None:
        ncols = min(n, 6)
    nrows = (n + ncols - 1) // ncols

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.bbox": "tight",
    })

    cell_w = min(7.2 / ncols, 1.45)
    fig_w = cell_w * ncols + 0.4
    fig_h = 2.6 * nrows + 0.3
    fig, all_axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), squeeze=False,
    )

    for idx, label in enumerate(labels):
        r, c = divmod(idx, ncols)
        ax = all_axes[r][c]
        _draw_one_panel(
            ax, label, ladders[label],
            truth_val=truth.get(label),
            show_ylabel=(c == 0),
        )

    # hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        all_axes[r][c].set_visible(False)

    # shared legend
    handles = [
        Line2D([0], [0], color=VERDICT_COLORS["align"], lw=2.4,
               label="gap dropped (credible)"),
        Line2D([0], [0], color=VERDICT_COLORS["artifact"], lw=2.4,
               label="moved, gap held/rose (artifact)"),
        Line2D([0], [0], color=VERDICT_COLORS["noop"], lw=2.4,
               label="no reweight (gate)"),
        Line2D([0], [0], marker="o", color="white",
               markeredgecolor=STATUS_COLORS["PASS"],
               markeredgewidth=1.8, markersize=7, lw=0, label="PASS"),
        Line2D([0], [0], marker="o", color="white",
               markeredgecolor=STATUS_COLORS["WARN"],
               markeredgewidth=1.8, markersize=7, lw=0, label="WARN_*"),
        Line2D([0], [0], marker="o", color="white",
               markeredgecolor=STATUS_COLORS["FAIL"],
               markeredgewidth=1.8, markersize=7, lw=0, label="FAIL_*"),
        Patch(facecolor=GAP_BAR, alpha=0.45,
              label="residual margin gap (right axis)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, -0.06 / nrows))
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.02)
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path)
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out_path
