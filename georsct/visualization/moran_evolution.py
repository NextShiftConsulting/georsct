"""Render Moran's I evolution across representation levels (R0, R1, R2).

Two-panel figure showing global Moran's I trajectories:
  Left:  regression targets (NFIP claims)
  Right: classification targets (311 reports, HWM presence)

Filled markers indicate statistical significance (Geary p < 0.05).
Sized for two-column TeX figure* at textwidth (~7in).

Accepts either pre-computed DataFrames (CSV workflow) or domain
TurbulenceResult objects (hex arch workflow):

    # From CSVs
    render_moran_evolution(lisa_df, geary_df, out_path)

    # From domain objects
    results = {("houston", "obs_nfip_event_claims", "r0"): turbulence_result, ...}
    lisa, geary = turbulence_results_to_frames(results)
    render_moran_evolution(lisa, geary, out_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from georsct.visualization.palette import (
    LEVEL_LABELS,
    LEVEL_ORDER,
    PAPER_RCPARAMS,
    SCENARIO_COLORS,
    SCENARIO_LABELS,
    TARGET_LABELS,
)

if TYPE_CHECKING:
    from georsct.domain.turbulence import TurbulenceResult


def turbulence_results_to_frames(
    results: dict[tuple[str, str, str], TurbulenceResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert domain TurbulenceResult objects to renderer-ready DataFrames.

    Args:
        results: Mapping of (scenario, target, level) -> TurbulenceResult.
            Typically produced by score_turbulence() per cell per level.

    Returns:
        (lisa_df, geary_df) matching the CSV schemas that
        render_moran_evolution() expects.
    """
    lisa_rows = []
    geary_rows = []
    for (scenario, target, level), tr in results.items():
        # Back-compute n_total from fraction_significant.
        # fraction_significant = n_sig / n_total, and
        # n_sig = n_hotspots + n_coldspots + n_outliers.
        n_sig = tr.n_hotspots + tr.n_coldspots + tr.n_outliers
        if tr.fraction_significant > 0:
            n_total = round(n_sig / tr.fraction_significant)
        else:
            n_total = n_sig  # all non-significant; use sig count as floor
        n_total = max(n_total, 1)

        lisa_rows.append({
            "scenario": scenario,
            "target": target,
            "level": level,
            "global_moran_I": tr.moran_i,
            "frac_HH": tr.n_hotspots / n_total,
            "frac_LL": tr.n_coldspots / n_total,
            "frac_outlier": tr.n_outliers / n_total,
            "frac_significant": tr.fraction_significant,
            "n_zctas": n_total,
        })
        interp = "positive_autocorrelation" if tr.geary_c < 1.0 else "negative_autocorrelation"
        geary_rows.append({
            "scenario": scenario,
            "target": target,
            "level": level,
            "geary_C": tr.geary_c,
            "geary_p": tr.geary_p,
            "interpretation": interp,
        })
    return pd.DataFrame(lisa_rows), pd.DataFrame(geary_rows)


def _draw_panel(
    ax: plt.Axes,
    lisa: pd.DataFrame,
    geary: pd.DataFrame,
    targets: list[str],
    title: str,
    show_ylabel: bool = True,
) -> None:
    """Draw one panel (regression or classification)."""
    x = np.arange(len(LEVEL_ORDER))

    for scenario in lisa["scenario"].unique():
        color = SCENARIO_COLORS.get(scenario, "#333333")
        label_base = SCENARIO_LABELS.get(scenario, scenario)

        for target in targets:
            sub = lisa[(lisa["scenario"] == scenario) & (lisa["target"] == target)]
            if sub.empty:
                continue

            sub = sub.set_index("level").reindex(LEVEL_ORDER)
            vals = sub["global_moran_I"].values

            gsub = geary[
                (geary["scenario"] == scenario) & (geary["target"] == target)
            ].set_index("level").reindex(LEVEL_ORDER)
            geary_p = gsub["geary_p"].values if not gsub.empty else [1.0] * 3

            target_label = TARGET_LABELS.get(target, target)
            if target == "obs_nfip_event_claims":
                ls, marker = "-", "o"
            elif target == "obs_has_311":
                ls, marker = "--", "s"
            else:
                ls, marker = ":", "D"

            ax.plot(
                x, vals, color=color, ls=ls, lw=1.8, marker=marker,
                markersize=5, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.4,
                label=f"{label_base} ({target_label})", zorder=3,
            )

            for i in range(len(x)):
                if np.isfinite(vals[i]) and geary_p[i] < 0.05:
                    ax.scatter(
                        x[i], vals[i], s=25, color=color,
                        edgecolors=color, linewidths=1.4,
                        marker=marker, zorder=4,
                    )

    ax.axhline(0, color="#999999", ls=":", lw=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS, fontsize=7)
    ax.set_title(title, fontsize=8.5, pad=6)
    ax.tick_params(axis="y", labelsize=7)
    if show_ylabel:
        ax.set_ylabel("Global Moran's $I$", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.12)


def render_moran_evolution(
    lisa: pd.DataFrame,
    geary: pd.DataFrame,
    out_path: str | Path,
    also_png: bool = True,
) -> Path:
    """Render two-panel Moran's I evolution figure.

    Args:
        lisa: DataFrame with columns: scenario, target, level,
            global_moran_I, frac_HH, frac_LL, frac_outlier,
            frac_significant, n_zctas.
        geary: DataFrame with columns: scenario, target, level,
            geary_C, geary_p, interpretation.
        out_path: PDF output path.
        also_png: Also write a same-stem .png at 200 dpi.

    Returns:
        Path to the written PDF.
    """
    plt.rcParams.update(PAPER_RCPARAMS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    _draw_panel(ax1, lisa, geary, ["obs_nfip_event_claims"],
                "Regression (NFIP claims)", show_ylabel=True)
    _draw_panel(ax2, lisa, geary, ["obs_has_311", "obs_has_hwm"],
                "Classification (311 / HWM)", show_ylabel=False)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    all_handles.append(
        Line2D([0], [0], marker="o", color="white",
               markerfacecolor="#333333", markeredgecolor="#333333",
               markeredgewidth=1.4, markersize=5, lw=0)
    )
    all_labels.append("filled = sig. ($p < 0.05$)")

    fig.legend(
        all_handles, all_labels, loc="lower center", ncol=4,
        fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.18),
    )
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path)
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return out_path
