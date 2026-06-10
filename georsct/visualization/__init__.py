"""Visualization modules for GeoRSCT paper figures.

All renderers accept DataFrames or dicts and return the output Path.
They do not load data from disk -- callers handle I/O.

Theme and structural colors imported from yrsn-analysis (canonical source)
via palette.py, with graceful fallback if yrsn-analysis is not installed.
Scenario-specific colors (Okabe-Ito) are georsct domain knowledge.

Install: pip install rsct[viz]

Modules:
    palette          -- colors, labels, rcParams (imports yrsn-analysis theme)
    model_ladder     -- R0->R1->R2 metric trajectories per cell
    moran_evolution  -- Global Moran's I across representation levels
    render_ladder    -- variance-stack ladder (re-exported from analysis)
"""

from georsct.analysis.render_ladder import render_ladder_panel
from georsct.visualization.model_ladder import render_model_ladder
from georsct.visualization.moran_evolution import (
    render_moran_evolution,
    turbulence_results_to_frames,
)

__all__ = [
    "render_ladder_panel",
    "render_model_ladder",
    "render_moran_evolution",
    "turbulence_results_to_frames",
]
