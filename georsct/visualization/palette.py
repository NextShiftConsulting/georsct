"""Shared palette and label maps for GeoRSCT paper figures.

Imports theme and structural colors from yrsn-analysis (canonical source)
when available. Scenario-specific colors and label maps are georsct domain
knowledge and live here.

Canonical source: yrsn-analysis/src/yrsn_analysis/themes/
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Import canonical theme from yrsn-analysis, fallback to local copies
# ---------------------------------------------------------------------------

try:
    from yrsn_analysis.themes.colors import (
        COLORBLIND_SAFE_PALETTE,
        DecisionColors,
        QualityColors,
        RSNColors,
    )
    from yrsn_analysis.themes.publication import PAPER_THEME, PublicationTheme
except ImportError:
    # Minimal fallback — enough for paper figures to render without
    # yrsn-analysis installed.  Keep in sync with canonical source.

    COLORBLIND_SAFE_PALETTE = [
        "#4A90E2", "#50C878", "#E67E22", "#9B59B6",
        "#E74C3C", "#F1C40F", "#34495E", "#95A5A6",
    ]

    class RSNColors:
        R = "#2ECC71"
        S = "#95A5A6"
        N = "#E74C3C"

    class DecisionColors:
        EXECUTE = "#4CAF50"
        REPAIR = "#FFC107"
        RE_ENCODE = "#FF9800"
        BLOCK = "#9E9E9E"
        REJECT = "#F44336"

    class QualityColors:
        HIGH = "#2ECC71"
        MODERATE = "#F39C12"
        LOW = "#E74C3C"

    @dataclass(frozen=True)
    class PublicationTheme:
        font_family: str = "Arial, sans-serif"
        font_size: int = 12
        title_font_size: int = 14
        tick_font_size: int = 10
        dpi: int = 300

    PAPER_THEME = PublicationTheme()


# ---------------------------------------------------------------------------
# GeoRSCT scenario palette (Okabe-Ito, colorblind-safe)
# These are georsct domain — scenario-to-color mapping for the flood paper.
# ---------------------------------------------------------------------------

SCENARIO_COLORS: dict[str, str] = {
    "houston": "#0072B2",
    "southwest_florida": "#D55E00",
    "nyc": "#009E73",
    "riverside_coachella": "#CC79A7",
    "new_orleans": "#E69F00",
}

SCENARIO_LABELS: dict[str, str] = {
    "houston": "Houston",
    "southwest_florida": "SW Florida",
    "nyc": "NYC",
    "riverside_coachella": "Riverside",
    "new_orleans": "New Orleans",
}

TARGET_LABELS: dict[str, str] = {
    "obs_nfip_event_claims": "NFIP claims",
    "obs_has_311": "311 reports",
    "obs_has_hwm": "HWM presence",
}

TARGET_MARKERS: dict[str, str] = {
    "obs_nfip_event_claims": "o",
    "obs_has_311": "s",
    "obs_has_hwm": "D",
}

LEVEL_ORDER: list[str] = ["r0", "r1", "r2"]
LEVEL_LABELS: list[str] = ["R0\n(static)", "R1\n(+hydro)", "R2\n(+temporal)"]

# Variance-stack status colors (Okabe-Ito)
STATUS_COLORS: dict[str, str] = {
    "PASS": "#009E73",
    "WARN": "#E69F00",
    "FAIL": "#D55E00",
}

VERDICT_COLORS: dict[str, str] = {
    "align": "#009E73",
    "artifact": "#E69F00",
    "noop": "#9A9A9A",
}

# Matplotlib rcParams for two-column TeX paper figures
PAPER_RCPARAMS: dict[str, object] = {
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.dpi": PAPER_THEME.dpi,
}
