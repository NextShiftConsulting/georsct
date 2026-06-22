"""Color-ramp and classification helpers for data-driven spatial symbology.

Zero-dependency (pure stdlib) utilities for building graduated and
categorized color stops from numeric data. Useful for:
  - SageMaker notebook certificate choropleth maps
  - Paper figure color scales (consistent with interactive viewers)
  - Any data-driven color mapping without matplotlib

Adapted from GeoLibre's color_ramp.py (MIT license, opengeos/GeoLibre).
Ramp names match common GIS conventions (viridis, spectral, rdylgn, etc.).
"""

from __future__ import annotations

import math
from typing import Any

# Named color ramps — anchor hex colors for interpolation.
COLOR_RAMPS: dict[str, list[str]] = {
    "viridis": ["#440154", "#31688e", "#35b779", "#fde725"],
    "plasma": ["#0d0887", "#9c179e", "#ed7953", "#f0f921"],
    "inferno": ["#000004", "#781c6d", "#ed6925", "#fcffa4"],
    "magma": ["#000004", "#721f81", "#f1605d", "#fcfdbf"],
    "cividis": ["#00204d", "#575d6d", "#a59c74", "#ffea46"],
    "turbo": ["#30123b", "#4777ef", "#1ccfd0", "#b9e642", "#fb8022", "#7a0403"],
    "spectral": ["#9e0142", "#f46d43", "#ffffbf", "#66c2a5", "#5e4fa2"],
    "blues": ["#eff6ff", "#93c5fd", "#2563eb", "#1e3a8a"],
    "greens": ["#f0fdf4", "#86efac", "#16a34a", "#14532d"],
    "oranges": ["#fff7ed", "#fdba74", "#f97316", "#7c2d12"],
    "reds": ["#fff5f0", "#fcae91", "#fb6a4a", "#cb181d", "#67000d"],
    "purples": ["#fcfbfd", "#bcbddc", "#807dba", "#54278f", "#3f007d"],
    "terrain": ["#333399", "#21bcb3", "#79d05a", "#e8e85a", "#a87b54", "#ffffff"],
    "rdylgn": ["#a50026", "#f46d43", "#ffffbf", "#66bd63", "#006837"],
    "rdylbu": ["#a50026", "#f46d43", "#ffffbf", "#74add1", "#313695"],
    "rdbu": ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"],
    "coolwarm": ["#3b4cc0", "#7b9ff9", "#dddcdc", "#f49a7b", "#b40426"],
    "greys": ["#ffffff", "#bdbdbd", "#636363", "#000000"],
}

_DEFAULT_RAMP = "viridis"


def get_color_ramp(name: str) -> list[str]:
    """Return a ramp's anchor colors, falling back to viridis."""
    return list(COLOR_RAMPS.get(name, COLOR_RAMPS[_DEFAULT_RAMP]))


def interpolate_hex(start: str, end: str, ratio: float) -> str:
    """Linearly interpolate between two #rrggbb colors."""
    def parse(v: str) -> tuple[int, int, int]:
        n = int(v.lstrip("#"), 16)
        return (n >> 16) & 255, (n >> 8) & 255, n & 255

    sr, sg, sb = parse(start)
    er, eg, eb = parse(end)
    return "#" + "".join(
        f"{round(s + (e - s) * ratio):02x}"
        for s, e in [(sr, er), (sg, eg), (sb, eb)]
    )


def sample_ramp(name: str, count: int) -> list[str]:
    """Sample count evenly spaced colors from a named ramp."""
    colors = get_color_ramp(name)
    if count <= 1:
        return [colors[-1]]
    result = []
    for i in range(count):
        scaled = (i / (count - 1)) * (len(colors) - 1)
        lo = math.floor(scaled)
        hi = min(len(colors) - 1, math.ceil(scaled))
        result.append(interpolate_hex(colors[lo], colors[hi], scaled - lo))
    return result


def equal_interval_breaks(vmin: float, vmax: float, count: int) -> list[float]:
    """Build count evenly spaced breaks across [vmin, vmax]."""
    if count <= 1:
        return [vmin]
    return [vmin + (vmax - vmin) * i / (count - 1) for i in range(count)]


def quantile_breaks(values: list[float], count: int) -> list[float]:
    """Build count quantile breaks from a numeric sample."""
    if count <= 0 or not values:
        return []
    sv = sorted(values)
    breaks = []
    for i in range(count):
        pos = (i / (count - 1)) * (len(sv) - 1) if count > 1 else 0
        lo = math.floor(pos)
        hi = min(len(sv) - 1, math.ceil(pos))
        breaks.append(sv[lo] + (sv[hi] - sv[lo]) * (pos - lo))
    return breaks


def graduated_stops(
    values: list[Any],
    *,
    class_count: int = 5,
    color_ramp: str = "viridis",
    scheme: str = "equal-interval",
) -> list[dict[str, Any]]:
    """Build graduated color stops from numeric values.

    Args:
        values: Raw column values (non-numeric entries are skipped).
        class_count: Number of classes (clamped to 2-12).
        color_ramp: A ramp name from COLOR_RAMPS.
        scheme: "equal-interval" or "quantile".

    Returns:
        List of {"value": float, "color": str} stop dicts.
    """
    count = min(12, max(2, int(class_count)))
    numeric = []
    for v in values:
        try:
            n = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(n):
            numeric.append(n)

    colors = sample_ramp(color_ramp, count)
    if not numeric:
        return [{"value": i, "color": c} for i, c in enumerate(colors)]

    vmin, vmax = min(numeric), max(numeric)
    if vmin == vmax:
        return [{"value": vmin, "color": colors[-1]}]

    breaks = (
        quantile_breaks(numeric, count)
        if scheme == "quantile"
        else equal_interval_breaks(vmin, vmax, count)
    )
    return [
        {"value": float(f"{v:.8g}"), "color": colors[i]}
        for i, v in enumerate(breaks)
    ]
