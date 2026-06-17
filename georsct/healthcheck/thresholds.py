"""Threshold presets for healthcheck diagnostics.

Resolved via ThresholdSource port (dependency inversion).
Default: StaticThresholdSource (no external dependencies).
Override: set_threshold_source() to plug in controlplane or API adapters.
"""

from __future__ import annotations

from georsct.ports.threshold_source import ThresholdPreset, ThresholdSource
from georsct.ports.static_thresholds import StaticThresholdSource

# Re-export for consumers
__all__ = [
    "ThresholdPreset",
    "GEOSPATIAL_CONUS27",
    "UNIVERSAL",
    "STRICT",
    "PRESETS",
    "resolve_preset",
    "set_threshold_source",
]

# ---------------------------------------------------------------------------
# Pluggable source (default: static snapshots)
# ---------------------------------------------------------------------------

_source: ThresholdSource = StaticThresholdSource()


def set_threshold_source(source: ThresholdSource) -> None:
    """Override the threshold source (e.g., controlplane or API adapter).

    Call once at application startup before any preset resolution.
    """
    global _source, GEOSPATIAL_CONUS27, UNIVERSAL, STRICT, PRESETS
    _source = source
    GEOSPATIAL_CONUS27 = _resolve("GEOSPATIAL_CONUS27")
    UNIVERSAL = _resolve("UNIVERSAL")
    STRICT = _resolve("STRICT")
    PRESETS = {
        "GEOSPATIAL_CONUS27": GEOSPATIAL_CONUS27,
        "UNIVERSAL": UNIVERSAL,
        "STRICT": STRICT,
    }


def _resolve(name: str) -> ThresholdPreset:
    """Resolve a preset from the active source, falling back to defaults."""
    preset = _source.get_preset(name)
    if preset is not None:
        return preset
    # Fallback to static if the source doesn't have this preset
    fallback = StaticThresholdSource()
    result = fallback.get_preset(name)
    if result is not None:
        return result
    return ThresholdPreset(preset_id=name)


# ---------------------------------------------------------------------------
# Named presets (resolved at import time from default source)
# ---------------------------------------------------------------------------

GEOSPATIAL_CONUS27: ThresholdPreset = _resolve("GEOSPATIAL_CONUS27")
UNIVERSAL: ThresholdPreset = _resolve("UNIVERSAL")
STRICT: ThresholdPreset = _resolve("STRICT")

PRESETS: dict[str, ThresholdPreset] = {
    "GEOSPATIAL_CONUS27": GEOSPATIAL_CONUS27,
    "UNIVERSAL": UNIVERSAL,
    "STRICT": STRICT,
}


def resolve_preset(name: str | None) -> ThresholdPreset:
    """Resolve a preset by name. Falls back to UNIVERSAL."""
    if name is None:
        return UNIVERSAL
    return _resolve(name)
