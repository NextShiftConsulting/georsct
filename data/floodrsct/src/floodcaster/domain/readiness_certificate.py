"""ReadinessView -- domain object passed to rendering adapters.

Pure data structure per hex_arch_target.md.
"""

from dataclasses import dataclass, field


@dataclass
class ReadinessView:
    """Domain result object consumed by map_renderer adapters."""

    scenario_id: str
    quality: float
    compatibility: float
    turbulence: float
    decision: str                    # EXECUTE | CAUTION | REFUSE
    spatial_annotations: list = field(default_factory=list)
    residual_hotspots: list = field(default_factory=list)
    evidence_warnings: list = field(default_factory=list)
