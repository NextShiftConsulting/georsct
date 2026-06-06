"""Scenario domain objects.

Pure data structures -- no I/O.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ScenarioConfig:
    """Declarative scenario definition.

    Post-hex-refactor, scenario builders become configs like this.
    Currently: build_event_dataset.py has per-scenario functions.
    """

    scenario_id: str
    region: str
    events: list[str]
    fips_codes: list[str]
    hazard_type: str = "flood"
    spatial_unit: str = "zcta"
    truth_source: str = "nfip_claims"
    truth_event_match: str = "high"       # high|medium|low
    diagnostic_status: str = "adjudicative"  # adjudicative|illustrative

    adapters: list[str] = field(default_factory=lambda: [
        "fema", "nlcd", "mrms", "nfip", "census", "wmatrix",
    ])
