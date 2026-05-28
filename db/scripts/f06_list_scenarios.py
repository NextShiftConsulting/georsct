"""
f06_list_scenarios.py — F6: list_scenarios()

Independent read used to bootstrap the dashboard before F1.
Error: PORTFOLIO_UNAVAILABLE if fewer than 3 core scenarios are present.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb

_SQL = (Path(__file__).parent.parent / "queries" / "f06_list_scenarios.sql").read_text()
CORE_MINIMUM = 3


class PortfolioUnavailable(Exception):
    """Fewer than CORE_MINIMUM core scenarios deployed."""


@dataclass
class ScenarioInfo:
    scenario_id: str
    display_name: str
    anchor_storm: str
    region: str
    flood_archetype: str
    primary_decision: str
    build_phase: str
    total_nfip_claims: Optional[int]
    total_loss_usd: Optional[float]
    affected_zctas: Optional[int]
    peak_extent_km2: Optional[float]
    fatality_count: Optional[int]
    snapshot_hours: list[int] = field(default_factory=list)
    in_scope_zcta_count: int = 0


def list_scenarios(con: duckdb.DuckDBPyConnection) -> list[ScenarioInfo]:
    """
    F6: list_scenarios — all supported scenarios with metadata.

    Raises:
        PortfolioUnavailable: fewer than 3 core scenarios are present.
    """
    rows = con.execute(_SQL).fetchall()
    cols = [d[0] for d in con.description]

    results = [ScenarioInfo(**dict(zip(cols, row))) for row in rows]

    core_count = sum(1 for s in results if s.build_phase == "core")
    if core_count < CORE_MINIMUM:
        raise PortfolioUnavailable(
            f"PORTFOLIO_UNAVAILABLE: only {core_count} core scenarios present, "
            f"need >= {CORE_MINIMUM}"
        )

    return results
