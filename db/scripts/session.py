"""
session.py — F1 and F2 wrappers: select_event and select_snapshot.

Raises typed errors matching the Floodcaster error taxonomy
(five-scenario delta §6).
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import duckdb

PHASE_1_SCENARIOS = {
    "harris_houston_urban",
    "orleans_jefferson_protected_basin",
    "riverside_coachella_desert_flash",
}


class UnknownScenario(Exception):
    """Scenario not in the supported set."""


class ScenarioNotDeployed(Exception):
    """Phase-2 scenario requested but not yet shipped."""


class SnapshotNotAvailable(Exception):
    """Requested snapshot_t is not in the scenario's snapshot list."""


@dataclass
class EventResult:
    scenario_id: str
    display_name: str
    anchor_storm: str
    flood_archetype: str
    build_phase: str
    available_snapshots: list[int]


@dataclass
class SessionState:
    session_id: str
    scenario_id: Optional[str]
    snapshot_t: Optional[int]
    selected_zcta_id: Optional[str]
    created_at: datetime
    updated_at: datetime


def select_event(con: duckdb.DuckDBPyConnection, scenario_id: str) -> EventResult:
    """
    F1: select_event — validate scenario and return its snapshot menu.

    Invalidates all prior cached certificate results (handled by the caller
    creating a new session_id).

    Raises:
        UnknownScenario: scenario_id not in supported set.
        ScenarioNotDeployed: phase-2 scenario requested before phase-2 ships.
    """
    sql = (Path(__file__).parent.parent / "queries" / "f01_select_event.sql").read_text()
    rows = con.execute(sql, {"scenario_id": scenario_id}).fetchall()

    if not rows:
        raise UnknownScenario(f"UNKNOWN_SCENARIO: {scenario_id!r}")

    row = rows[0]
    build_phase = row[4]
    if build_phase == "extension" and scenario_id not in PHASE_1_SCENARIOS:
        raise ScenarioNotDeployed(f"SCENARIO_NOT_DEPLOYED: {scenario_id!r}")

    return EventResult(
        scenario_id=row[0],
        display_name=row[1],
        anchor_storm=row[2],
        flood_archetype=row[3],
        build_phase=build_phase,
        available_snapshots=row[5],
    )


def select_snapshot(
    con: duckdb.DuckDBPyConnection,
    scenario_id: str,
    snapshot_t: int,
) -> int:
    """
    F2: select_snapshot — validate that snapshot_t is available for the scenario.

    Returns:
        precomputed_cert_count — 0 means certs must be fetched from DynamoDB.

    Raises:
        SnapshotNotAvailable: snapshot_t not in this scenario's snapshot list.
    """
    sql = (Path(__file__).parent.parent / "queries" / "f02_select_snapshot.sql").read_text()
    rows = con.execute(sql, {"scenario_id": scenario_id, "snapshot_t": snapshot_t}).fetchall()

    if not rows:
        raise SnapshotNotAvailable(
            f"SNAPSHOT_NOT_AVAILABLE: scenario={scenario_id!r} snapshot_t={snapshot_t}"
        )

    return rows[0][2]  # precomputed_cert_count


def create_session(con: duckdb.DuckDBPyConnection) -> str:
    """Create a new session row; return session_id."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    con.execute(
        "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
        [session_id, now, now],
    )
    return session_id


def update_session(
    con: duckdb.DuckDBPyConnection,
    session_id: str,
    scenario_id: Optional[str] = None,
    snapshot_t: Optional[int] = None,
    selected_zcta_id: Optional[str] = None,
) -> None:
    """Update session state after F1/F2/F4 calls."""
    now = datetime.now(timezone.utc)
    con.execute(
        """UPDATE sessions
           SET scenario_id = COALESCE(?, scenario_id),
               snapshot_t  = COALESCE(?, snapshot_t),
               selected_zcta_id = COALESCE(?, selected_zcta_id),
               updated_at  = ?
           WHERE session_id = ?""",
        [scenario_id, snapshot_t, selected_zcta_id, now, session_id],
    )


from pathlib import Path  # noqa: E402 — needed for SQL file resolution
