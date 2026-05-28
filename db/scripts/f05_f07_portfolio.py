"""
f05_f07_portfolio.py — F5: summarize_session and F7: summarize_portfolio

F5: per-session precision/recall against ground truth.
F7: cross-scenario reliability stats (the benchmark claim).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb

_QUERIES_DIR = Path(__file__).parent.parent / "queries"
_F05_SQL = (_QUERIES_DIR / "f05_summarize_session.sql").read_text()
_F07_SQL = (_QUERIES_DIR / "f07_summarize_portfolio.sql").read_text()

PHASE_1_MINIMUM = 3  # F7 requires >= 3 scenarios


class InsufficientScenarios(Exception):
    """F7 called with fewer than PHASE_1_MINIMUM scenarios evaluated."""


@dataclass
class SessionSummary:
    total_zctas: int
    zctas_targeted: int
    zctas_that_flooded: int
    zctas_that_didnt: int
    flooded_zctas_missed: int
    precision_at_execute: Optional[float]
    recall_at_execute: Optional[float]
    f1_at_execute: Optional[float]


@dataclass
class PerScenarioStats:
    scenario_id: str
    n_execute: int
    n_caution: int
    n_refuse: int
    precision_at_execute: Optional[float]
    recall_at_execute: Optional[float]
    ground_truth_alignment: Optional[float]
    dominant_failure_gate: Optional[str]


@dataclass
class PortfolioSummary:
    scenarios_evaluated: int
    scenarios_available: int
    per_scenario: list[PerScenarioStats] = field(default_factory=list)
    schema_fields_identical: bool = True
    scenarios_with_certs: int = 0


def summarize_session(
    con: duckdb.DuckDBPyConnection,
    session_id: str,
) -> SessionSummary:
    """
    F5: summarize_session — precision/recall for the active session.

    Precondition: list_zctas() called at least once (session_decision rows exist).

    Raises:
        ValueError: NO_DECISIONS_YET if no session_decision rows for session.
    """
    # Check precondition
    count = con.execute(
        "SELECT COUNT(*) FROM session_decision WHERE session_id = ?",
        [session_id],
    ).fetchone()[0]
    if count == 0:
        raise ValueError(f"NO_DECISIONS_YET: session {session_id!r} has no decisions")

    row = con.execute(_F05_SQL, {"session_id": session_id}).fetchone()
    cols = [d[0] for d in con.description]
    r = dict(zip(cols, row))

    return SessionSummary(
        total_zctas=r["total_zctas"],
        zctas_targeted=r["zctas_targeted"],
        zctas_that_flooded=r["zctas_that_flooded"],
        zctas_that_didnt=r["zctas_that_didnt"],
        flooded_zctas_missed=r["flooded_zctas_missed"],
        precision_at_execute=r.get("precision_at_execute"),
        recall_at_execute=r.get("recall_at_execute"),
        f1_at_execute=r.get("f1_at_execute"),
    )


def summarize_portfolio(
    con: duckdb.DuckDBPyConnection,
    session_id: str,
) -> PortfolioSummary:
    """
    F7: summarize_portfolio — cross-scenario reliability stats.

    This is the benchmark claim: gate logic behaves consistently across
    five distinct flood archetypes. Three archetypes is the minimum to make
    the claim falsifiable (five-scenario delta §2 F7).

    Raises:
        InsufficientScenarios: fewer than PHASE_1_MINIMUM scenarios evaluated.
    """
    # Check precondition: >= 3 distinct scenarios in session_decision
    n_scenarios = con.execute(
        "SELECT COUNT(DISTINCT scenario_id) FROM session_decision WHERE session_id = ?",
        [session_id],
    ).fetchone()[0]

    if n_scenarios < PHASE_1_MINIMUM:
        raise InsufficientScenarios(
            f"INSUFFICIENT_SCENARIOS: {n_scenarios} evaluated, "
            f"need >= {PHASE_1_MINIMUM}"
        )

    row = con.execute(_F07_SQL, {"session_id": session_id}).fetchone()
    if not row:
        raise InsufficientScenarios("INSUFFICIENT_SCENARIOS: no portfolio data")

    cols = [d[0] for d in con.description]
    r = dict(zip(cols, row))

    # per_scenario is JSON-aggregated in SQL
    import json
    per_scenario_raw = r.get("per_scenario") or []
    if isinstance(per_scenario_raw, str):
        per_scenario_raw = json.loads(per_scenario_raw)

    per_scenario = [
        PerScenarioStats(
            scenario_id=ps["scenario_id"],
            n_execute=ps.get("n_execute", 0),
            n_caution=ps.get("n_caution", 0),
            n_refuse=ps.get("n_refuse", 0),
            precision_at_execute=ps.get("precision_at_execute"),
            recall_at_execute=ps.get("recall_at_execute"),
            ground_truth_alignment=ps.get("ground_truth_alignment"),
            dominant_failure_gate=ps.get("dominant_failure_gate"),
        )
        for ps in per_scenario_raw
    ]

    return PortfolioSummary(
        scenarios_evaluated=r["scenarios_evaluated"],
        scenarios_available=r["scenarios_available"],
        per_scenario=per_scenario,
        schema_fields_identical=bool(r.get("schema_fields_identical", True)),
        scenarios_with_certs=r.get("scenarios_with_certs", 0),
    )
