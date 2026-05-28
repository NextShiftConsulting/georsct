"""
f04_select_zcta.py — F4: select_zcta()

Full certificate drill-down for one ZCTA:
  certificate + gate results + reason codes + infrastructure evidence + alternative arms
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb

_QUERIES_DIR = Path(__file__).parent.parent / "queries"
_SQL = (_QUERIES_DIR / "f04_select_zcta.sql").read_text()

# F4 SQL has 6 SELECT blocks separated by blank lines; split for individual execution
_SQL_BLOCKS = [b.strip() for b in _SQL.split("\n\n") if b.strip().startswith("SELECT")]

DEFAULT_ARM = "graphsage_v1"


@dataclass
class GateResultRow:
    gate_name: str
    gate_version: str
    gate_decision: str        # internal GateDecision vocab
    active_metric: str        # kappa_compat | kappa_modal_min
    active_value: float
    threshold_metric: str     # kappa_req
    threshold_value: float
    passed: bool
    gate_reason: Optional[str]
    evaluated_at: str


@dataclass
class AlternativeArmRow:
    embedding_arm: str
    r: Optional[float]
    s_sup: Optional[float]
    n: Optional[float]
    alpha: Optional[float]
    kappa_compat: Optional[float]
    sigma: Optional[float]
    n_ceiling: Optional[float]
    public_decision: Optional[str]
    gate_reached: Optional[str]
    would_change_decision: bool


@dataclass
class ModalSourceRow:
    source_name: str
    etag: str
    source_tier: int


@dataclass
class ZCTADetail:
    # Core cert fields (ADR-034 canonical names)
    certificate_id: str
    scenario_id: str
    snapshot_t: int
    zcta_id: str
    embedding_arm: str
    target: str
    snapshot_year: int
    r: float
    s_sup: float
    n: float
    alpha: float
    kappa_compat: float
    kappa_h: Optional[float]
    kappa_l: Optional[float]
    kappa_int: Optional[float]
    kappa_modal_min: Optional[float]
    sigma: float
    n_ceiling: Optional[float]
    n_ceiling_unavailable_reason: Optional[str]
    public_decision: str
    public_decision_source: str
    gate_reached: Optional[str]
    gate_reason: Optional[str]
    policy_id: str
    certificate_schema_version: str
    dataset_version: str
    live_sensors_used: bool
    computed_at: str
    # Normalised sub-resources
    reason_codes: list[str] = field(default_factory=list)
    gate_results: list[GateResultRow] = field(default_factory=list)
    infrastructure_evidence: Optional[dict] = None
    alternative_arms: list[AlternativeArmRow] = field(default_factory=list)
    modal_sources: list[ModalSourceRow] = field(default_factory=list)


def select_zcta(
    con: duckdb.DuckDBPyConnection,
    scenario_id: str,
    snapshot_t: int,
    zcta_id: str,
    embedding_arm: str = DEFAULT_ARM,
) -> Optional[ZCTADetail]:
    """
    F4: select_zcta — full certificate drill-down.

    Runs the six SQL blocks from f04_select_zcta.sql sequentially.

    Returns:
        ZCTADetail or None if the ZCTA has no certificate.

    Raises:
        ValueError: zcta_id not in scenario scope (ZCTA_NOT_IN_SCOPE).
    """
    params = {
        "scenario_id": scenario_id,
        "snapshot_t": snapshot_t,
        "zcta_id": zcta_id,
        "embedding_arm": embedding_arm,
    }

    # Validate ZCTA is in scope
    in_scope = con.execute(
        "SELECT 1 FROM scenario_zctas WHERE scenario_id = ? AND zcta_id = ?",
        [scenario_id, zcta_id],
    ).fetchone()
    if not in_scope:
        raise ValueError(f"ZCTA_NOT_IN_SCOPE: {zcta_id!r} is not in scenario {scenario_id!r}")

    # Block 1: core certificate
    cert_rows = con.execute(_SQL_BLOCKS[0], params).fetchall()
    if not cert_rows:
        return None  # ZCTA in scope but cert not yet computed

    cert_cols = [d[0] for d in con.description]
    cr = dict(zip(cert_cols, cert_rows[0]))

    detail = ZCTADetail(
        certificate_id=cr["certificate_id"],
        scenario_id=cr["scenario_id"],
        snapshot_t=cr["snapshot_t"],
        zcta_id=cr["zcta_id"],
        embedding_arm=cr["embedding_arm"],
        target=cr["target"],
        snapshot_year=cr["snapshot_year"],
        r=cr["r"], s_sup=cr["s_sup"], n=cr["n"], alpha=cr["alpha"],
        kappa_compat=cr["kappa_compat"],
        kappa_h=cr.get("kappa_h"), kappa_l=cr.get("kappa_l"),
        kappa_int=cr.get("kappa_int"), kappa_modal_min=cr.get("kappa_modal_min"),
        sigma=cr["sigma"],
        n_ceiling=cr.get("n_ceiling"),
        n_ceiling_unavailable_reason=cr.get("n_ceiling_unavailable_reason"),
        public_decision=cr["public_decision"],
        public_decision_source=cr["public_decision_source"],
        gate_reached=cr.get("gate_reached"),
        gate_reason=cr.get("gate_reason"),
        policy_id=cr["policy_id"],
        certificate_schema_version=cr["certificate_schema_version"],
        dataset_version=cr["dataset_version"],
        live_sensors_used=bool(cr["live_sensors_used"]),
        computed_at=str(cr["computed_at"]),
    )

    # Block 2: reason codes
    rc_rows = con.execute(_SQL_BLOCKS[1], params).fetchall()
    detail.reason_codes = [row[0] for row in rc_rows]

    # Block 3: gate results
    gr_rows = con.execute(_SQL_BLOCKS[2], params).fetchall()
    gr_cols = [d[0] for d in con.description]
    detail.gate_results = [
        GateResultRow(**dict(zip(gr_cols, row))) for row in gr_rows
    ]

    # Block 4: infrastructure evidence
    ie_rows = con.execute(_SQL_BLOCKS[3], params).fetchall()
    if ie_rows:
        ie_cols = [d[0] for d in con.description]
        detail.infrastructure_evidence = dict(zip(ie_cols, ie_rows[0]))

    # Block 5: alternative arms
    aa_rows = con.execute(_SQL_BLOCKS[4], params).fetchall()
    aa_cols = [d[0] for d in con.description]
    detail.alternative_arms = [
        AlternativeArmRow(**dict(zip(aa_cols, row))) for row in aa_rows
    ]

    # Block 6: modal source ETags
    ms_rows = con.execute(_SQL_BLOCKS[5], params).fetchall()
    ms_cols = [d[0] for d in con.description]
    detail.modal_sources = [
        ModalSourceRow(**dict(zip(ms_cols, row))) for row in ms_rows
    ]

    return detail
