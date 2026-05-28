"""
f03_list_zctas.py — F3: list_zctas()

Returns all in-scope ZCTAs with their certificates for a given
scenario/snapshot. Writes each decision to session_decision for F5/F7.

Ordering: alphabetic by zcta_id — NOT by kappa_compat.
Operator-facing ordering lives in policy_view_queue only (ADR-033 D1).
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

_SQL = (Path(__file__).parent.parent / "queries" / "f03_list_zctas.sql").read_text()

DEFAULT_ARM = "graphsage_v1"


@dataclass
class ZCTACertificate:
    zcta_id: str
    state_fips: Optional[str]
    county_fips: Optional[str]
    huc8_id: Optional[str]
    certificate_id: Optional[str]
    # Simplex
    r: Optional[float]
    s_sup: Optional[float]
    n: Optional[float]
    alpha: Optional[float]
    # Kappa (ADR-034 D1 names)
    kappa_compat: Optional[float]
    kappa_modal_min: Optional[float]
    # Geo-specific
    sigma: Optional[float]
    n_ceiling: Optional[float]
    n_ceiling_unavailable_reason: Optional[str]
    # Public projection (ADR-034 D2)
    public_decision: Optional[str]
    gate_reached: Optional[str]
    gate_reason: Optional[str]
    reason_codes: list[str] = field(default_factory=list)
    unavailable_reason: Optional[str] = None


def list_zctas(
    con: duckdb.DuckDBPyConnection,
    session_id: str,
    scenario_id: str,
    snapshot_t: int,
    embedding_arm: str = DEFAULT_ARM,
    georsct_parquet_path: Optional[str] = None,
) -> list[ZCTACertificate]:
    """
    F3: list_zctas — all ZCTAs for (scenario, snapshot) with their certs.

    Side-effect: writes a session_decision row for each ZCTA that has a
    certificate, so F5/F7 can compute precision/recall.

    Args:
        con:                 DuckDB connection.
        session_id:          Active session (from create_session).
        scenario_id:         e.g. 'harris_houston_urban'.
        snapshot_t:          Hours before landfall.
        embedding_arm:       Default 'graphsage_v1'.
        georsct_parquet_path: S3 or local path to georsct_table parquet.
                             Used to populate y_true / actually_flooded.
                             If None, those fields are left NULL.

    Returns:
        List of ZCTACertificate (ordered by zcta_id, not kappa_compat).
    """
    rows = con.execute(
        _SQL,
        {"scenario_id": scenario_id, "snapshot_t": snapshot_t, "embedding_arm": embedding_arm},
    ).fetchall()

    cols = [d[0] for d in con.description]
    results: list[ZCTACertificate] = []

    for row in rows:
        r = dict(zip(cols, row))
        cert = ZCTACertificate(
            zcta_id=r["zcta_id"],
            state_fips=r.get("state_fips"),
            county_fips=r.get("county_fips"),
            huc8_id=r.get("huc8_id"),
            certificate_id=r.get("certificate_id"),
            r=r.get("r"),
            s_sup=r.get("s_sup"),
            n=r.get("n"),
            alpha=r.get("alpha"),
            kappa_compat=r.get("kappa_compat"),
            kappa_modal_min=r.get("kappa_modal_min"),
            sigma=r.get("sigma"),
            n_ceiling=r.get("n_ceiling"),
            n_ceiling_unavailable_reason=r.get("n_ceiling_unavailable_reason"),
            public_decision=r.get("public_decision"),
            gate_reached=r.get("gate_reached"),
            gate_reason=r.get("gate_reason"),
            reason_codes=r.get("reason_codes") or [],
            unavailable_reason=r.get("unavailable_reason"),
        )
        results.append(cert)

        # Write session_decision for F5/F7 (only for ZCTAs with a cert)
        if cert.certificate_id and cert.public_decision:
            _write_session_decision(
                con=con,
                session_id=session_id,
                cert=cert,
                scenario_id=scenario_id,
                snapshot_t=snapshot_t,
                georsct_parquet_path=georsct_parquet_path,
            )

    return results


def _write_session_decision(
    con: duckdb.DuckDBPyConnection,
    session_id: str,
    cert: ZCTACertificate,
    scenario_id: str,
    snapshot_t: int,
    georsct_parquet_path: Optional[str],
) -> None:
    """Write (or replace) a session_decision row for this ZCTA."""
    y_true, actually_flooded = _lookup_ground_truth(
        con, cert.zcta_id, scenario_id, georsct_parquet_path
    )
    con.execute(
        """INSERT OR REPLACE INTO session_decision
           (session_id, certificate_id, zcta_id, scenario_id, snapshot_t,
            public_decision, y_true, actually_flooded, decided_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            session_id,
            cert.certificate_id,
            cert.zcta_id,
            scenario_id,
            snapshot_t,
            cert.public_decision,
            y_true,
            actually_flooded,
            datetime.now(timezone.utc),
        ],
    )


def _lookup_ground_truth(
    con: duckdb.DuckDBPyConnection,
    zcta_id: str,
    scenario_id: str,
    parquet_path: Optional[str],
) -> tuple[Optional[float], Optional[bool]]:
    """
    Fetch y_true from georsct_table.parquet via DuckDB S3 read.
    Returns (None, None) if parquet_path is not provided.
    """
    if not parquet_path:
        return None, None

    # Flood threshold varies by scenario archetype
    flood_thresholds = {
        "harris_houston_urban":               1_000_000,
        "orleans_jefferson_protected_basin":  500_000,
        "riverside_coachella_desert_flash":   100_000,
        "lee_charlotte_surge_evacuation":     5_000_000,
        "nyc_nj_urban_cloudburst":            200_000,
    }
    threshold = flood_thresholds.get(scenario_id, 1_000_000)

    rows = con.execute(
        f"""SELECT target_nfip_total_loss
            FROM read_parquet('{parquet_path}')
            WHERE zcta_id = ?""",
        [zcta_id],
    ).fetchone()

    if not rows or rows[0] is None:
        return None, None

    y_true = float(rows[0])
    return y_true, y_true >= threshold
