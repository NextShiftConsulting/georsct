"""
db_init.py — Initialise the DuckDB schema for Floodcaster.

Run once at startup (idempotent — uses CREATE TABLE IF NOT EXISTS).
Schema files are applied in numeric order.

Usage:
    python db_scripts/db_init.py [--db-path PATH]
    python db_scripts/db_init.py --db-path /tmp/floodcaster.duckdb

Default db-path: :memory: (in-process, for tests / local dev).
For ECS Fargate: mount an EFS volume and set --db-path to a persistent path,
or use :memory: and load certs from DynamoDB on each cold start.
"""

import argparse
import logging
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).parent.parent / "schema"
SCHEMA_FILES = sorted(SCHEMA_DIR.glob("*.sql"))


def init_db(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """
    Create all tables in schema order. Idempotent.

    Args:
        db_path: DuckDB database path. Use ':memory:' for in-process.

    Returns:
        Open DuckDB connection.
    """
    con = duckdb.connect(db_path)
    for schema_file in SCHEMA_FILES:
        log.info("Applying schema: %s", schema_file.name)
        sql = schema_file.read_text()
        con.executescript(sql)
    log.info("Schema initialised. Tables: %s", _list_tables(con))
    return con


def _list_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [row[0] for row in con.execute("SHOW TABLES").fetchall()]


def verify_adr034_compliance(con: duckdb.DuckDBPyConnection) -> None:
    """
    Assert that no retired ADR-034 field names exist in the schema.
    Raises AssertionError on violation.

    Retired identifiers (ADR-034):
        kappa_gate, kappa (bare field), decision (bare), reason (bare)
    """
    retired = {"kappa_gate", "kappa", "decision", "reason"}
    rows = con.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'main'
    """).fetchall()
    violations = [
        (t, c) for t, c in rows
        if c in retired
    ]
    assert not violations, (
        f"ADR-034 violation — retired field names found in schema: {violations}"
    )
    log.info("ADR-034 compliance check passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Initialise Floodcaster DuckDB schema")
    parser.add_argument("--db-path", default=":memory:", help="DuckDB path")
    args = parser.parse_args()

    con = init_db(args.db_path)
    verify_adr034_compliance(con)
    print(f"OK — {len(_list_tables(con))} tables created at {args.db_path}")
