-- Schema: sessions + session_decisions + modal_source_etags
-- ADR refs: ADR-024 (audit), five-scenario delta §I9 (decision determinism)
-- Engine: DuckDB (embedded in FastAPI)
--
-- sessions:          F1/F2 selection state (one active session per client)
-- session_decisions: F3 outputs written here; F5/F7 read from here
-- modal_source_etags: ETag snapshot at cert computation time (ADR-030 D4 audit block)

CREATE TABLE IF NOT EXISTS sessions (
    session_id        VARCHAR PRIMARY KEY,
    scenario_id       VARCHAR REFERENCES scenarios(scenario_id),
    snapshot_t        INTEGER,
    selected_zcta_id  VARCHAR(5),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Written by every F3 list_zctas() call; read by F5 and F7.
-- y_true and actually_flooded are denormalised from georsct_table.parquet
-- at write time to avoid a cross-store join during F5/F7.
CREATE TABLE IF NOT EXISTS session_decision (
    session_id        VARCHAR NOT NULL REFERENCES sessions(session_id),
    certificate_id    VARCHAR NOT NULL REFERENCES certificate(certificate_id),
    zcta_id           VARCHAR(5) NOT NULL,
    scenario_id       VARCHAR NOT NULL,
    snapshot_t        INTEGER NOT NULL,
    public_decision   VARCHAR NOT NULL CHECK (public_decision IN ('EXECUTE', 'CAUTION', 'REFUSE')),
    -- Ground truth from georsct_table.parquet (written at decision time)
    y_true            DOUBLE,
    actually_flooded  BOOLEAN,  -- y_true > scenario-specific flood threshold
    decided_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, certificate_id)
);

-- S3 ETag snapshot per modal source at cert computation time (ADR-030 D4)
CREATE TABLE IF NOT EXISTS modal_source_etag (
    certificate_id  VARCHAR NOT NULL REFERENCES certificate(certificate_id),
    source_name     VARCHAR NOT NULL,   -- 'georsct_table', 'nfip_claims', 'svi', etc.
    etag            VARCHAR NOT NULL,
    source_tier     INTEGER NOT NULL CHECK (source_tier IN (1, 2, 3)),
    -- Tier 1: Regulatory (FEMA, NFIP, levee certs)
    -- Tier 2: SOP (NOAA/NWS procedures, evacuation protocols)
    -- Tier 3: Technical (USGS gauges, NWM outputs, drainage graphs)
    PRIMARY KEY (certificate_id, source_name)
);
