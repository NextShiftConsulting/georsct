-- Schema: f40_vector + f40_field_status + f40_source_map + f40_audit
--         + policy_view_queue + policy_view_queue_item
-- ADR refs: ADR-033, ADR-035
-- Engine: DuckDB (embedded in FastAPI)
--
-- CRITICAL RULES (ADR-033):
--   F40 is composition-only. It must NOT emit a priority score, rank, or
--   weighted_priority. No resource_priority_score, priority_rank, risk_score.
--   Ordering belongs in policy_view_queue only, declared with a named policy_view
--   and ordering_rule.

CREATE TABLE IF NOT EXISTS f40_vector (
    f40_vector_id         VARCHAR PRIMARY KEY,
    scenario_id           VARCHAR NOT NULL REFERENCES scenarios(scenario_id),
    snapshot_t            INTEGER NOT NULL,
    zcta_id               VARCHAR(5) NOT NULL,
    certificate_id        VARCHAR NOT NULL REFERENCES certificate(certificate_id),

    -- Five non-scalar crisis fields (ADR-033 D1)
    risk_level            VARCHAR NOT NULL CHECK (risk_level IN ('high', 'moderate', 'low')),
    reliability_action    VARCHAR NOT NULL CHECK (reliability_action IN (
        'trust', 'review', 'escalate', 'suppress'
    )),
    population_context    VARCHAR CHECK (population_context IS NULL OR population_context IN (
        'high', 'medium', 'low'
    )),
    access_fragility      VARCHAR CHECK (access_fragility IS NULL OR access_fragility IN (
        'high', 'medium', 'low', 'not_applicable'
    )),
    equity_context        VARCHAR CHECK (equity_context IS NULL OR equity_context IN (
        'q1', 'q2', 'q3', 'q4'
    )),

    f40_schema_version    VARCHAR NOT NULL DEFAULT 'f40_vector_v1',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (scenario_id, snapshot_t, zcta_id)
);

-- Field-level status for missing or degraded F40 inputs (ADR-033 D5)
-- Handles the S5 access_fragility PENDING source case: NULL field +
-- structured field_status rendered as "Missing source data" — not "N/A".
CREATE TABLE IF NOT EXISTS f40_field_status (
    f40_vector_id         VARCHAR NOT NULL REFERENCES f40_vector(f40_vector_id),
    field_name            VARCHAR NOT NULL CHECK (field_name IN (
        'risk_level', 'reliability_action', 'population_context',
        'access_fragility', 'equity_context'
    )),
    status                VARCHAR NOT NULL CHECK (status IN (
        'valid', 'missing_source_data', 'not_applicable', 'degraded'
    )),
    missing_reason                VARCHAR,
    expected_available_version    VARCHAR,
    PRIMARY KEY (f40_vector_id, field_name)
);

-- Machine-readable provenance for every F40 field (ADR-033 D3)
-- F40 is composition-only; this table preserves the source for each field.
CREATE TABLE IF NOT EXISTS f40_source_map (
    f40_vector_id    VARCHAR NOT NULL REFERENCES f40_vector(f40_vector_id),
    field_name       VARCHAR NOT NULL CHECK (field_name IN (
        'risk_level', 'reliability_action', 'population_context',
        'access_fragility', 'equity_context'
    )),
    source_function  VARCHAR NOT NULL,   -- e.g. 'map_risk_level_from_public_decision'
    source_field     VARCHAR NOT NULL,   -- e.g. 'certificate.public_decision'
    transform        VARCHAR NOT NULL CHECK (transform IN (
        'passthrough', 'enum_remap', 'scenario_discriminated_enum_remap'
    )),
    source_version   VARCHAR NOT NULL,
    PRIMARY KEY (f40_vector_id, field_name)
);

-- Audit block per F40 response / batch (ADR-033 D2/D4)
-- Governance text lives here with stable IDs and locale tags.
CREATE TABLE IF NOT EXISTS f40_audit (
    f40_audit_id            VARCHAR PRIMARY KEY,
    scenario_id             VARCHAR NOT NULL,
    snapshot_t              INTEGER NOT NULL,
    recommended_use         VARCHAR NOT NULL,
    recommended_use_id      VARCHAR NOT NULL,
    required_display        BOOLEAN NOT NULL DEFAULT TRUE,
    audit_schema_version    VARCHAR NOT NULL,
    ip_status               VARCHAR NOT NULL,
    interpretation_note_id     VARCHAR NOT NULL,
    interpretation_note_locale VARCHAR NOT NULL DEFAULT 'en-US',
    interpretation_note_text   VARCHAR NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Downstream ordering layer (ADR-033 D1)
-- Ordering is allowed ONLY here, declared with a named policy_view + ordering_rule.
-- Certificates and F40 vectors must not sort by kappa_compat or any scalar proxy.
CREATE TABLE IF NOT EXISTS policy_view_queue (
    policy_view_queue_id  VARCHAR PRIMARY KEY,
    scenario_id           VARCHAR NOT NULL REFERENCES scenarios(scenario_id),
    snapshot_t            INTEGER NOT NULL,
    policy_view           VARCHAR NOT NULL,   -- e.g. 'resource_dispatch_72h'
    ordering_rule         VARCHAR NOT NULL,   -- e.g. 'risk_high_first_then_equity_q1'
    source                VARCHAR NOT NULL CHECK (source = 'F40_VECTOR'),
    policy_view_version   VARCHAR NOT NULL,
    note                  VARCHAR NOT NULL,
    created_by            VARCHAR NOT NULL,   -- 'crisis_orchestrator' | 'human_operator'
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS policy_view_queue_item (
    policy_view_queue_id  VARCHAR NOT NULL REFERENCES policy_view_queue(policy_view_queue_id),
    zcta_id               VARCHAR(5) NOT NULL,
    ordinal_position      INTEGER NOT NULL CHECK (ordinal_position > 0),
    recommended_action    VARCHAR,
    action_rationale      JSON NOT NULL DEFAULT '[]',
    PRIMARY KEY (policy_view_queue_id, ordinal_position),
    UNIQUE (policy_view_queue_id, zcta_id)
);
