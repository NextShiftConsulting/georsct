-- Schema: certificate + certificate_reason_code + gate_result
-- ADR refs: ADR-015, ADR-016, ADR-024, ADR-029, ADR-030, ADR-034
-- Engine: DuckDB (embedded in FastAPI)
--
-- FIELD NAME RULES (ADR-034 hard cutover — no aliases, no shims):
--   kappa_compat         NOT kappa or kappa_gate
--   public_decision      NOT decision
--   gate_reason          NOT reason
--   reason_codes         normalised to certificate_reason_code table

CREATE TABLE IF NOT EXISTS certificate (
    certificate_id         VARCHAR PRIMARY KEY,
    scenario_id            VARCHAR NOT NULL REFERENCES scenarios(scenario_id),
    snapshot_t             INTEGER NOT NULL,
    zcta_id                VARCHAR(5) NOT NULL,
    embedding_arm          VARCHAR NOT NULL,
    target                 VARCHAR NOT NULL,
    snapshot_year          INTEGER NOT NULL,

    -- Core simplex (R + S_sup + N = 1)
    r                      DOUBLE NOT NULL CHECK (r >= 0.0 AND r <= 1.0),
    s_sup                  DOUBLE NOT NULL CHECK (s_sup >= 0.0 AND s_sup <= 1.0),
    n                      DOUBLE NOT NULL CHECK (n >= 0.0 AND n <= 1.0),
    alpha                  DOUBLE NOT NULL CHECK (alpha >= 0.0 AND alpha <= 1.0),

    -- Kappa (ADR-034 D1)
    -- kappa_compat = R*(1-N), always present
    -- kappa_modal_min = min(kappa_H, kappa_L, kappa_int), multimodal only
    kappa_compat           DOUBLE NOT NULL CHECK (kappa_compat >= 0.0 AND kappa_compat <= 1.0),
    kappa_h                DOUBLE CHECK (kappa_h IS NULL OR (kappa_h >= 0.0 AND kappa_h <= 1.0)),
    kappa_l                DOUBLE CHECK (kappa_l IS NULL OR (kappa_l >= 0.0 AND kappa_l <= 1.0)),
    kappa_int              DOUBLE CHECK (kappa_int IS NULL OR (kappa_int >= 0.0 AND kappa_int <= 1.0)),
    kappa_modal_min        DOUBLE CHECK (kappa_modal_min IS NULL OR (kappa_modal_min >= 0.0 AND kappa_modal_min <= 1.0)),

    -- Geo-specific (ADR-030 D1)
    sigma                  DOUBLE NOT NULL CHECK (sigma >= 0.0),
    n_ceiling              DOUBLE CHECK (n_ceiling IS NULL OR (n_ceiling >= 0.0 AND n_ceiling <= 1.0)),
    n_ceiling_unavailable_reason VARCHAR,  -- required when n_ceiling IS NULL (I4)

    -- Public projection (ADR-034 D2)
    public_decision        VARCHAR NOT NULL CHECK (public_decision IN ('EXECUTE', 'CAUTION', 'REFUSE')),
    public_decision_source VARCHAR NOT NULL DEFAULT 'gate_projection_v1'
                           CHECK (public_decision_source IN ('gate_projection_v1')),

    -- Gate narrative (ADR-030 D1, ADR-034 D3)
    -- gate_reached: named string identifier per ADR-016; NULL on EXECUTE
    -- gate_reason:  human-readable text; NULL on EXECUTE
    -- reason_codes: normalised to certificate_reason_code (see below)
    gate_reached           VARCHAR,
    gate_reason            VARCHAR,

    -- Audit anchors (ADR-024, ADR-005)
    policy_id              VARCHAR NOT NULL,
    certificate_schema_version VARCHAR NOT NULL DEFAULT 'unified_certificate_v1',
    dataset_version        VARCHAR NOT NULL,
    live_sensors_used      BOOLEAN NOT NULL DEFAULT FALSE,
    computed_at            TIMESTAMPTZ NOT NULL,

    UNIQUE (scenario_id, snapshot_t, zcta_id, embedding_arm, target),

    FOREIGN KEY (scenario_id, snapshot_t) REFERENCES scenario_snapshots(scenario_id, snapshot_t),
    FOREIGN KEY (scenario_id, zcta_id)   REFERENCES scenario_zctas(scenario_id, zcta_id),

    -- I4: every cert has n_ceiling or an explanation
    CHECK (n_ceiling IS NOT NULL OR n_ceiling_unavailable_reason IS NOT NULL),

    -- gate_reached and gate_reason must both be NULL or both be non-NULL
    CHECK (
        (gate_reached IS NULL AND gate_reason IS NULL)
        OR (gate_reached IS NOT NULL)
    )
);

-- Normalised reason_codes (list[ReasonCode] from ADR-034 D3)
-- Relational, not stuffed in a JSON column — allows indexed queries.
CREATE TABLE IF NOT EXISTS certificate_reason_code (
    certificate_id VARCHAR NOT NULL REFERENCES certificate(certificate_id),
    reason_code    VARCHAR NOT NULL CHECK (reason_code IN (
        'CLEAN_EXECUTE',
        'WARN_INFORMATIONAL',
        'FALLBACK_USED',
        'RE_ENCODE_APPLIED',
        'REPAIR_APPLIED',
        'PASSED_AFTER_INTERVENTION',
        'INTEGRITY_FAILED',
        'CONSENSUS_FAILED',
        'ADMISSIBILITY_FAILED',
        'GROUNDING_FAILED',
        'KAPPA_COMPAT_BELOW_KAPPA_REQ',
        'KAPPA_MODAL_MIN_BELOW_KAPPA_REQ',
        'N_ABOVE_THRESHOLD',
        'COHERENCE_BELOW_THRESHOLD'
    )),
    sort_order     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (certificate_id, reason_code)
);

-- Internal gate results (ADR-034 D5)
-- gate_decision is the internal control vocabulary; it must NOT appear on certificate.
-- policy_id (on certificate) = what threshold bundle was used.
-- gate_version = which gate logic version evaluated the result.
CREATE TABLE IF NOT EXISTS gate_result (
    gate_result_id   VARCHAR PRIMARY KEY,
    certificate_id   VARCHAR NOT NULL REFERENCES certificate(certificate_id),
    gate_name        VARCHAR NOT NULL,
    gate_version     VARCHAR NOT NULL,   -- required per ADR-034 D5

    gate_decision    VARCHAR NOT NULL CHECK (gate_decision IN (
        'EXECUTE', 'REJECT', 'BLOCK', 'RE_ENCODE', 'REPAIR', 'WARN', 'FALLBACK'
    )),

    active_metric    VARCHAR NOT NULL CHECK (active_metric IN ('kappa_compat', 'kappa_modal_min')),
    active_value     DOUBLE NOT NULL CHECK (active_value >= 0.0 AND active_value <= 1.0),
    threshold_metric VARCHAR NOT NULL CHECK (threshold_metric IN ('kappa_req')),
    threshold_value  DOUBLE NOT NULL CHECK (threshold_value >= 0.0 AND threshold_value <= 1.0),
    passed           BOOLEAN NOT NULL,

    gate_reason      VARCHAR,           -- NULL on pass
    evaluated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Alternative embedding arms per ZCTA/scenario/snapshot (for F4 decision_walkthrough)
CREATE TABLE IF NOT EXISTS alternative_arm (
    certificate_id        VARCHAR NOT NULL REFERENCES certificate(certificate_id),
    embedding_arm         VARCHAR NOT NULL,
    r                     DOUBLE,
    s_sup                 DOUBLE,
    n                     DOUBLE,
    alpha                 DOUBLE,
    kappa_compat          DOUBLE,
    sigma                 DOUBLE,
    n_ceiling             DOUBLE,
    public_decision       VARCHAR CHECK (public_decision IN ('EXECUTE', 'CAUTION', 'REFUSE', NULL)),
    gate_reached          VARCHAR,
    would_change_decision BOOLEAN NOT NULL,
    PRIMARY KEY (certificate_id, embedding_arm)
);
