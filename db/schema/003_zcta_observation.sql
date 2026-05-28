-- Schema: zcta_observation + infrastructure_evidence
-- ADR refs: ADR-030 D2/D3, five-scenario delta §3/§4
-- Engine: DuckDB (embedded in FastAPI)
--
-- zcta_observation:      raw/derived facts per ZCTA per scenario run
-- infrastructure_evidence: scenario-discriminated union per §4 of five-scenario delta
--
-- These tables are SEPARATE from certificate on purpose.
-- Certificates store measurement facts and public projection.
-- Observations store the raw evidence that feeds certificate computation.

CREATE TABLE IF NOT EXISTS zcta_observation (
    observation_id            VARCHAR PRIMARY KEY,
    scenario_id               VARCHAR NOT NULL REFERENCES scenarios(scenario_id),
    snapshot_t                INTEGER NOT NULL,
    zcta_id                   VARCHAR(5) NOT NULL,

    observation_family        VARCHAR NOT NULL CHECK (observation_family IN (
        'weather', 'hydrology', 'population', 'equity', 'access', 'infrastructure'
    )),
    observation_name          VARCHAR NOT NULL,  -- e.g. 'rainfall_72h', 'river_stage'
    observation_value_num     DOUBLE,
    observation_value_text    VARCHAR,
    observation_value_json    JSON,

    unit                      VARCHAR,
    source_system             VARCHAR NOT NULL,  -- 'NOAA', 'USGS', 'CDC_PLACES', 'ACS', etc.
    source_version            VARCHAR NOT NULL,
    source_etag               VARCHAR,           -- S3 ETag for cache invalidation
    observed_at               TIMESTAMPTZ,

    quality_status            VARCHAR NOT NULL CHECK (quality_status IN (
        'valid', 'missing', 'stale', 'estimated', 'not_applicable'
    )),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (scenario_id, snapshot_t, zcta_id, observation_family, observation_name, source_system, source_version)
);

-- Scenario-discriminated infrastructure evidence (five-scenario delta §4)
-- One row per certificate. Non-applicable scenario fields are NULL.
-- Scenario is inferred from the linked certificate; the scenario_id column is
-- a denormalised copy for query convenience.
CREATE TABLE IF NOT EXISTS infrastructure_evidence (
    certificate_id              VARCHAR PRIMARY KEY REFERENCES certificate(certificate_id),
    scenario_id                 VARCHAR NOT NULL,

    -- S1: harris_houston_urban
    bayou_segment_id            VARCHAR,
    drainage_district_id        VARCHAR,
    impervious_surface_pct      DOUBLE,
    drainage_capacity_status    VARCHAR CHECK (drainage_capacity_status IN (
        'nominal', 'stressed', 'exceeded', 'unknown', NULL
    )),

    -- S2: orleans_jefferson_protected_basin
    levee_segment_id            VARCHAR,
    pump_station_ids            JSON,     -- list[str]
    pump_station_status         JSON,     -- list[operational|degraded|failed|unknown]
    subsidence_rate_mm_yr       DOUBLE,
    canal_proximity_m           DOUBLE,

    -- S3: riverside_coachella_desert_flash
    canyon_id                   VARCHAR,
    wash_segment_id             VARCHAR,
    burn_scar_overlap           BOOLEAN,
    upstream_catchment_km2      DOUBLE,
    road_access_status          VARCHAR CHECK (road_access_status IN (
        'open', 'at_risk', 'closed', 'unknown', NULL
    )),

    -- S4: lee_charlotte_surge_evacuation
    slosh_category              INTEGER CHECK (slosh_category IS NULL OR (slosh_category >= 1 AND slosh_category <= 5)),
    elevation_m_msl             DOUBLE,
    evacuation_route_status     VARCHAR CHECK (evacuation_route_status IN (
        'open', 'congested', 'closed', 'unknown', NULL
    )),
    coastal_distance_m          DOUBLE,

    -- S5: nyc_nj_urban_cloudburst
    sewer_shed_id               VARCHAR,
    basement_apartment_count    INTEGER,
    subway_station_ids          JSON    -- list[str]
    -- impervious_surface_pct is shared with S1
);
