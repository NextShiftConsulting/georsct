-- Schema: scenarios + scenario_snapshots + scenario_zctas
-- ADR refs: ADR-029, ADR-030, ADR-031, ADR-033
-- Engine: DuckDB (embedded in FastAPI)

CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id            VARCHAR PRIMARY KEY,
    display_name           VARCHAR NOT NULL,
    anchor_storm           VARCHAR NOT NULL,
    region                 VARCHAR NOT NULL,
    flood_archetype        VARCHAR NOT NULL,
    primary_decision       VARCHAR NOT NULL,
    build_phase            VARCHAR NOT NULL CHECK (build_phase IN ('core', 'extension')),
    -- Ground truth summary (for F7 back-test)
    total_nfip_claims      INTEGER,
    total_loss_usd         DOUBLE,
    affected_zctas         INTEGER,
    peak_extent_km2        DOUBLE,
    fatality_count         INTEGER
);

CREATE TABLE IF NOT EXISTS scenario_snapshots (
    scenario_id   VARCHAR NOT NULL REFERENCES scenarios(scenario_id),
    snapshot_t    INTEGER NOT NULL,  -- hours before landfall (e.g. 72, 48, 24, 0)
    PRIMARY KEY (scenario_id, snapshot_t)
);

-- ZCTAs in scope per scenario; scenario-fixed geo fields live here, not in certificates
CREATE TABLE IF NOT EXISTS scenario_zctas (
    scenario_id         VARCHAR NOT NULL REFERENCES scenarios(scenario_id),
    zcta_id             VARCHAR(5) NOT NULL,
    state_fips          VARCHAR(2),
    county_fips         VARCHAR(5),
    huc8_id             VARCHAR,
    huc8_coverage_pct   DOUBLE,
    PRIMARY KEY (scenario_id, zcta_id)
);

-- Seed data: five MVP1 scenarios
INSERT OR IGNORE INTO scenarios VALUES
    ('harris_houston_urban',
     'Houston 72-Hour Urban Flood Forecast',
     'Harvey 2017',
     'Harris County, TX',
     'urban drainage / bayou overflow',
     'Which high-risk neighbourhoods are ready for action?',
     'core', 786000, 17500000000.0, 1031, 1100.0, 36),
    ('orleans_jefferson_protected_basin',
     'New Orleans 72-Hour Protected Basin Forecast',
     'Ida 2021',
     'Orleans/Jefferson Parish, LA',
     'levee / pump / canal-dependent protected basin',
     'Which protected-basin ZCTAs face infrastructure failure risk?',
     'core', 422000, 8600000000.0, 284, 680.0, 13),
    ('riverside_coachella_desert_flash',
     'Riverside/Coachella 72-Hour Desert Flash Forecast',
     'Hilary 2023',
     'Riverside/San Bernardino County, CA',
     'mountain-canyon → desert-wash flash flooding',
     'Which canyon-adjacent ZCTAs are at flash-flood risk?',
     'core', 12000, 110000000.0, 47, 220.0, NULL),
    ('lee_charlotte_surge_evacuation',
     'Charlotte County 72-Hour Surge Forecast',
     'Ian 2022',
     'Lee/Charlotte County, FL',
     'storm surge / evacuation timing / coastal catastrophic loss',
     'Which coastal ZCTAs require immediate evacuation support?',
     'extension', 153000, 113000000000.0, 89, 510.0, 149),
    ('nyc_nj_urban_cloudburst',
     'NYC/NJ 72-Hour Urban Cloudburst Forecast',
     'Ida 2021 (Northeast remnants)',
     'NYC / NJ metro',
     'dense urban cloudburst / basement / subway / sewer overload',
     'Which basement-dense ZCTAs are at cloudburst inundation risk?',
     'extension', 22000, 900000000.0, 142, NULL, 13);

-- Snapshot hours per scenario (S3 adds extra snapshots per spec §1.2)
INSERT OR IGNORE INTO scenario_snapshots VALUES
    ('harris_houston_urban', 72),
    ('harris_houston_urban', 48),
    ('harris_houston_urban', 24),
    ('harris_houston_urban', 0),
    ('orleans_jefferson_protected_basin', 72),
    ('orleans_jefferson_protected_basin', 48),
    ('orleans_jefferson_protected_basin', 24),
    ('orleans_jefferson_protected_basin', 0),
    ('riverside_coachella_desert_flash', 96),
    ('riverside_coachella_desert_flash', 72),
    ('riverside_coachella_desert_flash', 48),
    ('riverside_coachella_desert_flash', 36),
    ('riverside_coachella_desert_flash', 24),
    ('riverside_coachella_desert_flash', 12),
    ('riverside_coachella_desert_flash', 0),
    ('lee_charlotte_surge_evacuation', 72),
    ('lee_charlotte_surge_evacuation', 48),
    ('lee_charlotte_surge_evacuation', 24),
    ('lee_charlotte_surge_evacuation', 0),
    ('nyc_nj_urban_cloudburst', 72),
    ('nyc_nj_urban_cloudburst', 48),
    ('nyc_nj_urban_cloudburst', 24),
    ('nyc_nj_urban_cloudburst', 0);
