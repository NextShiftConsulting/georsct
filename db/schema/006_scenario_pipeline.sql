-- Schema: scenario_pipeline_config + scenario_events
-- Extends 001_scenarios with pipeline-specific fields needed by
-- build_event_dataset.py, _validate_contract.py, and geo_qa.py.
--
-- The scenarios table (001) owns identity and metadata.
-- This table owns the data pipeline mapping.

CREATE TABLE IF NOT EXISTS scenario_pipeline_config (
    scenario_id     VARCHAR PRIMARY KEY REFERENCES scenarios(scenario_id),
    pipeline_name   VARCHAR NOT NULL UNIQUE,   -- short name used in pipeline code
    output_s3_key   VARCHAR,                   -- processed/{name}/{name}_event_features.parquet
    status          VARCHAR NOT NULL CHECK (status IN ('active', 'planned', 'archived'))
                    DEFAULT 'planned',
    -- Bounding box (EPSG:4326) for geo_qa spatial checks
    lat_min         DOUBLE,
    lat_max         DOUBLE,
    lon_min         DOUBLE,
    lon_max         DOUBLE
);

CREATE TABLE IF NOT EXISTS scenario_events (
    scenario_id     VARCHAR NOT NULL REFERENCES scenarios(scenario_id),
    event_name      VARCHAR NOT NULL,          -- pipeline key: harvey2017, ida2021, etc.
    dr_number       INTEGER,                   -- FEMA disaster declaration number
    storm_id        VARCHAR,                   -- HURDAT2 storm ID: AL092017
    window_start    DATE NOT NULL,             -- event window start
    window_end      DATE NOT NULL,             -- event window end
    s3_event_key    VARCHAR,                   -- S3 key used by fetchers (may differ from event_name)
    PRIMARY KEY (scenario_id, event_name)
);

-- Seed data: five scenarios

INSERT OR IGNORE INTO scenario_pipeline_config VALUES
    ('harris_houston_urban',              'houston',
     'processed/houston/houston_event_features.parquet',
     'active', 29.5, 30.1, -95.8, -95.0),

    ('orleans_jefferson_protected_basin', 'new_orleans',
     'processed/new_orleans/no_event_features.parquet',
     'active', 29.8, 30.1, -90.2, -89.9),

    ('riverside_coachella_desert_flash',  'riverside_coachella',
     'processed/riverside_coachella/rc_event_features.parquet',
     'active', 33.3, 34.1, -117.5, -115.3),

    ('lee_charlotte_surge_evacuation',    'southwest_florida',
     'processed/southwest_florida/swfl_event_features.parquet',
     'active', 25.8, 27.5, -82.8, -81.5),

    ('nyc_nj_urban_cloudburst',           'nyc',
     'processed/nyc/nyc_event_features.parquet',
     'active', 40.5, 40.9, -74.3, -73.7);

-- Events per scenario

INSERT OR IGNORE INTO scenario_events VALUES
    -- Houston
    ('harris_houston_urban', 'harvey2017', 4332, 'AL092017',
     '2017-08-25', '2017-09-02', 'harvey2017'),
    ('harris_houston_urban', 'imelda2019', 4466, 'AL112019',
     '2019-09-17', '2019-09-21', 'imelda2019'),
    ('harris_houston_urban', 'beryl2024',  4781, 'AL022024',
     '2024-07-08', '2024-07-12', 'beryl2024'),

    -- New Orleans
    ('orleans_jefferson_protected_basin', 'ida2021', 4611, 'AL092021',
     '2021-08-29', '2021-09-01', 'ida2021_nola'),  -- S3 key: NOLA-specific Ida fetch

    -- NYC
    ('nyc_nj_urban_cloudburst', 'ida2021_nyc', 4615, 'AL092021',
     '2021-09-01', '2021-09-04', 'ida2021_nyc'),

    -- Riverside-Coachella
    ('riverside_coachella_desert_flash', 'hilary2023', 4699, 'AL092023',
     '2023-08-20', '2023-08-22', 'hilary2023'),

    -- Southwest Florida
    ('lee_charlotte_surge_evacuation', 'ian2022',    4673, 'AL092022',
     '2022-09-28', '2022-09-29', 'ian2022'),
    ('lee_charlotte_surge_evacuation', 'helene2024', 4828, 'AL092024',
     '2024-09-26', '2024-09-27', 'helene2024'),
    ('lee_charlotte_surge_evacuation', 'milton2024', 4834, 'AL142024',
     '2024-10-09', '2024-10-10', 'milton2024');
