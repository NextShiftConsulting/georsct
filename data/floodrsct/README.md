# Series 035 — FloodRSCT: Flood Certificate Empirical Validation

RSCT certificate layer wrapping urban flood forecast models across five
scenarios. §8 of the FloodRSCT SIGSPATIAL 2026 paper.

**Data lock A:** June 1, 2026 — Houston minimum viable dataset frozen  
**Data lock B:** June 2, 2026 — All five scenarios frozen  
**Experiment execution:** June 1–3, 2026  
**Results lock:** June 4, 2026  
**Paper submission:** June 5, 2026  

**Dedicated S3 bucket:** `s3://swarm-floodrsct-data/`  
**Account:** 865679935554 (nsc-swarm)  
**All data acquisition:** SageMaker Processing jobs only. No local processing.  
**Commit discipline:** `git commit + push` before every job launch.

---

## Accessibility tiers

- **Tier A** — Free, immediate, no registration: USGS NWIS, NOAA, FEMA, Census
- **Tier B** — Free, requires account or one-time setup: local open-data portals, USACE
- **Tier C** — Requires partnership/permission: utility logs, proprietary forecasts

Only Tier A and B sources are used. Tier C dependencies (pump telemetry, road closure feeds,
evacuation route status) are documented as `operational_status_unavailable` in the
feature contract — not silently nulled or fabricated. See `FEATURE_CONTRACT.yaml`.

---

## Research Questions

| RQ | Question | Primary Scenario |
|----|----------|-----------------|
| RQ1 | Does the certificate downgrade high-accuracy forecasts? | Houston |
| RQ2 | Do gate-failure profiles differ across flood regimes? | Houston + NO + NYC |
| RQ3 | Does kappa-gate catch stress scenarios missed by raw accuracy? | New Orleans |
| RQ4 | Does Gate 3a fire on demographic confounders? | All scenarios |
| RQ5 | Is certificate behavior stable within an event but variable across events? | Houston |
| RQ6 | Does retrieval grounding improve certificate accuracy? | Houston |

---

## Scenarios

All five scenarios have complete data pipelines. Empirical evaluation
in §8 covers Houston (full, all six experiments) and New Orleans + NYC
(partial, Experiments 2 and 4). Riverside-Coachella and SW Florida are
validated for §6 block-size design claims (variogram inputs only).

### S1 — Houston, TX (full empirical coverage)
- **Archetype:** urban bayou/drainage flood
- **Unit:** Harris County ZCTA (132 ZCTAs, county FIPS 48201)
- **Events:** Harvey 2017 (DR-4332), Imelda 2019 (DR-4466), Beryl 2024 (DR-4781)
- **MVD:** 100+ ZCTAs × 2+ events; gauge stage, MRMS rainfall, NFIP claims, HWM
- See `01-Houston-Overview.md`

### S2 — New Orleans, LA (partial coverage)
- **Archetype:** protected-basin pump/levee flood
- **Unit:** Orleans Parish ZCTA (20 ZCTAs, county FIPS 22071)
- **Events:** Ida 2021 (DR-4611)
- **MVD:** 20 ZCTAs; USACE levee ratings, tidal stage, NFIP claims, pump evidence (hand-coded)
- See `02-New-Orleans-Overview.md`

### S3 — Riverside / Coachella Valley, CA (variogram inputs)
- **Archetype:** desert wash flash flood
- **Unit:** Riverside + Imperial County ZCTA (FIPS 06065 + 06025)
- **Events:** Hilary 2023 (DR-4699), AR Flood 2023
- **MVD:** 30+ ZCTAs; gauge peaks, MRMS rainfall, burn scar overlap, catchment area
- See `03-Riverside-CA-Overview.md`

### S4 — Lee / Charlotte, FL (variogram inputs)
- **Archetype:** coastal surge evacuation
- **Unit:** 6-county SW Florida ZCTA (FIPS 12021/12071/12115/12081/12057/12103)
- **Events:** Ian 2022 (DR-4673), Helene 2024 (DR-4828), Milton 2024 (DR-4834)
- **MVD:** 50+ ZCTAs; SLOSH surge, NOAA tides, elevation, NFIP claims
- See `04-Charlotte-Surge.md`

### S5 — NYC / NJ (partial coverage)
- **Archetype:** dense urban cloudburst drainage
- **Unit:** NYC 5-borough ZCTA (FIPS 36061/36047/36081/36005/36085)
- **Events:** Ida 2021 (DR-4615), Henri 2021
- **MVD:** 100+ ZCTAs; 311 flood complaints, NFIP claims, sewershed assignment, MTA stations
- `access_fragility` null until v24.003 (subway/sewer-shed load mapping incomplete)
- See `05-NJ-Cloudburst.md`

---

## Upstream Forecast Model

**Primary:** MaxFloodCast (Lee et al. 2024) — request sent to `lipai.huang@tamu.edu`
on May 27. Decision deadline: May 30.

**Fallback:** LSTM + XGBoost surrogate via `jobs/train_surrogate.py` on ml.g5.2xlarge.
Certificate layer is model-agnostic; either path produces `(zcta, event, pred_risk_score)`
rows fed into certificate evaluation.

---

## Feature Contract

`FEATURE_CONTRACT.yaml` is the single source of truth for all infrastructure
evidence and derived feature columns. Every field traces to an entry with:

- `source_type`: `public_fetch` | `derived` | `hand_coded` | `operational`
- `raw_s3_path` and `build_function` in `build_event_dataset.py`
- `field_status_behavior`: `present` | `missing_source_data` | `operational_status_unavailable`
- `version_target`: which data version populates this field

**Operational fields** — real-time telemetry that cannot be obtained from public
historical archives — are set to `"unknown"` with
`field_status = operational_status_unavailable`, not `missing_source_data`:

| Field | Scenario | Reason |
|-------|----------|--------|
| `drainage_capacity_status` | Houston | HCFCD pump/gate telemetry; no public archive |
| `road_access_status` | Riverside | CALTRANS real-time closures; no public archive |
| `evacuation_route_status` | SW Florida | FL county OES feeds; no public archive |
| `pump_station_status` | New Orleans | S&WB telemetry; partially hand-coded for Ida 2021 |

---

## S3 Layout

```
s3://swarm-floodrsct-data/
├── raw/
│   ├── geocertdb2026/          # copy of ZCTA features from swarm-yrsn-datasets
│   │   └── scenarios/{s}/      # scenario-filtered subsets
│   ├── usgs_nwis/              # gauge timeseries by scenario+event
│   ├── noaa_mrms/              # Stage IV hourly grib2 by event
│   ├── noaa_hrrr/              # HRRR 3-km QPF by event
│   ├── noaa_tides/             # tidal/coastal water levels (NO + SWFL)
│   ├── noaa_slosh/             # NHC SLOSH surge grids (SWFL events)
│   ├── hurdat2/                # NHC storm tracks
│   ├── usgs_stn/               # high-water marks (Harvey, Imelda)
│   ├── houston_311/            # Houston flood service requests
│   ├── nyc_311/                # NYC 311 flood complaints
│   ├── usace_levees/           # USACE National Levee Database
│   ├── nyc_sewersheds/         # NYC DEP sewer-shed polygons
│   ├── openfema/               # disaster declarations + event-specific claims
│   ├── nlcd/impervious/v2021/  # NLCD 2021 impervious surface GeoTIFFs
│   ├── dem/3dep/v1/            # USGS 3DEP 1/3 arc-second DEM tiles
│   ├── mtbs/perimeters/v2023/  # USGS MTBS burn perimeters (CA 2015-2023)
│   ├── nhdplus/catchments/v2/  # NHDPlus V2 catchment polygons (VPU 12, 18)
│   └── mta/subway_stations/v1/ # NYC MTA subway station locations
├── manifests/                  # dataset provenance (one manifest.json per source/version)
│   └── {dataset}/{version}/manifest.json
├── processed/
│   ├── houston/
│   ├── new_orleans/
│   ├── nyc/
│   ├── riverside_coachella/
│   └── southwest_florida/
├── model/
│   ├── maxfloodcast/           # if Lee et al. respond by May 30
│   └── surrogate/              # LSTM/XGB fallback (per scenario)
├── evidence/                   # hand-coded ground truth
│   ├── no_pump_stations_ida2021.csv
│   └── nyc_subway_flooding_ida2021.csv
└── code/s035/                  # SageMaker job code uploads
```

---

## Pre-existing Data (geocertdb2026)

Already built at ZCTA resolution nationally. Copied from
`swarm-yrsn-datasets/rsct_curriculum/series_018/processed/` by
`jobs/copy_geocertdb2026.py` — no re-download. Scenario subsets
generated for all five scenarios.

| Feature set | Column prefix |
|-------------|--------------|
| Census ACS demographics | `acs_*` |
| CDC Social Vulnerability Index | `svi_*` |
| FEMA flood zones | `pct_flood_zone_*` |
| NOAA storm events (all-time aggregated) | `noaa_*` |
| NFIP claims (all-time, not event-specific) | `nfip_*` |
| Topographic Wetness Index / slope | `twi_*` |

Event-specific NFIP claims come from `fetch_openfema_event.py`, not geocertdb2026.

---

## SageMaker Jobs

All raw fetch scripts write to `s3://swarm-floodrsct-data/raw/<source>/` and write a
manifest to `s3://swarm-floodrsct-data/manifests/<dataset>/<version>/manifest.json`.
`build_event_dataset.py` joins all raw inputs into analysis-ready `(unit, event)` tables.

### Raw fetch jobs

| Job script | Fetches | Output prefix | Instance |
|-----------|---------|---------------|---------|
| `copy_geocertdb2026.py` | ZCTA static features (reuse) | `raw/geocertdb2026/` | ml.m5.large |
| `fetch_usgs_nwis.py` | USGS gauge timeseries | `raw/usgs_nwis/` | ml.m5.large |
| `fetch_noaa_mrms.py` | NOAA Stage IV hourly precip | `raw/noaa_mrms/` | ml.m5.xlarge* |
| `fetch_noaa_hrrr.py` | HRRR 3-km QPF (surrogate input) | `raw/noaa_hrrr/` | ml.m5.xlarge* |
| `fetch_hurdat2.py` | NHC HURDAT2 storm tracks | `raw/hurdat2/` | ml.m5.large |
| `fetch_usgs_stn.py` | USGS STN high-water marks | `raw/usgs_stn/` | ml.m5.large |
| `fetch_houston_311.py` | Houston flood service requests | `raw/houston_311/` | ml.m5.large |
| `fetch_noaa_tides.py` | NOAA tides — New Orleans | `raw/noaa_tides/` | ml.m5.large |
| `fetch_noaa_tides_swfl.py` | NOAA tides — SW Florida | `raw/noaa_tides/` | ml.m5.large |
| `fetch_noaa_slosh.py` | NHC SLOSH surge grids | `raw/noaa_slosh/` | ml.m5.large |
| `fetch_usace_levees.py` | USACE National Levee Database | `raw/usace_levees/` | ml.m5.large |
| `fetch_nyc_311.py` | NYC 311 flood complaints | `raw/nyc_311/` | ml.m5.large |
| `fetch_nyc_sewersheds.py` | NYC DEP sewer-shed polygons | `raw/nyc_sewersheds/` | ml.m5.large |
| `fetch_openfema_event.py` | FEMA DR declarations + claims | `raw/openfema/` | ml.m5.large |
| `fetch_nlcd_impervious.py` | NLCD 2021 impervious GeoTIFFs | `raw/nlcd/` | ml.m5.large |
| `fetch_dem_elevation.py` | USGS 3DEP DEM tiles | `raw/dem/` | ml.m5.large |
| `fetch_mtbs_burn_scars.py` | USGS MTBS burn perimeters | `raw/mtbs/` | ml.m5.large |
| `fetch_nhdplus_catchments.py` | NHDPlus V2 catchments | `raw/nhdplus/` | ml.m5.xlarge |
| `fetch_mta_stations.py` | NYC MTA subway stations | `raw/mta/` | ml.m5.large |

\* ml.m5.xlarge for harvey2017 and ar_flood_2023 (large event windows); ml.m5.large for others.

### Processing jobs

| Job script | Purpose | Output prefix | Instance |
|-----------|---------|---------------|---------|
| `build_event_dataset.py` | Assemble (unit, event) tables | `processed/` | ml.m5.xlarge–2xlarge |
| `train_surrogate.py` | LSTM + XGB fallback model | `model/surrogate/` | ml.g5.2xlarge |

### Launchers

Every `scripts/launch_*.py` is a thin wrapper over `scripts/_launcher_base.py`.
Run locally after `git commit + push`. All accept `--dry-run` for validation.

```
scripts/
├── launch_copy_geocertdb2026.py
├── launch_fetch_usgs_nwis.py       --scenario {houston|new_orleans|nyc|...}
├── launch_fetch_noaa_mrms.py       --event {harvey2017|imelda2019|...}
├── launch_fetch_noaa_hrrr.py       --event {harvey2017|...}
├── launch_fetch_hurdat2.py
├── launch_fetch_usgs_stn.py        --event {harvey2017|imelda2019}
├── launch_fetch_houston_311.py
├── launch_fetch_noaa_tides.py
├── launch_fetch_noaa_tides_swfl.py
├── launch_fetch_noaa_slosh.py
├── launch_fetch_usace_levees.py    --scenario {new_orleans|nyc|southwest_florida}
├── launch_fetch_nyc_311.py
├── launch_fetch_nyc_sewersheds.py
├── launch_fetch_openfema_event.py
├── launch_fetch_nlcd_impervious.py
├── launch_fetch_dem_elevation.py
├── launch_fetch_mtbs_burn_scars.py
├── launch_fetch_nhdplus_catchments.py
├── launch_fetch_mta_stations.py
├── launch_build_event_dataset.py   --scenario {houston|new_orleans|...}
└── launch_train_surrogate.py       --scenario {houston|new_orleans|...}
```

---

## Data Acquisition Timeline

| Day | Activity | Key risk |
|-----|----------|----------|
| 1 — May 27 | Send MaxFloodCast request; launch geocertdb2026 copy, NWIS (Houston), HURDAT2, OpenFEMA | Lee et al. response time unknown |
| 2 — May 28 | MRMS (Harvey/Imelda/Beryl/Ida-NYC/Ian/Hilary); HRRR (same events); NLCD, DEM, MTBS, NHDPlus, MTA stations; NWIS (NO/NYC/RC/SWFL) | MRMS ~500 MB/event; bandwidth |
| 3 — May 29 | NOAA tides (NO + SWFL); USACE levees; STN HWMs; Houston 311; NYC 311; NYC sewersheds | NYC DEP may require setup |
| 4 — May 30 | MRMS (Helene/Milton/ar_flood_2023); SLOSH (Ian/Helene/Milton); HRRR (remaining events); **MaxFloodCast decision deadline** | SLOSH URL patterns may fail → MANUAL_DOWNLOAD_REQUIRED |
| 5 — May 31 | Hand-code `evidence/no_pump_stations_ida2021.csv`; hand-code `evidence/nyc_subway_flooding_ida2021.csv`; surrogate training if needed; verify SLOSH results | Hand-coding: 4-6 hrs per CSV |
| 6 — June 1 | `launch_build_event_dataset.py --scenario houston`; **DATA LOCK A** | All Houston jobs must have completed |
| 7 — June 2 | `launch_build_event_dataset.py` for NO/NYC/RC/SWFL; **DATA LOCK B** | |
| 8–9 — June 3–4 | Experiments 1–6; **RESULTS LOCK June 4** | No re-runs after lock |
| 10 — June 5 | Paper submission | |

---

## Output Schema

Each `processed/{scenario}/{scenario}_event_features.parquet` has four layers:

| Layer | Columns | Populated by |
|-------|---------|-------------|
| 1 — Upstream model | `pred_risk_score`, `pred_model_id` | Experiment / surrogate job |
| 2 — Certificate | `cert_r`, `cert_s`, `cert_n`, `cert_kappa`, `cert_action`, `cert_geometry_flags` | Experiment scripts |
| 3 — Retrieval grounding | `retrieval_grounding_score`, `retrieval_tier{1,2,3}_count` | Evidence registry scripts |
| 4 — Rationale | `rationale_text`, `rationale_acceptability` | Rationale generation scripts |
| Observability | `obs_gauge_count`, `obs_gauge_distance_km`, `obs_mrms_coverage_pct`, `obs_has_hwm`, `obs_has_311`, `obs_nfip_event_claims`, `obs_feature_modal_frac`, `obs_missing_sensor_flag` | `build_event_dataset.py` |

Certificate and retrieval columns are null at data-lock time. Experiment scripts
fill them in. This is intentional — the base table is data-only.

---

## §8 Empirical Scope and §9 Limitations

**§8 empirical scope:**

- Houston: full (Experiments 1–6)
- New Orleans: partial (Experiments 2, 4)
- NYC/NJ: partial (Experiments 2, 4)
- Riverside-Coachella + SW Florida: variogram/block-size validation only (§6 claims)

**§9 language (already drafted):**

> *"Empirical evaluation is presented for three of the five scenarios (Houston,
> New Orleans, New York/New Jersey); Riverside-Coachella and Southwest Florida
> are described as scenario designs in §6, and their empirical evaluation
> requires data-acquisition windows that exceed the present paper's scope;
> it is deferred to a companion technical report."*

> *"Pump-station availability data for New Orleans is hand-coded from Hurricane
> Ida 2021 post-event reports; a full operational evaluation would require
> partnership with the Sewerage and Water Board for real-time logs."*

> *"The upstream forecast model for Houston is MaxFloodCast [Lee et al. 2024] /
> a trained surrogate; the certificate layer is forecast-model-agnostic and could
> equivalently wrap any other Houston flood forecasting system."*

---

## Evidence Registry (Tier 1/2/3 for Experiment 6)

For RQ6 (retrieval grounding), the evidence registry needs ~200 items per scenario:
~50 Tier 1 (regulatory), ~100 Tier 2 (operational), ~50 Tier 3 (literature).
Cross-scenario: ~600 items total. Bulk-fetchable in 1-2 days.

| Tier | Sources |
|------|---------|
| 1 — Regulatory | FEMA FIRMs/LOMRs, NFIP bulletins, USACE EMs, Federal Register rulemakings |
| 2 — Operational | NWPS forecast archive, USGS gauge records, USACE NLD, NWS WWA archive, NOAA HRRR/Stage IV |
| 3 — Literature | Lee 2024, McEachran 2025, Nearing 2024, Najafi 2024, Roberts 2017, Ploton 2020, USACE/state DOT technical reports |
