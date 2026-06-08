---
license: cc-by-4.0
task_categories:
  - tabular-classification
  - tabular-regression
language:
  - en
tags:
  - flood
  - natural-hazards
  - geospatial
  - urban-resilience
  - climate
  - rsct
  - certification
size_categories:
  - 1K<n<10K
pretty_name: "One Flood Substrate, Six Decision Geometries: A Diagnostic Benchmark for Spatial AI"
authors:
  - Rudolph A Martin
configs:
  - config_name: houston
    data_files: data/houston/*.parquet
  - config_name: new_orleans
    data_files: data/new_orleans/*.parquet
  - config_name: nyc
    data_files: data/nyc/*.parquet
  - config_name: riverside_coachella
    data_files: data/riverside_coachella/*.parquet
  - config_name: southwest_florida
    data_files: data/southwest_florida/*.parquet
  - config_name: evidence
    data_files: data/evidence/*.csv
---

# One Flood Substrate, Six Decision Geometries: A Diagnostic Benchmark for Spatial AI

Multi-scenario, multi-event dataset for validating AI flood forecast certification
across five US urban flood regimes.

## Quick Start

```python
from datasets import load_dataset

# Load a single scenario
houston = load_dataset("rudymartin/floodrsct", "houston")

# Load all scenarios
for scenario in ["houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"]:
    ds = load_dataset("rudymartin/floodrsct", scenario)
    print(f"{scenario}: {ds['train'].num_rows} rows")
```

## Dataset Description

Each row represents a **(spatial unit, event)** pair -- a geographic unit (ZCTA or
sewershed) observed during a specific flood event. Features span demographics,
infrastructure, hydrology, meteorology, and flood outcomes.

### Scenarios

| Scenario | Region | Flood Regime | Events | Spatial Unit | Approx. Rows |
|----------|--------|-------------|--------|-------------|-------------|
| `houston` | Harris County, TX | Urban pluvial/tropical | Harvey 2017, Imelda 2019, Beryl 2024 | ZCTA (~132) | ~396 |
| `new_orleans` | Orleans Parish, LA | Compound surge + pluvial | Katrina 2005, Isaac 2012, Barry 2019, Ida 2021 | ZCTA (~20) | ~80 |
| `nyc` | NYC 5 boroughs | Urban pluvial/compound | Ida 2021, Henri 2021 | Sewershed | ~200+ |
| `riverside_coachella` | Riverside + Imperial Co., CA | Flash flood / arroyo | Hilary 2023, AR Flood 2023 | ZCTA | ~60+ |
| `southwest_florida` | 6-county SW FL | Coastal surge / tropical | Ian 2022, Helene 2024, Milton 2024 | ZCTA (~50) | ~150 |

### Temporal Classes (Causal Boundary)

Features are organized by temporal class, which defines their causal role:

| Class | Role | Examples |
|-------|------|---------|
| `invariant` | Physical geography (static) | Elevation, coastal distance, catchment area |
| `slow_drift` | Updated annually/decadally | ACS demographics, SVI, FEMA flood zones, NFIP history |
| `event_window` | Measured during the event | Rainfall (MRMS), peak stage, storm track, tidal surge |
| `post_event` | Ground truth / outcomes (**labels**) | High-water marks, 311 reports, NFIP event claims |
| `operational` | Real-time telemetry (unknowable at forecast time) | Pump status, road closures |

**Causal boundary:** Only `invariant`, `slow_drift`, and `event_window` features are
legitimate model inputs. `post_event` features are labels. Using `post_event` as inputs
is label leakage.

### Feature Groups

#### Demographics (ACS 2017-2021 5-year)
`acs_total_pop`, `acs_median_hh_income`, `acs_pct_below_poverty`,
`acs_pct_renter_occupied`, `acs_pct_no_vehicle`, `acs_median_home_value`,
`acs_median_year_built`, `acs_gini_index`, ...

#### Social Vulnerability (CDC SVI 2020)
`svi_overall`, `svi_socioeconomic`, `svi_household_disability`,
`svi_minority_language`, `svi_housing_transport`

#### Flood Zones (FEMA FIRM)
`flood_pct_zone_a`, `flood_pct_zone_x`, `flood_pct_zone_x500`, `flood_sfha`

#### Hydrology & Meteorology (event-window)
`rainfall_total_mm`, `peak_stage_ft`, `peak_flow_cfs`, `tidal_surge_max_m`,
`storm_distance_km`, `hrrr_qpf_total_mm`

#### Infrastructure
`impervious_pct`, `levee_condition_rating`, `upstream_catchment_km2`,
`subway_station_count` (NYC), `burn_scar_overlap` (Riverside)

#### Spatial W-Matrix Features
`zcta_degree`, `wlag_flood_zone_pct`, `wlag_population_density`,
`wlag_rainfall_mm`, `spatial_lag_residual_R0`

#### Observability
`obs_gauge_count`, `obs_gauge_distance_km`, `obs_mrms_coverage_pct`,
`obs_has_hwm`, `obs_has_311`, `obs_nfip_event_claims`

#### Outcomes (post_event labels)
`hwm_max_ft`, `flood_311_count`, `nfip_event_claims`

### Evidence (Hand-Coded Ground Truth)

The `evidence` config contains hand-coded CSV files from post-event reporting:

| File | Description |
|------|------------|
| `no_pump_stations_ida2021.csv` | New Orleans pump station status during Hurricane Ida 2021 |
| `nyc_subway_flooding_ida2021.csv` | NYC subway station flooding during Hurricane Ida 2021 |

## Data Sources

All source data is from US federal agencies and local government open data portals:

| Source | Agency | License |
|--------|--------|---------|
| ACS / Census | US Census Bureau | Public domain |
| SVI | CDC/ATSDR | Public domain |
| Flood zones (NFHL) | FEMA | Public domain |
| NFIP claims | FEMA OpenFEMA | Public domain |
| Stream gauges (NWIS) | USGS | Public domain |
| Precipitation (MRMS Stage IV) | NOAA | Public domain |
| Forecast (HRRR) | NOAA | Public domain |
| Storm tracks (HURDAT2) | NOAA NHC | Public domain |
| Tides & currents | NOAA CO-OPS | Public domain |
| High-water marks (STN) | USGS | Public domain |
| DEM (3DEP) | USGS | Public domain |
| Land cover (NLCD) | USGS MRLC | Public domain |
| Levees (NLD) | USACE | Public domain |
| NHDPlus catchments | EPA | Public domain |
| 311 service requests | City of Houston / NYC Open Data | Public domain |
| Subway stations | MTA / NYC Open Data | Public domain |
| Sewersheds | NYC DEP | Public domain |

## Raw Data Reproduction

The processed parquet files in this dataset are built from ~43.6 GB of raw data
on S3. To reproduce from scratch:

1. Clone the [georsct-flood](https://github.com/NextShiftConsulting/georsct-flood) repository
2. Run the fetch scripts in `floodrsct/scripts/launch_fetch_*.py` to download raw data
3. Run `floodrsct/scripts/launch_build_event_dataset.py --scenario <name>` per scenario

See `DATA_MANIFEST.md` in the repository for the complete raw data inventory.

## Citation

```bibtex
@article{martin2026floodrsct,
  title={One Flood Substrate, Six Decision Geometries: A Diagnostic Benchmark for Spatial AI},
  author={Martin, Rudolph A},
  year={2026},
  note={Preprint}
}
```

## License

CC-BY-4.0. All source data is US government public domain.
