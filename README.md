---
license: cc-by-4.0
task_categories:
  - tabular-regression
language:
  - en
tags:
  - geospatial
  - zcta
  - cdc-places
  - acs
  - public-health
  - socioeconomic
  - environmental
  - regression
  - benchmark
  - geoparquet
  - evaluation
  - representation-learning
  - model-evaluation
  - solver-evaluation
  - representation-solver-compatibility
  - spatial-generalization
  - spatial-lag
  - social-vulnerability
  - flood-risk
  - healthcare-access
  - geoai
size_categories:
  - 10K<n<100K
pretty_name: "GeoRSCT"
configs:
  - config_name: table
    data_files:
      - split: benchmark
        path: georsct_table.parquet
  - config_name: geometry
    data_files:
      - split: benchmark
        path: georsct_simplified_001.geoparquet
  - config_name: cdc_ci
    data_files:
      - split: benchmark
        path: cdc_places_ci.parquet
  - config_name: acs_moe
    data_files:
      - split: benchmark
        path: zcta_acs_margins_of_error.parquet
---

# GeoRSCT

**A geospatial regression benchmark for evaluating representation–solver compatibility.**

GeoRSCT is a benchmark and evaluation framework for studying when geospatial model performance reflects solver quality versus target difficulty, spatial leakage, aggregation effects, scale sensitivity, or representation–solver mismatch.

This initial public release (version **23.0.2**) includes 31,789 U.S. ZIP Code Tabulation Areas (ZCTAs), 63 solver-usable input features, 27 regression targets spanning health, socioeconomic, and environmental domains, simplified ZCTA geometries, coverage flags, uncertainty sidecars, and three geography-aware evaluation protocols.

The 63 solver-usable features include:

1. **33 ACS 2022 5-Year features** describing demographic, economic, housing, and social characteristics.
2. **14 spatial-lag ACS features** computed as queen-contiguity weighted neighbor means.
3. **16 geospatial enrichment features** covering social vulnerability, flood exposure, hospital/pharmacy access, and drive-time context.

The GeoRSCT data pipeline, build scripts, and validation notebooks are maintained in the companion GitHub repository at [https://github.com/NextShiftConsulting/georsct](https://github.com/NextShiftConsulting/georsct).

## Theoretical Foundation

GeoRSCT is the geospatial benchmark artifact for **Representation–Solver Compatibility Theory (RSCT)**.

The underlying RSCT theory is introduced in:

> Rudolph A. Martin. **Intelligence as Representation-Solver Compatibility: A General Theory of Representation-Dependent Reasoning.** Preprint, March 3, 2026. [Zenodo](https://doi.org/10.5281/zenodo.18854651) | [SSRN](https://ssrn.com/abstract=6339299)

RSCT argues that problem difficulty is not intrinsic to the problem alone. Instead, observed performance depends on the relationship among the problem, the encoding, and the solver:

```text
D = D(P, E, S)
```

GeoRSCT applies that theory to geospatial regression. The benchmark is designed to help determine whether observed performance reflects solver quality, target difficulty, spatial leakage, aggregation effects, scale sensitivity, or representation–solver incompatibility.

## Dataset Summary

GeoRSCT is an evaluation-ready ZCTA-level benchmark derived from public geospatial, public-health, socioeconomic, infrastructure, and environmental data sources. It is not a mirror of CDC PLACES. Instead, it curates 27 regression targets from multiple sources, 63 solver-usable input features, ZCTA boundary geometries, coverage flags, uncertainty sidecars, and three fixed evaluation split protocols for studying spatial generalization, target recoverability, and representation–solver compatibility.

GeoRSCT is designed for **evaluation diagnosis**, not only model ranking. A model score on this benchmark should be interpreted in relation to the target, the representation, the spatial split, and the solver family.

| Property | Value |
|---|---:|
| Version | 23.0.2 |
| ZCTAs | 31,789 |
| Main table columns | 106 analytic columns without geometry; 107 with geometry |
| Solver-usable input features | 63 |
| ACS features | 33 ACS 2022 5-Year features |
| Spatial-lag features | 14 queen-contiguity ACS neighbor-mean features |
| Geospatial enrichment features | 16 SVI, flood, healthcare-access, and drive-time context features |
| Target tasks | 27 |
| Health targets | 21 CDC PLACES 2023 estimates |
| Socioeconomic targets | 3 ACS-derived targets (income, home value, population density) |
| Environmental targets | 3 physical/remote-sensing targets (night lights, elevation, tree cover) |
| Evaluation protocols | 3 geography-aware protocols |
| Sidecar uncertainty files | CDC PLACES confidence intervals; ACS margins of error |
| CRS | EPSG:4326 / WGS 84 |

## What GeoRSCT Is For

Use GeoRSCT to study:

1. Whether a solver generalizes across geography.
2. Whether target difficulty dominates solver ranking.
3. Whether spatially structured splits change conclusions relative to random splits.
4. Whether a representation makes useful signal recoverable for a solver.
5. Whether benchmark scores reflect solver quality or geospatial artifacts.
6. Whether enriched geographic context changes recoverability relative to ACS-only baselines.
7. Whether uncertainty sidecars affect evaluation conclusions when margins of error or confidence intervals are incorporated.

GeoRSCT should not be interpreted as a leaderboard-only benchmark. Its primary purpose is to help evaluate what a geospatial score actually means.

## Evaluation Claims Supported

GeoRSCT supports claims about:

- geospatial regression performance under fixed geography-aware splits;
- target-level difficulty variation across health, socioeconomic, and environmental tasks;
- solver-family behavior under common ZCTA-level input representations;
- differences between county holdout, state holdout, and county-to-ZCTA super-resolution protocols;
- representation–solver compatibility under administratively aggregated U.S. geography;
- differences between ACS-only representations and expanded geospatial context representations;
- uncertainty-aware analysis when users explicitly incorporate the CDC confidence interval and ACS margin-of-error sidecars.

GeoRSCT does **not** support universal claims about global geospatial generalization, individual-level health prediction, causal inference, public-health surveillance deployment, clinical decision-making, or fairness outcomes for specific communities without additional validation.

## Why Geography-Aware Evaluation Matters

Geospatial rows are not independent examples. Nearby ZCTAs often share demographics, infrastructure, environmental exposure, housing markets, public-health patterns, and regional history. Random splits can therefore leak geographic information from training to test data.

GeoRSCT uses fixed geography-aware protocols because the central question is not only whether a model predicts well, but whether its performance reflects transferable structure rather than spatial proximity, administrative aggregation, scale effects, or target difficulty.

## Files

| File | Size | Description |
|---|---:|---|
| `georsct_simplified_001.geoparquet` | — | Full v23.0.2 dataset with simplified ZCTA boundary polygons |
| `georsct_table.parquet` | — | Same data without geometry for lightweight tabular use |
| `cdc_places_ci.parquet` | — | CDC PLACES 95% confidence intervals for 21 health targets; joins on `zcta_id` |
| `zcta_acs_margins_of_error.parquet` | — | ACS margins of error for ACS features; joins on `zcta_id` |
| `georsct_schema.json` | — | Column metadata, data types, missing-value counts, and summary statistics |
| `build_manifest.json` | — | Build provenance and dataset statistics, generated by the GeoRSCT pipeline |
| `georsct_checksums.sha256` | — | SHA-256 checksums for all files |
| `croissant.json` | — | MLCommons Croissant metadata for dataset discovery and machine-readable schema |
| `load_georsct.py` | — | Helper functions for loading, splitting, validating, and filtering by coverage |
| `quickstart.py` | — | Download verification and toy baseline |
| GitHub repository | — | Source code, pipeline, and validation notebooks at [https://github.com/NextShiftConsulting/georsct](https://github.com/NextShiftConsulting/georsct) |

Replace the size placeholders after the final v23.0.2 files are uploaded.

## Field Schema Overview

GeoRSCT is distributed as a row-per-ZCTA tabular benchmark (31,789 rows, one per 2020-vintage ZCTA in CONUS). ZCTAs are Census Bureau statistical areas that approximate USPS ZIP code service areas; multiple ZIP codes can map to the same ZCTA, and the mapping is not one-to-one. Each row includes identifiers, centroid coordinates, ACS input features, spatial-lag features, geospatial enrichment features, target labels, coverage flags, geography-aware split assignments, and optional geometry.

| Field group | Example columns | Type | Description |
|---|---|---|---|
| Identifier fields | `zcta_id`, `state_fips`, `county_fips` | string | Geographic identifiers used for joining, grouping, and split construction |
| Location fields | `latitude`, `longitude` | numeric | ZCTA centroid coordinates in EPSG:4326 |
| ACS input features | `acs_total_pop`, `acs_median_age`, `acs_pct_below_poverty`, ... | numeric | 33 American Community Survey 2022 5-Year features |
| Spatial-lag features | `lag_acs_total_pop`, `lag_acs_median_age`, `lag_acs_median_home_value`, ... | numeric | 14 queen-contiguity weighted neighbor means computed from ACS features |
| SVI enrichment features | `svi_socioeconomic`, `svi_household_disability`, `svi_minority_language`, `svi_housing_transport`, `svi_overall` | numeric | CDC/ATSDR Social Vulnerability Index context features |
| Flood enrichment features | `flood_pct_zone_a`, `flood_pct_zone_x500`, `flood_pct_zone_x` | numeric | FEMA NFHL flood-zone area percentages |
| Access enrichment features | `hifld_n_hospitals`, `hifld_nearest_hospital_km`, `hifld_n_pharmacies`, ... | numeric | HIFLD 2022 hospital, pharmacy, bed-count, and trauma-center access features |
| Drive-time enrichment features | `drive_min_to_nearest_hospital`, `drive_min_to_county_centroid` | numeric | OSRM road-network travel-time context |
| Health targets | `target_diabetes`, `target_obesity`, `target_smoking`, ... | numeric | 21 CDC PLACES 2023 model-based ZCTA health estimates |
| Socioeconomic targets | `target_income`, `target_home_value`, `target_population_density` | numeric | ACS-derived socioeconomic regression targets |
| Environmental targets | `target_night_lights`, `target_elevation`, `target_tree_cover` | numeric | Remote-sensing and physical-environment regression targets |
| Coverage flags | `has_cdc_places`, `has_income`, `has_home_value`, `has_cdc_ci` | boolean | Flags indicating whether target families and CDC confidence intervals are available |
| Evaluation splits | `split_imputation`, `split_extrapolation`, `split_superres` | categorical | Fixed geography-aware split assignments |
| Geometry | `geometry` | polygon | Simplified ZCTA boundary geometry, included only in the GeoParquet file |

For complete column names, data types, missing-value counts, and summary statistics, see `georsct_schema.json`.

## Getting Started

### 1. Install dependencies

```bash
pip install pandas pyarrow scikit-learn

# Optional, for geometry:
pip install geopandas pyogrio
```

### 2. Verify your download

```bash
python quickstart.py
```

This checks file integrity using SHA-256 checksums, validates the data, verifies row counts and split assignments, and runs a toy Ridge baseline on diabetes prediction to confirm the benchmark works end to end.

### 3. Load and use the lightweight table

```python
from load_georsct import load_georsct, get_split, feature_columns, target_columns

df = load_georsct("georsct_table.parquet")

# Pass target= to auto-drop rows where that target is missing.
train, val, test = get_split(
    df,
    protocol="imputation",
    fold=1,
    target="target_diabetes",
)

X_train = train[feature_columns(df)]
y_train = train["target_diabetes"]
```

By default, `feature_columns(df)` should return the 63 solver-usable v23.0.2 features: 33 ACS features, 14 spatial-lag features, and 16 enrichment features. To reproduce the original v23.001 ACS-only baseline, use only columns beginning with `acs_`.

### 4. Load with geometry

```python
geo = load_georsct("georsct_simplified_001.geoparquet")
geo.plot(column="target_obesity", legend=True)
```

### 5. Load uncertainty sidecars

```python
import pandas as pd

main = pd.read_parquet("georsct_table.parquet")
cdc_ci = pd.read_parquet("cdc_places_ci.parquet")
acs_moe = pd.read_parquet("zcta_acs_margins_of_error.parquet")

main_with_ci = main.merge(cdc_ci, on="zcta_id", how="left")
main_with_moe = main.merge(acs_moe, on="zcta_id", how="left")
```

The sidecars are optional. They are provided for uncertainty-aware analysis and do not need to be loaded for standard benchmark baselines.

## Handling Missing Values

Not all 31,789 ZCTAs have all 27 targets. When you pick a target, some rows may have `NaN`. You do **not** need to pre-filter the whole dataset; just handle missing values for your chosen target.

```python
# Option A: let get_split handle it.
train, val, test = get_split(
    df,
    protocol="imputation",
    fold=1,
    target="target_diabetes",
)

# Option B: manual dropna.
train = train.dropna(subset=["target_diabetes"])
```

Coverage flags tell you which targets are affected:

| Flag | Targets affected when False |
|---|---|
| `has_cdc_places` | All 21 `target_*` health columns |
| `has_income` | `target_income` |
| `has_home_value` | `target_home_value` |
| `has_cdc_ci` | CDC PLACES confidence interval columns in `cdc_places_ci.parquet` |

Environmental and physical-context targets (`target_elevation`, `target_tree_cover`, `target_night_lights`, `target_population_density`) have no missing values.

ACS input features have some missing values in median-value columns. See `georsct_schema.json` for exact counts. Handle these using your model's missing-value strategy, such as imputation, mean fill, or a model that handles missing values natively.

ACS margins of error are provided in the sidecar file `zcta_acs_margins_of_error.parquet` for uncertainty-aware analysis. They are not automatically propagated into benchmark scores unless the user explicitly incorporates them.

CDC PLACES confidence intervals are provided in `cdc_places_ci.parquet` for the 21 health targets. These join to the main table on `zcta_id`.

## Splitting Without the Helper

```python
import pandas as pd

df = pd.read_parquet("georsct_table.parquet")

target = "target_diabetes"
df_clean = df.dropna(subset=[target])

# Imputation protocol, fold 1.
train = df_clean[~df_clean["split_imputation"].isin(["valid1", "test"])]
val = df_clean[df_clean["split_imputation"] == "valid1"]
test = df_clean[df_clean["split_imputation"] == "test"]

feature_cols = [
    c for c in df_clean.columns
    if c.startswith("acs_")
    or c.startswith("lag_acs_")
    or c.startswith("svi_")
    or c.startswith("flood_")
    or c.startswith("hifld_")
    or c.startswith("drive_min_")
]
```

For an ACS-only baseline, use:

```python
acs_only_cols = [c for c in df_clean.columns if c.startswith("acs_")]
```

## Tasks

### Health Targets: CDC PLACES 2023

The 21 health targets are CDC PLACES model-based small-area estimates derived from BRFSS survey data using multilevel regression and poststratification. These are **modeled estimates, not direct measurements**.

| Target | Description |
|---|---|
| `target_annual_checkup` | Adults with annual checkup (%) |
| `target_arthritis` | Adults with arthritis (%) |
| `target_asthma` | Adults with current asthma (%) |
| `target_binge_drinking` | Adults who binge drink (%) |
| `target_bp_medicated` | Adults taking blood pressure medication (%) |
| `target_cancer` | Adults ever told they had cancer (%) |
| `target_cholesterol_screening` | Adults with cholesterol screening (%) |
| `target_chronic_kidney_disease` | Adults with chronic kidney disease (%) |
| `target_copd` | Adults with COPD (%) |
| `target_coronary_heart_disease` | Adults with coronary heart disease (%) |
| `target_dental_visit` | Adults with dental visit (%) |
| `target_diabetes` | Adults with diabetes (%) |
| `target_high_blood_pressure` | Adults with high blood pressure (%) |
| `target_high_cholesterol` | Adults with high cholesterol (%) |
| `target_mental_health_not_good` | Adults with frequent mental distress (%) |
| `target_obesity` | Adults with obesity (%) |
| `target_physical_health_not_good` | Adults with frequent physical distress (%) |
| `target_physical_inactivity` | Adults with no leisure-time physical activity (%) |
| `target_sleep_less_7hr` | Adults sleeping less than 7 hours (%) |
| `target_smoking` | Adults who currently smoke (%) |
| `target_stroke` | Adults ever had stroke (%) |

### Socioeconomic Targets

| Target | Source | Description |
|---|---|---|
| `target_income` | ACS 2022 | Median household income ($) |
| `target_home_value` | ACS 2022 | Median home value ($) |
| `target_population_density` | Census 2020 | Population per square kilometer |

### Environmental Targets

| Target | Source | Description |
|---|---|---|
| `target_night_lights` | VIIRS NASA/NOAA | Mean nighttime radiance, log10 nW/cm²/sr |
| `target_elevation` | USGS NED | Mean elevation, meters |
| `target_tree_cover` | Hansen GFC | Mean tree canopy cover (%) |

## Input Features

GeoRSCT v23.0.2 includes 63 solver-usable input features.

### 1. ACS Features

The 33 ACS features are prefixed with `acs_`. They include demographic, economic, housing, transportation, and social characteristics at the ZCTA level.

These features are the default ACS-only encoder/input representation and remain useful for controlled baselines.

### 2. Spatial-Lag Features

The 14 spatial-lag features are prefixed with `lag_acs_`. They are computed as queen-contiguity weighted neighbor means over selected ACS fields.

These are **spatial lags, not time lags**. They encode neighboring-area context, not temporal history.

### 3. Geospatial Enrichment Features

The 16 enrichment features include:

| Feature family | Example columns | Interpretation |
|---|---|---|
| SVI | `svi_socioeconomic`, `svi_overall` | Social vulnerability context |
| Flood | `flood_pct_zone_a`, `flood_pct_zone_x500`, `flood_pct_zone_x` | FEMA flood-zone exposure |
| HIFLD access | `hifld_n_hospitals`, `hifld_nearest_hospital_km`, `hifld_n_pharmacies` | HIFLD 2022 healthcare infrastructure and access |
| Drive time | `drive_min_to_nearest_hospital`, `drive_min_to_county_centroid` | Road-network access context |

The expanded v23.0.2 feature set is intended for stronger geospatial solver evaluation and representation–solver compatibility experiments. Reported benchmark comparisons should document whether they use ACS-only features or the full 63-feature representation.

## Evaluation Protocols

GeoRSCT includes three fixed protocols that test different aspects of spatial generalization.

### 1. Imputation: County Holdout

**Column:** `split_imputation`  
**Values:** `valid1` through `valid5`, `test`

This protocol uses 5-fold validation with county-level holdout. It tests geographic interpolation: can the model predict ZCTAs in unseen counties when other counties are observed?

### 2. Extrapolation: State Holdout

**Column:** `split_extrapolation`  
**Values:** `valid1` through `valid4`, `test`

This protocol uses state-level holdout. It tests distribution shift: can the model generalize to regions with different demographic and geographic characteristics?

### 3. Super-Resolution: County to ZCTA

**Column:** `split_superres`  
**Values:** `train`, `test`

This protocol trains on county-aggregated labels and predicts ZCTA-level values. It tests within-county heterogeneity recovery: can the model disaggregate coarse spatial signals?

### Why Fixed Geography-Aware Splits?

The splits are geographically structured, not random. This is deliberate:

- **Comparability:** Everyone evaluates on the same held-out ZCTAs.
- **Spatial leakage prevention:** Random splits can leak geographic information through nearby places.
- **Test set integrity:** The test fold is reserved for final evaluation. Use validation folds for model selection and hyperparameter tuning.

If you need custom splits for your own experiments, the underlying data supports them. But if you report GeoRSCT benchmark results, use the provided splits so others can reproduce and compare.

```python
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_parquet("georsct_table.parquet")

train, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.1, random_state=42)
```

## Coverage

Not all ZCTAs have all 27 targets. Coverage flags indicate availability:

| Flag | Meaning |
|---|---|
| `has_cdc_places` | CDC PLACES target values are available |
| `has_income` | ACS income target is available |
| `has_home_value` | ACS home-value target is available |
| `has_cdc_ci` | CDC PLACES confidence intervals are available in the sidecar file |

Known target coverage from the initial release remains:

| Flag | True | False | Reason for missing |
|---|---:|---:|---|
| `has_cdc_places` | 31,529 (99.2%) | 260 (0.8%) | CDC PLACES does not model these ZCTAs |
| `has_income` | 31,471 (99.0%) | 318 (1.0%) | ACS suppression or missing estimate |
| `has_home_value` | 31,244 (98.3%) | 545 (1.7%) | ACS suppression or few owner-occupied units |

Environmental and physical-context targets (`target_elevation`, `target_tree_cover`, `target_night_lights`, `target_population_density`) have 100% coverage.

Missing values are stored as `NaN` in the parquet files. Use coverage flags to filter evaluation sets when needed.

## Geometry

ZCTA boundary polygons are derived from Census TIGER/Line 2022, reprojected to EPSG:4326 and simplified with 0.001-degree tolerance while preserving topology. The table parquet omits geometry for lightweight use.

Use the original Census TIGER/Line files for high-precision cartographic or legal boundary analysis.

## Data Sources and Licensing

All source data is public domain or open license:

| Source | License | URL |
|---|---|---|
| CDC PLACES 2023 | Public domain | https://www.cdc.gov/places/ |
| ACS 2022 5-Year | Public domain | https://data.census.gov/ |
| CDC/ATSDR SVI 2022 | Public domain | https://www.atsdr.cdc.gov/placeandhealth/svi/ |
| FEMA NFHL | Public domain | https://www.fema.gov/flood-maps/national-flood-hazard-layer |
| DHS HIFLD 2022 | Public domain | https://hifld-geoplatform.hub.arcgis.com/ |
| VIIRS Nighttime Lights | Public domain | https://eogdata.mines.edu/products/vnl/ |
| USGS National Elevation Dataset | Public domain | https://www.usgs.gov/the-national-map-data-delivery |
| Hansen Global Forest Change | CC-BY-4.0 | https://glad.earthengine.app/view/global-forest-change |
| Census TIGER/Line 2022 | Public domain | https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html |
| OpenStreetMap / OSRM-derived drive-time context | ODbL data / OSRM software license | https://www.openstreetmap.org/ and https://project-osrm.org/ |

If using or redistributing the OSRM-derived drive-time features, preserve appropriate OpenStreetMap attribution and verify that downstream use complies with OpenStreetMap data licensing requirements.

## Related Resources

- **RSCT theory preprint:** Rudolph A. Martin, *Intelligence as Representation-Solver Compatibility: A General Theory of Representation-Dependent Reasoning* (2026). [Zenodo](https://doi.org/10.5281/zenodo.18854651) | [SSRN](https://ssrn.com/abstract=6339299)
- **HHS/CDC PLACES ZCTA GIS-friendly datasets:** official source releases for CDC PLACES ZCTA estimates.
- **PLACES on Data.gov:** official CDC/HHS PLACES ZCTA data release.

GeoRSCT differs from these source releases by packaging a curated multi-target regression benchmark with fixed split assignments, ACS encoder features, spatial-lag context, geospatial enrichment features, environmental and socioeconomic targets beyond PLACES, simplified ZCTA geometries, coverage flags, uncertainty sidecars, and evaluation metadata designed for representation–solver compatibility experiments.

## Data Integrity Validation

The 21 CDC PLACES health targets were cross-validated against the official HHS/CDC PLACES ZCTA release.

| Check | Result |
|---|---|
| Value match: 21 columns over 31,529 common ZCTAs | **100% exact match, rho = 1.0, max_diff = 0.0** |
| 260 missing-CDC ZCTAs where `has_cdc_places=False` | All absent from HHS release |
| ZCTA ID validity | All GeoRSCT ZCTAs appear in HHS or are documented missing-CDC |

Full methodology and results are provided in `VALIDATION_CROSS_CHECK.md` and `validation_report.json` when included in the repository.

## Dataset Creation

### Curation Rationale

GeoRSCT was created to provide a standardized evaluation benchmark for geospatial regression and representation–solver compatibility studies. Existing public datasets such as CDC PLACES and ACS are released as source data, not as evaluation-ready artifacts. Researchers working on geospatial prediction must independently download, clean, join, and split these sources, which introduces inconsistencies that make cross-paper comparison difficult.

GeoRSCT packages this pipeline into a single reproducible artifact with fixed geography-aware evaluation protocols, expanded geospatial context features, and optional uncertainty sidecars.

All build steps, joins, and validation checks for this initial public release are specified in the open-source pipeline in the GeoRSCT GitHub repository ([https://github.com/NextShiftConsulting/georsct](https://github.com/NextShiftConsulting/georsct)), so the benchmark can be regenerated and audited end to end.

### Source Data

All source data is produced by U.S. federal agencies or established research programs:

- **CDC PLACES 2023:** Model-based small-area health estimates at the ZCTA level, derived from the Behavioral Risk Factor Surveillance System using multilevel regression and poststratification. Produced by the CDC Division of Population Health.
- **American Community Survey 2022 5-Year Estimates:** Demographic, economic, housing, and social characteristics from the U.S. Census Bureau's ongoing household survey.
- **CDC/ATSDR Social Vulnerability Index 2022:** Social vulnerability context measures used as enrichment features.
- **FEMA National Flood Hazard Layer:** Flood-zone polygons used to estimate ZCTA area fractions in selected flood-risk categories.
- **DHS HIFLD 2022 healthcare infrastructure data:** 8,013 hospitals (7,634 open) from HIFLD Open Data 2022 snapshot. Hospital, pharmacy, bed-count, and trauma-center access context.
- **OSRM / OpenStreetMap-derived routing:** Drive-time context to nearest hospital and county centroid.
- **VIIRS Nighttime Lights:** Satellite-derived nighttime radiance composites from the Visible Infrared Imaging Radiometer Suite.
- **USGS National Elevation Dataset:** Digital elevation model from the U.S. Geological Survey.
- **Hansen Global Forest Change:** Satellite-derived tree canopy cover estimates from Landsat imagery, produced by the University of Maryland.
- **Census TIGER/Line 2022:** ZCTA boundary shapefiles from the U.S. Census Bureau.

### Annotations

No human annotation was performed for this benchmark. All target labels are one of the following:

- **Model-based estimates:** 21 CDC PLACES health measures derived from BRFSS survey data via multilevel regression and poststratification.
- **Survey aggregates:** ACS income and home value estimates.
- **Remote-sensing or physical-context measurements:** elevation, tree cover, nighttime lights, and population density.

### Personal and Sensitive Information

This dataset contains **no individual-level data**. All values are ZCTA-level aggregates. CDC PLACES estimates are modeled from survey data that has already been anonymized and aggregated. ACS estimates are published by the Census Bureau with disclosure avoidance applied. No personally identifiable information is present or recoverable from the benchmark.

## Considerations for Using the Data

### Social Impact

GeoRSCT enables research on geospatial health prediction, spatial generalization, target difficulty, and representation–solver compatibility. Such research could support better evaluation methodology for public-health resource allocation, health-equity analysis, epidemiological modeling, and geospatial machine learning.

However, predictions from models trained on this data should not be used as substitutes for actual health surveillance, clinical decision-making, causal policy analysis, or allocation decisions without additional validation and domain review.

### Discussion of Biases

- **BRFSS sampling bias:** CDC PLACES labels are derived from a telephone survey using landline and cell-phone respondents. Populations with lower phone access may be underrepresented in the underlying BRFSS data, though multilevel regression and poststratification partially mitigate this.
- **ACS non-response and uncertainty:** ACS estimates for small ZCTAs can have wide margins of error. ACS margins of error are provided as a sidecar, but benchmark baselines do not automatically propagate them unless explicitly stated.
- **Geographic coverage bias:** Some ZCTAs are not modeled by CDC PLACES. These are flagged with `has_cdc_places=False`, but their exclusion means the health-target subset slightly underrepresents the smallest and most unusual communities.
- **Temporal snapshot:** The benchmark reflects a single time period. ACS features are based on the 2018–2022 5-Year Estimates, while CDC PLACES 2023 incorporates BRFSS survey years used by that release.
- **Aggregation bias:** ZCTA-level values may hide within-ZCTA variation. Results may change under different spatial units such as counties, census tracts, or grid cells.
- **Target-source dependence:** The 21 health targets share a single source and methodology, so cross-task similarities may partly reflect shared modeling assumptions.
- **Spatial-lag interpretation risk:** Spatial-lag features encode neighboring-area context and should not be treated as temporal lags. They are useful for spatial modeling but must be documented clearly to avoid confusion with forecasting or leakage claims.
- **Derived-feature uncertainty:** Enrichment features such as healthcare access, flood exposure, and drive time are derived features. Their precision depends on source-data currency, geocoding quality, routing assumptions, and spatial overlay choices.

### Known Limitations

1. **CDC PLACES labels are modeled estimates, not direct measurements.** Cross-task correlations are partially methodological, not only epidemiological.
2. **Uncertainty sidecars are optional.** ACS margins of error and CDC confidence intervals are provided, but benchmark baselines do not automatically propagate them unless explicitly stated.
3. **Most targets share one source.** 21 of 27 tasks come from CDC PLACES, so apparent multi-task structure may reflect shared source methodology.
4. **Geometry is simplified.** The 0.001-degree simplification reduces file size but is not intended for high-precision boundary analysis.
5. **U.S.-only coverage.** Results may not generalize to other countries with different health systems, demographics, governance, and geographic data infrastructure.
6. **ZCTA-level scope.** ZCTAs are useful but imperfect administrative/geographic units. They should not be treated as the unique or “true” geography of the underlying phenomena.
7. **Low performance is not automatically solver failure.** It may reflect target difficulty, feature insufficiency, aggregation effects, spatial leakage, or measurement noise.
8. **Spatial-lag features are not temporal features.** They represent neighbor context and must be interpreted as such.
9. **Enrichment features are not causal controls.** They provide additional context but should not be interpreted as proving causal mechanisms.

## Additional Information

### Dataset Curators

GeoRSCT was curated by Next Shift Consulting as part of the RSCT (**Representation–Solver Compatibility Theory**) research program.

### Licensing

GeoRSCT is released under **CC-BY-4.0**.

| Source | License | Notes |
|---|---|---|
| CDC PLACES 2023 | Public domain under 17 USC 105 | U.S. government work. Data.gov may list ODbL as a platform default. |
| ACS 2022 5-Year | Public domain under 17 USC 105 | U.S. Census Bureau, U.S. government work. |
| CDC/ATSDR SVI 2022 | Public domain under 17 USC 105 | U.S. government work. |
| FEMA NFHL | Public domain under 17 USC 105 | U.S. government work. |
| DHS HIFLD 2022 | Public domain under 17 USC 105 | U.S. government work. HIFLD Open Data 2022 snapshot. |
| VIIRS Nighttime Lights | Public domain under 17 USC 105 | NASA/NOAA, U.S. government work. |
| USGS National Elevation Dataset | Public domain under 17 USC 105 | U.S. Geological Survey, U.S. government work. |
| Census TIGER/Line 2022 | Public domain under 17 USC 105 | U.S. Census Bureau, U.S. government work. |
| Hansen Global Forest Change | CC-BY-4.0 | University of Maryland. Most restrictive source license. |
| OpenStreetMap-derived routing features | ODbL attribution may apply | Preserve appropriate OSM attribution for derived routing context. |

**Why CC-BY-4.0:** Most source data are U.S. federal government works, which are public domain under 17 USC 105. The Hansen Global Forest Change dataset requires CC-BY-4.0, making it the binding minimum for this derivative benchmark. GeoRSCT adopts CC-BY-4.0 as the most restrictive source license.

### Changelog

| Version | Date | Changes |
|---|---|---|
| 23.0.2 | 2026-05-03 | Initial public release. 31,789 ZCTAs, 27 targets, 63 solver-usable input features (33 ACS, 14 spatial-lag, 16 geospatial enrichment), 3 geography-aware evaluation protocols, simplified ZCTA geometries, CDC PLACES confidence-interval sidecar, ACS margin-of-error sidecar, updated schema metadata, and expanded Croissant metadata. |

Version scheme: `{PLACES_vintage}.{release}`. For example, `23.0.2` means PLACES 2023, second internal build and first public release.

To pin a specific version in code:

```python
from datasets import load_dataset

ds = load_dataset("rudymartin/georsct", revision="v23.0.2")
```

### Citation

If you use the RSCT framing or representation–solver compatibility terminology, please also cite the theory preprint:

```bibtex
@misc{martin2026rsct,
  title        = {Intelligence as Representation-Solver Compatibility: A General Theory of Representation-Dependent Reasoning},
  author       = {Martin, Rudolph A.},
  year         = {2026},
  note         = {Preprint. Zenodo DOI: 10.5281/zenodo.18854651; SSRN: 6339299}
}
```

If you use the GeoRSCT dataset, please cite:

```bibtex
@dataset{georsct2026v23002,
  title        = {GeoRSCT: A Geospatial Regression Benchmark for Representation--Solver Compatibility},
  author       = {Martin, Rudolph A.},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/datasets/rudymartin/georsct},
  version      = {23.0.2},
  note         = {31,789 U.S. ZCTAs, 63 solver-usable input features, 27 regression targets, fixed geography-aware evaluation protocols, and uncertainty sidecars}
}
```
