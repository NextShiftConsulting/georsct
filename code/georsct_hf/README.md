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
  - config_name: cdc_ci
    data_files:
      - split: benchmark
        path: cdc_places_ci.parquet
  - config_name: acs_moe
    data_files:
      - split: benchmark
        path: zcta_acs_margins_of_error.parquet
  - config_name: noaa_long
    data_files:
      - split: benchmark
        path: noaa_storm_events_long.parquet
---

# GeoRSCT

**A geospatial regression benchmark for evaluating representation–solver compatibility.**

GeoRSCT is a benchmark and evaluation framework for studying when geospatial model performance reflects solver quality versus target difficulty, spatial leakage, aggregation effects, scale sensitivity, or representation–solver mismatch.

This release (version **24.0.1**) includes 31,789 U.S. ZIP Code Tabulation Areas (ZCTAs), 106 columns spanning 33 ACS features, 37 geospatial enrichment features, 27 regression targets, metadata, split assignments, and optional geometry. Three geography-aware evaluation protocols are included.

The solver-usable features include:

1. **33 ACS 2022 5-Year features** describing demographic, economic, housing, and social characteristics.
2. **37 geospatial enrichment features** covering social vulnerability (SVI), flood zones (FEMA), flood history (NOAA/NFIP), terrain hydrology (TWI), hospital/pharmacy access (HIFLD), and drive-time context.

Spatial-lag features are computed at runtime from the adjacency matrix and are not stored in the parquet.

The GeoRSCT data pipeline, build scripts, and validation notebooks are maintained in the companion GitHub repository at [https://github.com/NextShiftConsulting/georsct](https://github.com/NextShiftConsulting/georsct).

## Theoretical Foundation

GeoRSCT is the geospatial benchmark artifact for **Representation–Solver Compatibility Theory (RSCT)**.

The underlying RSCT theory is introduced in:

> Rudolph A. Martin. **Intelligence as Representation-Solver Compatibility: A General Theory of Representation-Dependent Reasoning.** Preprint, March 3, 2026. [Zenodo](https://doi.org/10.5281/zenodo.18854651) 
RSCT argues that problem difficulty is not intrinsic to the problem alone. Instead, observed performance depends on the relationship among the problem, the encoding, and the solver:

```text
D = D(P, E, S)
```

GeoRSCT applies that theory to geospatial regression. The benchmark is designed to help determine whether observed performance reflects solver quality, target difficulty, spatial leakage, aggregation effects, scale sensitivity, or representation–solver incompatibility.

## Dataset Summary

GeoRSCT is an evaluation-ready ZCTA-level benchmark derived from public geospatial, public-health, socioeconomic, infrastructure, and environmental data sources. It is not a mirror of CDC PLACES. Instead, it curates 27 regression targets from multiple sources, 70 solver-usable input features (33 ACS + 37 enrichment), ZCTA boundary geometries, coverage flags, uncertainty sidecars, and three fixed evaluation split protocols for studying spatial generalization, target recoverability, and representation–solver compatibility.

GeoRSCT is designed for **evaluation diagnosis**, not only model ranking. A model score on this benchmark should be interpreted in relation to the target, the representation, the spatial split, and the solver family.

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
| `georsct_table.parquet` | 17.8 MB | Main table: 31,789 ZCTAs x 106 columns (no geometry) |
| `georsct_simplified_001.geoparquet` | ~66 MB | Same + ZCTA boundary polygons (EPSG:4326, 0.001 deg simplified) |
| `cdc_places_ci.parquet` | 1.8 MB | CDC PLACES 95% CI sidecar: 44 fields |
| `zcta_acs_margins_of_error.parquet` | 7.9 MB | ACS 5-year MOE sidecar: 33 fields |
| `noaa_storm_events_long.parquet` | 1.1 MB | NOAA flood history 1996-2024: year-level rows for temporal experiments |

## Field Schema Overview

GeoRSCT is distributed as a row-per-ZCTA tabular benchmark (31,789 rows, one per 2020-vintage ZCTA in CONUS). ZCTAs are Census Bureau statistical areas that approximate USPS ZIP code service areas; multiple ZIP codes can map to the same ZCTA, and the mapping is not one-to-one. Each row includes identifiers, centroid coordinates, 33 ACS features, 37 geospatial enrichment features, 27 target labels, coverage flags, and geography-aware split assignments. Spatial-lag features are computed at runtime and are not stored in the parquet.

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

By default, `feature_columns(df)` returns the 70 solver-usable v24.0.1 features: 33 ACS + 37 enrichment. For an ACS-only baseline, filter to columns beginning with `acs_`.

### 4. Load with geometry

```python
import geopandas as gpd
geo = gpd.read_parquet("georsct_simplified_001.geoparquet")
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

# NOAA flood history sidecar (1996-2024, year-level rows for Experiment 1)
noaa_long = pd.read_parquet("noaa_storm_events_long.parquet")
# filter to temporal snapshots: 2018 (pre-Florence), 2019 (recovery), 2020 (pre-Isaias)
noaa_snap = noaa_long[noaa_long["year"].isin([2018, 2019, 2020])]
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
    or c.startswith("svi_")
    or c.startswith("flood_")
    or c.startswith("nfip_")
    or c.startswith("twi_")
    or c.startswith("slope_")
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

### Socioeconomic Targets

### Environmental Targets

## Input Features

GeoRSCT v24.0.1 includes 70 solver-usable input features: 33 ACS and 37 geospatial enrichment.

### 1. ACS Features

The 33 ACS features are prefixed with `acs_`. They include demographic, economic, housing, transportation, and social characteristics at the ZCTA level.

These features are the default ACS-only encoder/input representation and remain useful for controlled baselines.

### 2. Spatial-Lag Features

Spatial-lag features (`lag_acs_*`) are **not stored in the parquet**. They are computed at runtime from `zcta_adjacency.parquet` using queen-contiguity weighted neighbor means. To include them in training, call `compute_spatial_lags(df, acs_cols, adjacency)` from the build pipeline.

### 3. Geospatial Enrichment Features

The 37 enrichment features include (prefixes: `svi_`, `flood_`, `nfip_`, `twi_`, `slope_`, `hifld_`, `drive_`):

The expanded v24.0.1 feature set is intended for stronger geospatial solver evaluation and representation–solver compatibility experiments. Reported benchmark comparisons should document whether they use ACS-only (33) or full (70) features.

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

Target coverage:

Environmental and physical-context targets (`target_elevation`, `target_tree_cover`, `target_night_lights`, `target_population_density`) have 100% coverage.

Missing values are stored as `NaN` in the parquet files. Use coverage flags to filter evaluation sets when needed.

## Geometry

ZCTA boundary polygons are derived from Census TIGER/Line 2022, reprojected to EPSG:4326 and simplified with 0.001-degree tolerance while preserving topology. The table parquet omits geometry for lightweight use.

Use the original Census TIGER/Line files for high-precision cartographic or legal boundary analysis.

## Data Sources and Licensing

All source data is public domain or open license:

If using or redistributing the OSRM-derived drive-time features, preserve appropriate OpenStreetMap attribution and verify that downstream use complies with OpenStreetMap data licensing requirements.

## Related Resources

- **RSCT theory preprint:** Rudolph A. Martin, *Intelligence as Representation-Solver Compatibility: A General Theory of Representation-Dependent Reasoning* (2026). [Zenodo](https://doi.org/10.5281/zenodo.18854651) - **HHS/CDC PLACES ZCTA GIS-friendly datasets:** official source releases for CDC PLACES ZCTA estimates.
- **PLACES on Data.gov:** official CDC/HHS PLACES ZCTA data release.

GeoRSCT differs from these source releases by packaging a curated multi-target regression benchmark with fixed split assignments, ACS encoder features, spatial-lag context, geospatial enrichment features, environmental and socioeconomic targets beyond PLACES, simplified ZCTA geometries, coverage flags, uncertainty sidecars, and evaluation metadata designed for representation–solver compatibility experiments.

## Data Integrity Validation

The 21 CDC PLACES health targets were cross-validated against the official HHS/CDC PLACES ZCTA release.

Full methodology and results are provided in `VALIDATION_CROSS_CHECK.md` and `validation_report.json` when included in the repository.

## Dataset Creation

### Curation Rationale

GeoRSCT was created to provide a standardized evaluation benchmark for geospatial regression and representation–solver compatibility studies. Existing public datasets such as CDC PLACES and ACS are released as source data, not as evaluation-ready artifacts. Researchers working on geospatial prediction must independently download, clean, join, and split these sources, which introduces inconsistencies that make cross-paper comparison difficult.

GeoRSCT packages this pipeline into a single reproducible artifact with fixed geography-aware evaluation protocols, expanded geospatial context features, and optional uncertainty sidecars.

All build steps, joins, and validation checks are specified in the open-source pipeline in the GeoRSCT GitHub repository ([https://github.com/NextShiftConsulting/georsct](https://github.com/NextShiftConsulting/georsct)), so the benchmark can be regenerated and audited end to end.

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

**Why CC-BY-4.0:** Most source data are U.S. federal government works, which are public domain under 17 USC 105. The Hansen Global Forest Change dataset requires CC-BY-4.0, making it the binding minimum for this derivative benchmark. GeoRSCT adopts CC-BY-4.0 as the most restrictive source license.

### Changelog

Version scheme: `{major}.{minor}.{patch}`. v24.0.1 = first public release with v24 enrichment layers (NOAA, NFIP, TWI).

To pin a specific version in code:

```python
from datasets import load_dataset

ds = load_dataset("rudymartin/georsct", revision="v24.0.1")
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
@dataset{georsct2026v24001,
  title        = {GeoRSCT: A Geospatial Regression Benchmark for Representation--Solver Compatibility},
  author       = {Martin, Rudolph A.},
  year         = {2026},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/datasets/rudymartin/georsct},
  version      = {24.0.1},
  note         = {31,789 U.S. ZCTAs, 70 solver-usable input features (33 ACS + 37 enrichment), 27 regression targets, NOAA/NFIP/TWI flood layers, fixed geography-aware evaluation protocols, and uncertainty sidecars}
}
```
