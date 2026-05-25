# GeoCert Benchmark Data

27 geospatial regression tasks across 31,789 US ZIP Code Tabulation Areas (ZCTAs).

## Tasks

| Domain | Count | Source | Labels |
|--------|-------|--------|--------|
| Health | 21 | CDC PLACES 2023 | Model-based estimates (NOT direct measurements) |
| Socioeconomic | 4 | ACS 2022 5-Year, VIIRS, Census | Survey + remote sensing |
| Environmental | 2 | Hansen, USGS NED | Remote sensing + survey |

## S3 Artifacts

Bucket: `s3://swarm-yrsn-datasets/rsct_curriculum/series_018/processed/`

| File | Size | Description |
|------|------|-------------|
| `zcta_features_labels.parquet` | ~15 MB | Full dataset: 33 ACS features + 27 target labels |
| `geocert_splits.parquet` | 334 KB | Split assignments + coverage flags (standalone) |
| `cdc_places_coverage.json` | ~10 KB | Per-ZCTA CDC PLACES coverage manifest |
| `geocert_splits_provenance.json` | ~2 KB | Evaluation protocol descriptions |
| `sentinel_fix_provenance.json` | ~2 KB | ACS sentinel cleanup audit trail |

## Evaluation Protocols

Three evaluation protocols, each with held-out test sets:

| Protocol | Split Column | Folds | What It Tests |
|----------|-------------|-------|---------------|
| **Imputation** | `split_imputation` | 5-fold CV + test | County holdout — geographic interpolation |
| **Extrapolation** | `split_extrapolation` | 4-fold CV + test | State holdout — distribution shift |
| **Super-resolution** | `split_superres` | valid + test | County→ZCTA downscaling |

## Coverage

| Flag | Complete | Missing | Why Missing |
|------|----------|---------|-------------|
| `has_cdc_places` | 31,529 (99.2%) | 260 | CDC PLACES doesn't model these ZCTAs |
| `has_income` | 31,471 (99.0%) | 318 | ACS suppression (small population) |
| `has_home_value` | 31,244 (98.3%) | 545 | ACS suppression (few owner-occupied units) |

Environmental targets (elevation, tree cover, night lights, population density) have 100% coverage.

## Data Pipeline Scripts

In `v23/`. Run in order to reproduce the benchmark from scratch:

| Script | Purpose |
|--------|---------|
| `fix_acs_sentinels.py` | Replace Census sentinel values (-666666666) with NaN |
| `audit_cdc_places_coverage.py` | Audit + document CDC PLACES coverage per ZCTA |
| `build_benchmark_splits.py` | Build canonical split index with coverage flags |
| `build_geoparquet.py` | Join ZCTA boundaries with features/labels/splits |
| `build_release_package.py` | Build full release package (table parquet, schema, checksums) |
| `validate_against_hhs.py` | Cross-validate against official HHS/CDC PLACES on HF |

All scripts are idempotent with `--dry-run` support and S3 provenance logging.

## Licensing

All source data is public domain or open license:

| Source | License |
|--------|---------|
| CDC PLACES | Public domain (US government work) |
| ACS (Census Bureau) | Public domain (US government work) |
| VIIRS (NASA/NOAA) | Public domain (US government work) |
| USGS NED | Public domain (US government work) |
| Hansen Global Forest Change | CC-BY-4.0 |
| Census TIGER/Line | Public domain (US government work) |

## Known Limitations

1. **CDC PLACES labels are modeled estimates**, not direct measurements. They are small-area estimates derived from BRFSS survey data using multilevel regression and poststratification.
2. **ACS margins of error** are not propagated. Small ZCTAs (pop < 500) have wide confidence intervals.
3. **21 of 27 tasks share a single data source** (CDC PLACES). Cross-task correlations are partially methodological, not just epidemiological.
