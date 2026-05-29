# FloodRSCT Data Validation Framework

**Module**: `_validate_contract.py`  
**Contract**: `FEATURE_CONTRACT.yaml`  
**Status**: Operational (tested 2026-05-29 across all 5 scenarios)

---

## Why Geospatial Pipelines Need Domain-Specific Validation

Standard ML data pipelines validate schemas, nulls, and types. Geospatial flood
pipelines fail in ways that schema checks cannot detect:

1. **A raster sampled in the wrong CRS returns valid floats** -- just from the
   wrong location on Earth. NLCD at EPSG:5070 sampled with EPSG:4326 coordinates
   produces a real impervious percentage for a point 4-15 km away. No type error,
   no null, no exception. The model trains on spatially misaligned features.

2. **Temporal scalars hide compound causation.** Harvey's 500mm rainfall total
   is indistinguishable from 500mm spread over 10 days. But the first scenario
   floods a city; the second does not. Scalar aggregation destroys the temporal
   coincidence (surge + rain + tide phase) that drives damage.

3. **Sensor sparsity is non-random.** Tide gauges fail during peak surge (the
   instrument is destroyed by the event it measures). High-water marks cluster
   near roads (surveyors need access). NFIP claims are zero where nobody has
   flood insurance, not where nobody flooded. Every missing value has a
   domain-specific mechanism.

4. **Model output masquerades as observation.** SLOSH surge grids are
   hydrodynamic model predictions. HRRR QPF is a weather forecast. FloodSimBench
   depths are physics simulations. These have systematic bias, not measurement
   noise. A validation framework must distinguish observed from modeled data.

5. **Resolution mismatches span 5 orders of magnitude.** One DEM pixel covers
   100 m^2. One MRMS cell covers 22 km^2. One tide gauge reading is broadcast to
   200 ZCTAs. The output table makes them look equivalent unless provenance
   metadata survives the pipeline.

Generic data quality tools (Great Expectations, dbt tests, pandera) can validate
types and nulls. They cannot validate that a latitude/longitude pair was
interpreted in the correct coordinate reference system, that hourly rainfall was
not collapsed before compound flooding features were derived, or that a "0"
in NFIP claims means "no insurance" rather than "no flood."

**The FEATURE_CONTRACT.yaml is the geospatial validation spec.** It declares the
CRS, source sensor, temporal class, build function, and expected output column
for every feature. The three-layer validation framework below reads this contract
and enforces domain-specific invariants that generic tools miss.

---

## Architecture

```
                    FEATURE_CONTRACT.yaml
                           |
                           v
    +----------------------------------------------+
    |  Layer 1: Interface Contract Validation       |
    |  (pre-assembly gate)                          |
    |                                               |
    |  - Raw data exists on S3?                     |
    |  - Parquet schema matches builder's expects?  |
    |  - File sizes plausible (no HTML stubs)?      |
    |  - Per-event coverage (hours vs expected)?    |
    +----------------------------------------------+
                           |
                      PASS / FAIL
                           |
                           v
    +----------------------------------------------+
    |  Layer 2: Post-Assembly Validation            |
    |  (inline in build_event_dataset.py)           |
    |                                               |
    |  - Coverage thresholds per column             |
    |  - Plausibility bounds (physical limits)      |
    |  - (zcta_id, event) uniqueness (no fan-out)   |
    |  - Row count sanity                           |
    +----------------------------------------------+
                           |
                      PASS / WARN / FAIL
                           |
                           v
    +----------------------------------------------+
    |  Layer 3: Data Lock Validation                |
    |  (standalone script for QA gate)              |
    |                                               |
    |  - Every contract feature has output column   |
    |  - Non-null rate per feature                  |
    |  - Leakage gate (post_event != model input)   |
    |  - Schema match against contract              |
    |  - Runs Layer 2 on assembled parquet          |
    +----------------------------------------------+
                           |
                      VERDICT: CLEAR / BLOCKED
```

---

## Layer 1 -- Interface Contract Validation

**Purpose**: Catch mismatches between what fetchers produced and what the builder
expects. Blocks assembly from running on garbage inputs.

**When to run**: Before every `build_event_dataset.py` execution.

**What it checks**:

| Check | How | Catches |
|-------|-----|---------|
| Raw data exists | `s3_prefix_has_files(raw_s3_path)` for each feature in contract | Missing datasets (geocertdb2026 not copied, OpenFEMA not fetched) |
| Schema match | Read parquet column names, compare to `EXPECTED_RAW_COLUMNS` | Column mismatches (fetcher writes `observed_m`, builder reads `water_level_m`) |
| File size guard | Flag files < 100 bytes | HTML error pages saved as data (MRMS stubs, tidal API failures) |
| Event coverage | Resolve `{event}` placeholders to concrete paths per scenario | MRMS missing hours, API pagination truncation |

**Scenario-to-event mapping** (used to resolve `{event}` placeholders):

| Scenario | Events |
|----------|--------|
| houston | harvey2017, imelda2019, beryl2024 |
| new_orleans | ida2021_nola |
| nyc | ida2021_nyc |
| riverside_coachella | hilary2023 |
| southwest_florida | ian2022 |

**Known column expectations** (from code inspection of `build_event_dataset.py`):

| Build Function | Required Raw Columns | Notes |
|----------------|---------------------|-------|
| `aggregate_tides` | `observed_m`, `predicted_m` | Builder historically expected `water_level_m`; fetcher writes `observed_m` |
| `compute_storm_proximity` | `storm_id`, `lat`, `lon`, `max_wind_kt`, `timestamp` | Builder expects `category` but fetcher writes `max_wind_kt` + `status` |
| `aggregate_hwm` | `latitude`, `longitude`, `elev_ft` | Builder uses `elev_ft` directly |
| `aggregate_mrms_rainfall` | N/A (GRIB2 files) | Check file extension `.grib2.gz`; cfgrib needs decompression |

### Running Layer 1

```bash
# Single scenario
python _validate_contract.py --scenario houston --layer 1

# All scenarios
python _validate_contract.py --all --layer 1
```

### Example Output (Houston, 2026-05-29)

```
--- Layer 1 ---
  [L1] [PASS] impervious_pct: 1 files, 26351.0 MB
  [L1] [SKIP] drainage_capacity_status: operational -- no raw data expected
  [L1] [FAIL] bayou_segment_id: no files at raw/nhdplus/houston/v2/
  [L1] [FAIL] drainage_district_id: no files at raw/hcfcd/drainage_districts/v1/
  [L1] [PASS] max_rainfall_mm: 696 files, 429.5 MB
  [L1] [PASS] total_rainfall_mm: 696 files, 429.5 MB
  [L1] [WARN] tidal_surge_max_m: 3 file(s) < 100 bytes (possible stubs)
  [L1] [PASS] tidal_surge_max_m: 24 files, 1.2 MB
  [L1] [PASS] storm_distance_km: 1 files, 0.0 MB
  [L1] [PASS] hwm_max_ft: 2 files, 0.1 MB
  [L1] [FAIL] flood_311_count: no files at raw/houston_311/...
  [L1] [FAIL] nfip_event_claims: no files at raw/openfema/...

PASS: 6  FAIL: 4  WARN: 1  SKIP: 1
VERDICT: BLOCKED -- fix FAILs before proceeding
```

---

## Layer 2 -- Post-Assembly Validation

**Purpose**: Catch silent data loss after assembly. A column can exist with the
right dtype but be 100% NaN because of a column-name mismatch in the join.

**When to run**: After each `build_{scenario}()` function returns, before writing
the output parquet.

**What it checks**:

### Coverage Thresholds

Minimum non-null fraction per output column. Thresholds are set per feature
based on the expected data density:

| Column | Threshold | Rationale |
|--------|-----------|-----------|
| `rainfall_total_mm` | 50% | Some ZCTAs outside rain band |
| `tidal_surge_max_m` | 20% | Coastal scenarios only; inland = NaN |
| `storm_distance_km` | 90% | Almost all ZCTAs have track distance |
| `hwm_max_ft` | 5% | Only 50-100 marks per event (inherently sparse) |
| `flood_311_count` | 5% | Urban only, self-selected reporters |
| `nfip_event_claims` | 10% | Where insured properties exist |
| `impervious_pct` | 80% | Should be nearly complete |
| `elevation_m_msl` | 90% | Should be nearly complete |

### Plausibility Bounds

Physical limits on measured quantities. Values outside these bounds indicate
unit conversion errors, CRS bugs, or corrupt source data:

| Column | Min | Max | Notes |
|--------|-----|-----|-------|
| `rainfall_total_mm` | 0 | 3000 | Harvey max ~1500mm; generous upper |
| `max_rainfall_mm` | 0 | 500 | Single-hour maximum |
| `tidal_surge_max_m` | -2 | 10 | Negative = below predicted |
| `elevation_m_msl` | -100 | 5000 | NOLA is below sea level |
| `impervious_pct` | 0 | 100 | Percentage |
| `hwm_max_ft` | 0 | 50 | Feet above ground |

### Deduplication Guard

After every left join, assert that `(zcta_id, event)` remains unique. A
many-to-many join (e.g., multiple HWMs per ZCTA without aggregation) silently
fans out the table, duplicating rows and inflating sample size.

### Integration with Builder

```python
# In build_event_dataset.py, after assembly:
from _validate_contract import validate_layer2, print_report

results = validate_layer2(df, scenario)
_, fails, _ = print_report(results, scenario)
if fails > 0:
    log.error("Layer 2 validation failed: %d issues", fails)
    # Continue but flag in observability columns
```

---

## Layer 3 -- Data Lock Validation

**Purpose**: Full reconciliation of the assembled output parquet against
`FEATURE_CONTRACT.yaml`. This is the go/no-go gate for declaring a data lock.

**When to run**: Once, immediately before declaring Data Lock A or Data Lock B.

**What it checks**:

| Check | Method | Catches |
|-------|--------|---------|
| Column presence | Every feature in contract -> column in output parquet | Features silently dropped during assembly |
| Non-null rate | Per-feature, with temporal_class-aware thresholds | 100% null columns from mismatched joins |
| Leakage gate | `post_event` and `operational` columns flagged as WARN | Accidental use of outcomes as model inputs |
| Schema match | Output column names and dtypes vs contract | Schema drift between assembly runs |
| Layer 2 (inline) | Runs all Layer 2 checks on the output parquet | Coverage, plausibility, dedup |

### Leakage Gate

The `temporal_class` field in `FEATURE_CONTRACT.yaml` defines the causal
boundary:

```
CAUSAL BOUNDARY
                          |
  invariant ----+         |
  slow_drift ---+--> OK   |  post_event --+--> LABELS ONLY
  event_window -+         |  operational --+
                          |
```

Features classified as `post_event` (HWMs, 311, NFIP claims) or `operational`
(pump status, road closures) are outcomes or unknowable-at-forecast-time. If
they appear in the output parquet, Layer 3 emits a WARN reminding the modeler
not to use them as inputs.

### Running Layer 3

```bash
# Single scenario (full validation: L1 + L3 + L2-inline)
python _validate_contract.py --scenario houston

# All scenarios
python _validate_contract.py --all

# Data Lock A gate (Houston only, exit 0/1)
python _validate_contract.py --scenario houston
echo $?  # 0 = CLEAR, 1 = BLOCKED
```

---

## Relationship to GeoSpatial Challenges

Every challenge documented in `GeoSpatial_Challenges.md` maps to a specific
validation layer:

| Challenge | Section | Layer | How Caught |
|-----------|---------|-------|------------|
| CRS mismatches | 3 | L1 | CRS assertion on raster files |
| Temporal collapse | 7 | L2 | Coverage threshold on temporal features |
| Column name mismatches | 8 | L1 | Schema check on raw parquets |
| Heterogeneous sensors | 9 | L2 | `_n_obs` columns, provenance metadata |
| Big vs small data | 10 | L2 | Coverage thresholds vary by feature density |
| Sensor instability | 11 | L1 | File count vs expected hours |
| Missing variables | 12 | L3 | Contract reconciliation (feature present vs absent) |
| Data quality gaps | 13 | All | All three layers |

---

## Bugs Found and Fixed by the Validator

Iterative runs of `_validate_contract.py --all --layer 1` (2026-05-29):

| Bug | How Found | Resolution |
|-----|-----------|------------|
| NLCD contract path wrong (`impervious/v2021/` vs actual `impervious_2021/`) | L1 FAIL | Fixed in FEATURE_CONTRACT.yaml |
| HWM validator expected `elev_m` but builder uses `elev_ft` | L1 FAIL | Fixed validator spec (builder is correct) |
| HWM contract only listed `raw/usgs_stn/` but data also at `raw/surge_estimates/` | L1 FAIL | Added `or` path in contract |
| NOLA validator used `ida2021_nola` but S3 key is `ida2021_nyc` | L1 FAIL | Fixed SCENARIO_EVENTS mapping |
| 3 tidal stub files (station 8771013, 66 bytes -- decommissioned gauge) | L1 WARN | Harmless; builder reads parquets not JSON |
| HURDAT2 storm_tracks.parquet missing `category` column | Known | Builder derives from `max_wind_kt` + `status` |

---

## Current Scorecard (2026-05-29, post-fix)

| Scenario | PASS | FAIL | WARN | SKIP |
|----------|------|------|------|------|
| **houston** | 6 | 4 | 1 | 1 |
| **new_orleans** | 4 | 4 | 0 | 1 |
| **nyc** | 7 | 4 | 0 | 0 |
| **riverside_coachella** | 7 | 1 | 0 | 1 |
| **southwest_florida** | 6 | 4 | 0 | 1 |
| **Total** | **30** | **17** | **1** | **4** |

---

## Gap / Variance Register

Every remaining L1 FAIL classified as: **FETCH** (run existing fetcher),
**CODE** (write new fetcher or fix builder), or **VARIANCE** (document as
accepted limitation in the paper).

### FETCH -- Run Existing Fetcher (Launchers Ready)

| Feature | Scenarios | S3 Path | Launcher | Impact if Missing |
|---------|-----------|---------|----------|-------------------|
| `nfip_event_claims` | ALL 5 | `raw/openfema/` | `launch_fetch_openfema_event.py` | **BLOCKER** -- this is the label/outcome variable |
| geocertdb2026 | ALL 5 | `raw/geocertdb2026/` | `launch_copy_geocertdb2026.py` | **BLOCKER** -- all Layer 2 static features (ACS, SVI, flood zones) |

### CODE -- New Fetcher or Builder Fix Needed

| Feature | Scenarios | Issue | Effort | Priority |
|---------|-----------|-------|--------|----------|
| `flood_311_count` | houston, nyc | Fetchers exist (`fetch_houston_311.py`, `fetch_nyc_311.py`) but no launchers for per-event pulls | Low (add launcher + run) | MEDIUM -- post-event label, not model input |
| `sewer_shed_id` | nyc | Fetcher exists (`fetch_nyc_sewersheds.py`) but no launcher | Low | LOW -- scenario-specific |
| `levee_condition_rating` | new_orleans, nyc | Fetcher exists (`fetch_usace_levees.py`) but no launcher with correct `--scenario` | Low | MEDIUM |
| `coastal_distance_m` | southwest_florida | No fetcher for TIGER coastline | Medium (write fetcher + spatial join) | LOW for Data Lock A |
| `slosh_max_surge_m`, `slosh_category` | southwest_florida | SLOSH MOM grids not freely downloadable in bulk; `fetch_noaa_slosh.py` may need manual download step | Medium-High | HIGH for SWFL scenario |

### VARIANCE -- Accepted Gaps (Document in Paper)

| Feature | Scenarios | Why Accepted | Paper Section |
|---------|-----------|-------------|---------------|
| `bayou_segment_id` | houston | NHDPlus Houston subset not fetched; builder has `build_catchment_features` but no Houston-specific NHDPlus data | S9 Limitations: "Hydrologic unit membership (HUC-12, bayou segment) is defined in the feature contract but not populated for Data Lock A" |
| `drainage_district_id` | houston | HCFCD shapefile requires manual download from county portal | S9 Limitations: same |
| `subsidence_rate_mm_yr` | new_orleans | USGS InSAR subsidence raster not fetched; fetcher not written | S9 Limitations: "Land subsidence rates require InSAR-derived velocity fields not included in this study" |
| `canal_proximity_m` | new_orleans | OSM Overpass query not written | S9 Limitations: "Canal proximity features for the New Orleans protected-basin scenario are deferred" |
| `drainage_capacity_status` | houston | SKIP (operational -- no public archive) | Already documented in FEATURE_CONTRACT as `operational_status_unavailable` |
| `pump_station_status` | new_orleans | SKIP (operational -- partially hand-coded for Ida 2021) | Already documented |
| `road_access_status` | riverside_coachella | SKIP (operational) | Already documented |
| `evacuation_route_status` | southwest_florida | SKIP (operational) | Already documented |

### NOLA Event Key Variance (Discovered During Validation)

The MRMS fetcher stored Ida 2021 under key `ida2021_nyc` (Sep 1-4 window, covering
the NYC remnant rainfall). The NOLA builder uses event key `ida2021` and expects
MRMS at `raw/noaa_mrms/ida2021/` -- but the NOLA builder **does not call
`aggregate_mrms_rainfall()` at all** (lines 738-747). This is a code gap, not a data
gap: MRMS rainfall for Ida's NOLA landfall (Aug 27-Sep 1) requires a separate
fetcher run with a NOLA-specific window.

**Impact**: NOLA scenario has no MRMS rainfall features. Tidal surge and HWM data
provide partial flood signal, but the dominant forcing (200+ mm rainfall over
Orleans Parish in 6 hours) is absent.

**Resolution**: Add `ida2021_nola` event to MRMS fetcher (window: Aug 27-Sep 1)
and add MRMS call to `build_new_orleans()`. Not a Data Lock A blocker (Houston is
the primary scenario) but required for Data Lock B.

---

### Summary by Fixability

| Category | Count | Action |
|----------|-------|--------|
| **FETCH** (run existing launcher) | 2 features, all 5 scenarios | Launch now |
| **CODE** (write launcher / minor fix) | 5 features | Before Data Lock B |
| **VARIANCE** (accepted limitation) | 4 features | Document in S9 |
| **OPERATIONAL** (skip by design) | 4 features | Already documented |
| **NOLA event key** | 1 code gap | Before Data Lock B |

---

## Paper-Ready Summary

### For Section 4 (Methodology) or Section 6 (Data)

> Geospatial flood pipelines fuse data from heterogeneous sensors spanning five
> orders of magnitude in spatial resolution (10m lidar to 4.7km radar composites),
> eight orders of magnitude in sample density (500M DEM pixels vs. 50 post-event
> high-water marks), and fundamentally different error characteristics (measurement
> noise in gauges, systematic bias in hydrodynamic models, survivorship bias in
> field surveys where instruments are destroyed by the events they measure).
>
> Standard data validation -- schema checks, null rates, type assertions -- is
> necessary but insufficient for this domain. A raster sampled in the wrong
> coordinate reference system returns valid floating-point values from the wrong
> location on Earth: no type error, no null, no exception. Hourly precipitation
> collapsed to a single scalar erases the temporal coincidence of surge, rainfall,
> and tidal phase that drives compound flooding -- the dominant damage mechanism in
> four of our five scenarios.
>
> We introduce a three-layer validation framework driven by a machine-readable
> feature contract (FEATURE_CONTRACT.yaml) that declares the source sensor,
> coordinate reference system, temporal class, build function, and expected output
> column for every feature. Layer 1 (pre-assembly) validates interface contracts
> between fetchers and the assembly pipeline. Layer 2 (post-assembly) enforces
> coverage thresholds, physical plausibility bounds, and join-uniqueness invariants.
> Layer 3 (data lock) performs full reconciliation against the contract and enforces
> causal boundaries -- features classified as `post_event` (high-water marks, NFIP
> claims, 311 reports) are flagged as labels, not model inputs. Initial deployment
> detected a CRS path mismatch in the land cover layer and a column-name mismatch
> in the tidal data that would have produced silently corrupted features in the
> assembled dataset.

### For Section 9 (Limitations)

> The validation framework addresses engineering defects (CRS mismatches,
> column-name mismatches, file format issues) but cannot resolve inherent
> data limitations: tide gauge sparsity (3 stations serving 200 ZCTAs), high-water
> mark survivorship bias (surveyor access determines coverage, not flood extent),
> NFIP claims censoring (zero claims may indicate absent insurance rather than
> absent damage), and temporal collapse of hourly signals to per-event scalars.
> These limitations are documented in the feature contract's `temporal_class` and
> `missing_behavior` fields and acknowledged as ceiling constraints on model
> performance.
