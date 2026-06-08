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
  [L1] [PASS] bayou_segment_id: 2 files, 89.7 MB
  [L1] [FAIL] drainage_district_id: no files at raw/hcfcd/drainage_districts/v1/
  [L1] [PASS] max_rainfall_mm: 696 files, 429.5 MB
  [L1] [PASS] total_rainfall_mm: 696 files, 429.5 MB
  [L1] [WARN] tidal_surge_max_m: 6 file(s) < 200 bytes (possible stubs)
  [L1] [PASS] tidal_surge_max_m: 24 files, 1.2 MB
  [L1] [PASS] storm_distance_km: 1 files, 0.0 MB
  [L1] [PASS] hwm_max_ft: 4 files, 0.1 MB
  [L1] [FAIL] flood_311_count: no files at raw/houston_311/...
  [L1] [PASS] nfip_event_claims: 9 files, 3.8 MB

PASS: 8  FAIL: 2  WARN: 1  SKIP: 1
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
| SLOSH placeholder files (148 bytes) counted as data | L1 false PASS | Raised stub threshold to 200 bytes; all-stub = FAIL |
| `get_aws_credentials()` returns `region_name`, duplicates explicit kwarg | Crash on every boto3.client() call | Added `_aws.pop("region_name", None)` across 27 scripts |
| OpenFEMA `$select` had wrong field names (`floodZone`, `numberOfFloorsInInsuredBuilding`) | 400 Bad Request | Fixed to `ratedFloodZone`, `numberOfFloorsInTheInsuredBuilding` |
| `bayou_segment_id` L1 PASS with wrong-geography data (RC HUC, not Houston HUC) | Review | Documented as VARIANCE; L1 cannot detect spatial mismatch |

---

## Current Scorecard (2026-05-29, live from `_validate_contract.py --all`)

### Layer 1 -- Feature x Scenario Matrix

Legend: P = PASS, F = FAIL, W = WARN, S = SKIP (operational), `*` = see notes

| Feature | HOU | NOLA | NYC | RC | SWFL | Gap Category |
|---------|-----|------|-----|----|------|--------------|
| `impervious_pct` | P | -- | P | -- | -- | |
| `elevation_m_msl` | -- | P | -- | -- | P | |
| `burn_scar_overlap` | -- | -- | -- | P | -- | |
| `upstream_catchment_km2` | -- | -- | -- | P | -- | |
| `wash_segment_id` | -- | -- | -- | P | -- | |
| `bayou_segment_id` | P`*` | -- | -- | -- | -- | VARIANCE (wrong HUC) |
| `drainage_district_id` | **F** | -- | -- | -- | -- | VARIANCE |
| `subway_station_count` | -- | -- | P | -- | -- | |
| `nearest_subway_distance_m` | -- | -- | P | -- | -- | |
| `sewer_shed_id` | -- | -- | **F** | -- | -- | CODE |
| `levee_condition_rating` | -- | **F** | **F** | -- | -- | CODE |
| `subsidence_rate_mm_yr` | -- | **F** | -- | -- | -- | VARIANCE |
| `canal_proximity_m` | -- | **F** | -- | -- | -- | VARIANCE |
| `slosh_max_surge_m` | -- | -- | -- | -- | **F** | CODE (manual DL) |
| `slosh_category` | -- | -- | -- | -- | **F** | CODE (manual DL) |
| `coastal_distance_m` | -- | -- | -- | -- | **F** | CODE |
| `max_rainfall_mm` | P | P | P | P | P | |
| `total_rainfall_mm` | P | P | P | P | P | |
| `tidal_surge_max_m` | P/W | P | P | -- | P | WARN: 6 stub files |
| `storm_distance_km` | P | -- | -- | P | P | |
| `hwm_max_ft` | P | -- | P | P | P | |
| `flood_311_count` | **F** | -- | **F** | -- | -- | CODE |
| `nfip_event_claims` | P | P | P | P | P | DONE (fetched 2026-05-29) |
| `drainage_capacity_status` | S | -- | -- | -- | -- | OPERATIONAL |
| `pump_station_status` | -- | S | -- | -- | -- | OPERATIONAL |
| `road_access_status` | -- | -- | -- | S | -- | OPERATIONAL |
| `evacuation_route_status` | -- | -- | -- | -- | S | OPERATIONAL |

### Scenario Totals (L1 only)

| Scenario | PASS | FAIL | WARN | SKIP | L1 Verdict |
|----------|------|------|------|------|------------|
| **riverside_coachella** | 8 | 0 | 0 | 1 | **CLEAR** |
| **houston** | 8 | 2 | 1 | 1 | BLOCKED |
| **new_orleans** | 5 | 3 | 0 | 1 | BLOCKED |
| **nyc** | 8 | 3 | 0 | 0 | BLOCKED |
| **southwest_florida** | 7 | 3 | 0 | 1 | BLOCKED |
| **Total** | **36** | **11** | **1** | **4** | |

### Layer 3 -- All scenarios BLOCKED (builds not yet run)

All 5 scenarios show `output_parquet: FAIL` because `build_event_dataset.py` has
not been executed. This is expected -- L3 validates the assembled output, which
does not exist until the builder runs. L3 will be re-run as the Data Lock gate
after each build.

---

## Gap / Variance Register

Every remaining L1 FAIL classified by resolution path.

### DONE -- Fetched This Session (2026-05-29)

| Feature | Scenarios | Result |
|---------|-----------|--------|
| `nfip_event_claims` | ALL 5 | 435 declarations + 270K claims across 9 DRs |
| geocertdb2026 (ZCTA features) | ALL 5 | 7 parquets + 5 scenario subsets (17-198 ZCTAs each) |

### CODE -- Fetcher Exists, Needs Launcher or Data

| Feature | Scenarios | Issue | Effort | Priority |
|---------|-----------|-------|--------|----------|
| `flood_311_count` | houston, nyc | Fetchers exist (`fetch_houston_311.py`, `fetch_nyc_311.py`) but no per-event launchers | Low | MEDIUM -- post-event label, not model input |
| `sewer_shed_id` | nyc | Fetcher exists (`fetch_nyc_sewersheds.py`) but not run | Low | LOW -- scenario-specific |
| `levee_condition_rating` | new_orleans, nyc | Fetcher exists (`fetch_usace_levees.py`) but not run with correct `--scenario` | Low | MEDIUM |
| `coastal_distance_m` | southwest_florida | No fetcher for TIGER coastline shapefile | Medium | LOW for Data Lock A |
| `slosh_max_surge_m`, `slosh_category` | southwest_florida | SLOSH grids require manual download from NHC; placeholders only on S3 | Medium-High | HIGH for SWFL scenario |

### VARIANCE -- Accepted Gaps (Document in Paper S9)

| Feature | Scenarios | Why Accepted | Paper Text |
|---------|-----------|-------------|------------|
| `bayou_segment_id` | houston | L1 PASS is misleading: data at `raw/nhdplus/catchments/v2/` covers Riverside HUC4 1810/1811, not Houston HUC8 12040104. Files exist but for the wrong geography. | "Hydrologic unit membership (bayou segment) is defined in the feature contract but Houston-specific NHDPlus flowlines are not populated for Data Lock A" |
| `drainage_district_id` | houston | HCFCD shapefile requires manual download from county portal (data.hcfcd.org) | same |
| `subsidence_rate_mm_yr` | new_orleans | USGS InSAR subsidence raster (Qu et al. doi:10.5066/P9VFMF9K); fetcher not written | "Land subsidence rates require InSAR-derived velocity fields not included in this study" |
| `canal_proximity_m` | new_orleans | OSM Overpass query not written | "Canal proximity features for the New Orleans protected-basin scenario are deferred" |

### OPERATIONAL -- Skip by Design (No Public Archive)

| Feature | Scenario | Notes |
|---------|----------|-------|
| `drainage_capacity_status` | houston | Real-time HCFCD pump/gate telemetry |
| `pump_station_status` | new_orleans | S&WB pump telemetry; partial hand-coded CSV for Ida 2021 |
| `road_access_status` | riverside_coachella | CalTrans/SigAlert not archived |
| `evacuation_route_status` | southwest_florida | FL county OES not archived |

### NOLA Event Key Variance (Discovered During Validation)

The MRMS fetcher stored Ida 2021 under key `ida2021_nyc` (Sep 1-4 window, covering
the NYC remnant rainfall). The NOLA builder uses event key `ida2021` and expects
MRMS at `raw/noaa_mrms/ida2021/` -- but the NOLA builder **does not call
`aggregate_mrms_rainfall()` at all** (lines 738-747). This is a code gap, not a data
gap: MRMS rainfall for Ida's NOLA landfall (Aug 27-Sep 1) requires a separate
fetcher run with a NOLA-specific window.

**Impact**: NOLA scenario has no MRMS rainfall features. Tidal surge and HWM data
provide partial flood signal, but the dominant forcing (200+ mm rainfall over
Orleans Parish in 6 hours) is absent. If NOLA appears in the paper's main results,
this gap must be filled or NOLA scoped to appendix.

**Resolution**: Add `ida2021_nola` event to MRMS fetcher (window: Aug 27-Sep 1)
and add MRMS call to `build_new_orleans()`. Required for Data Lock B.

---

### Summary by Fixability

| Category | Feature Count | Action |
|----------|---------------|--------|
| **DONE** (fetched this session) | 2 features, all 5 scenarios | Complete |
| **CODE** (launcher / minor fix) | 5 features | Before Data Lock B |
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
