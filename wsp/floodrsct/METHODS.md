# FloodRSCT Data Processing Methods

## Pipeline Overview

Each scenario follows a five-stage pipeline. Stages 1-4 answer "is the dataset
correctly built?" Stage 5 answers "is the dataset adequately supported across
the strata that matter for the experiments?"

```
SCENARIO (e.g. southwest_florida)
 |
 |  Events: ian2022, helene2024, milton2024
 |
 |  STAGE 1: FETCH (per source, per event)
 |  Each fetch job writes raw data to S3 independently.
 |
 |  Event-level sources              Static sources
 |  (one fetch per event)            (one fetch per scenario)
 |  +-----------------------+        +-----------------------+
 |  | fetch_noaa_mrms.py    |        | fetch_dem_elevation.py|
 |  |   -> raw/noaa_mrms/   |        |   -> raw/dem/3dep/    |
 |  +-----------------------+        +-----------------------+
 |  | fetch_noaa_tides.py   |        | fetch_noaa_slosh.py   |
 |  |   -> raw/noaa_tides/  |        |   -> raw/noaa_slosh/  |
 |  +-----------------------+        +-----------------------+
 |  | fetch_noaa_hrrr.py    |        | fetch_tiger_coast.py  |
 |  |   -> raw/noaa_hrrr/   |        |   -> raw/tiger/       |
 |  +-----------------------+        +-----------------------+
 |  | fetch_surge_hwm.py    |        | fetch_openfema.py     |
 |  |   -> raw/surge_est./  |        |   -> raw/openfema/    |
 |  +-----------------------+        +-----------------------+
 |  | fetch_gpm_imerg.py    |
 |  |   -> raw/gpm_imerg/   |
 |  +-----------------------+
 |  | fetch_smap.py         |
 |  |   -> raw/smap_soil/   |
 |  +-----------------------+
 |  | fetch_hurdat2.py      |
 |  |   -> raw/hurdat2/     |
 |  +-----------------------+
 |
 |  STAGE 2: GATE (presence)
 |  +---------------------------------------------------+
 |  | validate_data_readiness.py                        |
 |  |   Audits every (event, dataset) cell against S3.  |
 |  |   Must show 0 MISSING before proceeding.          |
 |  |   Distinguishes: not_applicable vs source_empty   |
 |  +---------------------------------------------------+
 |
 |  STAGE 3: ASSEMBLE (join raw sources into per-event rows)
 |  +---------------------------------------------------+
 |  | build_event_dataset.py --scenario southwest_florida|
 |  |                                                   |
 |  |  For each event (ian2022, helene2024, milton2024):|
 |  |    1. Load ZCTA geometries from geocertdb2026     |
 |  |    2. Aggregate MRMS grids -> total_rainfall_mm   |
 |  |    3. Join tidal surge (max across stations)      |
 |  |    4. Spatial join HWMs to nearest ZCTA           |
 |  |    5. Compute storm_distance_km from HURDAT2      |
 |  |    6. Sample DEM at ZCTA centroids                |
 |  |    7. Sample SLOSH MOM at ZCTA centroids          |
 |  |    8. Join NFIP claims by reportedZipcode         |
 |  |    9. Merge all features into (zcta_id, event) row|
 |  |                                                   |
 |  |  Output:                                          |
 |  |    processed/southwest_florida/                    |
 |  |      swfl_event_features.parquet                  |
 |  +---------------------------------------------------+
 |
 |  STAGE 4: VALIDATE (integrity)
 |  +---------------------------------------------------+
 |  | strat_sampler_qa.py --scenario southwest_florida   |
 |  |   Range checks, cross-column consistency,          |
 |  |   key integrity, constant column detection         |
 |  |   -> evidence/qa/strat_sampler_seed{N}.json       |
 |  +---------------------------------------------------+
 |
 |  STAGE 5: STRATIFIED COVERAGE (sufficiency + quality)
 |  +---------------------------------------------------+
 |  | stratified_coverage_audit.py                       |
 |  |   --scenario southwest_florida                     |
 |  |                                                   |
 |  |  18 audits in two layers:                         |
 |  |                                                   |
 |  |  LAYER 0 -- Dataset-support probes                |
 |  |  (is the substrate admissible for evaluation?)    |
 |  |    P1. Per-event support (transfer)               |
 |  |    P2. Coastal vs inland (heterogeneity)          |
 |  |    P3. Levee-protected vs not (heterogeneity)     |
 |  |    P4. County group sizes (blocked CV)            |
 |  |    P5. Adjacency coverage (spatial-lag)           |
 |  |    P6. Outcome signal per stratum (ceiling)       |
 |  |                                                   |
 |  |  LAYER 1 -- GeoRSCT mode audits                  |
 |  |  (which failure modes are active?)                |
 |  |    A.1 Autocorrelation leakage (split bias)       |
 |  |    A.2 Geographic heterogeneity (stratum CV)      |
 |  |    A.3 Smooth-map illusion (NOT_READY)            |
 |  |    B.1 MAUP / partition drift (ZCTA boundaries)   |
 |  |    B.2 Scale mismatch (broadcast/coarse)          |
 |  |    B.3 Crosswalk gap (join hit rates)             |
 |  |    C.1 Vintage drift (feature age vs event)       |
 |  |    C.2 CRS inconsistency (projection/datum)       |
 |  |    C.3 Spatial missingness bias (systematic NaN)  |
 |  |    D.1 Ceiling-aggregation drift (NOT_READY)      |
 |  |    D.2 Architecture flattening (NOT_READY)        |
 |  |    D.3 Interp/extrap mismatch (distribution gap)  |
 |  |                                                   |
 |  |  -> evidence/qa/coverage_audit_{scenario}.json    |
 |  +---------------------------------------------------+
```

Not all sources apply to every scenario. The feature contract
(`FEATURE_CONTRACT.yaml`) defines which sources are required per scenario.
Stage 2 enforces this before assembly.

---

## Storm Surge Estimation

### Problem

NOAA CO-OPS tide stations provide two products for computing storm surge:
- **Observed water levels** (`hourly_height`): measured values at the gauge
- **Predicted tides** (`predictions`): astronomical tide model output

Storm surge is traditionally computed as `surge = observed - predicted`. However,
many stations -- particularly in the Gulf of Mexico -- do not serve predictions
for the NAVD or STND datums. During Hurricane Ida (2021), 3 of 4 New Orleans
stations returned empty prediction responses, leaving surge as NaN despite having
complete observed water level records.

### Method: Pre-Storm Baseline Anomaly

When predictions are unavailable, surge is computed as the **anomaly from a
pre-storm baseline**:

```
baseline = median(observed_m[0:24h])
surge_m  = observed_m - baseline
```

**Rationale:**
- The first 24 hours of each event window precedes landfall in every scenario
  (event windows start 2-3 days before peak impact by design).
- The median is robust to outliers from early outer-band effects.
- The anomaly captures the storm signal: how much water level rose above
  pre-storm conditions at that specific gauge.

**Priority order:**
1. Prediction-based surge (`observed - predicted`) when predictions exist
2. Baseline anomaly (`observed - median(first 24h)`) as fallback

The fallback is applied per-station. A station with predictions uses method 1;
a station without predictions uses method 2. Both can coexist within the same
event.

### Limitations

- The baseline anomaly does not remove the astronomical tide signal. For
  diurnal/semi-diurnal tide stations, the anomaly includes ~0.3-0.6m of tidal
  variation that would be removed by method 1.
- If the pre-storm period itself has anomalous water levels (e.g., from a
  preceding weather system), the baseline is biased.
- The 24-hour window assumes hourly observations. Stations with gaps in the
  first 24h use fewer samples for the median.

### Affected Scenarios

| Scenario | Stations | Predictions Available | Surge Method |
|----------|----------|----------------------|--------------|
| Houston  | 3 | Partial | Mixed |
| New Orleans | 4 | 0 of 4 | All baseline anomaly |
| NYC | 3 | 3 of 3 | All prediction-based |
| SW Florida | 2 | 2 of 2 | All prediction-based |
| Riverside-Coachella | 0 | N/A | No tidal component |

---

## MRMS Precipitation Aggregation

### Method

Multi-Radar Multi-Sensor (MRMS) hourly precipitation grids are aggregated to
ZCTA-level total rainfall:

1. **Download**: Grib2 files (gzip-compressed) from Iowa State Mesonet archive
2. **Decode**: cfgrib reads each file into a 2D precipitation array (3500 x 7000
   for CONUS grid, ~0.01 degree resolution)
3. **Accumulate**: Running sum across all hourly files (not stored in memory
   simultaneously -- each grid is added to the accumulator and discarded)
4. **Assign to ZCTA**: For each ZCTA centroid, find the nearest MRMS grid point
   by Euclidean distance on lat/lon coordinates

**Coordinate handling**: cfgrib returns latitude and longitude as 1D coordinate
arrays. These are expanded via `np.meshgrid` before flattening to match the 2D
precipitation grid shape.

**Parallelism**: Files are decoded in parallel using `ProcessPoolExecutor` with
`min(4, cpu_count)` workers. Each worker creates its own `boto3.client` (not
shared across processes). Worker count is capped at 4 to limit concurrent memory
usage (~100 MB per CONUS grid).

### Product Name Boundary

The MRMS product name changed between 2020 and 2021:

| Year Range | Product | Algorithm |
|------------|---------|-----------|
| Pre-2021 | `GaugeCorr_QPE_01H` | Original gauge correction |
| 2021+ | `MultiSensor_QPE_01H_Pass2` | Multi-sensor second pass |

The two products are **not directly comparable** — gauge correction algorithms
differ. The fetcher (`fetch_noaa_mrms_v2.py`) auto-selects via `_product_name()`.
Cross-product comparison (e.g., a 2017 event vs a 2024 event) should note this
in uncertainty discussions.

### Sentinel Values (Erratum 2026-06-03)

MRMS grib2 files use **negative sentinel values** for quality-flagged pixels:

| Value | Meaning |
|-------|---------|
| -3.0 | No data (radar gap, no coverage) |
| -2.0 | Below threshold (trace precipitation) |
| -1.0 | Range-folded (ambiguous radar return) |

**Bug discovered**: Prior to commit `f382892`, the accumulation code summed
raw pixel values without masking sentinels. Over a multi-day event window (e.g.,
Harvey 2017 with 432 hourly files), sentinel pixels accumulated to large negative
totals:

| Accumulated Value | Cause |
|-------------------|-------|
| -1296 | -3 x 432 hours (no-data pixel, all hours) |
| -432 | -3 x 144 hours (partial coverage) |
| -360 | -3 x 120 hours (partial coverage) |

This produced `rainfall_total_mm` values of -622 (mean) across Houston ZCTAs,
and propagated into `wlag_rainfall_mm` via the spatial lag computation. All
positive precipitation signal was overwhelmed by sentinel accumulation.

**Fix**: Clamp `arr < 0` to 0 in `_process_one_grib()` before returning the
array to the accumulator. This is correct because MRMS QPE values are physically
non-negative (precipitation rate in mm/hr). Any negative value is a quality flag,
not a measurement.

**Impact**: All five scenarios require rebuild of `*_event_features.parquet`.
See "Rebuild Process" below.

### Rebuild Process (Post-Sentinel Fix)

After applying the sentinel fix, all scenario event features must be rebuilt to
ensure consistent, correct rainfall values across the experiment:

**Scenarios and expected MRMS file counts:**

| Scenario | Events | MRMS Files | Instance | Est. Wall Clock |
|----------|--------|------------|----------|-----------------|
| houston | harvey2017, imelda2019, beryl2024 | ~1,400 | ml.m5.4xlarge | ~45 min |
| southwest_florida | ian2022, helene2024, milton2024 | ~525 | ml.m5.4xlarge | ~20 min |
| new_orleans | ida2021_nola, barry2019_nola, isaac2012_nola, henri2021 | ~700 | ml.m5.4xlarge | ~25 min |
| nyc | ida2021_nyc | ~170 | ml.m5.2xlarge | ~10 min |
| riverside_coachella | hilary2023, ar_flood_2023 | ~850 | ml.m5.4xlarge | ~30 min |

**Rebuild command (per scenario):**

```bash
cd data/floodrsct/scripts
python launch_build_event_dataset.py --scenario houston --dry-run   # verify config
python launch_build_event_dataset.py --scenario houston             # launch
```

**Rebuild order**: All five are independent — launch in parallel if quota allows.

**Downstream impact**: After event features are rebuilt, any R0/R1 baselines that
used rainfall features must also be re-run to update model diagnostics. The R0
results are invalidated because the rainfall feature carried negative sentinel
noise instead of signal.

**Verification after rebuild:**

```python
import pandas as pd
df = pd.read_parquet("houston_event_features.parquet")
r = df["rainfall_total_mm"].dropna()
assert (r >= 0).all(), f"Negative rainfall values remain: min={r.min()}"
assert r.max() > 0, "All rainfall is zero — MRMS decode may have failed"
# Harvey peak: ~1500mm over 4 days near Port Arthur
assert r.max() > 100, f"Max rainfall {r.max():.0f}mm — implausibly low for Harvey"
```

### Limitations

- Nearest-centroid assignment does not account for ZCTA shape or area. Large
  ZCTAs may have significant within-ZCTA precipitation variation.
- MRMS is radar-based and can underestimate precipitation in regions with poor
  radar coverage or beam blockage (mountainous terrain).
- Clamping sentinels to 0 loses the distinction between "no data" (-3) and
  "measured zero rainfall" (0.0). The `obs_mrms_coverage_pct` field partially
  mitigates this — ZCTAs with low coverage likely had sentinel-dominated pixels.
- The -2 sentinel (below threshold) is also clamped to 0. This is physically
  correct (trace precipitation is negligible at ZCTA scale) but loses the
  "precipitation detected but below reporting threshold" signal.

---

## Tidal Water Level Aggregation

### Current Method

All tide stations within a scenario are loaded. The **maximum observed water
level** and **maximum surge** across all stations are broadcast to every ZCTA in
the scenario. This is a conservative (worst-case) estimate.

### Known Limitation

This approach produces a constant value across all ZCTAs. A future version should
assign each ZCTA the value from its nearest tide station, weighted by distance.

---

## SLOSH MOM Surge Hazard

### Background

NHC SLOSH (Sea, Lake and Overland Surges from Hurricanes) produces two product
types relevant to flood risk:

- **MEOW** (Maximum Envelope of Water): surge envelope for a single hypothetical
  storm with fixed parameters (category, forward speed, track angle, landfall point)
- **MOM** (Maximum of MEOWs): cell-wise maximum across thousands of MEOWs for a
  given basin and storm category. This is the worst-case surge planning surface.

MOM grids are **basin-specific and category-specific, not storm-specific**. A Cat 3
MOM for the Tampa Bay (TBW) basin is the same regardless of whether the actual
storm is Milton 2024 or a hypothetical future hurricane. MOM is a static hazard
surface, not an event measurement.

### Why MOM, Not Per-Storm Products

The earlier approach attempted to download per-storm SLOSH products (advisory-level
P-Surge archives) via NHC URLs. This failed because:

1. NHC does not publish downloadable SLOSH grids at stable per-storm URLs
2. Ian, Helene, and Milton hit **three different SLOSH basins** (FTM/Fort Myers,
   APF/Big Bend, TBW/Tampa Bay) -- the Ian GeoTIFFs could not be reused
3. P-Surge is an operational forecast product with ~21 advisory snapshots per storm;
   selecting which advisory to use introduces subjective bias

MOM avoids all three problems: one download covers all basins, all categories, and
the product is deterministic (no advisory selection required).

### Method

1. **Download**: National SLOSH MOM inundation grid from NHC
   (`US_SLOSH_MOM_Inundation_v4.zip`). Single GeoTIFF covering Texas to Maine,
   all SLOSH basins, per Saffir-Simpson category (Cat 1-5).
2. **Store**: `s3://swarm-floodrsct-data/raw/noaa_slosh/mom_national/`
3. **Derive** (in `build_event_dataset.py`):
   - For each SW Florida ZCTA centroid, sample the MOM raster at that location
   - `slosh_max_surge_m`: maximum MOM inundation depth (feet converted to meters)
     at the ZCTA centroid for the storm's actual Saffir-Simpson category at landfall
   - `slosh_category`: the Saffir-Simpson category used for lookup
   - Guard: raster pixel values >= 99 are treated as no-data (SLOSH uses high
     sentinel values for dry cells outside the inundation envelope)

### Assembly Pipeline

The assembly step (`build_event_dataset.py --scenario southwest_florida`) joins
all pre-fetched raw data into the final processed parquet. Raw data must already
exist on S3 before assembly runs. The assembly job:

1. Bootstraps pip packages (rasterio, geopandas, xarray, etc.)
2. Assembles all SW Florida events (ian2022, helene2024, milton2024) by joining
   MRMS, tides, HWM, HURDAT2, DEM, SLOSH, and NFIP sources per ZCTA
3. For SLOSH: reads the Cat 4/3 MOM GeoTIFF from S3, samples at ZCTA centroids,
   converts feet to meters, guards pixel values >= 99 as no-data

### Temporal Classification

`slosh_max_surge_m` is classified as **invariant** (not event_window). The MOM
surface does not change between storms -- it represents the pre-computed worst-case
envelope for the basin geometry. The event-specific element is only the category
lookup key (e.g., Ian = Cat 4, Milton = Cat 3, Helene = Cat 4).

### Limitations

- MOM is a worst-case envelope, not an actual hindcast. Real surge from a specific
  storm is typically lower than the MOM value for that category.
- MOM resolution varies by basin (~250m to ~1km). Coastal ZCTAs smaller than the
  grid cell may receive imprecise values.
- Inland ZCTAs beyond the MOM inundation extent receive 0 (no surge), which is
  correct but may mask riverine flooding contributions.
- The national GeoTIFF is large (~500 MB). Only the SW Florida scenario uses SLOSH
  features; other scenarios do not have surge model fields.

### Basin Coverage

| Storm | Year | SLOSH Basin | Landfall Location |
|-------|------|-------------|-------------------|
| Ian | 2022 | FTM (Fort Myers) | Cayo Costa / Fort Myers Beach |
| Milton | 2024 | TBW (Tampa Bay) | Siesta Key / Sarasota |
| Helene | 2024 | APF (Apalachee Bay) | Perry / Keaton Beach |

All three basins are covered by the national MOM grid.

### Validation (2026-05-30)

End-to-end validation on SageMaker (`ml.m5.4xlarge`, job
`s035-build-events-southwest-florida-20260530-030117`):

| Event | Category | GeoTIFF | Centroids Sampled | ZCTA Coverage | Notes |
|-------|----------|---------|-------------------|---------------|-------|
| ian2022 | Cat 4 | us_Category4_MOM_Inundation_HIGH.tif | 198 | 56.4% (114/202) | Higher coverage — Cat 4 inundates more coastline |
| helene2024 | Cat 4 | us_Category4_MOM_Inundation_HIGH.tif | 198 | 56.4% (114/202) | Same GeoTIFF as Ian (MOM is invariant) |
| milton2024 | Cat 3 | us_Category3_MOM_Inundation_HIGH.tif | 198 | 39.1% (79/202) | Lower coverage — Cat 3 surge envelope is smaller |

Key observations:
- `src.sample()` reads only queried pixels; no full-raster load (66 GB uncompressed).
- Value >= 99 correctly guarded as NaN (levee-protected areas per NHC metadata).
- Cat 4 > Cat 3 coverage is physically expected (stronger storms push surge further inland).
- 202 ZCTAs total, 198 with valid centroids in geocertdb2026. 4 ZCTAs lack centroid coordinates.
- Output: 606 rows x 152 columns in `swfl_event_features.parquet`.

---

## NLCD Impervious Surface

### Source

NLCD 2021 Impervious Surface (CONUS), published by MRLC. The national raster
covers the lower 48 states at 30m resolution in EPSG:5070 (Albers Equal Area).

- **Raw file**: `nlcd_2021_impervious_l48.img` (26 GB, Erdas Imagine HFA format)
- **S3 path**: `raw/nlcd/impervious_2021/`
- **Scenarios**: houston, nyc

### Format Conversion

The source raster is distributed in Erdas Imagine HFA format (`.img`). Pip-installed
rasterio links against a minimal GDAL build that does not include the HFA driver.
The python GDAL bindings (`osgeo.gdal`) segfault when reading `.img` directly in
the SageMaker container environment (Ubuntu 22.04, pytorch-training image).

The build pipeline converts `.img` to GeoTIFF at runtime using `gdal_translate`:

```
gdal_translate -of GTiff -co COMPRESS=LZW input.img output.tif
```

This is a **lossless format conversion** — pixel values, CRS (EPSG:5070), nodata
value, data type (uint8), and resolution (30m) are all preserved exactly. LZW is
lossless compression (like zip). The only change is the container format: HFA → GeoTIFF.
The resulting `.tif` is natively readable by pip-installed rasterio without additional
drivers.

The original `.img` is deleted from `/tmp` after conversion to free disk space
(26 GB `.img` + converted `.tif` would exceed volume on smaller instances).

### Method

For each ZCTA centroid:
1. Reproject centroid from EPSG:4326 (WGS84) to EPSG:5070 (raster native CRS)
2. Extract a 1 km buffer window (500m in each direction) around the centroid
3. Compute mean impervious percentage from valid pixels (0-100, excluding nodata=127)
4. Output: `impervious_pct` (0-100 scale, percentage of impervious surface)

### Limitations

- Centroid-based sampling may not capture the full heterogeneity of large ZCTAs
- NLCD 2021 vintage may not reflect post-2021 development
- The 30m resolution is coarser than building-level impervious data but appropriate
  for ZCTA-level analysis

---

## NFIP Claims

Disaster-specific NFIP claims are joined to ZCTAs by the `reportedZipcode` field
in OpenFEMA data, filtered to the disaster declaration number (DR-XXXX) for each
event. Claims are aggregated as count, total building loss, total contents loss,
and mean loss per claim.

---

## Stratified Sampler QA

Post-assembly validation uses seed-controlled random sampling to verify data
quality across scenarios. Five probe families:

1. **Range checks**: Physical plausibility bounds for 20+ columns
2. **Cross-column consistency**: Logical rules (e.g., rainfall > 0 implies MRMS
   coverage > 0)
3. **Column family coherence**: Related columns should have similar null rates
4. **Key integrity**: No duplicate (zcta_id, event) pairs
5. **Constant column detection**: Flags columns where every value is identical

Each run is parameterized by `--seed` and `--n-samples` for reproducibility.
Results are written as JSON evidence to S3 for experiment traceability.

---

## Data Readiness Classification

`validate_data_readiness.py` audits every (event, dataset) cell against S3 and
classifies gaps into distinct statuses:

| Status | Meaning | Example |
|--------|---------|---------|
| `fetched` | Data present on S3 | harvey2017/mrms (432 files) |
| `missing` | Expected data not found; action required | -- |
| `not_applicable` | Feature category does not apply to this event type | ar_flood_2023/hurdat2 |
| `source_empty` | Our specific data source returned zero records; other sources may exist | beryl2024/hwm |

### Accepted No-Data Cases

**ar_flood_2023 / hurdat2** (`not_applicable`): HURDAT2 is a tropical/subtropical
cyclone best-track dataset (NOAA). The 2023 California atmospheric river is not a
tropical cyclone and has no HURDAT2 track. The `storm_distance_km` feature is null
for AR events by definition.

**beryl2024 / hwm** (`source_empty`): USGS STN Flood Event Viewer (event 342)
returned zero high-water mark records for Hurricane Beryl (2024). This does not
mean no HWM observations exist -- the NHC Beryl report documents survey teams
finding 5-7 ft storm-surge inundation between Matagorda and Freeport, and
HCFCD/NWS surveys recorded marks up to 10.2 ft NAVD88. These observations are
not in STN and would require separate ingestion from NHC/HCFCD/NWS sources.

**ar_flood_2023 / tides** (`not_applicable`): Inland atmospheric river event;
coastal tidal stations are not relevant to the Riverside-Coachella scenario.
