# GeoSpatial Challenges

**Pipeline**: FloodRSCT build_event_dataset.py  
**Status**: Issues identified during Data QC (2026-05-29)  
**Severity**: These affect spatial accuracy of all surrogate model features

---

## 1. HAND (Height Above Nearest Drainage) — MISSING

### Problem

The pipeline uses raw DEM elevation (mean within 500m buffer of ZCTA centroid) as a flood susceptibility feature. Raw elevation is nearly meaningless for flood risk — a point at 30m elevation next to a river is far more flood-prone than a point at 30m on a ridgeline.

### What HAND Is

Height Above Nearest Drainage (HAND) computes, for each pixel:
1. Find the nearest drainage channel (flow accumulation > threshold)
2. Trace the flow path from this pixel down to that channel
3. HAND = pixel_elevation - channel_elevation

Result: a per-pixel "flood susceptibility" in meters. HAND < 5m = high risk. HAND > 15m = minimal risk regardless of absolute elevation.

### Current State

- **Available**: NOAA/OWP publishes pre-computed 10m HAND rasters for CONUS (derived from 3DEP DEM + NHDPlus drainage network)
- **Not fetched**: No fetcher exists for HAND data
- **Not in contract**: FEATURE_CONTRACT.yaml has `elevation_mean_m` but no HAND feature

### Suggested Solution

1. Add HAND fetcher targeting NOAA/OWP HAND tiles (GeoTIFF, EPSG:5070, 10m)
2. Compute per-ZCTA statistics: `hand_mean_m`, `hand_p10_m` (10th percentile = most flood-prone areas), `hand_pct_below_5m` (fraction of ZCTA in high-risk zone)
3. Reproject centroid/geometry to EPSG:5070 before zonal stats (same CRS fix pattern as NLCD -- Section 3 applies here)
4. Add to FEATURE_CONTRACT.yaml as `temporal_class: invariant`

### Impact If Not Fixed

DEM elevation alone has weak predictive power for flood damage. Two ZCTAs at identical elevation but different HAND values will have vastly different flood outcomes. The surrogate model will underperform on flat coastal areas where elevation varies little but HAND varies significantly.

---

## 2. HUC (Hydrologic Unit Code) — WRONG SPATIAL UNIT

### Problem

The pipeline aggregates all features to ZCTA (ZIP Code Tabulation Area). ZCTAs are postal/demographic boundaries with no hydrologic meaning. A single ZCTA can span multiple watersheds, or a single watershed can span many ZCTAs.

### What HUC Levels Are

USGS Hydrologic Unit Codes define nested watershed boundaries:

| Level | Name | Count (CONUS) | Typical Area |
|-------|------|---------------|--------------|
| HUC-2 | Region | 18 | ~400,000 km^2 |
| HUC-4 | Subregion | 222 | ~40,000 km^2 |
| HUC-6 | Basin | 378 | ~25,000 km^2 |
| HUC-8 | Subbasin | 2,270 | ~1,800 km^2 |
| HUC-10 | Watershed | 18,700 | ~500 km^2 |
| HUC-12 | Subwatershed | 101,000 | ~100 km^2 |

HUC-12 is the natural spatial unit for flood hydrology — it represents the area that drains to a single pour point.

### Current State

- **NHDPlus catchments fetched**: 2 files on S3 (`nhdplus_catchments/`) — these are NHDPlus catchments (finer than HUC-12, ~160,000 units)
- **Not used in assembly**: `build_event_dataset.py` aggregates everything to ZCTA, never references HUC or catchment boundaries
- **ZCTA-HUC crosswalk**: Does not exist in pipeline

### Suggested Solution

**Option A (Recommended for Data Lock A):** Keep ZCTA as primary unit but add HUC-12 membership:
1. Spatial join each ZCTA centroid to its HUC-12
2. Add `huc12_id` as a grouping feature (for the surrogate model to learn watershed-level effects)
3. Add `huc8_id` for regional watershed identity

**Option B (Post-Lock improvement):** Dual-resolution model:
1. Aggregate hydrologic features (rainfall, HAND, drainage) at HUC-12
2. Aggregate socioeconomic features (311, NFIP, impervious) at ZCTA
3. Join via spatial intersection weights

### Impact If Not Fixed

Rainfall that falls in an upstream HUC-12 flows downstream — but the ZCTA-only model has no concept of upstream contributing area. Two adjacent ZCTAs may receive identical local rainfall but vastly different flooding because one sits below a large upstream catchment.

---

## 3. EPSG/CRS Mismatches — SILENT REPROJECTION FAILURES

### Problem

`build_event_dataset.py` assumes all inputs are in EPSG:4326 (WGS84 lat/lon). Several datasets are in different coordinate reference systems. No reprojection is performed at assembly time.

### Affected Datasets

| Dataset | Native CRS | Pipeline Assumes | Error Magnitude |
|---------|-----------|------------------|-----------------|
| NLCD Impervious | EPSG:5070 (Albers Equal Area) | EPSG:4326 | 4-15 km positional error |
| MRMS Stage IV | HRAP (polar stereographic) | EPSG:4326 | 4-8 km at mid-latitudes |
| 3DEP DEM | EPSG:4269 (NAD83) | EPSG:4326 | <1m (negligible for this use) |
| NHDPlus | EPSG:4269 | EPSG:4326 | <1m (negligible) |
| SLOSH surge grids | Basin-local projections | N/A (not yet fetched) | Variable |

### Specific Failure Modes

**NLCD (EPSG:5070 Albers):**
- `build_impervious_features()` reads the .img raster and samples at ZCTA centroid lat/lon
- But pixel coordinates in EPSG:5070 are in meters (easting/northing), not degrees
- Sampling at (29.7, -95.3) in an Albers grid reads a pixel in the wrong hemisphere
- Result: NaN or nonsense values for all impervious surface features

**MRMS (HRAP polar stereographic):**
- `aggregate_mrms_rainfall()` uses haversine nearest-neighbor to find the closest MRMS grid cell to each ZCTA centroid
- Iowa State Mesonet GRIB2 files typically encode lat/lon coordinates in metadata; cfgrib reads these, so `ds["latitude"]` and `ds["longitude"]` should already be in degrees
- **Likely not a 4-8 km error** -- but worth a one-file verification (open one GRIB2, confirm coordinates are geographic)
- If MRMS coordinates were raw HRAP x/y (unlikely given source), mismatch would be ~4-8 km at mid-latitudes

### Suggested Solution

1. **At fetch time (preferred):** Each fetcher reprojects to EPSG:4326 before writing to S3. NHDPlus and MTBS fetchers already do this correctly.
2. **At assembly time (fallback):** `build_event_dataset.py` reads the CRS from each input file and reprojects on the fly using `rasterio.warp.transform` or `pyproj.Transformer`.
3. **Validation:** Add a CRS assertion at the top of each aggregation function:
   ```python
   assert src.crs.to_epsg() == 4326, f"Expected EPSG:4326, got {src.crs}"
   ```

### Impact If Not Fixed

NLCD features will be garbage (wrong pixels entirely). MRMS rainfall assignment will be noisy (4-8 km offset at best). The surrogate model will learn from spatially misaligned features, degrading all predictions.

---

## 4. Resolution Mismatches — NO QUALITY FLAGS

### Problem

Features derived from vastly different spatial resolutions are aggregated to the same ZCTA without any indication of representativeness or uncertainty.

### Resolution Inventory

| Source | Native Resolution | Aggregation Method | Samples per ZCTA |
|--------|-------------------|-------------------|-----------------|
| 3DEP DEM | 10 m | 500m buffer mean | ~7,800 pixels |
| NLCD Impervious | 30 m | Point sample at centroid | 1 pixel |
| MRMS Rainfall | ~4.7 km | Nearest centroid | 1-4 cells |
| Tidal Stations | Point (1-6 per region) | Nearest within 100km, broadcast | 1 station value |
| USGS HWMs | Point (sparse, <100 per event) | Haversine nearest within 5km | 0-3 marks |
| HRRR QPF | 3 km | Not yet assembled | 1-9 cells |

### Specific Problems

**Tidal broadcast:**
- Houston has 3 tide gauges for ~200 ZCTAs
- All ZCTAs within 100km of a station get that station's exact value
- 60+ inland ZCTAs (>50 km from coast) receive tidal values that are physically meaningless for them
- No distance-decay or "this value is 80km from source" flag

**MRMS single-cell assignment:**
- A ZCTA spanning 50 km^2 gets rainfall from one 4.7 km grid cell (22 km^2)
- Convective storms can have 100mm/hr rainfall gradients over 5 km
- Result: Systematic underestimate of peak rainfall for large ZCTAs

**HWM sparsity:**
- Only ~50-100 high-water marks per event, distributed along accessible roads/structures
- Most ZCTAs have 0 HWMs within 5km
- Pipeline assigns NaN (correct) but this makes HWM features >80% missing

### Suggested Solution

1. **Add quality/confidence columns** for each aggregated feature:
   - `mrms_rainfall_n_cells`: Number of MRMS cells covering this ZCTA
   - `tidal_station_dist_km`: Distance from ZCTA centroid to assigned station
   - `hwm_count_5km`: Number of HWMs within 5km
   - `dem_pixel_count`: Number of DEM pixels in buffer

2. **Area-weighted aggregation for rasters:**
   - MRMS: Area-weighted mean of all cells intersecting ZCTA polygon (not nearest centroid)
   - NLCD: Zonal statistics over full ZCTA polygon (mean impervious fraction)
   - DEM: Zonal statistics over full ZCTA polygon

3. **Distance-decay for point sources:**
   - Tidal: IDW (inverse distance weighted) from all stations within range, with a `max_dist_km` cutoff beyond which value = NaN
   - HWM: Keep as-is (NaN if no nearby marks) but document the sparsity

### Impact If Not Fixed

The surrogate model will treat a tidal value 5km from a gauge identically to one 90km away. It will learn from MRMS rainfall that systematically misrepresents spatial variability within large ZCTAs. Without quality flags, there is no way to weight observations by reliability during training.

---

## 5. Spatial Aggregation Methods — DOCUMENTED FAILURE MODES

### Current Methods in build_event_dataset.py

| Function | Method | Failure Mode |
|----------|--------|--------------|
| `aggregate_mrms_rainfall()` | Haversine nearest-centroid | Misses rainfall gradient across ZCTA |
| `aggregate_tides()` | Nearest station, broadcast to all ZCTAs | Inland ZCTAs get meaningless coastal values |
| `aggregate_hwm()` | Haversine nearest within 5km | >80% NaN; biased toward road-accessible locations |
| `build_elevation_features()` | 500m buffer around centroid | Ignores ZCTA shape; misses floodplain edges |
| `build_impervious_features()` | 500m buffer zonal mean | CRS bug made this sample wrong pixels (FIXED) |
| `compute_storm_proximity()` | Haversine to HURDAT2 track points | Correct for this use case |

### Recommended Upgrades (Priority Order)

1. **NLCD: Centroid point sample -> Zonal mean** (CRITICAL — current method samples 1 pixel from wrong CRS)
2. **MRMS: Nearest cell -> Area-weighted zonal mean** (HIGH — convective gradients matter)
3. **DEM: 500m buffer -> Full ZCTA zonal stats** (MEDIUM — buffer misses floodplain geometry)
4. **Tides: Broadcast -> IDW with distance cutoff** (MEDIUM — physically meaningless for inland ZCTAs)
5. **HWMs: Keep as-is** (LOW — sparsity is inherent to the data source)

---

## 6. Summary: Data Lock A Implications

### Can proceed to Data Lock A (June 1) with known limitations:

| Issue | Severity for Houston | Workaround |
|-------|---------------------|------------|
| No HAND | HIGH | DEM alone is weak predictor; acknowledge in paper limitations |
| No HUC membership | MEDIUM | ZCTA-only model is publishable but suboptimal |
| NLCD CRS wrong | CRITICAL | Must fix before assembly — features are garbage |
| MRMS CRS ambiguous | HIGH | Verify GRIB2 files have lat/lon coordinates (cfgrib may auto-reproject) |
| Tidal broadcast | LOW for Houston | 3 gauges cover the bay area reasonably |
| Resolution flags missing | LOW | Publishable without; add as "future work" |

### Must-fix before Data Lock A:

1. ~~NLCD CRS reprojection~~ **FIXED** -- pyproj.Transformer added to `build_impervious_features()` and `build_elevation_features()`
2. MRMS coordinate verification (open one GRIB2, confirm ds["latitude"]/ds["longitude"] are in degrees)

### Should-fix before SIGSPATIAL submission:

3. HAND layer (adds significant predictive power; straightforward to fetch)
4. HUC-12 membership column
5. Area-weighted zonal statistics for MRMS and NLCD

---

## 7. Temporal Collapse — INFORMATION DESTRUCTION DURING AGGREGATION

### Problem

Every time-varying data source is collapsed to a single scalar per (ZCTA, event) row. The temporal structure — when things happened relative to each other — is destroyed before it reaches the surrogate model.

### What Gets Collapsed

| Source | Native Granularity | What Builder Keeps | What's Lost |
|--------|-------------------|-------------------|-------------|
| MRMS rainfall | Hourly (432 files for Harvey) | `rainfall_total_mm` (sum) | Hyetograph shape, peak intensity, duration of heavy rain |
| NOAA tides | Hourly (~400 records) | `max_surge_m` (single max) | Timing of peak surge, tidal phase, duration above flood stage |
| HURDAT2 track | 6-hourly fixes | `storm_min_dist_km` (min distance) | Duration of exposure, approach speed, track curvature |
| USGS gauges | 15-min to hourly | `peak_stage_ft` (max) | Time-to-peak, recession curve, duration above flood stage |
| 311 reports | Individual timestamps | `complaints_311_count` (count) | Temporal distribution, onset delay relative to storm |

### Why This Matters for Flood Prediction

Flood damage is driven by **temporal coincidence**, not independent maxima:

1. **Harvey (2017):** Catastrophic because 500mm of rain fell over 4 days while astronomically high tides blocked bayou drainage. A model seeing `rainfall_total_mm=500` and `max_surge_m=1.2` separately cannot learn that the damage was caused by their overlap — the rain couldn't drain because the tide was high simultaneously.

2. **Ida NYC (2021):** 80mm fell in a single hour (hourly intensity, not total, drove subway flooding). The total rainfall for the event was moderate. `rainfall_total_mm` underweights this because it sums over the full window including dry hours.

3. **Ian (2022):** Storm surge arrived hours before peak rainfall. ZCTAs that flooded from surge alone vs. surge + rain had different damage patterns. The model sees identical `max_surge_m` for both.

### Specific Aggregation Failures

**MRMS `rainfall_total_mm`:**
- Harvey: 432 hourly files summed to one number. The difference between "500mm over 4 days" and "500mm in 12 hours" is the difference between managed flooding and catastrophe. Both produce the same `rainfall_total_mm`.
- A ZCTA that received 50mm/hr for 10 hours is more damaged than one that received 10mm/hr for 50 hours, despite identical totals. Peak intensity matters more than total for urban flash flooding.

**Tides `max_surge_m`:**
- Takes the single maximum across all stations and all hours, broadcasts to every ZCTA.
- Timing of peak surge relative to peak rainfall is the key driver. If surge peaks at low tide + 1.2m, the absolute water level is lower than if surge peaks at high tide + 0.8m. The tidal phase is discarded.

**Tides + MRMS temporal join (never performed):**
- The pipeline never aligns hourly rainfall with hourly water level. The compound flooding signal (rain + tide at the same hour) is the most predictive feature for coastal flood damage, and it doesn't exist in the output schema.

### Suggested Solutions

**Option A (Data Lock A — minimal):** Add derived temporal features that preserve some structure without changing the schema:
1. `rainfall_peak_hourly_mm` — max single-hour rainfall (captures intensity, not just total)
2. `rainfall_duration_hrs` — hours with >1mm precipitation (captures storm duration)
3. `surge_duration_above_1m_hrs` — hours where surge exceeded 1m (captures exposure duration)
4. `surge_rainfall_overlap_hrs` — hours where surge > 0.5m AND rainfall > 10mm simultaneously (compound flooding proxy)

These are computable from the raw hourly files already on S3 without changing the aggregation unit.

**Option B (Post-Lock — requires schema change):** Add a temporal feature vector per (ZCTA, event):
1. Store 6-hour or 12-hour binned rainfall/surge as array columns in parquet
2. The surrogate model receives a short time series, not a scalar
3. This is the path to learning temporal coincidence effects

**Option C (Post-Lock — dual resolution):** Separate the temporal and spatial problems:
1. Temporal features at station/grid-cell level (full hourly resolution, no spatial aggregation)
2. Spatial features at ZCTA level (static infrastructure, demographics)
3. Join via a temporal-spatial attention mechanism in the surrogate model

### Impact If Not Fixed

The surrogate model cannot distinguish between:
- 500mm over 4 days (manageable) vs. 500mm in 12 hours (catastrophic)
- Peak surge during high tide (devastating) vs. peak surge during low tide (survivable)
- Simultaneous surge + rain (compound flooding) vs. sequential (less damaging)

These are the dominant drivers of flood damage severity. A model trained on temporal scalars will have a ceiling on predictive accuracy that no amount of hyperparameter tuning can overcome. The limitation should be stated in the paper's methodology section.

### Timestamp Handling Audit (Verified 2026-05-29)

UTC handling across the pipeline is consistent:
- NOAA tides: requested as `time_zone: "gmt"`, stored as tz-naive datetime64 (implicitly UTC)
- MRMS: filenames encode UTC hour, event windows use `tzinfo=timezone.utc`
- HURDAT2: NHC data is UTC; fetcher writes tz-naive; builder defensively localizes with `tz_localize("UTC")`
- Event windows in fetchers (broad) vs. peak_window in builder (narrow) are correctly nested

One bug: Imelda peak_window in build_event_dataset.py uses year 2017 instead of 2019 (line 607).

No UTC conversion errors were found. The problem is not timezone handling — it is the deliberate destruction of temporal resolution during scalar aggregation.

---

## 8. Column Name Mismatches — SILENT DATA LOSS

### Problem

Several `build_event_dataset.py` aggregation functions reference column names that don't match what the fetcher actually writes. Because all functions have graceful fallbacks (return NaN/empty), these mismatches produce a dataset that looks complete but has entire feature groups silently zeroed out.

### Mismatches Found (2026-05-29)

| Function | Expects | Fetcher Writes | Result |
|----------|---------|---------------|--------|
| `aggregate_tides()` | `water_level_m` | `observed_m` | All tides = NaN |
| `aggregate_tides()` | `station_id` column | Not present | Falls back to S3 key string |
| `compute_storm_proximity()` | `category` column | `max_wind_kt` + `status` | `storm_landfall_category` = NaN always |
| `aggregate_mrms_rainfall()` | Uncompressed `.grb2` | Gzip-compressed `.grib2.gz` | cfgrib fails silently, all rainfall = NaN |
| `aggregate_mrms_rainfall()` | Caps at 200 files | Harvey has 432 | Drops 232 hours (post-landfall) |
| `load_nfip_event_claims()` | `amountPaidOnBuildingClaim` | Column presence varies | Falls back to count (mislabeled as dollar loss) |

### Why These Are Dangerous

The pipeline "succeeds" — no crashes, no error exits. The output parquet has all expected columns. But tides, storm proximity, rainfall, and NFIP loss are NaN or wrong for reasons that are invisible without reading the code. A downstream modeler would see "sparse data" and attribute it to source gaps, not column name bugs.

### Suggested Fix

Add a post-assembly validation step that checks non-null rates for critical columns and warns if they fall below expected thresholds:

```python
EXPECTED_NON_NULL = {
    "rainfall_total_mm": 0.5,   # at least 50% of ZCTAs should have rainfall
    "max_surge_m": 0.3,          # coastal scenarios only
    "storm_min_dist_km": 0.9,    # almost all ZCTAs should have storm distance
    "peak_stage_ft": 0.1,        # gauge coverage is sparse but not zero
}
```

---

## 9. Heterogeneous Sensors -- DATA MODALITY INVENTORY

### Problem

The pipeline fuses 20+ data sources spanning fundamentally different sensor modalities -- radar composites, tide gauges, satellite imagery, administrative records, census surveys, hydrodynamic models, and hand-surveyed field marks. Each modality has its own spatial footprint, temporal cadence, error characteristics, and failure modes. The assembly pipeline treats them uniformly: download, extract one scalar, left-join to ZCTA. No metadata about the sensor origin survives into the output schema.

A downstream modeler or reviewer seeing `rainfall_total_mm` next to `impervious_pct` has no way to know that one comes from a 4.7 km radar composite updated hourly and the other from a 30m satellite classification updated once per decade.

### Complete Source-of-Field Inventory

#### Gridded Raster (Remote Sensing / Model Output)

| Source | Sensor/Platform | Native Resolution | Temporal Cadence | Format | Fetcher | Fields Derived |
|--------|----------------|-------------------|------------------|--------|---------|----------------|
| MRMS Stage IV | WSR-88D radar network + rain gauges (NOAA/NWS composite) | ~4.7 km (HRAP grid) | Hourly | GRIB2 (gzip) | `fetch_noaa_mrms_v2.py` | `rainfall_total_mm`, `max_rainfall_mm` |
| NLCD Impervious | Landsat 8/9 OLI (USGS/MRLC classification) | 30 m | Decadal (2021 vintage) | GeoTIFF / IMG | `fetch_nlcd_impervious.py` | `impervious_pct` |
| 3DEP DEM | Lidar + photogrammetry (USGS National Map) | 1/3 arc-sec (~10 m) | Static (latest available) | GeoTIFF | `fetch_3dep_dem.py` | `elevation_m_msl` |
| SLOSH Surge | Hydrodynamic model (NHC SLOSH MOM/MEOW) | ~200-500 m (basin-local grid) | Per-storm scenario | NetCDF/raster | `fetch_noaa_slosh.py` | `slosh_max_surge_m`, `slosh_category` |
| HRRR QPF | NWP model (NOAA HRRR v4) | 3 km | Hourly forecasts | GRIB2 | `fetch_noaa_hrrr.py` | (not yet assembled) |
| USGS Subsidence | InSAR (Sentinel-1 derived velocity) | ~90 m | Multi-year composite | GeoTIFF | `fetch_usgs_subsidence_no.py` (planned) | `subsidence_rate_mm_yr` |
| FloodSimBench | Physics-based urban flood model (1m synthetic) | 1 m | Per-scenario composite | GeoTIFF | `fetch_floodsimbench.py` | Inundation depth validation |

#### Point Time Series (In-Situ Gauges)

| Source | Sensor/Platform | Spatial Coverage | Temporal Cadence | Format | Fetcher | Fields Derived |
|--------|----------------|------------------|------------------|--------|---------|----------------|
| NOAA Tides | Acoustic/pressure tide gauges (CO-OPS stations) | 1-6 stations per region | 6-minute (aggregated to hourly) | JSON API -> Parquet | `fetch_noaa_tides.py` | `tidal_surge_max_m` |
| USGS NWIS | Stage/discharge gauges (USGS streamflow network) | ~50-200 per scenario | 15-min to hourly | JSON API -> Parquet | `fetch_usgs_nwis.py` | `peak_stage_ft` (planned) |

#### Point Observations (Field Survey / Crowd-Sourced)

| Source | Collection Method | Spatial Coverage | Temporal Resolution | Format | Fetcher | Fields Derived |
|--------|-------------------|------------------|---------------------|--------|---------|----------------|
| USGS HWMs | Post-event field survey (hand-placed marks) | ~50-100 per event, road-accessible | Single observation per event | JSON API -> Parquet | `fetch_surge_hwm.py` | `hwm_max_ft` |
| Houston 311 | Citizen phone/web reports | Urban only, self-selected | Individual timestamps | CSV (manual download) | `fetch_houston_311.py` | `flood_311_count` |
| NYC 311 | Citizen phone/web reports | Urban only, self-selected | Individual timestamps | NYC Open Data API | `fetch_nyc_311.py` | `flood_311_count` |

#### Point Track (Meteorological Best-Track)

| Source | Sensor/Platform | Spatial Coverage | Temporal Cadence | Format | Fetcher | Fields Derived |
|--------|----------------|------------------|------------------|--------|---------|----------------|
| HURDAT2 | NHC post-analysis best track (aircraft recon + satellite + surface obs) | Along-track fixes | 6-hourly | Fixed-width text -> Parquet | `fetch_hurdat2.py` | `storm_distance_km`, `storm_landfall_category` |

#### Vector Geometry (Boundary / Infrastructure)

| Source | What It Represents | Geometry Type | Vintage | Format | Fetcher | Fields Derived |
|--------|-------------------|---------------|---------|--------|---------|----------------|
| NHDPlus V2 | Hydrologic drainage network + catchments | Polyline + Polygon | Static (NHDPlus V2) | Shapefile/GeoParquet | `fetch_nhdplus_catchments.py` | `upstream_catchment_km2`, `bayou_segment_id`, `wash_segment_id` |
| TIGER Coastline | US coastline boundary | Polyline | 2020 census | Shapefile | (planned) | `coastal_distance_m` |
| NYC Sewersheds | Combined/separate sewer drainage areas | Polygon | Current | GeoPackage | `fetch_nyc_sewersheds.py` | `sewer_shed_id` |
| USACE Levees | Federal levee system locations + condition | Polyline + attributes | Current inspection | API -> Parquet | `fetch_usace_levees.py` | `levee_condition_rating` |
| HCFCD Districts | Harris County drainage/watershed management areas | Polygon | Current | Open Data portal | (planned) | `drainage_district_id` |
| OSM Canals (NOLA) | Drainage canals in Orleans Parish | Polyline | Current | Overpass API | (planned) | `canal_proximity_m` |
| MTBS Burn Scars | Post-fire burn perimeters (2015-2023) | Polygon | Annual | Shapefile/GeoParquet | `fetch_mtbs_burn_scars.py` | `burn_scar_overlap` |
| FEMA NFHL | Special Flood Hazard Areas (100-yr/500-yr zones) | Polygon | Varies by county | Shapefile | (in geocertdb2026) | `flood_zone_pct_A`, `flood_zone_pct_X` |

#### Tabular (Census / Survey / Administrative Records)

| Source | What It Represents | Spatial Unit | Vintage | Format | Fetcher | Fields Derived |
|--------|-------------------|-------------|---------|--------|---------|----------------|
| geocertdb2026 | ACS demographics, SVI, TWI, flood zones (31,789 ZCTAs, 106 columns) | ZCTA | 2020 ACS / 2022 SVI / 2021 NLCD | Parquet | `copy_geocertdb2026.py` | All Layer 2 static features |
| OpenFEMA NFIP | Flood insurance claims per disaster declaration | Reported address -> ZCTA | Per-event | API -> Parquet | `fetch_openfema_event.py` | `nfip_event_claims` |
| MTA Stations | Subway station locations | Point | Current | NYC Open Data API | `fetch_mta_stations.py` | `subway_station_count`, `nearest_subway_distance_m` |

#### Foundation Model Weights (EO Experts -- Not Tabular Data)

| Source | Architecture | Training Data | Role in Pipeline | Fetcher |
|--------|-------------|---------------|-----------------|---------|
| Prithvi-EO-2.0 | ViT (IBM/NASA) | HLS Sentinel-2 + Landsat | Primary geospatial expert | `fetch_prithvi_eo2.py` |
| TerraMind-base-Flood | Multimodal (IBM/ESA) | ImpactMesh-Flood (S1/S2/DEM) | Benchmark challenger expert | `fetch_terramind_flood.py` |
| Sen1Floods11 | N/A (benchmark dataset) | 11 global flood events, SAR + optical | Segmentation benchmark | `fetch_sen1floods11.py` |
| ImpactMesh-Flood | N/A (benchmark dataset) | S1RTC/S2L2A/DEM/masks | TerraMind training data | `fetch_impactmesh_flood.py` |

### Why This Matters

**1. Error propagation is modality-dependent.** A radar composite (MRMS) has spatially correlated errors across neighboring grid cells. A tide gauge has uncorrelated measurement noise. A census survey has sampling uncertainty at small-population ZCTAs. A physics model (SLOSH) has systematic bias from simplified bathymetry. Treating all as equally reliable scalars hides the dominant error source.

**2. Spatial representativeness varies by orders of magnitude.** One MRMS "observation" covers 22 km^2. One NLCD pixel covers 900 m^2. One tide gauge represents a single point broadcast to 200 ZCTAs. One HWM covers a 1m^2 mark on a building. The output table makes them look equivalent.

**3. Temporal cadence mismatch drives information loss.** Hourly sensors (MRMS, tides, NWIS) are collapsed to scalars. Decadal snapshots (NLCD, ACS) are treated as invariant. 6-hourly track data (HURDAT2) is reduced to minimum distance. The pipeline has no concept of temporal alignment across modalities (see Section 7).

**4. Some "data" is actually model output.** SLOSH surge grids are hydrodynamic model predictions, not observations. HRRR QPF is a numerical weather prediction forecast. FloodSimBench depths are physics-based simulations. These have fundamentally different error characteristics than observed data -- model bias, not measurement noise.

**5. Missing data has different semantics per modality.** HWM NaN means "no surveyor visited this ZCTA" (missing not at random -- surveyors go to damaged areas). Tidal NaN means "no station within 100 km" (geographic gap). NFIP claims = 0 could mean "no flood damage" or "no one had flood insurance" (censored data). The pipeline treats all NaN identically.

### What the Output Schema Should Track

For each derived feature, the output parquet should carry at minimum:

```
{feature_name}_source_type:     raster | gauge | survey | model | admin | census
{feature_name}_native_res:      "4700m" | "30m" | "point" | "zcta"
{feature_name}_n_obs:           number of raw observations contributing to this value
{feature_name}_dist_to_source:  distance (m) from ZCTA centroid to nearest observation
```

This enables the surrogate model (or a reviewer) to weight features by reliability and identify ZCTAs where a given feature is extrapolated beyond its support.

### Impact If Not Fixed

Without sensor provenance metadata, the surrogate model treats a ZCTA with 7,800 DEM pixels, 4 MRMS cells, 1 tide gauge at 80km, and 0 HWMs as having the same data quality as a ZCTA with 7,800 DEM pixels, 4 MRMS cells, 1 tide gauge at 2km, and 3 HWMs at 500m. The model cannot learn to down-weight unreliable inputs. A SIGSPATIAL reviewer will ask "how do you handle sensor heterogeneity?" -- this section is the answer and the plan.

---

## 10. Big Data vs Small Data -- SAMPLE SIZE ASYMMETRY

### Problem

The pipeline joins sources with sample sizes spanning 8 orders of magnitude. Some features are derived from millions of pixels; others from a handful of field observations. The output table flattens this into one row per (ZCTA, event), hiding a fundamental tension: the model is trained on ~200-800 ZCTAs per scenario but some features have rich support while others are near-vacuous.

### The Asymmetry

| Source | Raw Observations | ZCTAs Covered | Obs-per-ZCTA | Character |
|--------|-----------------|---------------|-------------|-----------|
| 3DEP DEM | ~500M pixels (98 tiles, 32.5 GB) | All | ~7,800 pixels | Big data: dense, uniform, high-confidence |
| NLCD Impervious | ~200M pixels per state | All | ~3,000 pixels | Big data: dense, uniform |
| MRMS Rainfall | 432 hourly grids (Harvey) | All | 1-4 cells x 432 hours | Medium: full spatial coverage, coarse resolution |
| geocertdb2026 | 31,789 ZCTAs x 106 columns | All | 1 row (census/survey) | Medium: complete coverage, survey-sampled |
| NOAA Tides | ~400 hourly readings x 3 stations | ~200 (broadcast) | 1 station value | Small data: 3 points represent 200 ZCTAs |
| USGS NWIS | ~2,000 readings x 50-200 gauges | ~10-30% of ZCTAs | 0-2 gauges | Small data: sparse, non-random placement |
| USGS HWMs | 50-100 marks per event | ~5-15% of ZCTAs | 0-3 marks | Very small: post-hoc, biased toward damage |
| 311 Reports | 1,000-10,000 per event | Urban ZCTAs only | 0-50 reports | Small, biased: self-selected reporters |
| OpenFEMA NFIP | Varies by event | Insured properties only | 0-100+ claims | Censored: 0 claims != 0 damage |

### Why This Is a Limitation (Not Fixable)

This asymmetry is inherent to the problem domain. You cannot conjure more tide gauges or more high-water marks. NOAA operates 3 stations in Galveston Bay because that is all the infrastructure that exists. HWMs are sparse because field survey teams are finite and access roads flood.

**What the paper should state:** "Feature reliability varies by orders of magnitude. DEM-derived features are supported by thousands of pixels per ZCTA; tidal features by a single station broadcast to hundreds of ZCTAs. We document this asymmetry in Table N and note that model performance on tidal features should be interpreted with caution."

### Mitigations (Within Reach)

1. **Observation count columns** (Section 9): `{feature}_n_obs` lets the model learn to down-weight thin features
2. **Hierarchical priors**: Scenario-level tidal signal as prior, ZCTA-level as residual (requires model architecture change)
3. **Explicit missingness encoding**: Distinguish "measured zero" from "no sensor" (NaN vs 0 for 311, NFIP)

---

## 11. Sensor Instability -- TEMPORAL COVERAGE GAPS AND FORMAT DRIFT

### Problem

Several data sources have known coverage gaps, format changes, or instrument discontinuities within the event windows used by this pipeline. The fetchers treat each source as stable and uniform. They are not.

### Known Instabilities

| Source | Instability | Events Affected | Impact |
|--------|-------------|-----------------|--------|
| MRMS product rename | `GaugeCorr_QPE_01H` (<=2020) -> `MultiSensor_QPE_01H_Pass2` (>=2021) | Ida 2021, Ian 2022, Hilary 2023, Beryl 2024 | Fetcher handles this (`_product_name()` switch). But the two products use different gauge correction algorithms -- they are not directly comparable across the 2020/2021 boundary. |
| MRMS archive gaps | Iowa State Mesonet occasionally missing hours; EMC fallback returns HTML error pages with HTTP 200 | All events | Fetcher has `MIN_FILE_SIZE` guard (10 KB) to reject HTML stubs. But missing hours create temporal gaps in rainfall accumulation. |
| NOAA CO-OPS station outages | Tide gauges go offline during extreme events (surge damage, power loss) | Harvey (Galveston), Ian (Fort Myers) | The most important hours (peak surge) are the most likely to be missing. Fetcher returns NaN; builder uses max of available hours. Peak surge may be underestimated. |
| HURDAT2 real-time vs post-analysis | 2024 storms (Beryl) may still be preliminary best-track, not final post-analysis | Beryl 2024 | Track positions and intensities may be revised. The HURDAT2 file used (hurdat2-1851-2023-051124.txt) ends at 2023 -- Beryl 2024 is NOT in this file. |
| NLCD vintage lag | 2021 NLCD reflects ~2019-2020 imagery | Events 2021-2024 | Post-2020 development (new subdivisions, parking lots) not captured. Houston metro adds ~1,500 acres impervious per year. |
| Census ACS vintage | 2020 5-year ACS in geocertdb2026 | Events 2022-2024 | Population shifts post-COVID not reflected. Some ZCTAs gained/lost 10%+ population 2020-2024. |
| OpenFEMA claims lag | NFIP claims take 6-18 months to close | Beryl 2024 | Claims data may be incomplete for recent events. |

### Why This Is a Limitation (Partially Fixable)

Instrument outages during extreme events are intrinsic -- you cannot fix a tide gauge that was destroyed by the surge you are trying to measure. Format changes are handled in code but introduce methodological discontinuity.

**Fixable:** Document which hours/stations are missing per event. Add `{feature}_coverage_pct` column.

**Not fixable:** The most extreme observations are the most likely to be missing (survivorship bias in instrumentation). This is a fundamental limitation of in-situ flood measurement.

---

## 12. Missing Variables -- KNOWN PREDICTORS NOT IN THE PIPELINE

### Problem

Flood damage research identifies several strong predictors that are absent from the pipeline, either because no public data source exists, because the fetcher has not been written, or because the variable requires manual curation.

### Missing Variable Inventory

| Variable | Why It Matters | Public Source | Status | Fixable? |
|----------|---------------|---------------|--------|----------|
| **HAND** (Height Above Nearest Drainage) | Dominant predictor of fluvial flood susceptibility; raw DEM is nearly meaningless without it | NOAA/OWP pre-computed 10m HAND rasters | No fetcher written | YES -- see Section 1 |
| **Soil permeability / SSURGO** | Controls infiltration rate; impervious surface alone misses pervious-but-saturated soil | USDA SSURGO/gSSURGO database | No fetcher written | YES -- gSSURGO is freely available |
| **Antecedent soil moisture** | Saturated soil before a storm dramatically increases runoff | NASA SMAP L4 (9 km, daily) | No fetcher written | YES -- but 9 km resolution is very coarse for ZCTA-level |
| **Storm drainage capacity** | Engineered drainage determines urban flood threshold | Municipal GIS (varies by city) | Houston: HCFCD has partial data. Others: no public source | PARTIAL -- Houston only, not standardized |
| **Building first-floor elevation** | Determines whether flooding = damage | FEMA elevation certificates (per-building) | Not publicly aggregated to ZCTA | NO -- individual building records, no bulk download |
| **Insurance penetration rate** | NFIP claims = 0 is ambiguous (no damage vs. no insurance) | FEMA policy-in-force counts by ZIP | Available but not fetched | YES -- would disambiguate NFIP outcome variable |
| **Compound flooding interaction** | Simultaneous surge + rain + tide phase | Derived from existing sources (MRMS + tides + HURDAT2) | Raw data exists; temporal join not performed | YES -- see Section 7 |
| **Urban canopy / tree cover** | Intercepts rainfall, slows runoff | NLCD Tree Canopy (30m, same source as impervious) | Same NLCD download, different product | YES -- trivial to add |
| **Flow accumulation / contributing area** | Upstream area draining through a ZCTA | Derivable from 3DEP DEM + NHDPlus flow direction | Raw data on S3; computation not implemented | YES -- standard GIS workflow |
| **Historical flood frequency** | How often a location has flooded before | USGS flood frequency statistics (PeakFQ) | Available per gauge, not per ZCTA | PARTIAL -- sparse gauge coverage |

### What This Means for the Paper

**Acknowledged limitations (state in Section 6 / Limitations):**
- Building-level elevation data is unavailable at scale
- Storm drainage capacity is operationally observed, not publicly archived (see FEATURE_CONTRACT `operational` class)
- Antecedent soil moisture at useful resolution requires a separate remote sensing pipeline

**Addressable before Data Lock A (June 1):**
- HAND (strongest single addition to predictive power)
- Compound flooding temporal features (Section 7, Option A)

**Addressable before SIGSPATIAL submission (June 5):**
- Insurance penetration rate (disambiguates NFIP outcome)
- Tree canopy (same NLCD source, minimal code)
- Flow accumulation (data already on S3)

---

## 13. Data Quality -- NO SYSTEMATIC VALIDATION FRAMEWORK

### Problem

The pipeline has no systematic data quality checks between fetch and assembly. Each fetcher writes to S3 with a manifest (record count, CRS, source URL), but no fetcher validates the *content* of what it downloaded. The assembly pipeline reads whatever is on S3 and trusts it.

### Quality Gaps by Source

| Source | What Could Go Wrong | Current Guard | What's Missing |
|--------|-------------------|---------------|----------------|
| MRMS GRIB2 | Corrupt/truncated files; HTML error pages saved as .grib2.gz | `MIN_FILE_SIZE = 10 KB` | No GRIB2 header validation; no check that precipitation values are physically plausible (0-500 mm/hr) |
| 3DEP DEM | Tiles with NoData voids (coastal edges, water bodies) | None | No check for NoData fraction; a tile that is 80% ocean voids still "succeeds" |
| NLCD Impervious | Projection mismatch (EPSG:5070 vs expected 4326) | None (bug -- Section 3) | No CRS assertion at read time |
| NOAA Tides | Station returns partial data (instrument outage during peak) | None | No completeness check (expected vs actual hourly records) |
| HURDAT2 | Storms of interest not found (wrong basin ID, file not updated for 2024) | Parser logs warning if 0 records | No assertion that expected storms are present |
| HWMs | Duplicate HWM IDs; elevation in wrong datum | None | No dedup; no datum check (NAVD88 vs local) |
| geocertdb2026 | Source parquet files missing from S3 (copy_geocertdb2026 not yet run) | `NoSuchKey` catch logs warning, continues | Silent success with missing files = incomplete Layer 2 |
| OpenFEMA | API pagination incomplete; rate limiting truncates results | None | No expected-vs-actual claim count validation |
| 311 Reports | Duplicate reports for same address; non-flood complaints included | None | No dedup; keyword filter may be too broad or too narrow |

### What a Validation Framework Would Look Like

**Post-fetch validation** (run after each fetcher, before assembly):

```python
# Example: validate_mrms.py
def validate_mrms(event: str):
    files = list_s3_keys(f"raw/noaa_mrms/{event}/")
    expected_hours = compute_expected_hours(EVENT_WINDOWS[event])
    
    assert len(files) >= expected_hours * 0.9, f"Coverage: {len(files)}/{expected_hours}"
    
    for f in sample(files, min(10, len(files))):
        ds = open_grib2(f)
        assert ds["tp"].max() < 500, f"Implausible precip: {ds['tp'].max()} mm"
        assert ds["tp"].min() >= 0, f"Negative precip in {f}"
```

**Pre-assembly gate** (run once before build_event_dataset.py):

```python
REQUIRED_DATASETS = {
    "houston": ["mrms/harvey2017", "tides/harvey2017", "hurdat2", "geocertdb2026"],
}
for ds in REQUIRED_DATASETS[scenario]:
    assert s3_prefix_has_files(f"raw/{ds}/"), f"Missing required dataset: {ds}"
```

### Why This Is a Limitation (Partially Fixable)

Some quality issues are fixable with validation code (plausibility bounds, completeness checks, CRS assertions). Others are inherent to the data sources (HWM datum inconsistency, 311 self-selection bias, NFIP censoring). The paper should distinguish between engineering defects (fixable) and data limitations (acknowledged).

**Fixable before Data Lock A:** Add `validate_data_lock_a.py` with the pre-assembly gate (dataset presence + basic completeness).

**Limitation for the paper:** "We do not perform pixel-level quality control on radar precipitation or satellite-derived land cover. Source-level QC is performed by the issuing agencies (NOAA/NWS for MRMS, USGS/MRLC for NLCD)."

---

## 14. Validation Method -- CONTRACT-DRIVEN THREE-LAYER FRAMEWORK

### Insight

`FEATURE_CONTRACT.yaml` already declares `build_function`, `output_column`, and `source_dataset` for every feature. This is not just documentation -- it is a machine-readable validation spec. The contract tells us exactly what each fetcher should produce, what each builder function should consume, and what columns should appear in the final output. Every mismatch documented in Sections 8, 11, and 13 is detectable by reading the contract and comparing it to reality.

### Architecture

Three validation layers, each gated before the next stage runs:

```
  FEATURE_CONTRACT.yaml
         |
         v
  Layer 1: Interface Contract Validation    (pre-assembly gate)
         |
         v
  Layer 2: Post-Assembly Validation         (built into builder)
         |
         v
  Layer 3: Data Lock Validation             (standalone QA script)
```

### Layer 1 -- Interface Contract Validation (Pre-Assembly Gate)

**When:** Before `build_event_dataset.py` runs. Blocks assembly if critical inputs are broken.

**What it checks:**

| Check | Source of Truth | Catches |
|-------|---------------|---------|
| S3 key exists for each `raw_s3_path` in contract | `FEATURE_CONTRACT.yaml` | Missing datasets (geocertdb2026 not copied, OpenFEMA not fetched) |
| Column names in raw parquet match what `build_function` expects | Fetcher output vs builder code | Column mismatches -- Section 8 (`observed_m` vs `water_level_m`, missing `category`) |
| File format matches expected (gzip detection, CRS assertion) | Contract `notes` + file headers | MRMS .grib2.gz handed to cfgrib without decompression -- Section 8 |
| File count >= expected minimum per event window | `EVENT_WINDOWS` dict + contract | MRMS missing hours, incomplete API pagination |
| CRS of raster files matches expected | Contract `notes` (EPSG codes) | NLCD EPSG:5070 vs assumed 4326 -- Section 3 |

```python
# validate_contract.py (importable module)
def validate_layer1(scenario: str, contract: list[dict]) -> list[ValidationResult]:
    """Pre-assembly gate. Returns list of PASS/FAIL/WARN per feature."""
    results = []
    for feature in contract:
        if feature["source_type"] == "operational":
            results.append(SKIP(feature, "operational -- no raw data expected"))
            continue

        # 1. Check raw data exists on S3
        raw_path = feature["raw_s3_path"]
        if raw_path and not s3_prefix_has_files(raw_path):
            results.append(FAIL(feature, f"No files at {raw_path}"))
            continue

        # 2. Check column names (parquet files only)
        if raw_path and raw_path.endswith(".parquet"):
            schema = read_parquet_schema(raw_path)
            expected_col = feature["output_column"]
            # Check the *input* columns the build_function reads
            # (derived from code inspection, stored in contract)
            ...

        # 3. Check CRS (raster files only)
        ...

        results.append(PASS(feature))
    return results
```

### Layer 2 -- Post-Assembly Validation (Built Into Builder)

**When:** After each `build_*` / `aggregate_*` function completes, before the final left-join.

**What it checks:**

| Check | Threshold | Catches |
|-------|-----------|---------|
| Non-null rate per output column | Feature-specific (see below) | Silent NaN floods from column mismatches or format errors |
| Row count matches expected ZCTA count for scenario | Exact match to geocertdb2026 scenario subset | Dropped or duplicated ZCTAs during joins |
| Value range plausibility | Feature-specific physical bounds | Negative rainfall, elevation > 8848m, surge > 20m |
| No duplicate (zcta_id, event) rows after each join | 0 duplicates | Merge fan-out from missing dedup (Section 8) |

**Expected non-null rates (minimum):**

```python
COVERAGE_THRESHOLDS = {
    # Event-window features -- most ZCTAs should have values
    "rainfall_total_mm":        0.50,   # some ZCTAs outside rain band
    "max_rainfall_mm":          0.50,
    "tidal_surge_max_m":        0.20,   # coastal scenarios only; inland = NaN
    "storm_distance_km":        0.90,   # almost all ZCTAs have track distance

    # Post-event labels -- inherently sparse
    "hwm_max_ft":               0.05,   # only 50-100 marks per event
    "flood_311_count":          0.10,   # urban only, self-selected
    "nfip_event_claims":        0.30,   # where insured properties exist

    # Static features -- should be complete
    "impervious_pct":           0.90,
    "elevation_m_msl":          0.95,
}
```

### Layer 3 -- Data Lock Validation (Standalone Script)

**When:** Run once before declaring Data Lock A (June 1) or Data Lock B (June 2). Produces a go/no-go report.

**What it checks:**

| Check | Method | Output |
|-------|--------|--------|
| Every feature in `FEATURE_CONTRACT.yaml` with `version_target <= current` has a non-null column in the output parquet | Read contract + read output parquet | Feature-level PASS/FAIL table |
| `temporal_class` boundary respected | No `post_event` or `operational` features used as model inputs | Leakage gate |
| Cross-scenario consistency | Same static features (geocertdb2026) produce same values across scenarios for overlapping ZCTAs | Drift detection |
| Output parquet schema matches contract | Column names, dtypes | Schema drift gate |
| Manifest reconciliation | Every `raw_s3_path` in contract has a corresponding manifest in `raw/manifests/` | Provenance chain |

```python
# validate_data_lock_a.py (standalone runner)
def main():
    contract = load_contract("FEATURE_CONTRACT.yaml")
    scenarios = ["houston", "southwest_florida", "nyc", "new_orleans", "riverside_coachella"]

    report = {}
    for scenario in scenarios:
        l1 = validate_layer1(scenario, contract)
        output_path = f"processed/{scenario}/{scenario}_event_features.parquet"
        if s3_key_exists(output_path):
            l3 = validate_layer3(scenario, contract, output_path)
        else:
            l3 = [FAIL("output_parquet", "Not yet assembled")]
        report[scenario] = {"layer1": l1, "layer3": l3}

    print_report(report)
    # Exit 1 if any FAIL in layer 1 or layer 3
    sys.exit(1 if any_failures(report) else 0)
```

### Implementation Plan

| Deliverable | File | Depends On | Target |
|-------------|------|-----------|--------|
| Validation module (Layers 1+2) | `jobs/validate_contract.py` | `FEATURE_CONTRACT.yaml` | Before Data Lock A |
| Layer 2 integration | `jobs/build_event_dataset.py` (import + inline checks) | `validate_contract.py` | Before first assembly run |
| Data Lock runner (Layer 3) | `scripts/validate_data_lock_a.py` | Assembled output parquets | June 1 |

### Relationship to Challenges

| Challenge Section | Validation Layer That Catches It |
|-------------------|--------------------------------|
| 3. CRS Mismatches | Layer 1 (CRS assertion on raster files) |
| 7. Temporal Collapse | Layer 2 (coverage threshold on temporal features) |
| 8. Column Name Mismatches | Layer 1 (schema check on raw parquets) |
| 10. Big Data vs Small Data | Layer 2 (`_n_obs` columns, coverage thresholds) |
| 11. Sensor Instability | Layer 1 (file count vs expected hours) |
| 12. Missing Variables | Layer 3 (contract reconciliation -- feature present vs absent) |
| 13. Data Quality | All three layers |

---

## References

- HAND methodology: Nobre et al. (2011), "Height Above the Nearest Drainage"
- NOAA/OWP HAND: https://cfim.ornl.gov/data/
- WBD (Watershed Boundary Dataset): https://www.usgs.gov/national-hydrography/watershed-boundary-dataset
- MRMS grid specification: https://mrms.nssl.noaa.gov/
- NLCD projection: EPSG:5070 (NAD83 / Conus Albers)
