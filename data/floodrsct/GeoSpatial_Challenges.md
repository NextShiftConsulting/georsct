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

## References

- HAND methodology: Nobre et al. (2011), "Height Above the Nearest Drainage"
- NOAA/OWP HAND: https://cfim.ornl.gov/data/
- WBD (Watershed Boundary Dataset): https://www.usgs.gov/national-hydrography/watershed-boundary-dataset
- MRMS grid specification: https://mrms.nssl.noaa.gov/
- NLCD projection: EPSG:5070 (NAD83 / Conus Albers)
