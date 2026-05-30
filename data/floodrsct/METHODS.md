# FloodRSCT Data Processing Methods

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

### Limitations

- Nearest-centroid assignment does not account for ZCTA shape or area. Large
  ZCTAs may have significant within-ZCTA precipitation variation.
- MRMS is radar-based and can underestimate precipitation in regions with poor
  radar coverage or beam blockage (mountainous terrain).

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
