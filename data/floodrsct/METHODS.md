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
