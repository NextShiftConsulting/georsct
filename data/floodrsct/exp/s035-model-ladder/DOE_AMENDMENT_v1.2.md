# DOE Amendment v1.2: Verified Feature Engineering + Kappa Diagnostics

**Amends:** DOE_LOCKED.md v1.1 (2026-05-29)
**Date:** 2026-06-01
**Reason:** R1/R2 feature bundles in v1.1 were aspirational. This amendment
replaces them with concrete features derivable from verified raw data on S3.
Adds kappa proxy diagnostics as the mechanism for H4.

---

## Change 1: R1 Feature Bundle (replaces §Representation Bundles, R1 row)

### What v1.1 said
> R1 = R0 + HUC/catchment area, NHDPlus stream density, HCFCD district,
> levee rating, sewershed

### What actually exists on S3

| Raw Source | S3 Location | Size | Coverage |
|-----------|-------------|------|----------|
| NHDPlus catchments VPU 12 | `raw/nhdplus/catchments/v2/catchments_vpu12.parquet` | 15 MB | Houston ✓ |
| NHDPlus catchments VPU 18 | `raw/nhdplus/catchments/v2/catchments_vpu18.parquet` | 71 MB | Riverside ✓ |
| NHDPlus catchments VPU 02 | **NOT ON S3** | ~50 MB | NYC — needs fetch |
| NHDPlus catchments VPU 03 | **NOT ON S3** | ~80 MB | SW Florida — needs fetch |
| NHDPlus catchments VPU 08 | **NOT ON S3** | ~60 MB | New Orleans — needs fetch |
| NHDPlus flowlines | **NOT ON S3** | — | Stream density impossible without this |
| HCFCD districts | `raw/hcfcd/drainage_districts/v1/hcfcd_districts.parquet` | 1 MB | Houston only, already wired |
| USACE levees (NOLA) | `raw/usace_levees/new_orleans_levees.parquet` | 18 KB | Join never ran |
| USACE levees (NYC) | `raw/usace_levees/nyc_levees.parquet` | 10 KB | Join never ran |
| NYC sewersheds | `raw/nyc_sewersheds/nyc_sewersheds.gpkg` | 2.8 MB | Join never ran |
| NLCD impervious | `raw/nlcd/` | 26 GB | Zonal stats never ran |

### Revised R1 features

R1 = R0 + the following columns:

**Universal (all scenarios):**

| Column | Source | Computation | Expected Coverage |
|--------|--------|-------------|-------------------|
| `nhd_catchment_area_km2` | NHDPlus catchment polygons | ZCTA centroid → spatial join → catchment area_sq_km | >90% (some centroids miss) |
| `slope_basin_slope` | Already in parquet | Direct use | 85-99% |
| `slope_stream_slope` | Already in parquet | Direct use | 85-99% |
| `twi_acc_twi` | Already in parquet | Direct use | 85-99% |
| `twi_tot_twi` | Already in parquet | Direct use | 85-99% |
| `drive_min_to_county_centroid` | Already in parquet | Direct use | 85-99% |
| `drive_min_to_county_seat` | Already in parquet | Direct use | 85-99% |
| `drive_min_to_nearest_hospital` | Already in parquet | Direct use | 85-99% |
| `hifld_n_hospital_beds` | Already in parquet | Direct use | 85-99% |
| `hifld_n_pharmacies` | Already in parquet | Direct use | 85-99% |
| `hifld_nearest_pharmacy_km` | Already in parquet | Direct use | 85-99% |
| `hifld_nearest_trauma_center_km` | Already in parquet | Direct use | 85-99% |
| `flood_deaths` | Already in parquet | Direct use | 85-99% |
| `flood_injuries` | Already in parquet | Direct use | 85-99% |
| `nfip_total_building_loss` | Already in parquet | Direct use | 85-99% |
| `nfip_total_contents_loss` | Already in parquet | Direct use | 85-99% |

**Scenario-specific:**

| Column | Scenarios | Source | Computation |
|--------|-----------|--------|-------------|
| `upstream_catchment_km2` | Houston, Riverside | NHDPlus (already wired) | Direct use |
| `hcfcd_drainage_district` | Houston only | HCFCD parquet | Already wired as bayou_segment_id |
| `levee_nearest_km` | NOLA, NYC | USACE levees | ZCTA centroid → nearest levee segment (haversine) |
| `levee_condition_rating` | NOLA, NYC | USACE levees | Condition rating of nearest levee |
| `sewershed_type` | NYC only | NYC sewersheds gpkg | ZCTA centroid → spatial join → combined/separate/NA |
| `slosh_max_surge_m` | SW Florida | Already in parquet (51%) | Direct use |

**Dropped from v1.1:**

| Feature | Why Dropped |
|---------|-------------|
| NHDPlus stream density | No flowline layer on S3; fetching NHDPlus flowlines is a separate 2+ GB download per VPU |
| Sewershed (non-NYC) | No national sewershed dataset; city-specific data only available for NYC |

**R1 total: R0 (36 features) + 16 universal + 1-3 scenario-specific = 52-55 features**

### Prerequisites (SageMaker jobs)

1. **Job A: Fetch NHDPlus VPUs 02, 03, 08** — extend `fetch_nhdplus_catchments.py`
2. **Job B: Build R1 supplement** — new `build_r1_features.py`:
   - Spatial join: ZCTA centroids → NHDPlus catchments → nhd_catchment_area_km2
   - Spatial join: ZCTA centroids → USACE levees → levee_nearest_km, levee_condition_rating
   - Spatial join: ZCTA centroids → NYC sewersheds → sewershed_type
   - Output: `processed/{scenario}/{scenario}_r1_supplement.parquet`

---

## Change 2: R2 Feature Bundle (replaces §Representation Bundles, R2 row)

### What v1.1 said
> R2 = R1 + peak 1h/3h rainfall, storm duration, surge-rain overlap,
> lag between peaks

### What actually exists on S3

| Raw Source | Files | Total Size | Events Covered |
|-----------|-------|-----------|----------------|
| MRMS hourly grib2 | 2,156 | 1.2 GB | harvey2017(432), ian2022(189), ida2021_nola(168), ida2021_nyc(72), hilary2023(96), milton2024(168), helene2024(168), beryl2024(144), henri2021(96), imelda2019(120), ar_flood_2023(503) |
| Tide gauge CSVs | 78 | 2.9 MB | All events, 6-12 stations each |
| HWMs | 10 events | 0.2 MB | Varies by event |
| HURDAT2 tracks | 2 | 6.7 MB | All named storms |

### Revised R2 features

R2 = R1 + the following event-dynamic columns:

| Column | Source | Computation | Expected Coverage |
|--------|--------|-------------|-------------------|
| `peak_1h_mm` | MRMS hourly grib2 | Max single-hour rainfall at ZCTA centroid | >90% |
| `peak_3h_mm` | MRMS hourly grib2 | Max 3-hour rolling sum at ZCTA centroid | >90% |
| `peak_6h_mm` | MRMS hourly grib2 | Max 6-hour rolling sum at ZCTA centroid | >90% |
| `storm_duration_h` | MRMS hourly grib2 | Hours where rainfall > 1 mm at ZCTA centroid | >90% |
| `time_to_peak_h` | MRMS hourly grib2 | Hours from first rain to peak at ZCTA centroid | >90% |
| `rainfall_intensity_cv` | MRMS hourly grib2 | CV of hourly rainfall during storm hours | >90% |
| `tide_peak_m` | Tide gauge CSVs | Peak water level from nearest station | Coastal only (~50%) |
| `surge_rain_lag_h` | Tide + MRMS | Hours between peak rainfall and peak surge | Coastal only (~50%) |
| `storm_approach_speed_kph` | HURDAT2 | Storm translation speed at closest approach | Named storms only |

**R2 total: R1 (52-55 features) + 9 temporal = 61-64 features**

### Computation approach

The existing `_mrms_spatial_aggregate()` in `build_event_dataset.py` already:
- Downloads + decodes hourly grib2 files via ProcessPoolExecutor
- Samples at ZCTA centroids via nearest-neighbor on the CONUS grid
- Accumulates a running sum

The change: instead of discarding per-hour values after summing, accumulate
a (n_zctas × n_hours) matrix. Memory: 600 ZCTAs × 500 hours × 4 bytes =
1.2 MB — trivial. Then compute temporal statistics from the matrix.

### Prerequisites (SageMaker job)

**Job C: Build R2 temporal features** — new `build_r2_features.py`:
1. For each event: load all hourly MRMS grib2 files
2. Sample each grid at ZCTA centroids → build hourly rainfall matrix
3. Compute rolling maxima, duration, timing statistics per ZCTA
4. Load tide gauge CSVs → assign nearest station per ZCTA → compute surge timing
5. Output: `processed/{scenario}/{scenario}_r2_supplement.parquet`

---

## Change 3: Kappa Proxy Diagnostics (new section, implements H4 mechanism)

### Motivation

DOE v1.1 H4 says "audit flags predict representation uplift" but does not
specify which flags or how they are computed. This amendment defines four
kappa proxy diagnostics computed from R0 outputs that serve as the audit
flags for H4.

### Four kappa proxies from R0

All computed per (scenario, target, solver) cell from R0 results:

**1. kappa_leakage** — Autocorrelation leakage indicator (maps to GeoRSCT A.1)

```
kappa_leakage = 1 - (metric_random - metric_spatial) / max(metric_random, 0.01)
```

- Low → random split inflated by spatial autocorrelation → spatial features
  (R1) should help because the model is exploiting neighbor similarity that
  won't generalize
- High → performance robust to spatial blocking → R0 already captures spatial
  structure

**2. kappa_transfer** — Cross-event generalization (maps to GeoRSCT D.3)

```
kappa_transfer = max(0, metric_leave_event / max(metric_spatial, 0.01))
```

- Low → model can't generalize across events → event-specific features (R2)
  should help
- High → static features transfer across events → R2 adds noise

**3. kappa_solver** — Solver agreement (model uncertainty diagnostic)

```
kappa_solver = 1 - |metric_hgbdt - metric_ridge| / max(|metric_hgbdt|, |metric_ridge|, 0.01)
```

- Low → linear and nonlinear solvers disagree → complex representation
  structure the simple model can't capture
- High → linear signal dominates → R0 may suffice

**4. kappa_residual_spatial** — Residual clustering (maps to GeoRSCT A.2/B.1)

```
kappa_residual_spatial = 1 - |Moran's I on HistGBDT residuals|
```

Requires adjacency matrix. If `zcta_adjacency.parquet` unavailable, use
k=5 nearest-neighbor spatial weights from centroid coordinates.

- Low → prediction errors cluster geographically → R1 spatial features should
  reduce clustering
- High → errors spatially random → model captures spatial structure adequately

### Money table design (paper Figure 2)

Each row = one (scenario, target) combination. Columns:

| scenario | target | kappa_leak | kappa_xfer | kappa_solve | kappa_resid | R0_metric | R1_metric | R2_metric | R0→R1_pct | R1→R2_pct |
|----------|--------|------------|------------|-------------|-------------|-----------|-----------|-----------|-----------|-----------|

H4 tests:
- Spearman(kappa_leakage, R0→R1 uplift) — do leaky models gain more from R1?
- Spearman(kappa_transfer, R1→R2 uplift) — do transfer-failing models gain more from R2?
- Spearman(kappa_residual_spatial, R0→R1 uplift) — do spatially-clustered-error models gain more from R1?

### Sample size caveat

Maximum usable (scenario, target) cells:
- Houston: 3 targets (NFIP, 311, HWM)
- SW Florida: 1 target (NFIP only)
- NYC: 2 targets (NFIP partial, 311)
- Riverside: 1 target (NFIP partial)
- New Orleans: dropped (n=20)

Total: **7 cells.** Spearman on n=7 is marginal. Report:
1. Effect size (rho) with bootstrap 95% CI (10,000 resamples)
2. Exact permutation p-value (not asymptotic)
3. Scatter plot with labeled points for visual inspection
4. Honest statement: "with 7 scenario-target pairs, we report observed
   associations; confirmation requires more scenarios"

If we compute per-solver as well (HistGBDT and Ridge independently), n
doubles to 14. But solver isn't independent — report both pooled and
stratified.

---

## Change 4: Revised Experiment Matrix

| Phase | Script | Input | Output | Prereqs | Instance | Est. Duration |
|-------|--------|-------|--------|---------|----------|---------------|
| A | `fetch_nhdplus_catchments.py` (extended) | EPA NHDPlus V2.1 | VPU 02/03/08 parquets on S3 | None | ml.m5.large | ~15 min |
| B | `build_r1_features.py` (new) | NHDPlus + levees + sewersheds + assembled parquet | `{scenario}_r1_supplement.parquet` | Phase A | ml.m5.xlarge | ~30 min/scenario |
| C | `build_r2_features.py` (new) | MRMS grib2 + tide CSVs + assembled parquet | `{scenario}_r2_supplement.parquet` | None | ml.m5.xlarge | ~60 min/scenario |
| 1 | `train_r0_baseline.py` (existing) | Assembled parquet | Folds + R0 metrics + predictions | None | ml.m5.xlarge | ~5 min/scenario |
| 2 | `train_r1_hydrology.py` (new) | Assembled + R1 supplement + R0 folds | R1 metrics + predictions | Phase B, Phase 1 | ml.m5.xlarge | ~5 min/scenario |
| 3 | `train_r2_temporal.py` (new) | Assembled + R1 + R2 supplements + R0 folds | R2 metrics + predictions | Phase C, Phase 2 | ml.m5.xlarge | ~5 min/scenario |
| 4 | `compute_kappa_diagnostics.py` (new) | R0 predictions + adjacency | Kappa proxy table | Phase 1 | local or ml.m5.large | ~5 min |
| 5 | `compute_uplift_table.py` (new) | All results JSONs + kappa table | Money table + H2-H4 evidence | Phase 3, Phase 4 | local | ~2 min |

### Critical path

```
          Phase A (NHDPlus fetch, 15 min)
               |
               v
Phase 1 ----> Phase B (R1 features, 30 min) ----> Phase 2 (R1 train)
(R0 train)         |                                    |
   |               |                                    v
   +----------> Phase 4 (kappa diagnostics)      Phase 3 (R2 train)
                                                       |
Phase C (R2 features, 60 min) -------------------------+
                                                       |
                                                       v
                                                 Phase 5 (money table)
```

Parallel lanes:
- Phase A + Phase C + Phase 1 can all run simultaneously
- Phase B depends on Phase A only
- Phase 2 depends on Phase B + Phase 1 (folds)
- Phase 3 depends on Phase C + Phase 2
- Phase 4 depends on Phase 1 only
- Phase 5 depends on Phase 3 + Phase 4

**Wall-clock estimate: ~2.5 hours** (if A, C, 1 run in parallel, then B, then 2, then 3+4 in parallel, then 5)

---

## Change 5: New Orleans downgraded to illustrative

New Orleans has 20 ZCTAs — too few for meaningful ML (5-fold CV = 4 rows per
test fold). Downgrade from "modelable scenario" to "illustrative example" in
the paper. The paper shows the certificate audit for NOLA as a data-quality
case study, not a prediction result.

Modelable scenarios: Houston (396), SW Florida (606), NYC (422), Riverside (172).

---

## Change 6: Target availability by scenario

| Scenario | obs_nfip_event_claims | obs_has_311 | obs_has_hwm | Modelable targets |
|----------|----------------------|-------------|-------------|-------------------|
| Houston | 100% | 100% (via complaints_311_count) | 100% (via hwm_count) | 3 |
| SW Florida | 100% | 0% | 0% | 1 |
| NYC | 50% | 100% | 0% | 2 |
| Riverside | 50% | 0% | 0% | 1 |

Total modelable cells: 7 (scenario × target combinations with sufficient data).

---

## Change Control

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-05-29 | Initial DOE (DRAFT) |
| v1.1 | 2026-05-29 | LOCKED for execution |
| v1.2 | 2026-06-01 | Verified R1/R2 features from S3 inventory; added kappa proxy design; downgraded NOLA; added assembly job specs |
