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
| ~~`nfip_total_building_loss`~~ | REMOVED | Target leakage | — |
| ~~`nfip_total_contents_loss`~~ | REMOVED | Target leakage | — |

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

**1. diag_leakage** — Autocorrelation leakage indicator (maps to GeoRSCT A.1)

```
diag_leakage = 1 - (metric_random - metric_spatial) / max(metric_random, 0.01)
```

- Low → random split inflated by spatial autocorrelation → spatial features
  (R1) should help because the model is exploiting neighbor similarity that
  won't generalize
- High → performance robust to spatial blocking → R0 already captures spatial
  structure

**2. diag_transfer** — Cross-event generalization (maps to GeoRSCT D.3)

```
diag_transfer = max(0, metric_leave_event / max(metric_spatial, 0.01))
```

- Low → model can't generalize across events → event-specific features (R2)
  should help
- High → static features transfer across events → R2 adds noise

**3. diag_solver** — Solver agreement (model uncertainty diagnostic)

```
diag_solver = 1 - |metric_hgbdt - metric_ridge| / max(|metric_hgbdt|, |metric_ridge|, 0.01)
```

- Low → linear and nonlinear solvers disagree → complex representation
  structure the simple model can't capture
- High → linear signal dominates → R0 may suffice

**4. diag_residual_spatial** — Residual clustering (maps to GeoRSCT A.2/B.1)

```
diag_residual_spatial = 1 - |Moran's I on HistGBDT residuals|
```

Requires adjacency matrix. If `zcta_adjacency.parquet` unavailable, use
k=5 nearest-neighbor spatial weights from centroid coordinates.

- Low → prediction errors cluster geographically → R1 spatial features should
  reduce clustering
- High → errors spatially random → model captures spatial structure adequately

### Money table design (paper Figure 2)

Each row = one (scenario, target) combination. Columns:

| scenario | target | diag_leak | diag_xfer | diag_solve | diag_resid | R0_metric | R1_metric | R2_metric | R0→R1_pct | R1→R2_pct |
|----------|--------|------------|------------|-------------|-------------|-----------|-----------|-----------|-----------|-----------|

H4 tests:
- Spearman(diag_leakage, R0→R1 uplift) — do leaky models gain more from R1?
- Spearman(diag_transfer, R1→R2 uplift) — do transfer-failing models gain more from R2?
- Spearman(diag_residual_spatial, R0→R1 uplift) — do spatially-clustered-error models gain more from R1?

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
| 4 | `compute_diagnostics.py` (new) | R0 predictions + adjacency | Kappa proxy table | Phase 1 | local or ml.m5.large | ~5 min |
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

## Change 7: Progressive Kappa Cascade (replaces one-shot kappa design in Change 3)

### Motivation

Change 3 computed kappa diagnostics from R0 only. A PhD reviewer should be able
to verify: (1) the diagnostic predicted the fix BEFORE the fix was applied, and
(2) the diagnostic changed AFTER the fix — not just that uplift correlated with
a post-hoc metric. Progressive recomputation at each level provides both the
prediction and the confirmation.

### Cascade Protocol

Kappa diagnostics are computed THREE times — once per representation level:

| Phase | Input | Output | What It Proves |
|-------|-------|--------|----------------|
| 4a | R0 predictions | `diagnostics_r0.json` | Pre-registered predictions: which scenarios should benefit from R1 |
| 4b | R1 predictions | `diagnostics_r1.json` | Confirmation: did R1 fix the flagged spatial issues? |
| 4c | R2 predictions | `diagnostics_r2.json` | Confirmation: did R2 fix the flagged temporal issues? |

### Diagnostic Movement Table (paper Figure 3)

Each row = one (scenario, target) cell. Shows kappa values moving through levels:

```
scenario | target | diag_leak_R0 | diag_leak_R1 | diag_leak_R2 | diag_xfer_R0 | diag_xfer_R1 | diag_xfer_R2 | diag_resid_R0 | diag_resid_R1 | diag_resid_R2
```

**Expected behavior if the cascade works:**

| Diagnostic | R0 -> R1 expected | R1 -> R2 expected | Interpretation |
|-----------|-------------------|-------------------|----------------|
| diag_leakage | INCREASE (spatial structure now captured) | STABLE (R2 doesn't target spatial) | R1 fixed the spatial autocorrelation exploitation |
| diag_transfer | STABLE (R1 doesn't target temporal) | INCREASE (event dynamics now captured) | R2 fixed the cross-event generalization failure |
| diag_residual_spatial | INCREASE (errors decorrelate) | STABLE or INCREASE | R1 removed geographic clustering of errors |
| diag_solver | INCREASE or STABLE | INCREASE or STABLE | Feature addition resolves model disagreement |

**Disconfirmation patterns (equally reportable):**
- diag_leakage doesn't increase after R1: spatial features didn't address the autocorrelation
- diag_transfer doesn't increase after R2: temporal features didn't help generalization
- Kappa goes DOWN: added features introduced noise or new failure modes

### Revised Experiment Matrix (replaces Change 4)

| Phase | Script | Input | Output | Prereqs |
|-------|--------|-------|--------|---------|
| 1 | `train_r0_baseline.py` | Assembled parquet | R0 metrics + predictions parquet | None |
| 4a | `compute_diagnostics.py --level r0` | R0 predictions | `diagnostics_r0.json` | Phase 1 |
| 2 | `train_r1_hydrology.py` | Assembled + R1 supplement + R0 folds | R1 metrics + predictions parquet | Data team + Phase 1 |
| 4b | `compute_diagnostics.py --level r1` | R1 predictions | `diagnostics_r1.json` | Phase 2 |
| 3 | `train_r2_temporal.py` | Assembled + R1 + R2 supplements + R0 folds | R2 metrics + predictions parquet | Data team + Phase 2 |
| 4c | `compute_diagnostics.py --level r2` | R2 predictions | `diagnostics_r2.json` | Phase 3 |
| 5 | `compute_uplift_table.py` | All results + all kappa files | Money table + cascade table + H2-H4 evidence | All above |

### Critical Path (revised)

```
Phase 1 (R0) -> Phase 4a (kappa R0) -> Phase 2 (R1) -> Phase 4b (kappa R1) -> Phase 3 (R2) -> Phase 4c (kappa R2) -> Phase 5
```

Phases are now SEQUENTIAL by design. This is intentional: each kappa computation
must complete before the next training phase starts, to establish the temporal
ordering that proves predictions were not post-hoc.

**S3 timestamps prove ordering.** Each kappa file's S3 LastModified timestamp
precedes the training run that tests its predictions.

---

## Change 8: Anti-Cherry-Picking Protocol

### Motivation

With 7 cells, 4 kappa proxies, and 2 uplift measurements (R0->R1, R1->R2), a
reviewer should be confident results are not cherry-picked from 8+ possible
correlations. This protocol makes cherry-picking structurally impossible.

### Pre-Registration (before each training phase)

Before R1 trains, `diagnostics_r0.json` is uploaded to S3 with:

```json
{
  "level": "r0",
  "timestamp": "ISO-8601 (proves this precedes R1)",
  "predictions": {
    "r1_should_help_most": ["houston", "nyc"],
    "r1_should_help_least": ["southwest_florida"],
    "ordering_criterion": "diag_leakage ascending (lowest = most predicted uplift)"
  },
  "kappa_values": { ... },
  "flag_threshold": "median_split"
}
```

Similarly, before R2 trains, `diagnostics_r1.json` predicts which scenarios
should benefit from R2 (based on diag_transfer).

### Flag Threshold: Median Split (pre-committed)

No threshold tuning. For each kappa proxy:
- Cells with kappa BELOW median = "flagged" (predicted to benefit from the fix)
- Cells with kappa ABOVE median = "unflagged" (predicted to not benefit)

Median split is chosen because:
- With n=7, any fixed threshold risks having 0 or 7 cells in one group
- Median guarantees 3-4 in each group
- No degrees of freedom for the researcher to exploit

### All-Cells Reporting (mandatory)

The money table reports EVERY cell, not just those that confirm the hypothesis.
Required columns:

```
scenario | target | flagged_by | predicted_uplift | observed_uplift | correct?
```

Where `correct?` = (flagged AND uplift > median) OR (unflagged AND uplift <= median).

**Hit rate** = fraction of cells where prediction matched. Report with exact
binomial CI. For n=7, even 7/7 correct has wide CI — report honestly.

### Multiple Comparison Correction

> **AMENDED (v1.7/v1.8):** Primary test is now fold-level Wilcoxon (single
> test, no correction needed). Cell-level associations are exploratory.

**Primary test (H2a):** Single fold-level Wilcoxon signed-rank (R0 vs R1).
No multiple comparison correction required.

**Exploratory cell-level (H2b, H4):**
8 associations: 4 diagnostics x 2 transitions. At n=7:

1. **All 8 Spearman correlations** with individual bootstrap 95% CIs
2. **Effect sizes and directional consistency** (fraction positive)
3. **kappa_geom** (not diag_leakage) as the cell-level predictor for H2b
4. **Holm-Bonferroni** reported for transparency but noted as decorative at n=7

### Negative Results Protocol

If the primary hypothesis fails (H2a Wilcoxon p >= 0.05 or Cohen's d < 0.2):

- **DO report** the null result with effect size and CI
- **DO report** whether any of the 7 exploratory tests showed signal
- **DO report** whether the cascade movement table shows expected patterns
  (kappas improving after fixes) even if the prediction-uplift correlation is weak
- **DO NOT** switch the primary hypothesis post-hoc
- **Frame as:** "The diagnostic cascade shows interpretable movement, but the
  prediction-uplift link requires more scenarios to confirm"

### Prediction Parquets (new output from training scripts)

All training scripts save per-row predictions for Moran's I computation:

```
results/s035/{level}_{scenario}_predictions.parquet
```

Schema:
- `zcta_id`: str
- `event`: str
- `target`: str (column name)
- `solver`: str
- `split`: str (spatial_blocked only — reference split)
- `fold`: str
- `y_true`: float
- `y_pred`: float

Only spatial_blocked split predictions are saved (the reference split for all
kappa computations). Random and leave-event-out metrics are used for kappa
formulas but their per-row predictions are not needed.

### Kappa Formula Standardization

All kappa formulas use a "higher is better" primary metric:
- Regression targets: R2 score
- Classification targets: ROC-AUC

This avoids sign confusion with RMSE (lower = better). R2 can be negative
(worse than mean predictor), which is informative, not problematic.

---

## Change 9: RSCT Certification Layer (Phase 4.5)

### Motivation

The s035 experiment uses yrsn's `compute_kappa_spatial` for Moran's I, but the
experiment currently stops at kappa diagnostics. The paper claims GeoRSCT can
**certify** representation quality — so each (scenario, target, level) cell
should receive an RSN certificate from yrsn's production certificate pipeline.
This closes the loop: the same system that diagnoses also certifies.

### Architecture: yrsn as Measurement Plane

```
s035 training scripts          yrsn (measurement plane)
─────────────────────          ─────────────────────────
train_r0_baseline.py ────┐
train_r1_hydrology.py ───┼──> compute_diagnostics.py ──> kappa proxies
train_r2_temporal.py ────┘            │
                                      v
                              compute_certificates.py (NEW)
                                      │
                         ┌────────────┼────────────┐
                         v            v            v
                    YRSNCertificate  alpha    sigma/omega
                    (R, S_sup, N)   (quality)  (stability)
```

yrsn provides the **measurement and certification infrastructure**. The solvers
remain standalone sklearn/torch — transparent for the paper, auditable by
reviewers. yrsn interprets their outputs.

### Certificate Computation Per Cell

For each (scenario, target, level) cell, compute an RSN certificate using
yrsn's `YRSNCertificate` from `yrsn.core.certificates.core`:

```python
from yrsn.core.certificates.core import YRSNCertificate
from yrsn.core.quality.alpha import compute_alpha
from yrsn.core.quality.omega import compute_omega
```

**Mapping solver outputs → RSN certificate:**

| Certificate Field | Source | Formula |
|-------------------|--------|---------|
| R | Spatial-blocked R2 (HistGBDT) | `R = clamp(r2_spatial_blocked, 0, 1)` |
| S_sup | Random-vs-spatial gap (leakage) | `S_sup = clamp(r2_random - r2_spatial, 0, 1)` |
| N | 1 - R - S_sup | Simplex closure: `N = 1 - R - S_sup` |
| alpha | R / (R + N) | `compute_alpha(R, N)` — quality signal |
| omega | 1 - S_sup | `compute_omega(S_sup)` — in-distribution confidence |
| kappa | From diagnostics | Mean of 4 kappa proxies for the cell |
| sigma | Across-fold kappa variance | `np.std([kappa_fold_1, ..., kappa_fold_k])` |

**Why this mapping works:**
- R = spatial-blocked performance → how much real signal the representation captures
- S_sup = leakage → how much "superfluous" signal inflates random-split performance
- N = residual → noise / unexplained
- R + S_sup + N = 1 by construction (simplex guaranteed)

**Edge cases:**
- Negative R2: set R=0, distribute to N. `N = 1 - S_sup`.
- S_sup > 1: clamp to 1, set R=0, N=0 (pure leakage).

### Certificate Evolution Table (paper Figure 4)

One row per cell, showing certificate movement across levels:

```
scenario | target | R_R0 | S_R0 | N_R0 | kappa_R0 | R_R1 | S_R1 | N_R1 | kappa_R1 | R_R2 | S_R2 | N_R2 | kappa_R2 | verdict
```

**Expected patterns:**
- R increases R0→R1→R2 (representation captures more signal)
- S_sup decreases R0→R1 (spatial features reduce leakage)
- N decreases overall (less unexplained variance)
- kappa increases (quality gates would pass more cells at higher levels)

**Verdict column:**
- `PASS`: kappa ≥ 0.7 at final level (yrsn gate 4 threshold)
- `WARN`: 0.5 ≤ kappa < 0.7
- `FAIL`: kappa < 0.5

### New Script: `compute_certificates.py`

Phase 4.5 — runs after each kappa diagnostics phase, before next training phase.

```
Usage:
    python compute_certificates.py --level r0 --upload
    python compute_certificates.py --level r1 --upload
    python compute_certificates.py --level r2 --upload
```

Input: `results/s035/{level}_{scenario}.json` + `diagnostics_{level}.json`
Output: `results/s035/certificates_{level}.json`

Uses:
- `yrsn.core.certificates.core.YRSNCertificate` — certificate construction
- `yrsn.core.quality.alpha.compute_alpha` — alpha computation
- `yrsn.core.quality.omega.compute_omega` — omega computation

Does NOT use:
- Gate pipeline (SequentialGatekeeper) — we report kappa vs threshold, not enforce
- DGM — that's Change 10
- Rotor/embeddings — certificates are from solver metrics, not embedding decomposition

### Paper Value

This makes the paper claim concrete:

> "Each representation level receives an RSN certificate from the RSCT quality
> pipeline. Certificate R (signal captured) increases from 0.XX at R0 to 0.XX
> at R2, while S_sup (leakage) decreases from 0.XX to 0.XX. Of 7 cells, X
> achieve kappa ≥ 0.7 (PASS) at R2 vs Y at R0 — demonstrating that the
> representation ladder progressively resolves quality failures the diagnostic
> identified."

### Revised Phase Ordering

```
Phase 1 (R0) -> Phase 4a (kappa R0) -> Phase 4.5a (cert R0) -> Phase 2 (R1) -> Phase 4b (kappa R1) -> Phase 4.5b (cert R1) -> Phase 3 (R2) -> Phase 4c (kappa R2) -> Phase 4.5c (cert R2) -> Phase 5
```

Certification adds ~1 min per level (no training, just arithmetic on existing results).

---

## Change 10: DGM as Orchestrator (Paper Section 8)

### Motivation

The experiment ladder tests representations (R0/R1/R2) × solvers (HistGBDT/Ridge)
independently. In production, a system should **route** each ZCTA to the optimal
(representation, solver) combination. yrsn's Dual Graph Morph (DGM) is exactly
this routing layer — it uses certificate quality signals to dispatch typed morph
operators. Demonstrating DGM on s035 outputs shows the system that diagnosed the
problem can also prescribe the fix.

### DGM Is Not a Solver

DGM does not train or predict. It is the **control plane** that routes inputs to
solvers based on quality certificates. The s035 experiment provides the evidence
DGM needs to make routing decisions:

```
┌─────────────────────────────────────────────────────────┐
│  DGM Control Plane (yrsn.core.dgm_unified)              │
│                                                         │
│  G_R: ZCTA nodes with certificate (R, S_sup, N, kappa) │
│  G_S: Solver nodes {HistGBDT_R0, HistGBDT_R1, ...}     │
│  phi: Morphism routing ZCTAs → optimal solver           │
│                                                         │
│  Morph operators (from certificate signals):            │
│    PRUNING    → kappa < 0.3: drop cell (too noisy)     │
│    RE_ENCODE  → S_sup high: upgrade representation     │
│    REPAIR     → sigma high: ensemble solvers            │
│    EXECUTE    → kappa >= 0.7: proceed with best solver │
│                                                         │
│  Input: s035 certificates + solver results              │
│  Output: per-ZCTA routing recommendation                │
└─────────────────────────────────────────────────────────┘
```

### Mapping s035 Cells to DGM

| DGM Concept | s035 Equivalent |
|-------------|-----------------|
| G_R event node | (scenario, target, zcta_id) — each ZCTA is a node with certificate from Phase 4.5 |
| G_S agent node | (solver, level) combination — e.g., HistGBDT_R1 |
| phi morphism | Certificate-based routing: which (solver, level) is best for this ZCTA? |
| MorphType.PRUNING | kappa < 0.3 → ZCTA is unpredictable, flag for human review |
| MorphType.EVENT_EXPANSION | S_sup high at R0 → upgrade to R1 (add spatial features) |
| MorphType.REPAIR | sigma high → ensemble HistGBDT + Ridge (hedge disagreement) |
| MorphType.VERIFICATION | kappa ≥ 0.7 → certify and proceed |

### DGM Routing Table (paper Table 4)

Post-hoc analysis on s035 results. For each (scenario, target) cell:

```
scenario | target | cert_R0 | cert_R1 | cert_R2 | morph_decision | recommended_arm | actual_best_arm | correct?
```

Where `morph_decision` is one of:
- `EXECUTE@R0` — R0 certificate passes, no upgrade needed
- `RE_ENCODE→R1` — R0 certificate fails on S_sup, upgrade to R1
- `RE_ENCODE→R2` — R1 certificate fails on diag_transfer, upgrade to R2
- `REPAIR@R1` — R1 solvers disagree (low diag_solver), ensemble
- `PRUNING` — All levels fail, flag for review

### What This Tests (H5 — new exploratory hypothesis)

**H5 (exploratory):** DGM morph routing, informed by progressive certificates,
selects the same (representation, solver) arm that exhaustive comparison shows is
best.

This is NOT a trained routing model. DGM uses fixed certificate thresholds from
yrsn's gate pipeline:
- kappa ≥ 0.7 → EXECUTE (yrsn gate 4 threshold)
- S_sup > 0.2 → RE_ENCODE (representation leakage)
- sigma > 0.15 → REPAIR (solver instability from oobleck)
- kappa < 0.3 → PRUNING (noise-dominated)

**Hit rate:** Does DGM's recommendation match exhaustive best? Report as:
- Fraction of cells where DGM selects the arm with highest R2
- Fraction where DGM selects within 0.02 R2 of best (near-optimal)

With 7 cells this is descriptive, not inferential. The paper frames it as
proof-of-concept, not validated production routing.

### New Script: `compute_dgm_routing.py`

Phase 6 — runs after Phase 5 (needs all certificates + all results).

```
Usage:
    python compute_dgm_routing.py --upload
```

Input:
- `results/s035/certificates_{r0,r1,r2}.json` — progressive certificates
- `results/s035/{level}_{scenario}.json` — solver results per arm

Output: `results/s035/dgm_routing.json`

Uses:
- `yrsn.core.dgm_unified.DualGraphSystem` — graph construction
- `yrsn.core.dgm_unified.MorphType` — typed morph operators
- `yrsn.core.certificates.core.YRSNCertificate` — certificate reading
- Thresholds from `yrsn_controlplane.SequentialGatekeeper` defaults

Does NOT use:
- `run_dual_graph()` directly — s035 has no agent nodes, only solver nodes.
  We construct a simplified DGM with ZCTA→solver routing only.
- Training or inference — DGM is routing logic, not a model.

### Paper Framing

Section 8 (Discussion) or Section 7 (Extended Results):

> "We demonstrate that the RSCT Dual Graph Morph, using progressive certificates
> from the diagnostic cascade, routes each ZCTA to the representation level that
> the kappa diagnostics identified as necessary. Of 7 cells, DGM's RE_ENCODE
> morph correctly identified X cells needing representation upgrade, while
> EXECUTE correctly identified Y cells where R0 sufficed. The routing hit rate
> (Z/7) suggests that certificate-driven orchestration can replace exhaustive
> grid search over the representation × solver space."

If hit rate is low:

> "DGM routing matched exhaustive best in X/7 cells. Mismatches occurred where
> [specific kappa diagnostic was miscalibrated / threshold was too conservative].
> This identifies calibration requirements for production deployment of
> certificate-driven routing."

### Revised Complete Phase Ordering

```
Phase 1 (R0 train)
  -> Phase 4a (kappa R0)
    -> Phase 4.5a (cert R0)
      -> Phase 2 (R1 train)
        -> Phase 4b (kappa R1)
          -> Phase 4.5b (cert R1)
            -> Phase 3 (R2 train)
              -> Phase 4c (kappa R2)
                -> Phase 4.5c (cert R2)
                  -> Phase 5 (money table + hypothesis tests)
                    -> Phase 6 (DGM routing analysis)
```

Phase 6 adds ~2 min (routing logic on existing certificates, no training).

**Total wall-clock estimate: ~2.5 hours + 10 min overhead** (certification and
routing are arithmetic on existing results).

---

## Change 11: FEMA FAST Integration (Feature Source + External Validation)

### Motivation

s035 trains on NFIP insurance claims as the primary target. Insurance claims
reflect what was *filed*, not what was *damaged* — subject to policy uptake
bias, claim processing delays, and coverage limits. FEMA's Flood Assessment
Structure Tool (FAST) provides an independent **engineering estimate** of
flood losses using Hazus depth-damage functions applied to every structure.
This creates two opportunities:

1. **Feature source:** FAST's per-structure damage estimates aggregated to
   ZCTA become representation features — the output of a physics-based
   engineering model, distinct from both statistical features (R0) and raw
   hydrology (R1).

2. **External validation:** Correlation between s035 model-ladder predictions
   and FAST engineering estimates validates that the statistical model
   captures real flood damage signal, not just insurance filing patterns.

### What FAST Is

FEMA Flood Assessment Structure Tool — Hazus depth-damage applied per structure:

- **Input A:** Building inventory (occupancy type, sqft, first-floor height,
  foundation type, stories, replacement cost) — from NSI 2.0
- **Input B:** Flood depth raster (.tiff, feet) — event-specific depth grids
- **Process:** Assigns Hazus depth-damage functions per building, extracts
  flood depth at each structure location, computes dollar losses
- **Output:** Per-structure economic loss ($), building damage state, debris
- **Speed:** ~10,000 structures/second
- **Implementation:** `sphere-flood` (`HazusFloodAnalysis`) wrapped by
  `floodcaster` — NOT HazPy. The Hazus engine is already in our stack.

### FAST as Feature Source (R1.5 — Engineering Model Features)

FAST outputs aggregated to ZCTA become a new representation class. These are
NOT raw measurements (R0) or hydrology (R1) or temporal dynamics (R2) — they
are **engineering model outputs** that encode Hazus domain knowledge about
how flood depth translates to structural damage.

**FAST-derived ZCTA features:**

| Column | Aggregation | Signal |
|--------|-------------|--------|
| `fast_mean_loss_per_sqft` | Mean(structure_loss / sqft) per ZCTA | Average damage intensity |
| `fast_max_loss_usd` | Max(structure_loss) per ZCTA | Worst-case structure |
| `fast_pct_damaged` | Count(loss > 0) / Count(all) per ZCTA | Damage penetration rate |
| `fast_total_loss_usd` | Sum(structure_loss) per ZCTA | Total ZCTA exposure |
| `fast_median_depth_ft` | Median(flood_depth_at_structure) per ZCTA | Typical inundation |
| `fast_n_structures` | Count(structures) per ZCTA | Building density |

**Where FAST features sit in the ladder:**

```
R0: Static tabular (36 features)
R1: + Hydrology/infrastructure (52-55 features)
R1.5: + FAST engineering model outputs (6 features)  <-- NEW
R2: + Temporal event dynamics (61-64 features)
```

R1.5 is between R1 and R2 because FAST features are event-specific (need
flood depth per event) but derived from static building inventory + engineering
functions — they don't capture temporal dynamics (rainfall timing, surge lag).

**Alternative: FAST as R2 supplement.** If flood depth rasters are only
available for events where we also have MRMS temporal data, FAST features can
be folded into R2 instead of creating a separate level. The choice depends on
data availability — if FAST features exist for ALL events (using static BFE
depth instead of event-specific depth), they belong at R1.5. If only for events
with depth grids, they go into R2.

### FAST as External Validation Target (Phase 7)

After the model ladder completes (R0→R1→R2 predictions), compare predictions
against FAST's independent engineering estimates:

**Validation protocol:**

For each (scenario, event) pair where FAST outputs exist:

1. Aggregate FAST per-structure losses to ZCTA level → `fast_total_loss_zcta`
2. Compare against s035 model predictions:
   - Spearman rho(predicted_nfip_claims, fast_total_loss_zcta)
   - Per-level: does correlation improve R0→R1→R2?
3. Compare against observed NFIP claims:
   - Spearman rho(observed_nfip_claims, fast_total_loss_zcta) — baseline
   - If s035 prediction correlates with FAST better than raw NFIP correlates
     with FAST → model captures damage signal beyond insurance filing patterns

**Validation table (paper Table 5):**

```
scenario | event | rho(NFIP_obs, FAST) | rho(pred_R0, FAST) | rho(pred_R1, FAST) | rho(pred_R2, FAST) | interpretation
```

**Expected patterns:**
- rho(NFIP_obs, FAST) is the ceiling — perfect model can't exceed this
  (NFIP and FAST measure different things)
- rho(pred, FAST) increases R0→R1→R2 → model progressively captures
  engineering-validated damage signal
- If rho(pred_R2, FAST) > rho(NFIP_obs, FAST) → model generalizes beyond
  insurance filing patterns (strong result)

**H6 (exploratory):** Model-ladder predictions at R2 correlate more strongly
with FAST engineering estimates than R0 predictions do, validating that
representation upgrades capture physical damage signal.

### Data Prerequisites — Verified Against S3

**Flood depth rasters: AVAILABLE (Houston, NYC, coastal)**

S3 inventory (2026-06-01) confirms depth rasters already on S3:

| Source | S3 Location | Coverage | Format | Status |
|--------|-------------|----------|--------|--------|
| **FloodSimBench** (Chrimerss et al.) | `raw/floodsimbench/6hr_max/` | Houston (7 tiles: HOU001-HOU007) + NYC (2 tiles: NYC001-NYC002), 10 return periods each (1-yr to 1000-yr) | MaxDepth .tif, ~40 MB/tile | **READY** — 78 files, 3 GB |
| **SLOSH MOM** (national) | `raw/noaa_slosh/mom_national/` | Cat 1-5 inundation HIGH, all coastal scenarios | Inundation .tif, ~1.2 GB/cat | **READY** — 5 categories, 5.5 GB |
| **3DEP DEM** | `raw/dem/3dep/v1/houston/` (14 tiles), `new_orleans/` | Houston, New Orleans | 1/3 arc-second elevation .tif | **READY** — for BFE-ground depth |

**FloodSimBench is the primary FAST input.** It provides physics-based 2D
flood simulation MaxDepth grids — exactly what FAST expects. The 10 return
periods (1-yr through 1000-yr, 48mm to 181mm 6-hr rainfall) let us run FAST
across a range of severity levels, not just one event.

**SLOSH MOM** provides surge inundation for coastal storms — applicable to
SW Florida (Ian, Helene, Milton) and NYC coastal flooding.

**NSI 2.0: NOT ON S3 — data team must fetch**

| Dataset | S3 target | Source | Size | Status |
|---------|-----------|--------|------|--------|
| NSI 2.0 structures — Houston | `raw/nsi/v2/houston_structures.parquet` | [NSI 2.0 API](https://nsi.sec.usace.army.mil/nsiapi/) | ~50 MB | **NEEDED** |
| NSI 2.0 structures — NYC | `raw/nsi/v2/nyc_structures.parquet` | NSI 2.0 API | ~80 MB | **NEEDED** |
| NSI 2.0 structures — SW Florida | `raw/nsi/v2/southwest_florida_structures.parquet` | NSI 2.0 API | ~40 MB | **NEEDED** (stretch) |

NSI 2.0 fields required by FAST: `fd_id, x, y, occtype, sqft, found_type,
num_story, found_ht, val_struct, val_cont, med_yr_blt`

NSI 2.0 is publicly available via USACE API — no license issues. Bulk download
per county FIPS is straightforward.

### Scenario Scoping (verified against S3 data)

| Scenario | Depth Source | Quality | FAST Role | Tier |
|----------|-------------|---------|-----------|------|
| **Houston** | FloodSimBench 7 tiles x 10 return periods | Physics-based 2D sim, full metro | Feature source + validation | **Primary** |
| **NYC** | FloodSimBench 2 tiles (Manhattan only) | Physics-based, partial coverage | Validation only (partial) | **Primary** |
| **SW Florida** | SLOSH MOM Cat 1-5 | Surge inundation only, no pluvial | Validation only (coastal) | **Stretch** |
| **Riverside** | None on S3 | No depth raster available | **EXCLUDED** from FAST | N/A |
| **NOLA** | Illustrative only (n=20) | N/A | N/A | N/A |

**Houston is the headline FAST scenario.** Full metro coverage, 10 severity
levels, physics-based depth grids. NYC adds a second city but only Manhattan.
SW Florida is stretch — SLOSH surge is a different flooding mechanism than
the pluvial flooding FloodSimBench models.

### Architecture Boundary: floodcaster vs s035

Generic flood analysis capabilities are open-source API calls in
`github/floodcaster`. Experiment-specific scripts in s035 call floodcaster.

**floodcaster additions (reusable, open-source):**

| New Module | Migrates From | Capability |
|------------|--------------|------------|
| `floodcaster/nsi_sources.py` | `rsct-geocert/data/floodrsct/jobs/fetch_nsi_structures.py` | Fetch NSI 2.0 buildings by bbox/FIPS from USACE API, map to sphere schema |
| `floodcaster/aggregation.py` | New | Spatial join buildings to ZCTA polygons, compute group stats (sum, mean, count, percentile) |
| `floodcaster/analysis.py` update | New function | `run_nsi_flood_analysis()` — NSI buildings + depth raster -> per-building losses via sphere `HazusFloodAnalysis` |

**Existing sphere classes used by floodcaster:**
- `sphere.core.schemas.fast_buildings.FastBuildings` / `NsiBuildings`
- `sphere.flood.analysis.hazus_flood.HazusFloodAnalysis`
- `sphere.flood.default_vulnerability.DefaultFloodVulnerability`

**s035 scripts (experiment-specific, call floodcaster):**

**`run_fast_zcta.py`** — Phase 7a. Selects s035 scenarios + return periods,
calls `floodcaster.analysis.run_nsi_flood_analysis()` and
`floodcaster.aggregation.aggregate_by_zcta()`, extracts 6 ZCTA features.
Output: `processed/{scenario}/{scenario}_fast_zcta.parquet`

**`compute_fast_validation.py`** — Phase 7b. Reads ZCTA aggregates + s035
predictions, computes Spearman correlations across levels (H6). Output:
`results/s035/fast_validation.json`

### Revised Complete Phase Ordering

```
Phase 1 (R0 train)
  -> Phase 4a (kappa R0) -> Phase 4.5a (cert R0)
    -> Phase 2 (R1 train)
      -> Phase 4b (kappa R1) -> Phase 4.5b (cert R1)
        -> [Optional] Phase 2.5 (R1.5 train with FAST features)
          -> Phase 3 (R2 train)
            -> Phase 4c (kappa R2) -> Phase 4.5c (cert R2)
              -> Phase 5 (money table + hypothesis tests)
                -> Phase 6 (DGM routing)
                  -> Phase 7 (FAST validation)
```

Phase 2.5 is conditional on FAST features being ready (NSI + depth rasters
on S3). If not ready by launch, skip R1.5 and use FAST only for validation
(Phase 7). FAST validation runs last because it needs all predictions from
all levels.

### Paper Value

**As features (Section 5, Results):**

> "At R1.5, we add FAST engineering model outputs — per-ZCTA damage estimates
> from Hazus depth-damage functions applied to National Structure Inventory
> buildings. This representation encodes domain knowledge about how flood depth
> translates to structural damage. R1.5 improves R2 by X% / does not improve,
> suggesting that [engineering model signal is/is not] complementary to
> statistical features."

**As validation (Section 6, Validation):**

> "To validate that the model ladder captures physical flood damage signal
> rather than insurance filing patterns, we compare predictions against FEMA
> FAST engineering estimates — an independent physics-based damage model.
> Spearman correlation between R2 predictions and FAST estimates (rho = 0.XX)
> exceeds R0 correlation (rho = 0.XX), confirming that representation upgrades
> capture engineering-validated damage structure."

**As limitation (Section 8, if FAST correlation is weak):**

> "FAST validation reveals a ceiling: NFIP claims and engineering damage
> estimates correlate at rho = 0.XX, reflecting fundamental differences between
> insurance-reported and physics-modeled losses. The model ladder's prediction-
> FAST correlation (rho = 0.XX) approaches this ceiling at R2, suggesting the
> remaining gap is in the target definition, not the representation."

---

## Change 12: GeoParquet Standardization

### Problem

All spatial outputs in the pipeline write plain parquet with float lat/lon
columns. No GeoParquet RFC metadata. Downstream tools (DuckDB spatial, QGIS,
kepler.gl, geopandas `read_parquet`) cannot recognize these as spatial data
without manual coordinate-column guessing.

Current state (audited 2026-06-01):
- Raw fetchers (fetch_tiger_coastline, fetch_nhdplus, etc.) use geopandas
  `to_parquet()` which writes WKB geometry — but the downstream builders
  strip geometry to scalar lat/lon before writing final outputs
- Assembled event parquets are plain pandas DataFrames
- `zcta_adjacency.parquet` has no geometry at all (edge list only)
- ZCTA centroids exist as float columns (`centroid_lat`, `centroid_lon`) but
  no GeoParquet Point geometry

### Requirement

All spatial parquets produced by s035 MUST use GeoParquet format:

```python
import geopandas as gpd

# When writing any parquet with spatial coordinates:
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
gdf.to_parquet(path)  # Writes GeoParquet with RFC metadata
```

### What Changes

| Output | Current | Required |
|--------|---------|----------|
| `{scenario}_event_features.parquet` | pandas, float lat/lon | GeoParquet, Point geometry, EPSG:4326 |
| `{scenario}_r1_supplement.parquet` | pandas, float lat/lon | GeoParquet, Point geometry, EPSG:4326 |
| `{scenario}_r2_supplement.parquet` | pandas, float lat/lon | GeoParquet, Point geometry, EPSG:4326 |
| `{scenario}_fast_zcta.parquet` | (new) | GeoParquet, Point geometry, EPSG:4326 |
| `zcta_adjacency.parquet` | edge list, no geometry | Add ZCTA centroid Points for both endpoints |
| `{level}_{scenario}_predictions.parquet` | pandas, zcta_id only | Add ZCTA centroid Point geometry |

### Why This Matters for the Paper

1. **Reproducibility:** GeoParquet is the OGC standard (adopted 2023). A
   reviewer running `gpd.read_parquet()` gets a GeoDataFrame with correct CRS
   — no guessing which column is latitude.

2. **Visualization:** Prediction maps, residual maps, and certificate maps
   can be rendered directly from predictions parquet without a separate
   coordinate join.

3. **Spatial operations:** W-matrix construction (Change 13) needs geometry.
   If predictions are GeoParquet, `libpysal.weights.Queen.from_dataframe(gdf)`
   works out of the box.

4. **SIGSPATIAL venue:** A spatial computing paper that doesn't use spatial
   data formats is a red flag for reviewers.

### Implementation

Data team adds `geopandas` to SageMaker requirements (already available in
most SageMaker images). Training scripts add a centroid join before writing
prediction parquets. No changes to model logic.

**Cost:** ~10 lines per script. Centroid lookup from `zcta_features_labels.parquet`
(already loaded).

---

## Change 13: Spatial W-Matrix Features (R1 Representation Fix)

### Problem

R1 claims to add "spatial features" but the current R1 bundle (Change 1) has
**zero spatial structure**. All R1 features are point attributes at each ZCTA
(catchment area, slope, TWI, infrastructure counts). None encode the spatial
*relationships* between ZCTAs.

Audited 2026-06-01:
- `spatial_autocorrelation_range_km` is declared in all 5 regime configs
  (houston.yaml, etc.) but **never read by any Python code** — dead config
- `zcta_adjacency.parquet` (Queen's contiguity edge list) is on S3 but only
  used for post-hoc Moran's I in `compute_diagnostics.py`
- No `libpysal` or `pysal` imported anywhere in the pipeline
- No spatial lag features, no spatial error terms, no eigenvector spatial
  filtering, no variogram features
- Spatial blocking for CV uses county/ZIP3 groups, not W-matrices
- `audit_mode_a3_smooth_map.py` (spatial smoothing audit) is marked NOT_READY

This is the gap diag_leakage is designed to detect: models exploit spatial
autocorrelation in random splits but have no explicit spatial features to
capture it in spatial-blocked splits. Adding W-matrix features to R1 is the
fix R1 was always supposed to provide.

### W-Matrix Construction

Use the existing `zcta_adjacency.parquet` to build a proper spatial weights
matrix via libpysal:

```python
from libpysal.weights import W

# From existing adjacency edge list
adj_df = load_adjacency(s3)  # Already in _coverage_common.py
neighbors = adj_df.groupby('zcta_id_1')['zcta_id_2'].apply(list).to_dict()
w = W(neighbors)
w.transform = 'r'  # Row-standardized
```

### W-Matrix Derived Features (added to R1)

| Column | Formula | Signal |
|--------|---------|--------|
| `wlag_nfip_claims` | W * y (spatial lag of target) | Neighbor flood activity — strongest spatial signal |
| `wlag_flood_zone_pct` | W * flood_zone_pct_in_sfha | Neighborhood flood zone density |
| `wlag_population_density` | W * acs_population_density | Neighborhood urbanization |
| `wlag_median_income` | W * acs_median_household_income | Neighborhood socioeconomic context |
| `wlag_impervious_pct` | W * impervious surface fraction | Neighborhood runoff potential |
| `spatial_lag_residual_R0` | W * residual(R0) | Spatial error correction from R0 predictions |
| `zcta_degree` | Count of queen-contiguous neighbors | Connectivity / edge vs interior |
| `zcta_mean_neighbor_dist_km` | Mean centroid distance to neighbors | Spatial density of urban fabric |

**R1 total (revised): R0 (36) + 16 universal + 1-3 scenario-specific + 8 W-matrix = 61-63 features**

### Spatial Lag of Target: Leakage Risk

`wlag_nfip_claims` (W * y) is the most powerful spatial feature but carries
leakage risk in spatial-blocked CV because neighbors may span block boundaries.

**Mitigation:**
1. Compute spatial lag WITHIN each CV fold's training set only — never from
   test ZCTAs. This means spatial lag is a per-fold computed feature, not a
   static column.
2. For ZCTAs at block boundaries, use only within-block neighbors for the lag.
3. Report with and without `wlag_nfip_claims` as an ablation — this is the
   single feature most likely to drive R0→R1 uplift AND most likely to be
   accused of leakage. The ablation defuses the criticism.

```python
# Per-fold spatial lag (no leakage)
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
    y_train = y.iloc[train_idx]
    # Compute lag only from training ZCTAs
    w_train = w.subset(train_idx)  # libpysal subset
    lag_train = w_train.sparse @ y_train.values
    lag_test = np.nan  # Test ZCTAs get no lag (or lag from training neighbors only)
```

### Spatial Error Correction: R0 Residual Lag

`spatial_lag_residual_R0` is computed AFTER R0 trains (Phase 1):
1. Get R0 predictions on spatial-blocked folds
2. Compute residuals: `r = y_true - y_pred_R0`
3. Compute spatial lag of residuals: `W * r`
4. This becomes an R1 feature: "how wrong was R0 for my neighbors?"

This is the spatial error model (SEM) insight encoded as a feature. If R0
errors cluster spatially (low diag_residual_spatial), this feature captures
the pattern R0 missed.

**Critical:** This feature creates a dependency: R1 features require R0
predictions. This is already the case in the sequential pipeline
(Phase 1 → Phase 2) but now it's a data dependency, not just a phase ordering.

### Impact on Kappa Diagnostics

If W-matrix features work correctly:
- diag_leakage should INCREASE R0→R1 (spatial autocorrelation now captured
  by explicit features, not leaked through random splits)
- diag_residual_spatial should INCREASE R0→R1 (residual clustering reduced
  by spatial error correction feature)
- diag_solver may INCREASE (spatial lag gives Ridge something to work with
  that previously only HistGBDT could learn implicitly)

This makes the kappa cascade story much stronger — the diagnostics flag
spatial problems at R0, W-matrix features at R1 fix them, and the kappa
movement proves it.

### Dependencies

| Dependency | Status |
|-----------|--------|
| `libpysal` | Not in SageMaker requirements — add to `requirements.txt` |
| `zcta_adjacency.parquet` on S3 | **READY** — already used by compute_diagnostics.py |
| R0 predictions (for residual lag) | Produced by Phase 1 |
| GeoParquet (Change 12) | Needed for `W.from_dataframe()` convenience, but edge list works without it |

### Paper Value

W-matrix features are the mechanism that makes R1 work. Without them, R1 is
just "more point attributes" — a reviewer would rightfully ask why adding
slope and hospital distance reduces spatial autocorrelation. With W-matrix
features, R1 explicitly encodes spatial structure:

> "R1 augments R0 with two types of spatial information: (1) point attributes
> from hydrology and infrastructure (catchment area, TWI, levee proximity),
> and (2) spatial lag features derived from a Queen's contiguity weight matrix
> (neighbor flood claims, neighbor flood zone density, R0 residual spatial
> lag). The latter explicitly addresses the spatial autocorrelation that
> diag_leakage detects at R0."

---

## Change Control

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-05-29 | Initial DOE (DRAFT) |
| v1.1 | 2026-05-29 | LOCKED for execution |
| v1.2 | 2026-06-01 | Verified R1/R2 features from S3 inventory; added kappa proxy design; downgraded NOLA; added assembly job specs |
| v1.3 | 2026-06-01 | Progressive kappa cascade; anti-cherry-picking protocol; prediction parquets; sequential phase ordering; multiple comparison correction |
| v1.4 | 2026-06-01 | RSCT certification layer (yrsn as measurement plane); DGM orchestrator routing analysis; H5 exploratory hypothesis; revised 6-phase pipeline |
| v1.5 | 2026-06-01 | FEMA FAST scoped to Houston (primary) + NYC (primary) + SW Florida (stretch); verified depth rasters on S3 (FloodSimBench + SLOSH MOM); NSI 2.0 only missing data |
| v1.6 | 2026-06-01 | GeoParquet standardization (all spatial outputs); W-matrix spatial features in R1 (spatial lag, residual correction); libpysal dependency; dead config cleanup |
| v1.7 | 2026-06-02 | Pre-registered primary outcomes; NFIP temporal gate (IBNR boundary); storm features moved to R2; fold-level Wilcoxon as primary test; random-features ablation; git provenance in launchers |
| v1.8 | 2026-06-02 | Kappa independence: geometry-only kappa (Phase 0.5); model-derived diagnostics demoted to diagnostic fields, not kappa inputs; causal boundary guard on training features; R1/R2 supplements required |
| v1.9 | 2026-06-02 | Statistical reporting framework: three-axis distinction (RSN simplex / kappa_geom / bootstrap CI); cell-level bootstrap CIs on aggregate uplift and certificate signals; bootstrap framed as uncertainty not inference |
| v1.10 | 2026-06-05 | Spatially-blocked paired loss analysis (Statistical-Considerations.md §14): supplementary test using per-ZCTA loss deltas aggregated to county-level spatial blocks; four-layer separation (spatial dependence / inference / certificate / DGM); naming rules; applies to H2a, H3, H5 |
| v1.11 | 2026-06-05 | Event-level dependence (Statistical-Considerations.md §14.5): two-stage aggregation rule for multi-event scenarios; cascade effect table for adding events post-DOE (R2 hard regeneration, NFIP historical regeneration, fold balance re-verification); V4.10/V4.11/V5.15/V5.16 verification gates |

---

## Kappa Independence (v1.8)

**Principle:** kappa measures geometric compatibility (D*/D), not model
quality.  It must have **zero computational dependency** on the RSN
simplex (R, S_sup, N), fold metrics, predictions, residuals, or model
scores.

### What changed

The prior design computed kappa from `diag_leakage`, which is
`1 - (random_metric - spatial_metric) / |random_metric|` — a monotone
transform of S_sup.  This made kappa algebraically dependent on the
simplex, defeating its purpose as an independent diagnostic axis.

### New design

kappa is computed in **Phase 0.5** (`compute_geometry_kappa.py`), which
runs **before any model training**.  It reads only:

| Input | Source | Model-free? |
|-------|--------|-------------|
| Adjacency graph | `zcta_adjacency.parquet` | Yes |
| County crosswalk | `zcta_county_crosswalk.parquet` | Yes |
| Feature coverage | assembled parquet (column nullity) | Yes |
| Target availability | assembled parquet (target nullity) | Yes |

The current implementation uses four compatibility terms, all scaled so
that 1 = more compatible and 0 = less compatible:

| Term | Measures | Safe inputs |
|------|----------|-------------|
| `spatial_connectivity` | Connectedness of scenario ZCTAs in adjacency graph | Adjacency graph only |
| `support_coverage` | Feature and target observability for eligible rows | Parquet schema + nullity |
| `scale_stability` | Aggregation stability proxy (ZCTA vs county variance) | Feature values + crosswalk |
| `administrative_alignment` | ZCTA-county boundary alignment (admin proxy) | Crosswalk only |

`kappa_geom = mean(available compatibility terms)`

The current `scale_stability` and `administrative_alignment` terms are
conservative proxies.  Future versions may replace or supplement them
with hydrologic topology, basin/catchment alignment, or explicit MAUP
sensitivity tests.

### What is NOT kappa

The following remain as **diagnostic fields** in `compute_diagnostics.py`
but are explicitly excluded from kappa computation:

- `diag_leakage` — random vs spatial metric gap (model-derived)
- `diag_transfer` — leave-event-out vs spatial ratio (model-derived)
- `diag_solver` — HistGBDT vs Ridge agreement (model-derived)
- `diag_residual_spatial` — Moran's I on residuals (model-derived)

These are useful for understanding model behavior.  They are not
geometric compatibility.

### Paper language

> We distinguish model-quality decomposition from geometric compatibility.
> The RSN simplex summarizes observed model behavior under validation.
> In contrast, kappa is computed from pre-evaluation problem geometry and
> data-support structure, before model training or scoring.  It is not a
> function of RSN coordinates, validation scores, fold gaps, predictions,
> or residuals.  This separation prevents kappa from becoming a disguised
> quality metric and preserves its role as an independent diagnostic axis.

---

## Pre-Registered Primary Outcomes (v1.7)

**Rationale:** With 3 targets x 4 scenarios x 2 solvers x 3 splits, there are
24+ headline numbers per arm. Without pre-registration, the risk of cherry-picking
the best-looking comparison is high. These designations are locked before any
Phase 1 results are examined.

### Primary outcome per hypothesis

| Hypothesis | Primary Metric | Primary Target | Primary Split | Primary Solver | Justification |
|------------|---------------|----------------|---------------|----------------|---------------|
| H1 (gate) | R2 | obs_nfip_event_claims | spatial_blocked | histgbdt | Regression R2 on the insurance target under spatial blocking is the hardest test |
| H2 (R0->R1) | Paired fold delta (R2 metric) | obs_nfip_event_claims | spatial_blocked | histgbdt | Same target/split/solver; uplift measured as fold-level Wilcoxon signed-rank |
| H3 (R1->R2) | Paired fold delta (R2 metric) | obs_nfip_event_claims | spatial_blocked | histgbdt | Consistent with H2 |

### Secondary outcomes (reported but not used for PASS/FAIL)

- obs_has_311 (ROC-AUC, classification) -- reported for Houston/NYC only
- obs_has_hwm (ROC-AUC, classification) -- reported for Houston only
- Ridge solver results -- reported as linear-probe diagnostic
- Random and leave-event-out splits -- reported for diag_leakage and diag_transfer computation

### Null baseline

- R0 with `--random-features`: same fold structure, same targets, same solvers,
  but feature matrix replaced with N(0,1) noise. If random-features R0 matches
  real-features R0, the features carry no signal.

---

## Statistical Reporting Framework (v1.9)

### Three Diagnostic Axes

The paper reports results along three orthogonal axes.  Each measures a
different quantity at a different stage of the pipeline.  They must not
be conflated.

**Axis 1: RSN simplex (observed model behavior)**

R + S_sup + N = 1.  Computed AFTER model training and validation.
Summarizes how much signal (R), leakage (S_sup), and noise (N) the
model exhibits under spatial-blocked cross-validation.  This is an
empirical decomposition of solver performance — it changes when the
representation or solver changes.

**Axis 2: kappa_geom (pre-training geometric compatibility)**

Computed in Phase 0.5, BEFORE any model trains.  Measures how
geometrically amenable the (scenario, target) cell is to the
representation, based on spatial connectivity, feature coverage,
scale stability, and administrative alignment.  Has zero computational
dependency on RSN coordinates, fold metrics, predictions, or residuals.
See v1.8 for full specification and runtime guard.

**Axis 3: Bootstrap CIs (uncertainty on reported aggregate effects)**

Non-parametric bootstrap over (scenario x target) experiment cells.
Reports 95% confidence intervals on mean aggregate uplift and on mean
certificate signals.  These are **uncertainty intervals on effect size**,
not a replacement for the pre-registered fold-level Wilcoxon primary
test (v1.7).

### Role of Each Axis in the Paper

| Axis | Role | Where Reported |
|------|------|----------------|
| RSN simplex | Describes what the model does | Certificate tables (Figures 4, Appendix) |
| kappa_geom | Predicts which cells are hard, before training | Geometry kappa table (Appendix H) |
| Bootstrap CI | Quantifies how uncertain the aggregate effects are | Money table, certificate summary, Appendix |

### What Bootstrap CIs Do and Do Not Claim

The bootstrap resamples experiment cells (scenario x target), not
individual rows or folds.  The decision object is the cell — each cell
contributes one uplift value.

**Do report:**
- Mean uplift with 95% CI (e.g., "R0->R1 mean uplift = 12.3%, 95% CI [4.1%, 20.5%]")
- Fraction of bootstrap samples with positive mean (directional evidence)
- Certificate signal means with 95% CIs (e.g., "mean R = 0.42, 95% CI [0.35, 0.49]")
- n_cells, n_bootstrap, seed for reproducibility

**Do NOT claim:**
- That the CI constitutes a hypothesis test (it does not)
- That the CI replaces the pre-registered Wilcoxon test (it does not)
- Population-level inference from n=7 cells (report as observed uncertainty)

### Paper Language

> We report three orthogonal diagnostic quantities.  The RSN simplex
> (R, S_sup, N) decomposes observed model performance after validation.
> The geometric compatibility index kappa_geom is computed before model
> training from spatial structure and data support alone.  Bootstrap
> 95% confidence intervals over experiment cells quantify uncertainty on
> aggregate effects.  Primary inference follows from the pre-registered
> fold-level Wilcoxon signed-rank test; the bootstrap CI characterizes
> effect-size uncertainty for reporting purposes.

### Implementation

- `compute_uplift_table.py`: `bootstrap_cell_uplift()` produces
  `cell_bootstrap_ci` block in `money_table.json`, with per-transition
  (R0->R1, R1->R2) mean, CI, n_cells, pct_positive.
- `compute_certificates.py`: `summary_bootstrap_ci` block with per-signal
  (R, S_sup, N, alpha, omega, kappa, tau, sigma) mean and 95% CI.
- Both use n_bootstrap=10000, seed=42, percentile method.
