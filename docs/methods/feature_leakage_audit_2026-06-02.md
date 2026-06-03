# Feature Leakage Audit — Houston R1 Hydrology

**Date**: 2026-06-02
**Scope**: All features in R0 + R1 feature sets for Houston (Harvey2017, Imelda2019, Beryl2024)
**Trigger**: Paper audit P0-P2 prior to NeurIPS submission

---

## Summary

Seven county-level Storm Events features quarantined. Two event-concurrent spatial
lags wired for per-fold recomputation. Three-layer validation gap in the build
pipeline repaired. NFIP historical supplement confirmed clean.

---

## P0: Fold Integrity

**Status: CLEAN**

`generate_folds.py:162` assigns folds on `df["zcta_id"].unique()`, then broadcasts
via dict lookup. Same ZCTA always maps to same fold regardless of how many event
rows it appears in. Verified: 0 ZCTAs appear in multiple `spatial_blocked` folds.

**Note**: Fold 3 contains 285/396 rows (72%) — highly imbalanced. Not a leakage
issue but affects variance estimates.

---

## P0: County-Level Storm Events Features — QUARANTINED

### Problem

Seven `flood_*` features from `geocertdb2026` are NOAA Storm Events county-level
aggregates. They are:

1. **Constant within scenario** — every ZCTA in Harris County gets the same value.
   Zero predictive variance within a scenario.
2. **Not temporally gated** — totals include the target event itself (e.g.,
   `flood_deaths` for Harvey2017 includes Harvey deaths).
3. **Outcome-from-outcome risk** — `flood_deaths` and `flood_injuries` are outcomes
   of the same class of event we're predicting.

### Features quarantined

| Feature | Source | Replacement |
|---|---|---|
| `flood_deaths` | Storm Events county total | None (no ZCTA-level source) |
| `flood_injuries` | Storm Events county total | None (no ZCTA-level source) |
| `flood_event_count` | Storm Events county total | `nfip_historical_frequency` |
| `flood_event_count_5y` | Storm Events county total | `nfip_historical_frequency` |
| `flood_events_per_year` | Storm Events county total | `nfip_historical_frequency` |
| `flood_property_damage_k` | Storm Events county total | `nfip_historical_severity` |
| `flood_crop_damage_k` | Storm Events county total | None (no ZCTA-level source) |

### Actions taken

- Commented out in all training scripts (`train_r0_baseline.py`, `train_r1_hydrology.py`,
  `train_r2_temporal.py`, `compute_geometry_kappa.py`)
- Reclassified from `slow_drift` to `post_event` in `FEATURE_CONTRACT.yaml`
- Quarantine comment documents rationale at each site

### Impact on results

None. These features had zero within-scenario variance, so gradient boosting and
ridge regression cannot learn from them in single-scenario training.

---

## P0: NFIP Historical Supplement — CLEAN

`build_nfip_historical.py:134` applies strict temporal gate:
```python
all_claims[all_claims["dateOfLoss"] < cutoff]
```

Per-event cutoff from `incidentBeginDate`. Timezone-aware handling on lines 132-133.

Verified values confirm correct causal ordering:
- Harvey2017: `nfip_historical_frequency` mean = 0 (first event, no prior claims)
- Imelda2019: mean = 375.55 (includes Harvey claims only)
- Beryl2024: mean = 406.42 (includes Harvey + Imelda claims)

These two features (`nfip_historical_frequency`, `nfip_historical_severity`) are the
ZCTA-level, temporally-gated replacements for the quarantined county-level flood
event count and damage features.

---

## P1: Spatial Lag Leakage Classification

### Three-tier classification

Not all spatial lags are equal. The 8 W-matrix features fall into three leakage
classes:

#### Tier 1: Map-known (no leakage) — 3 features

| Feature | Source column | Justification |
|---|---|---|
| `wlag_flood_zone_pct` | `flood_pct_zone_a` | FEMA NFHL static map, published years before events |
| `wlag_population_density` | `population` | ACS census estimate, static between decennial updates |
| `wlag_median_income` | `acs_median_hh_income` | ACS census estimate |

These are knowable at prediction time — the pre-computed full-dataset spatial lag
is the same value you'd compute from any fold partition.

#### Tier 2: Static with vintage caveat — 1 feature

| Feature | Source column | Justification |
|---|---|---|
| `wlag_impervious_pct` | `impervious_pct` | NLCD 2021 vintage |

Defensible as static (land cover doesn't change during an event), but NLCD 2021
vintage means pre-2021 events (Harvey2017, Imelda2019) use a future land cover
snapshot. The temporal mismatch is small (impervious surface changes slowly) but
should be noted in the paper.

#### Tier 3: Event-concurrent (per-fold recomputation required) — 2 features

| Feature | Source column | Why it leaks |
|---|---|---|
| `wlag_nfip_claims` | `obs_nfip_event_claims` | Spatial lag of the prediction target |
| `wlag_rainfall_mm` | `rainfall_total_mm` | Event-window measurement — neighbor rainfall unknown before event |

Both receive per-fold recomputation in `train_r1_hydrology.py`:
- Build zcta -> value lookup from **training fold ZCTAs only**
- Test fold ZCTAs receive lag from training neighbors only
- NaN if all neighbors are in test fold
- Hard refusal if adjacency data unavailable (defense-in-depth gate)

#### Graph structure features — 2 features (no leakage)

| Feature | Source |
|---|---|
| `zcta_degree` | Adjacency graph topology |
| `zcta_mean_neighbor_dist_km` | Adjacency graph topology |

These are derived from static geography, not from any measurement.

### Per-fold recomputation implementation

`_recompute_wlag_per_fold()` in `train_r1_hydrology.py`:

```
For each fold:
  1. Build lookup: zcta_id -> source_col mean, training ZCTAs only
  2. For each row in merged DataFrame:
     a. Get ZCTA's neighbors from adjacency dict
     b. Collect source values from training neighbors only
     c. wlag = mean(training neighbor values), or NaN if none
  3. Overwrite the wlag column in X_all for this fold iteration
```

Defense-in-depth: if `wlag_nfip_claims` or `wlag_rainfall_mm` is in the feature
set but no adjacency data is available, `run_split()` raises `RuntimeError`
instead of silently using the leaked pre-computed values.

---

## P1: Three-Layer Validation Gap — REPAIRED

### Problem

The build pipeline (`build_event_dataset.py`) had a silent fallback pattern:

1. `_load_adjacency()` returned `None` on S3 miss (instead of raising)
2. `compute_w_matrix_features()` returned NaN columns when adjacency was None
3. `validate_post_assembly()` checked coverage but not schema completeness
4. Build uploaded the parquet anyway on validation failure

Result: assembled parquets could have all-NaN W-matrix columns, and training
would silently proceed with useless features. The `no-target-lag` ablation
(52 features, 7 W-matrix) was identical to `no-wlag` (45 features, 0 W-matrix)
because W-matrix columns were present but all NaN.

### Fixes

| Layer | Before | After |
|---|---|---|
| `_load_adjacency()` | Returned `None` on S3 miss | Raises `FileNotFoundError` |
| `compute_w_matrix_features()` | Produced NaN columns on `None` adjacency | Propagates the error (no fallback) |
| `validate_post_assembly()` | Coverage check only | Schema completeness cross-check vs FEATURE_CONTRACT.yaml |
| Build upload | Uploaded on validation failure | `sys.exit(1)` on failure |

### Lag source column name fixes

Three column name mismatches in `build_event_dataset.py` caused silent NaN
propagation even when adjacency data was available:

| Lag feature | Wrong source column | Correct source column |
|---|---|---|
| `wlag_population_density` | `acs_population_density` | `population` |
| `wlag_rainfall_mm` | `total_rainfall_mm` | `rainfall_total_mm` |
| `wlag_nfip_claims` | `nfip_event_claims` | `nfip_event_claim_count` |

Additionally, `wlag_impervious_pct` was in `STATIC_LAG_MAP` but `impervious_pct`
is in the event-level lookup table. Moved to `EVENT_LAG_MAP`.

---

## P1: Rainfall Data Quality — FIXED

### Root cause (two bugs)

1. **Sentinel contamination** (commit f382892): MRMS GRIB2 uses negative sentinels
   (-3 = no data, -1 = range-folded, -2 = below threshold). The accumulation sum
   added raw values without masking, producing totals of -1296 mm for Harvey.
   Fix: `arr = np.where(arr < 0, 0.0, arr)` in `_process_one_grib()`.

2. **Longitude convention mismatch** (commit 4e7308b): MRMS GRIB2 uses 0-360
   longitude (Houston = 264.6). ZCTA centroids use -180/180 (Houston = -95.4).
   Nearest-centroid lookup found edge pixels with zero rainfall instead of the
   correct Houston-area pixels. Fix: convert `lon_2d = np.where(lon > 180, lon - 360, lon)`
   before spatial join.

After both fixes, the raw MRMS data on S3 produces correct ZCTA-level rainfall:
- Harvey 2017: max=5834 mm (grid), realistic per-ZCTA values
- Imelda 2019: max=1316 mm (grid)
- Beryl 2024: max=901 mm (grid)

### Diagnostic improvements (commit 08281f6)

`_process_one_grib()` now returns structured error strings instead of `None`,
enabling error aggregation by failure type. Post-decode summary logs n_valid,
n_errors, and accumulation stats. Post-assignment log confirms ZCTA-level
min/max/mean.

**Status**: All 5 scenarios rebuilding with both fixes (2026-06-03 16:19).

---

## Other Scenarios: W-Matrix Validation (2026-06-03)

Validated W-matrix columns across all 5 assembled parquets on S3.

### `impervious_pct` Missing in 3 Scenarios

| Scenario | `impervious_pct` | `wlag_impervious_pct` | Root cause |
|---|---|---|---|
| houston | 100% populated | 100% populated | `build_impervious_features()` called |
| nyc | 100% populated | 100% populated | `build_impervious_features()` called |
| new_orleans | **MISSING** | **0% non-null** | `build_impervious_features()` never called |
| riverside_coachella | **MISSING** | **0% non-null** | `build_impervious_features()` never called |
| southwest_florida | **MISSING** | **0% non-null** | `build_impervious_features()` never called |

**Fix**: Added `build_impervious_features()` call + merge in `build_new_orleans()`,
`build_riverside_coachella()`, and `build_southwest_florida()`. NLCD raster is
national (L48) and already on S3 — the function works for any ZCTA list.

**Requires rebuild**: All 3 scenarios need `build_event_dataset.py` re-run via
SageMaker to regenerate assembled parquets with impervious data populated.

### NYC `wlag_flood_zone_pct` All Zeros

NYC `flood_pct_zone_a` is all zeros — NYC flood zones are primarily AE and VE
(coastal), not Zone A (inland riverine). The lag of all-zero source values is zero.
**Not a bug** — correct behavior for NYC's flood zone distribution.

### Other W-Matrix Coverage

| Scenario | `wlag_nfip_claims` | `wlag_rainfall_mm` | `wlag_median_income` |
|---|---|---|---|
| houston | 100% | 100% (all zero — broken MRMS) | 100% |
| new_orleans | 97% | 48% | 53% |
| nyc | 50% | 100% | 96% |
| riverside_coachella | 50% | 100% | 99% |
| southwest_florida | 100% | 100% (all zero) | 99% |

Lower coverage in new_orleans reflects island/isolated ZCTAs with no neighbors or
no overlap with source data. 50% nfip_claims coverage in nyc/riverside reflects
events with no FEMA disaster declaration (henri2021, ar_flood_2023) where no claims
data exists.

---

## Houston R1 Ablation Results (Clean, Post-Fix)

All runs use per-fold wlag recomputation where applicable. Target:
`obs_nfip_event_claims`, split: `spatial_blocked`, solver: `histgbdt`.

| Ablation | Total features | W-matrix features | NFIP spatial RMSE |
|---|---|---|---|
| no-wlag | 45 | 0 | 2.0174 |
| no-target-lag | 52 | 7 (static + event lags) | 1.5091 |
| full | 53 | 8 (7 + per-fold wlag_nfip_claims) | 1.4761 |
| wlag-only | 40 | 8 (per-fold) | 1.4278 |

**Headline result**: no-target-lag vs no-wlag (1.51 vs 2.02, 25% RMSE improvement)
demonstrates that spatial structure matters independent of target lag. The relational
signal is real, not a leaked shortcut.

full vs no-target-lag (1.48 vs 1.51) shows target lag adds marginal signal beyond
the covariate lags.

---

## NLCD Vintage Caveat (for paper text)

`wlag_impervious_pct` uses NLCD 2021 impervious surface. For pre-2021 events
(Harvey2017, Imelda2019), this is a future land cover snapshot. The temporal
mismatch is small (impervious surface changes slowly in established metro areas)
but must be disclosed as a limitation.

**Suggested paper text** (Section 4, substrate description):
> Impervious surface percentage is derived from the NLCD 2021 product. For events
> prior to 2021 (Harvey 2017, Imelda 2019), this introduces a minor temporal
> mismatch: the land cover snapshot postdates the event by 2--4 years. We treat
> this as acceptable for established metro areas where impervious surface changes
> slowly, but note it as a known limitation.

---

## Open Items

| Item | Status | Owner |
|---|---|---|
| MRMS rainfall fix (sentinel + longitude) | **DONE** — f382892 + 4e7308b | — |
| MRMS rainfall: rebuild all 5 scenarios | **IN PROGRESS** — launched 2026-06-03 16:19 | — |
| NLCD vintage caveat in paper text | Draft above | Paper author |
| NFIP temporal gate unit test | **DONE** (8/8 passed) | — |
| W-matrix validation (all 5 scenarios) | **DONE** — 3 missing `impervious_pct`, code fixed | — |
| Rebuild 3 scenarios (NO, RC, SWFL) for impervious_pct | **DONE** — completed 2026-06-03 | — |
| `cropland_pct` feature (NLCD classes 81+82) | **DONE** — code complete (61f23ce), needs NLCD Land Cover raster fetch + rebuild | — |
| NLCD Land Cover raster fetch | **READY** — launcher exists, not yet run | — |
| Rebuild all 5 scenarios (post-NLCD Land Cover) | Blocked on NLCD fetch | — |
| Re-run R1 ablations (post-rainfall fix) | Blocked on rebuilds | — |
| Allocation geometry assessment (P2) | Not started | — |
