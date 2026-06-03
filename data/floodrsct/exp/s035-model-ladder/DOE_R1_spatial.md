# DOE: R1 Spatial — Hydrology + W-Matrix Treatment

**Experiment:** s035-model-ladder / Phase 2
**Role:** First treatment arm — adds spatial structure to fix diag_leakage/diag_residual_spatial flags from R0
**Status:** DESIGNED (DOE phase, not launched)
**Depends on:** R0 complete (fold assignments + R0 predictions for residual lag)

---

## Hypothesis

**H2a (primary):** R1 representation improves over R0 on the primary target
under spatial-blocked CV.

**Primary test (v1.7):** Fold-level Wilcoxon signed-rank on paired
(R0_fold_metric, R1_fold_metric) across ~35 fold observations pooled
from all modelable cells. Requires both p < 0.05 AND Cohen's d > 0.2.

**H2b (exploratory):** kappa_geom (computed pre-training, Phase 0.5)
predicts which cells benefit more from R1. Spearman rho(kappa_geom, uplift)
with bootstrap 95% CI. Reported as observed association (n=9 cells).

**Note (v1.8):** The original H2 used diag_leakage as predictor, but
diag_leakage shares R0_spatial metric with the uplift calculation, creating
a room-to-improve confound. kappa_geom has zero model dependency and
replaces diag_leakage as the cell-level predictor. diag_leakage remains
a diagnostic field reported in the cascade table.

**Secondary:** diag_leakage increases R0→R1 (spatial autocorrelation captured
by explicit features, not leaked through splits).

---

## Design Matrix

| Factor | Levels | Type |
|--------|--------|------|
| Scenario | houston, southwest_florida, nyc, riverside_coachella | Observational |
| Target | obs_nfip_event_claims, obs_has_311, obs_has_hwm | Fixed per scenario |
| Solver | HistGBDT, Ridge | Fixed |
| Split | random, spatial_blocked, leave_event_out | Fixed |
| Fold | Same fold assignments as R0 | Controlled (from Phase 1) |
| Features | R0 + R1 universal (16) + R1 scenario-specific (1-3) + W-matrix (8) | Treatment |

**Treatment vs control:** R1 features added ON TOP of R0. Same folds, same
solver, same target. ONLY the feature set changes.

---

## Independent Variable: R1 Feature Bundle

### R1 Universal Features (16 columns, all scenarios)

| Column | Source | Signal |
|--------|--------|--------|
| nhd_catchment_area_km2 | NHDPlus V2 spatial join | Watershed scale |
| slope_basin_slope | Already in parquet | Terrain gradient |
| slope_stream_slope | Already in parquet | Channel gradient |
| twi_acc_twi | Already in parquet | Topographic wetness |
| twi_tot_twi | Already in parquet | Total wetness index |
| drive_min_to_county_centroid | Already in parquet | Accessibility |
| drive_min_to_county_seat | Already in parquet | Accessibility |
| drive_min_to_nearest_hospital | Already in parquet | Infrastructure access |
| hifld_n_hospital_beds | Already in parquet | Healthcare capacity |
| hifld_n_pharmacies | Already in parquet | Healthcare access |
| hifld_nearest_pharmacy_km | Already in parquet | Healthcare distance |
| hifld_nearest_trauma_center_km | Already in parquet | Emergency access |
| flood_deaths | Already in parquet | Historical severity |
| flood_injuries | Already in parquet | Historical severity |
| ~~nfip_total_building_loss~~ | REMOVED | Target leakage: cumulative NFIP payout with no temporal gate |
| ~~nfip_total_contents_loss~~ | REMOVED | Target leakage: cumulative NFIP payout with no temporal gate |

### R1 Scenario-Specific Features (1-3 columns)

| Column | Scenarios | Source |
|--------|-----------|--------|
| upstream_catchment_km2 | Houston, Riverside | NHDPlus |
| hcfcd_drainage_district | Houston | HCFCD parquet (label-encoded) |
| levee_nearest_km | NOLA, NYC | USACE levees |
| levee_condition_rating | NOLA, NYC | USACE levees |
| sewershed_name | NYC | NYC sewersheds gpkg (label-encoded) |
| slosh_max_surge_m | SW Florida | Already in parquet |

### W-Matrix Spatial Features (8 columns) — NEW

Built from Queen's contiguity via `zcta_adjacency.parquet` + libpysal:

| Column | Formula | Signal | Leakage Risk |
|--------|---------|--------|-------------|
| wlag_nfip_claims | W * target (per-fold, train only) | Neighbor flood activity | HIGH — ablation required |
| wlag_flood_zone_pct | W * flood_zone_pct_in_sfha | Neighborhood flood exposure | Low |
| wlag_population_density | W * acs_population_density | Neighborhood urbanization | Low |
| wlag_median_income | W * acs_median_household_income | Neighborhood SES | Low |
| wlag_impervious_pct | W * impervious_pct | Neighborhood runoff | Low |
| spatial_lag_residual_R0 | W * residual(R0, histgbdt) | Spatial error correction | Medium — uses R0 preds |
| zcta_degree | Count(queen neighbors) | Connectivity | None |
| zcta_mean_neighbor_dist_km | Mean(centroid distance to neighbors) | Spatial density | None |

**R1 total: R0 (36) + 16 universal + 1-3 scenario + 8 W-matrix = 61-63 features**

---

## Spatial Lag Leakage Protocol

### wlag_nfip_claims (target lag)

This is the most powerful and most dangerous feature. The spatial lag of the
target variable uses neighbor target values as input — which could leak
information across CV fold boundaries.

**Mitigation (mandatory):**
1. Compute spatial lag PER FOLD using TRAINING ZCTAs only
2. Test ZCTAs receive lag from training neighbors only (or NaN if all neighbors
   are in test fold)
3. Report results WITH and WITHOUT wlag_nfip_claims as ablation
4. If uplift is entirely driven by wlag_nfip_claims, the story is "neighbor
   activity predicts flood risk" not "hydrology features help"

```python
# Per-fold spatial lag — no leakage
for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
    y_train_vals = y.iloc[train_idx].values
    # Build W restricted to training ZCTAs
    w_fold = w.subset(ids=train_zcta_ids)
    lag_train = w_fold.sparse @ y_train_vals
    # Test: lag from training neighbors only
    lag_test = compute_lag_from_subset(w, test_zcta_ids, y_train_vals)
```

### spatial_lag_residual_R0 (residual lag)

Uses R0 predictions (Phase 1 output) to compute residuals, then spatially
lags them. Risk: residuals encode R0's view of the data, which could create
a subtle feedback loop. Mitigation: use out-of-fold R0 predictions only
(each ZCTA's residual comes from the fold where it was in the test set).

---

## Dependent Variables

| Variable | Metric | Comparison |
|----------|--------|------------|
| R1 absolute performance | R2 (spatial_blocked, histgbdt) | Report |
| R0→R1 uplift | R2_R1 - R2_R0, same fold/solver/target | Primary |
| Uplift percentage | (R2_R1 - R2_R0) / max(abs(R2_R0), 0.01) * 100 | For money table |
| Kappa movement | diag_leakage_R1 - diag_leakage_R0 | Diagnostic confirmation |

---

## Controlled Variables

Same as R0 — all hyperparameters, fold assignments, targets, and split
protocols are IDENTICAL. Only the feature set changes.

---

## Outputs

| Artifact | S3 Key | Format |
|----------|--------|--------|
| Results | `results/s035/r1_{scenario}.json` | JSON |
| Predictions | `results/s035/r1_{scenario}_predictions.parquet` | GeoParquet |
| R1 supplement | `processed/{scenario}/{scenario}_r1_supplement.parquet` | GeoParquet |

---

## Success Criteria

| Criterion | Threshold | Action if FAIL |
|-----------|-----------|----------------|
| H2a primary: R1 > R0 (fold-level Wilcoxon) | p < 0.05 AND Cohen's d > 0.2 | Report as null (R1 does not improve on R0) |
| H2b exploratory: kappa_geom predicts uplift | Spearman rho, bootstrap CI | Report as observed association (n=7 marginal) |
| H2 secondary: mean uplift > 0 when flagged | uplift_flagged > 0 | R1 features don't help flagged scenarios |
| diag_leakage increases R0→R1 | delta > 0 for majority of cells | Spatial features didn't reduce autocorrelation |
| At least 1 scenario shows > 5% uplift | uplift > 5% | Marginal improvement only |

---

## Kill Rules

- ALL uplifts negative (R1 worse than R0 everywhere) → R1 features are noise;
  investigate feature quality, drop noisy columns, proceed to R2 without R1
- wlag_nfip_claims ablation shows ALL uplift from target lag → report honestly;
  R1 story is "neighbor claims predict" not "hydrology helps"
- > 30% of R1 feature columns are NaN → supplement join failed, fix pipeline

---

## Ablations (mandatory)

| Ablation | Features Removed | Tests |
|----------|------------------|-------|
| R1 no-wlag | Remove all 8 W-matrix features | Is uplift from point features or spatial structure? |
| R1 no-target-lag | Remove wlag_nfip_claims only | Is target lag driving everything? |
| R1 wlag-only | R0 + 8 W-matrix features (no hydro) | Is hydrology needed beyond spatial structure? |

Report all 3 ablations in the money table alongside full R1.

---

## Compute

| Resource | Value |
|----------|-------|
| Instance | ml.m5.xlarge |
| Storage | 10 GB EBS |
| Est. duration | ~5 min per scenario |
| Dependencies | libpysal (W-matrix), R0 predictions (residual lag) |
| GPU | NOT NEEDED |

---

## DO NOT Constraints

- Do NOT change fold assignments from R0
- Do NOT tune hyperparameters
- Do NOT compute spatial lag from test fold ZCTAs
- Do NOT drop R0 features (R1 = R0 + additions)
- Do NOT use GPU instances
