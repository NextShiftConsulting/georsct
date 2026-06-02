# DOE: R0 Baseline — Static Tabular Control

**Experiment:** s035-model-ladder / Phase 1
**Role:** Control arm (all subsequent levels measured against R0)
**Status:** DESIGNED (DOE phase, not launched)

---

## Hypothesis

**H1:** HistGBDT on R0 features achieves R2 > 0 (better than mean predictor)
on at least 2 of 3 targets under spatial-blocked CV.

If H1 FAILS on all targets → data quality too low for any modeling; STOP.

---

## Design Matrix

| Factor | Levels | Type |
|--------|--------|------|
| Scenario | houston, southwest_florida, nyc, riverside_coachella | Observational (fixed) |
| Target | obs_nfip_event_claims (log1p), obs_has_311 (binary), obs_has_hwm (binary) | Fixed per scenario |
| Solver | HistGBDT, Ridge | Fixed (2 levels) |
| Split | random, spatial_blocked, leave_event_out | Fixed (3 protocols) |
| Fold | 5-fold CV (spatial_blocked), 80/20 (random), k-1 (leave-event-out) | Fixed |
| Features | R0 bundle only (33 static tabular) | Control — held constant |

**Total cells:** 5 scenarios x 7 targets x 2 solvers x 3 splits = 210 runs
(not all targets available in all scenarios — actual: ~105 runs)
Note: New Orleans promoted to modelable (66 ZCTAs x 4 events = 264 rows).

---

## Independent Variables

None at R0 — this is the control. R0 establishes the baseline against which
R1 and R2 are measured.

## Dependent Variables

| Variable | Metric | Higher = Better? |
|----------|--------|-----------------|
| Prediction accuracy (regression) | R2 score | Yes |
| Prediction accuracy (classification) | ROC-AUC | Yes |
| Prediction error | RMSE (log1p scale) | No |
| Baseline ratio | RMSE / naive_mean_RMSE | No (< 1.0 = skill) |

## Controlled Variables

| Variable | Value | Rationale |
|----------|-------|-----------|
| HistGBDT max_iter | 200 | DOE: no tuning |
| HistGBDT max_depth | 6 | DOE: no tuning |
| HistGBDT learning_rate | 0.1 | DOE: no tuning |
| Ridge alpha | 1.0 | DOE: no tuning |
| Random seed | 42 | Reproducibility |
| Imputation (Ridge only) | median | DOE: no tuning |
| Scaling (Ridge only) | StandardScaler | DOE: no tuning |
| Fold assignments | Fixed, saved to S3 | Shared across R0/R1/R2 |

---

## Feature Bundle: R0 (33 features)

**Source:** `processed/{scenario}/{scenario}_event_features.parquet` (assembled)
+ `processed/{scenario}/{scenario}_nfip_historical.parquet` (temporally-gated NFIP)

| Category | Features | Count |
|----------|----------|-------|
| Demographics (ACS) | population, median_income, pct_poverty, pct_owner_occupied, housing_units, ... | ~10 |
| Social Vulnerability (SVI) | theme1-4 percentiles | 4 |
| Flood Zones (FEMA NFHL) | pct_in_sfha, pct_zone_ae, pct_zone_x, pct_zone_vh | ~6 |
| Terrain (TWI) | twi_acc_twi, twi_tot_twi, slope_basin_slope, slope_stream_slope | 4 |
| Infrastructure (HIFLD) | n_hospitals, n_pharmacies, nearest_pharmacy_km, nearest_trauma_km | ~6 |
| NFIP Historical (gated) | nfip_historical_frequency, nfip_historical_severity | 2 |
| Land Cover | impervious_pct (where available) | ~1 |

**Temporal gating (IBNR boundary):** NFIP historical features are computed from
claims with `dateOfLoss` strictly before the target event's `incidentBeginDate`.
Same-event claims are excluded. Built by `build_nfip_historical.py`.

**What R0 does NOT have:**
- No spatial structure (no W-matrix, no neighbor features)
- No hydrology (no catchment area, no stream density)
- No temporal dynamics (no rainfall, no surge timing, no storm track)
- No engineering model outputs (no FAST)

---

## Split Protocols

### Random 80/20 (diagnostic only)
- Purpose: measure A.1 spatial autocorrelation leakage
- Implementation: stratified by target, seed=42
- NOT used for headline results — only for diag_leakage computation

### Spatial-Blocked 5-Fold CV (primary)
- Purpose: main benchmark, eliminates spatial leakage
- Implementation: greedy bin-packing by county FIPS (preferred) or ZIP3 prefix (fallback)
- All headline metrics reported from this split

### Leave-Event-Out (transfer diagnostic)
- Purpose: measure D.3 cross-event generalization
- Implementation: hold out one entire event, train on remaining
- Used for diag_transfer computation

---

## Outputs

| Artifact | S3 Key | Format |
|----------|--------|--------|
| Fold assignments | `folds/{scenario}_folds.parquet` | Parquet with fold_id columns |
| Fold metadata | `folds/{scenario}_folds_meta.json` | JSON (block strategy, counts) |
| Results | `results/s035/r0_{scenario}.json` | JSON (all runs, metrics, params) |
| Predictions | `results/s035/r0_{scenario}_predictions.parquet` | GeoParquet (spatial_blocked only, per-row y_true/y_pred) |

### Predictions Schema

| Column | Type | Description |
|--------|------|-------------|
| zcta_id | str | ZCTA identifier |
| event | str | Event name |
| target | str | Target column name |
| solver | str | histgbdt or ridge |
| split | str | spatial_blocked (only) |
| fold | str | Fold identifier |
| y_true | float | Observed value |
| y_pred | float | Predicted value |
| geometry | Point | ZCTA centroid (EPSG:4326) |

---

## Success Criteria

| Criterion | Threshold | Action if FAIL |
|-----------|-----------|----------------|
| H1: R2 > 0 on >= 2 targets (HistGBDT, spatial_blocked) | R2 > 0.0 | STOP — data insufficient |
| RMSE/baseline < 1.0 on primary target | < 1.0 | STOP — no measurable skill |
| At least 1 scenario trains successfully | Runs complete | Fix data, retry |

---

## Kill Rules

- H1 FAIL on ALL targets in ALL scenarios → abort experiment series
- HistGBDT train-vs-val R2 gap > 0.5 → severe overfitting, reduce max_depth
  (but this violates "no tuning" — report as limitation)
- > 50% of runs fail (data errors, NaN explosion) → fix data pipeline

---

## Diagnostic Outputs (consumed by Phase 4a)

R0 results are consumed by `compute_diagnostics.py --level r0` to produce:

| Kappa Proxy | R0 Inputs Used |
|-------------|----------------|
| diag_leakage | R2(random) vs R2(spatial_blocked), HistGBDT |
| diag_transfer | R2(leave_event_out) vs R2(spatial_blocked), HistGBDT |
| diag_solver | R2(spatial_blocked, histgbdt) vs R2(spatial_blocked, ridge) |
| diag_residual_spatial | Per-row predictions (spatial_blocked, histgbdt) + adjacency matrix → Moran's I |

These diagnostics are the PRE-REGISTRATION for what R1 should fix. They are
uploaded to S3 BEFORE R1 trains, establishing temporal ordering proof.

---

## Compute

| Resource | Value |
|----------|-------|
| Instance | ml.m5.xlarge (4 vCPU, 16 GB) |
| Storage | 10 GB EBS |
| Est. duration | ~5 min per scenario |
| Est. cost | < $0.10 per scenario |
| GPU | NOT NEEDED (tabular models only) |

---

## DO NOT Constraints

- Do NOT tune hyperparameters — defaults are frozen
- Do NOT add features after R0 baseline is locked
- Do NOT impute targets (drop rows with NaN target)
- Do NOT use GPU instances
- Do NOT change fold assignments after R0 runs
- Do NOT use random split metrics for headline results
