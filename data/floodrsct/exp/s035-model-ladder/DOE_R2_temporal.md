# DOE: R2 Temporal — Event Dynamics Treatment

**Experiment:** s035-model-ladder / Phase 3
**Role:** Second treatment arm — adds temporal event dynamics to fix diag_transfer flags from R1
**Status:** DESIGNED (DOE phase, not launched)
**Depends on:** R1 complete (fold assignments + R1 predictions)

---

## Hypothesis

**H3:** When diag_transfer is low at R1 (poor cross-event generalization),
R2 representation produces larger uplift than when diag_transfer is high.

**Primary test:** Spearman rho(diag_transfer_R1, uplift_R1_to_R2) < 0
(lower diag_transfer = more uplift from R2).

**Secondary:** diag_transfer increases R1→R2 (event dynamics now captured,
improving cross-event generalization).

---

## Design Matrix

| Factor | Levels | Type |
|--------|--------|------|
| Scenario | houston, southwest_florida, nyc, riverside_coachella | Observational |
| Target | obs_nfip_event_claims, obs_has_311, obs_has_hwm | Fixed per scenario |
| Solver | HistGBDT, Ridge | Fixed |
| Split | random, spatial_blocked, leave_event_out | Fixed |
| Fold | Same fold assignments as R0 | Controlled |
| Features | R1 + R2 temporal (9 features) | Treatment |

---

## Independent Variable: R2 Temporal Feature Bundle (9 columns)

All event-level features joined on [zcta_id, event]:

| Column | Source | Computation | Coverage |
|--------|--------|-------------|----------|
| peak_1h_mm | MRMS hourly grib2 | Max single-hour rainfall at ZCTA centroid | >90% |
| peak_3h_mm | MRMS hourly grib2 | Max 3-hour rolling sum | >90% |
| peak_6h_mm | MRMS hourly grib2 | Max 6-hour rolling sum | >90% |
| storm_duration_h | MRMS hourly grib2 | Hours with rainfall > 1mm | >90% |
| time_to_peak_h | MRMS hourly grib2 | Hours from first rain to peak | >90% |
| rainfall_intensity_cv | MRMS hourly grib2 | CV of hourly rainfall | >90% |
| tide_peak_m | Tide gauge CSVs | Peak water level, nearest station | Coastal only (~50%) |
| surge_rain_lag_h | Tide + MRMS | Hours between peak rainfall and peak surge | Coastal only (~50%) |
| storm_approach_speed_kph | HURDAT2 | Storm translation speed at closest approach | Named storms only |

**R2 total: R1 (61-63) + 9 temporal = 70-72 features**

**R2 join key:** [zcta_id, event] — event-level, not ZCTA-level like R1.
This means R2 features vary across events for the same ZCTA.

---

## Dependent Variables

| Variable | Metric | Comparison |
|----------|--------|------------|
| R2 absolute performance | R2 score (spatial_blocked, histgbdt) | Report |
| R1→R2 uplift | R2_R2 - R2_R1, same fold/solver/target | Primary |
| R0→R2 total uplift | R2_R2 - R2_R0 | Full ladder gain |
| Uplift percentage | (R2_R2 - R2_R1) / max(abs(R2_R1), 0.01) * 100 | For money table |
| Kappa movement | diag_transfer_R2 - diag_transfer_R1 | Diagnostic confirmation |

---

## Controlled Variables

Identical to R0 and R1 — same hyperparameters, folds, targets, splits.

---

## Temporal Feature Quality Concerns

| Concern | Mitigation |
|---------|-----------|
| Coastal features (tide, surge_rain_lag) are NaN for ~50% of ZCTAs | HistGBDT handles NaN natively; Ridge imputes median. Report ablation with/without coastal features. |
| storm_approach_speed_kph is NaN for non-named storms (AR flood 2023) | Same NaN handling. Feature importance will show if this matters. |
| Leave-event-out split removes one entire event — R2 features trained on remaining events may not transfer | This IS the test. If diag_transfer improves R1→R2, temporal features help generalization despite event removal. |
| MRMS grib2 spatial resolution (1 km) vs ZCTA scale (~10 km) | Nearest-neighbor extraction at centroid. MAUP-like but consistent across ZCTAs. |

---

## Outputs

| Artifact | S3 Key | Format |
|----------|--------|--------|
| Results | `results/s035/r2_{scenario}.json` | JSON |
| Predictions | `results/s035/r2_{scenario}_predictions.parquet` | GeoParquet |
| R2 supplement | `processed/{scenario}/{scenario}_r2_supplement.parquet` | GeoParquet |

---

## Success Criteria

| Criterion | Threshold | Action if FAIL |
|-----------|-----------|----------------|
| H3 primary: rho(diag_transfer_R1, uplift) < 0 | Spearman rho < 0 | Report as null |
| H3 secondary: mean R1→R2 uplift > 0 | uplift > 0 | Temporal features don't help |
| diag_transfer increases R1→R2 | delta > 0 majority | Event dynamics not captured |
| Leave-event-out R2 improves over R1 | R2_leo > R1_leo | Transfer improved |

---

## Kill Rules

- ALL R1→R2 uplifts negative → temporal features are noise
- R2 worse than R0 on leave-event-out → temporal features hurt transfer
- > 60% of R2 features are NaN for a scenario → MRMS coverage gap, exclude that scenario from R2 analysis

---

## Ablations

| Ablation | Features Removed | Tests |
|----------|------------------|-------|
| R2 no-coastal | Remove tide_peak_m, surge_rain_lag_h | Is uplift from rainfall timing or surge interaction? |
| R2 rainfall-only | Keep only peak_1h/3h/6h_mm + duration + time_to_peak | Minimal temporal signal |

---

## Compute

| Resource | Value |
|----------|-------|
| Instance | ml.m5.xlarge |
| Storage | 10 GB EBS |
| Est. duration | ~5 min per scenario |
| Dependencies | R1 predictions, R2 supplement on S3 |
| GPU | NOT NEEDED |

---

## DO NOT Constraints

- Do NOT change fold assignments from R0
- Do NOT tune hyperparameters
- Do NOT drop R0 or R1 features (R2 = R1 + additions)
- Do NOT impute targets
- Do NOT use GPU instances
- Do NOT recompute R1 features when adding R2
