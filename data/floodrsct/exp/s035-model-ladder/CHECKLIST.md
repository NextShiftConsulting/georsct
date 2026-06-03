# S035-Model-Ladder Pre-Flight Checklist

**Bucket:** `s3://swarm-floodrsct-data/`
**Job scripts:** `data/floodrsct/jobs/`
**Launchers:** `data/floodrsct/scripts/`
**DOE version:** v1.6 (DOE_AMENDMENT_v1.2.md)

---

## 1. Git State

- [ ] `git status` -- working tree clean
- [ ] `git log -1 --oneline` -- commit hash: `__________`
- [ ] `git push`

---

## 2. Data Prerequisites

### Assembled Parquets

| Scenario | S3 Key | Status |
|----------|--------|--------|
| houston | `processed/houston/houston_event_features.parquet` | [ ] |
| new_orleans | `processed/new_orleans/no_event_features.parquet` | [x] |
| southwest_florida | `processed/southwest_florida/swfl_event_features.parquet` | [x] |
| nyc | `processed/nyc/nyc_event_features.parquet` | [ ] |
| riverside_coachella | `processed/riverside_coachella/rc_event_features.parquet` | [ ] |

### Reference Data

- [ ] `raw/geocertdb2026/zcta_county_crosswalk.parquet`
- [ ] `raw/geocertdb2026/zcta_adjacency.parquet` (optional, folds fall back to ZIP3)

### FAST Data (Phase 7)

- [ ] `raw/floodsimbench/6hr_max/*.tif` (78 files, 3 GB) -- FloodSimBench depth grids
- [ ] `raw/noaa_slosh/mom_national/*.tif` (5 cats, 5.5 GB) -- SLOSH MOM surge
- [ ] `raw/nsi/v2/houston_structures.parquet` -- NSI 2.0 (data team must fetch)
- [ ] `raw/nsi/v2/nyc_structures.parquet` -- NSI 2.0 (data team must fetch)
- [ ] `raw/nsi/v2/southwest_florida_structures.parquet` -- NSI 2.0 (stretch, data team)

---

## 3. Phase 0: Audit Battery (existing orchestrator)

```bash
python scripts/launch_strat_sampler_qa.py --scenario houston
```

- [ ] All P1-P6 support probes PASS
- [ ] Results at `evidence/qa/coverage_audit_{scenario}.json`

---

## 3.5. Phase 0.5: Geometry Kappa (v1.8 — pre-training)

Computes kappa_geom from problem geometry ONLY (spatial connectivity,
feature coverage, scale stability, admin alignment).  Must run BEFORE
any model training to establish independence from RSN coordinates.

```bash
python scripts/launch_compute_geometry_kappa.py --dry-run
python scripts/launch_compute_geometry_kappa.py
```

- [ ] `results/s035/geometry_kappa.json` on S3
- [ ] kappa_geom has zero dependency on predictions, residuals, or fold metrics
- [ ] S3 timestamp precedes Phase 1 R0 training timestamps

---

## 4. Phase 1: R0 Baseline Training

### Per Scenario (repeat for: houston, southwest_florida, nyc, riverside_coachella)

```bash
python scripts/launch_train_r0_baseline.py --scenario {scenario} --dry-run
python scripts/launch_train_r0_baseline.py --scenario {scenario}
```

- [ ] houston: `results/s035/r0_houston.json` on S3
- [ ] southwest_florida: `results/s035/r0_southwest_florida.json` on S3
- [ ] nyc: `results/s035/r0_nyc.json` on S3
- [ ] riverside_coachella: `results/s035/r0_riverside_coachella.json` on S3
- [ ] `folds/{scenario}_folds.parquet` with 3 fold columns, no NaN
- [ ] At least 2 cells have R2 > 0 (H1 gate)
- [ ] Prediction parquets saved: `results/s035/r0_{scenario}_predictions.parquet`

---

## 5. Phase 4a: Kappa Diagnostics R0

```bash
python scripts/launch_compute_diagnostics.py --level r0
```

- [ ] `results/s035/diagnostics_r0.json` uploaded to S3
- [ ] Uploaded BEFORE Phase 2 starts (pre-registration timestamp)
- [ ] 4 kappa proxies computed per cell
- [ ] Pre-registration predictions (flagged/unflagged) recorded

---

## 6. Phase 4.5a: Certificates R0

```bash
python scripts/launch_compute_certificates.py --level r0
```

- [ ] `results/s035/certificates_r0.json` on S3
- [ ] R + S_sup + N = 1 for every cell (simplex check)
- [ ] alpha, omega, kappa, sigma populated per cell

---

## 7. Phase 2: R1 Hydrology Training

### Per Scenario

```bash
python scripts/launch_train_r1_hydrology.py --scenario {scenario} --dry-run
python scripts/launch_train_r1_hydrology.py --scenario {scenario}
```

- [ ] houston: `results/s035/r1_houston.json` on S3
- [ ] southwest_florida: `results/s035/r1_southwest_florida.json` on S3
- [ ] nyc: `results/s035/r1_nyc.json` on S3
- [ ] riverside_coachella: `results/s035/r1_riverside_coachella.json` on S3
- [ ] W-matrix features present (8 columns: wlag_*, spatial_lag_*, zcta_degree, zcta_mean_neighbor_dist_km)
- [ ] W-matrix spatial lag computed per-fold, train-only (no leakage)
- [ ] Prediction parquets saved

---

## 8. Phase 4b: Kappa Diagnostics R1

```bash
python scripts/launch_compute_diagnostics.py --level r1
```

- [ ] `results/s035/diagnostics_r1.json` uploaded BEFORE Phase 3
- [ ] Pre-registration predictions for R2 recorded

---

## 9. Phase 4.5b: Certificates R1

```bash
python scripts/launch_compute_certificates.py --level r1
```

- [ ] `results/s035/certificates_r1.json` on S3
- [ ] Simplex check: R + S_sup + N = 1

---

## 10. Phase 2.5: R1.5 FAST Features (CONDITIONAL)

**Skip if NSI 2.0 not on S3. Go directly to Phase 3.**

```bash
python scripts/launch_run_fast_zcta.py --scenario houston --return-period 100yr
```

- [ ] NSI 2.0 confirmed on S3 (houston + nyc minimum)
- [ ] floodcaster `run_nsi_flood_analysis()` available
- [ ] 6 FAST ZCTA features computed per scenario
- [ ] `processed/{scenario}/{scenario}_fast_zcta.parquet` on S3

---

## 11. Phase 3: R2 Temporal Training

### Per Scenario

```bash
python scripts/launch_train_r2_temporal.py --scenario {scenario} --dry-run
python scripts/launch_train_r2_temporal.py --scenario {scenario}
```

- [ ] houston: `results/s035/r2_houston.json` on S3
- [ ] southwest_florida: `results/s035/r2_southwest_florida.json` on S3
- [ ] nyc: `results/s035/r2_nyc.json` on S3
- [ ] riverside_coachella: `results/s035/r2_riverside_coachella.json` on S3
- [ ] Temporal features present (9 columns: mrms_*, tide_*, hurdat2_*)
- [ ] Coastal NaN handling documented
- [ ] Prediction parquets saved

---

## 12. Phase 4c + 4.5c: Kappa Diagnostics + Certificates R2

```bash
python scripts/launch_compute_diagnostics.py --level r2
python scripts/launch_compute_certificates.py --level r2
```

- [ ] `results/s035/diagnostics_r2.json` on S3
- [ ] `results/s035/certificates_r2.json` on S3
- [ ] Certificate evolution across R0->R1->R2 inspectable

---

## 13. Phase 5: Money Table + Hypothesis Tests

```bash
python scripts/launch_compute_uplift_table.py
```

- [ ] `results/s035/uplift_table.json` on S3
- [ ] All 9 cells reported (no cherry-picking)
- [ ] H2a primary test: fold-level Wilcoxon signed-rank (R0 vs R1), p < 0.05 AND Cohen's d > 0.2
- [ ] H2b exploratory: Spearman rho(kappa_geom, uplift) with bootstrap CI (n=9)
- [ ] 8 exploratory cell-level associations reported with bootstrap CIs
- [ ] Negative results reported honestly if H2a fails

---

## 14. Phase 6: DGM Routing (Exploratory)

```bash
python scripts/launch_compute_dgm_routing.py
```

- [ ] `results/s035/dgm_routing.json` on S3
- [ ] Routing table: morph_decision + recommended_arm + actual_best_arm per cell
- [ ] Hit rate with binomial CI
- [ ] H5 framed as proof-of-concept (n=7 too small for inference)

---

## 15. Phase 7: FAST External Validation

```bash
python scripts/launch_run_fast_zcta.py --scenario houston --all-return-periods
python scripts/launch_compute_fast_validation.py
```

- [ ] FAST ZCTA aggregates on S3 for Houston (primary) + NYC (primary)
- [ ] Validation table: rho(NFIP_obs,FAST), rho(R0,FAST), rho(R1,FAST), rho(R2,FAST)
- [ ] Multiple return period robustness check (10yr, 50yr, 100yr, 500yr)
- [ ] `results/s035/fast_validation.json` on S3
- [ ] Kill rule check: rho(NFIP_obs, FAST) > 0 for each scenario

---

## 16. Execution Record

| Phase | Date | Commit | Job Name | Duration | Result |
|-------|------|--------|----------|----------|--------|
| 0 (audit) | | | | | |
| 1 (R0) | 2026-05-29 | 54015bc | s035-r0-baseline-southwest-florida-20260530-045744 | pending | pending |
| 4a (kappa R0) | | | | | |
| 4.5a (cert R0) | | | | | |
| 2 (R1) | | | | | |
| 4b (kappa R1) | | | | | |
| 4.5b (cert R1) | | | | | |
| 2.5 (R1.5 FAST) | | | | | conditional |
| 3 (R2) | | | | | |
| 4c (kappa R2) | | | | | |
| 4.5c (cert R2) | | | | | |
| 5 (money table) | | | | | |
| 6 (DGM) | | | | | |
| 7a (FAST ZCTA) | | | | | |
| 7b (FAST valid.) | | | | | |
