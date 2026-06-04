# S035-Model-Ladder Pre-Flight Checklist

**Bucket:** `s3://swarm-floodrsct-data/`
**Job scripts:** `data/floodrsct/jobs/`
**Launchers:** `data/floodrsct/scripts/`
**DOE version:** v1.9 (DOE_AMENDMENT_v1.2.md)

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

### Per Scenario (repeat for: houston, southwest_florida, nyc, riverside_coachella, new_orleans)

```bash
python scripts/launch_train_r0_baseline.py --scenario {scenario} --dry-run
python scripts/launch_train_r0_baseline.py --scenario {scenario}
```

- [x] houston: `results/s035/r0_houston.json` on S3
- [x] southwest_florida: `results/s035/r0_southwest_florida.json` on S3
- [x] nyc: `results/s035/r0_nyc.json` on S3
- [x] riverside_coachella: `results/s035/r0_riverside_coachella.json` on S3
- [x] new_orleans: `results/s035/r0_new_orleans.json` on S3
- [x] `folds/{scenario}_folds.parquet` with 3 fold columns, no NaN
- [x] At least 2 cells have R2 > 0 (H1 gate)
- [x] Prediction parquets saved: `results/s035/r0_{scenario}_predictions.parquet`

---

## 5. Phase 4a: Kappa Diagnostics R0

```bash
python scripts/launch_compute_diagnostics.py --level r0
```

- [x] `results/s035/diagnostics_r0.json` uploaded to S3
- [x] Uploaded BEFORE Phase 2 starts (pre-registration timestamp)
- [x] 4 kappa proxies computed per cell
- [x] Pre-registration predictions (flagged/unflagged) recorded

---

## 6. Phase 4.5a: Certificates R0

```bash
python scripts/launch_compute_certificates.py --level r0
```

- [x] `results/s035/certificates_r0.json` on S3
- [x] R + S_sup + N = 1 for every cell (simplex check)
- [x] alpha, omega, kappa, sigma populated per cell

---

## 7. Phase 2: R1 Hydrology Training

### Per Scenario

```bash
python scripts/launch_train_r1_hydrology.py --scenario {scenario} --dry-run
python scripts/launch_train_r1_hydrology.py --scenario {scenario}
```

- [x] houston: `results/s035/r1_hydrology_houston.json` on S3
- [x] southwest_florida: `results/s035/r1_hydrology_southwest_florida.json` on S3
- [x] nyc: `results/s035/r1_hydrology_nyc.json` on S3
- [x] riverside_coachella: `results/s035/r1_hydrology_riverside_coachella.json` on S3
- [x] new_orleans: `results/s035/r1_hydrology_new_orleans.json` on S3
- [x] W-matrix features present (8 columns: wlag_*, spatial_lag_*, zcta_degree, zcta_mean_neighbor_dist_km)
- [x] W-matrix spatial lag computed per-fold, train-only (no leakage)
- [x] Prediction parquets saved

---

## 8. Phase 4b: Kappa Diagnostics R1

```bash
python scripts/launch_compute_diagnostics.py --level r1
```

- [x] `results/s035/diagnostics_r1.json` uploaded BEFORE Phase 3
- [x] Pre-registration predictions for R2 recorded

---

## 9. Phase 4.5b: Certificates R1

```bash
python scripts/launch_compute_certificates.py --level r1
```

- [x] `results/s035/certificates_r1.json` on S3
- [x] Simplex check: R + S_sup + N = 1

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

- [x] houston: `results/s035/r2_houston.json` on S3
- [x] southwest_florida: `results/s035/r2_southwest_florida.json` on S3
- [x] nyc: `results/s035/r2_nyc.json` on S3
- [x] riverside_coachella: `results/s035/r2_riverside_coachella.json` on S3
- [x] new_orleans: `results/s035/r2_new_orleans.json` on S3
- [x] Temporal features present (9 columns: mrms_*, tide_*, hurdat2_*)
- [x] Coastal NaN handling documented
- [x] Prediction parquets saved

---

## 12. Phase 4c + 4.5c: Kappa Diagnostics + Certificates R2

```bash
python scripts/launch_compute_diagnostics.py --level r2
python scripts/launch_compute_certificates.py --level r2
```

- [x] `results/s035/diagnostics_r2.json` on S3
- [x] `results/s035/certificates_r2.json` on S3
- [x] Certificate evolution across R0->R1->R2 inspectable

---

## 13. Phase 5: Money Table + Hypothesis Tests

```bash
python scripts/launch_compute_uplift_table.py
```

- [x] `results/s035/money_table.json` on S3 (40 KB, 9 cells)
- [x] All 9 cells reported (no cherry-picking)
- [x] H2a primary test: fold-level Wilcoxon signed-rank (R0 vs R1)
- [x] H2b exploratory: Spearman rho(kappa_geom, uplift) with bootstrap CI (n=9)
- [x] 8 exploratory cell-level associations reported with bootstrap CIs
- [x] Negative results reported honestly if H2a fails

---

## 14. Phase 6: DGM Routing (Exploratory)

```bash
python scripts/launch_compute_dgm_routing.py
```

- [x] `results/s035/dgm_routing.json` on S3 (7.5 KB, 11 cells)
- [x] Routing table: morph_decision + recommended_arm + actual_best_arm per cell
- [x] Hit rate with binomial CI (0% -- cert field mismatch, proof-of-concept)
- [x] H5 framed as proof-of-concept (n=11)

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
| 1 (R0) | 2026-06-03 | 915ba42 | s035-r0-baseline-{scenario} | ~5 min each | COMPLETE (5/5 scenarios) |
| 4a (diag R0) | 2026-06-03 | 915ba42 | s035-diagnostics-r0 | ~3 min | COMPLETE (11 cells) |
| 4.5a (cert R0) | 2026-06-03 | 915ba42 | s035-certificates-r0 | ~3 min | COMPLETE (11 cells) |
| 2 (R1) | 2026-06-03 | 915ba42 | s035-r1-hydrology-{scenario} | ~5 min each | COMPLETE (5/5 scenarios) |
| 4b (diag R1) | 2026-06-03 | 915ba42 | s035-diagnostics-r1 | ~3 min | COMPLETE (11 cells) |
| 4.5b (cert R1) | 2026-06-03 | 915ba42 | s035-certificates-r1 | ~3 min | COMPLETE (11 cells) |
| 2.5 (R1.5 FAST) | -- | -- | -- | -- | SKIPPED (NSI 2.0 not on S3) |
| 3 (R2) | 2026-06-03 | 915ba42 | s035-r2-temporal-{scenario} | ~5 min each | COMPLETE (5/5 scenarios) |
| 4c (diag R2) | 2026-06-03 | 915ba42 | s035-diagnostics-r2 | ~3 min | COMPLETE (11 cells) |
| 4.5c (cert R2) | 2026-06-03 | 915ba42 | s035-certificates-r2 | ~3 min | COMPLETE (11 cells) |
| 5 (money table) | 2026-06-03 | 5d83a25 | s035-uplift-table | ~3 min | COMPLETE (9 cells, 40 KB) |
| 6 (DGM) | 2026-06-03 | 5d83a25 | s035-dgm-routing | ~3 min | COMPLETE (11 cells, proof-of-concept) |
| LISA (standalone) | 2026-06-03 | cc6faba | s035-lisa-{level}-{scenario} | ~3 min each | COMPLETE (30 parquets) |
| Sidecar LISA | 2026-06-03 | 4a74171 | s035-sidecar-lisa | ~5 min | COMPLETE (34 parquets + rollup) |
| Sidecar GWR | 2026-06-04 | a53b230 | s035-sidecar-gwr | ~6 min | COMPLETE (4/5 local R2 parquets) |
| Sidecar Geary | 2026-06-03 | 4a74171 | s035-sidecar-geary (via all) | ~5 min | COMPLETE (32 cells) |
| Fig 5 (LISA) | 2026-06-04 | a53b230 | s035-fig5-lisa-houston | ~3 min | COMPLETE (PDF 1.8 MB + SVG) |
| Fig 6 (GWR) | 2026-06-04 | a53b230 | s035-fig6-gwr-houston | ~3 min | COMPLETE (PDF 800 KB + SVG) |
| 7a (FAST ZCTA) | -- | -- | -- | -- | DEFERRED (NSI 2.0 not on S3) |
| 7b (FAST valid.) | -- | -- | -- | -- | DEFERRED |
