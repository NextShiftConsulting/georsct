# S035 Checklist by R-Level (Ground Truth)

**Generated:** 2026-06-10 from SageMaker job history + S3 artifacts + git log
**5 scenarios:** houston (HOU), southwest_florida (SWFL), nyc, riverside_coachella (RC), new_orleans (NOLA)

---

## R0 — Static Tabular Baseline (~33-36 features)

**DOE:** DOE_LOCKED.md (H1: baseline skill) — LOCKED
**Hypothesis:** HistGBDT on R0 achieves RMSE < naive mean on >= 2/3 targets

### SageMaker Jobs (reverse chronological)

| Date | Job | Status |
|------|-----|--------|
| 06-07 | s035-build-events-{5 metros} | Completed (5/5) |
| 06-05 | s035-r0-baseline-{5 metros} | Completed (5/5) |
| 06-06 | s035-r0-baseline-nyc (NYC rebuild w/ Sandy+2023) | Completed |

### S3 Artifacts

- [x] r0_{5 metros}.json — training results (06-07 reruns, 16-38 KB)
- [x] r0_{5 metros}_predictions.parquet (06-07)
- [x] r0_{5 metros}_lisa.json + .parquet (06-03)
- [x] diagnostics_r0.json (06-07)
- [x] certificates_r0.json + .parquet (06-10, data uploaded, print crash after)
- [x] geometry_kappa.json (06-05) — pre-training

### **Status: COMPLETE**

---

## R1 — Spatial/Hydrologic (+hydrology, W-matrix; ~58-63 features)

**DOE:** DOE_R1_spatial.md (H2: R1 > R0) — LOCKED
**Hypothesis:** Fold-level Wilcoxon R0 vs R1, p<0.05 AND Cohen's d>0.2

### SageMaker Jobs

| Date | Job | Status |
|------|-----|--------|
| 06-06 | s035-r1-full-nyc (NYC rebuild) | Completed |
| 06-05 | s035-r1-full-{5 metros} | Completed (5/5) |

### S3 Artifacts

- [x] r1_hydrology_{5 metros}.json + predictions (06-05, NYC 06-06)
- [x] r1_{no_target_lag, no_wlag, wlag_only}_houston — ablations (06-03)
- [x] r1_{5 metros}_lisa.json + .parquet (06-03)
- [x] diagnostics_r1.json (06-07)
- [x] certificates_r1.json + .parquet (06-10, data uploaded, print crash after)

### **Status: COMPLETE**

---

## R2 — Temporal Event (+precipitation, tide, storm tracks; ~67-72 features)

**DOE:** DOE_R2_temporal.md (H3: R2 > R1) — LOCKED
**Hypothesis:** R2 RMSE < R1 RMSE by > 3% on primary target

### SageMaker Jobs

| Date | Job | Status |
|------|-----|--------|
| 06-06 | s035-r2-temporal-nyc (NYC rebuild) | Completed |
| 06-05 | s035-r2-temporal-{5 metros} | Completed (5/5) |

### S3 Artifacts

- [x] r2_{5 metros}.json + predictions (06-05, NYC 06-06)
- [x] r2_{5 metros}_lisa.json + .parquet (06-03)
- [x] diagnostics_r2.json (06-07)
- [x] certificates_r2.json + .parquet (06-10, data uploaded, print crash after)

### **Status: COMPLETE**

---

## Cross-Level (R0-R2) Analysis

### SageMaker Jobs

| Date | Job | Status |
|------|-----|--------|
| 06-08 | s035-dgm-routing | Completed |
| 06-08 | s035-certificates-r{0,1,2} (coherence rerun) | Completed (3/3) |
| 06-07 | s035-verdicts | Completed |
| 06-07 | s035-fast-validation | Completed |
| 06-07 | s035-region-all (sidecar: LISA+Geary+GWR) | Completed (3 runs) |
| 06-07 | s035-fig5-lisa-houston, fig6-gwr-houston | Completed |
| 06-07 | s035-fast-zcta-{houston,nyc} | Completed (2/2) |
| 06-06 | s035-uplift-table (money table) | Completed (3 runs) |
| 06-06 | s035-diagnostics-r{0,1,2} | Completed (6 runs, 2 per level) |
| 06-06 | s035-certificates-r{0,1,2} | Completed (6 runs) |
| 06-06 | s035-gearbox-warmup | Completed (2 runs) |

### S3 Artifacts

- [x] money_table.json (06-06, 44 KB) — H2/H3/H4
- [x] evidence/h2_evidence.json, h3, h4 (06-06)
- [x] dgm_routing.json (06-08) — H5
- [x] verdicts.json (06-07)
- [x] fast_validation.json (06-07)
- [x] sidecar/{lisa,geary,gwr}_results + rollup + figures

### **Status: COMPLETE**

---

## R3 — Certificate-Gated Feature Admission (DGM-controlled block routing)

**DOE:** DOE_R3_certificate_gated_admission.md — **DRAFT (never locked)**
**Hypothesis:** DGM-gated admission outperforms blind enrichment

### SageMaker Jobs (reverse chronological)

| Date | Job | Status | Notes |
|------|-----|--------|-------|
| 06-10 | s035-r3-block-tests-{5 metros} | **Completed (5/5)** | 3rd attempt, after kappa fix |
| 06-10 | s035-certificates-r{0,1,2} (rerun w/ rsct v0.3.0) | Failed (3/3) | Print bug `c['kappa']`; data uploaded OK |
| 06-09 | s035-r3-block-tests-{5 metros} batch 2 | Failed (5/5) | stale wheel |
| 06-09 | s035-r3-block-tests-{4 metros} batch 1 | Failed (4/4) | missing SW FL |
| 06-06 | s035-r3-block-tests-{5 metros} | Completed (5/5) | earlier version |
| 06-06 | s035-r3-money-table | Completed (2 runs) | |
| 06-06 | s035-r3-certified-{5 metros} | Completed (5/5) | training |
| 06-06 | s035-r3-block-admission | Completed | DGM admission trace |
| 06-06 | s035-r3-block-tests-nyc (solo) | Completed | NYC rebuild |
| 06-06 | s035-r3-order-robust-nyc | Completed | |
| 06-05 | s035-r3-order-robust-{5 metros} | Completed (3/5, 2 stopped+rerun) | |
| 06-05 | s035-r3-block-tests-{5 metros} (1st gen) | Completed (5/5) | |
| 06-05 | s035-r3-block-admission (1st gen) | Completed | |
| 06-05 | s035-r3-certified-{5 metros} (1st gen) | Completed (5/5) | |
| 06-05 | s035-r3-money-table (1st gen) | Completed | |
| 06-05 | s035-dgm-routing (1st gen) | Completed | |

### S3 Artifacts

- [x] r3_{5 metros}.json + predictions (06-06)
- [x] r3_block_tests_{5 metros}.json (06-10, latest run)
- [x] r3_block_certificates_{5 metros}.json (06-10)
- [x] r3_order_robustness_{5 metros}.json (06-05/06)
- [x] r3_candidate_graph.json (06-05)
- [x] r3_feature_registry.json (06-05, 51 KB)
- [x] r3_dgm_admission_trace.json (06-06, 120 KB)
- [x] r3_gear_summary.json (06-06)
- [x] r3_block_admission_table.json (06-06, 132 KB)
- [x] r3_money_table.json (06-06, 10 KB)
- [x] r3_hypothesis_evidence.json (06-06)

### Missing

- [ ] DOE never locked — executed as DRAFT
- [ ] No R3 LISA diagnostics (no sidecar spatial analysis)
- [ ] No certificates_r3.json (level-level certs; only block-level certs exist)
- [ ] No diagnostics_r3.json

### **Status: DATA COMPLETE, ANALYSIS GAPS, DOE NOT LOCKED**

---

## R4 — Vision-Language Model (map + text -> structured risk; 7 VLMs)

**DOE:** DOE_R4_vlm.md — **DESIGN (never locked)**
**Hypotheses:** H7 (VLM signal rho>0.3), H8 (inter-rater rho>0.7), H9 (kappa correlation)

### SageMaker Jobs (reverse chronological)

| Date | Job | Status | Notes |
|------|-----|--------|-------|
| 06-09 | s035-quality-gpt4o-{5 metros} | Completed (5/5) | 2nd quality run |
| 06-09 | s035-quality-{qwen,nova,jina,gemini-flash}-{5 metros} | Completed (20/20) | 2nd quality run |
| 06-09 | s035-vlm-gpt4o-{4 metros} (rerun HOU,NYC,RC,SWFL) | Completed (4/4) | missing NOLA (already had it) |
| 06-09 | s035-zcta-evidence-{5 metros} | Completed (5/5) | evidence docs |
| 06-08 | s035-zcta-maps-{5 metros} | Completed (5/5, 2 batches) | map rendering |
| 06-08 | s035-vlm-{qwen,nova,jina}-{5 metros} | Completed (15/15) | |
| 06-08 | s035-vlm-gemini-{pro,flash}-{5 metros} | Completed (10/10) | |
| 06-08 | s035-vlm-gpt4o-{5 metros} (1st batch) | Completed (5/5) | |
| 06-07 | s035-quality-{gpt4o,gemini,jina,nova,qwen}-{5 metros} | Completed (25/25) | 1st quality run |

### S3 Artifacts — VLM Inference (7 providers x 5 metros = 35 parquets)

| Provider | HOU | SWFL | NYC | RC | NOLA |
|----------|:---:|:----:|:---:|:--:|:----:|
| gemini (1.5) | [x] 06-02 | [x] | [x] | [x] | [x] |
| gemini_2_5_pro | [x] 06-08 | [x] | [x] | [x] | [x] |
| gemini_3_5_flash | [x] 06-08 | [x] | [x] | [x] | [x] |
| gpt4o | [x] 06-09 | [x] | [x] | [x] | [x] 06-08 |
| jina | [x] 06-08 | [x] | [x] | [x] | [x] |
| nova | [x] 06-08 | [x] | [x] | [x] | [x] |
| qwen | [x] 06-08 | [x] | [x] | [x] | [x] |

### S3 Artifacts — Quality + Calibration (5 providers x 5 metros = 25 each)

- [x] r4_quality_{gemini_3_5_flash,gpt4o,jina,nova,qwen}_{5 metros} (06-09, supersedes 06-07 runs)
- [x] r4_calibration_{gemini_3_5_flash,gpt4o,jina,nova,qwen}_{5 metros} (06-09)
- [x] maps/{5 metros}/ + manifests (06-08)

### Missing

- [ ] DOE never locked — executed as DESIGN
- [ ] r4_money_table.json is **STALE** (06-02) — predates all 06-08/09 VLM runs; needs regen
- [ ] No diagnostics_r4.json
- [ ] No certificates_r4.json
- [ ] No R4 LISA diagnostics
- [ ] **H7 not tested** — no rho(VLM_score, NFIP_claims) analysis
- [ ] **H8 not tested** — no pairwise inter-rater agreement analysis
- [ ] **H9 not tested** — no kappa-diagnostic comparison
- [ ] **No H7/H8/H9 evidence JSONs**
- [ ] Tier-2 relevance quorum (commit 6e753f3) — code written 06-10, never run

### **Status: RAW DATA COMPLETE (85 parquets + 50 quality/cal), ALL ANALYSIS MISSING**

---

## R5 — VLA/Agent Evidence + Action (harness evolution)

**DOE:** None
**Git:** 3 commits, code only

| Date | Commit | Description |
|------|--------|-------------|
| 06-08 | 1cc9004 | docs(r5): add R4 vs R5 architectural comparison |
| 06-07 | 02de901 | feat(r5): wire simplex into evolution protocol |
| 06-05 | cc73bd5 | feat(r5): add harness evolution protocol |

### SageMaker Jobs: **NONE**
### S3 Artifacts: **NONE**

### **Status: NOT STARTED (code scaffold only)**

---

## Known Bugs

| Bug | Impact | Fix |
|-----|--------|-----|
| `compute_certificates.py:414` `KeyError: 'kappa'` | Print crash after successful S3 upload; SageMaker reports Failed but data is correct | Change `c['kappa']` to `c['kappa_compat']` |

---

## Job Counts by Phase

| Phase | Completed | Failed | Stopped | Total |
|-------|-----------|--------|---------|-------|
| Data fetch (NYC rebuild) | 8 | 1 | 0 | 9 |
| Build events | 6 | 0 | 0 | 6 |
| R0 baseline | 6 | 0 | 0 | 6 |
| R1 hydrology | 6 | 0 | 0 | 6 |
| R2 temporal | 6 | 0 | 0 | 6 |
| Diagnostics R0-R2 | 8 | 0 | 0 | 8 |
| Certificates R0-R2 | 12 | 6 | 0 | 18 |
| Gearbox warmup | 2 | 0 | 0 | 2 |
| Uplift/money table | 4 | 0 | 0 | 4 |
| FAST ZCTA | 3 | 4 | 0 | 7 |
| FAST validation | 1 | 1 | 0 | 2 |
| Verdicts | 2 | 0 | 1 | 3 |
| Sidecar (LISA/Geary/GWR/figs) | 5 | 2 | 0 | 7 |
| DGM routing | 2 | 0 | 0 | 2 |
| R3 certified training | 10 | 0 | 0 | 10 |
| R3 block admission | 2 | 0 | 0 | 2 |
| R3 block tests | 16 | 9 | 0 | 25 |
| R3 order robustness | 8 | 0 | 2 | 10 |
| R3 money table | 3 | 0 | 0 | 3 |
| R4 ZCTA maps | 10 | 0 | 0 | 10 |
| R4 ZCTA evidence | 5 | 0 | 0 | 5 |
| R4 VLM inference | 39 | 0 | 0 | 39 |
| R4 quality | 50 | 0 | 0 | 50 |
| **TOTAL** | **~210** | **~23** | **~3** | **~236** |

---

## Summary

| Level | Data | Analysis | DOE | Status |
|-------|------|----------|-----|--------|
| **R0** | 5/5 train + pred + LISA + diag + cert | H1 complete | LOCKED | **COMPLETE** |
| **R1** | 5/5 + 3 ablations + LISA + diag + cert | H2 complete | LOCKED | **COMPLETE** |
| **R2** | 5/5 + LISA + diag + cert | H3 complete | LOCKED | **COMPLETE** |
| **R0-R2** | money table + verdicts + DGM + FAST + sidecar | H4/H5/H6 complete | -- | **COMPLETE** |
| **R3** | 5/5 train + 5/5 block tests + block certs + order robust | No LISA, no diag, no level certs | **DRAFT** | **DATA DONE, ANALYSIS GAP** |
| **R4** | 35 VLM + 25 quality + 25 calibration | Stale money table, H7-H9 untested | **DESIGN** | **RAW DATA DONE, ANALYSIS NOT DONE** |
| **R5** | 0 | 0 | None | **NOT STARTED** |
