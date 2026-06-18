# PhD Best-Practice Benchmark Completion Checklist

**Experiment:** s035-model-ladder
**Date:** 2026-06-07
**Target venue:** ACM SIGSpatial
**DOE version:** v1.9 (DOE_AMENDMENT_v1.2.md)

---

## A. Reproducibility

| # | Item | Status | Evidence |
|---|------|--------|----------|
| A1 | Code versioned and committed (git hash traceability) | DONE | Git repo `georsct`, latest commit `23f8ca9`; CHECKLIST.md has hash placeholder for per-run recording |
| A2 | Random seeds fixed | DONE | `seed=42` specified in DOE_R0_baseline.md, enforced in all training scripts |
| A3 | Dependencies pinned (requirements.txt) | PARTIAL | `requirements.txt` exists with `>=` lower bounds (e.g. `scikit-learn>=1.4`) but no upper bounds or lockfile. Not a fully reproducible pin. |
| A4 | Full pipeline re-runnable from raw data to final figures | PARTIAL | Scripts exist for all phases; SageMaker launchers documented in CHECKLIST.md. However, no single `make all` or `snakemake` workflow exists. Requires manual phase-by-phase execution. |
| A5 | Fold assignments saved and reused across levels | DONE | `s3://swarm-floodrsct-data/folds/{scenario}_folds.parquet` (5 scenarios, all present with metadata JSONs) |
| A6 | Hyperparameters explicitly stated (no tuning) | DONE | DOE_R0_baseline.md: HistGBDT max_iter=200, max_depth=6, lr=0.1; Ridge alpha=1.0 |
| A7 | Wall-clock time and instance type documented | GAP | CHECKLIST.md has placeholder fields but no actual execution records filled in |

---

## B. Statistical Rigor

| # | Item | Status | Evidence |
|---|------|--------|----------|
| B1 | Hypotheses pre-registered in DOE | DONE | H1-H4 in DOE_LOCKED.md; H5-H8 in DOE_R3_certificate_gated_admission.md; H7-H9 in DOE_R4_vlm.md |
| B2 | Effect sizes reported alongside p-values | DONE | Cohen's d reported in all fold-level tests (e.g. H2 pooled d=0.27, H3 pooled d=0.85) in `money_table.json` |
| B3 | Multiple comparison corrections applied | DONE | Statistical-Considerations.md explicitly justifies no correction needed (single pre-registered primary test H2a); secondary tests clearly labeled exploratory |
| B4 | Confidence intervals provided | PARTIAL | Per-cell bootstrap CIs present in `money_table.json` (6 entries in `cell_bootstrap_ci`); per-fold CIs in cell_tests. But R3 money table has `r3_headline_ci` and `r3_full_ci` per row. Missing: pooled CI for the primary H2 test. |
| B5 | Sample size justified (power analysis) | GAP | No formal power analysis document found. Statistical-Considerations.md states n=45 paired observations for Wilcoxon but does not compute post-hoc power or justify sufficiency formally. |
| B6 | Null/negative results reported | DONE | H2 pooled verdict is "INCONCLUSIVE" (p=0.11); R3 H5 shows "NO_DEGRADATION" (negative); per-cell tests include negative deltas; DOE kill rules evaluated |
| B7 | Spatially-blocked paired loss test (V4.5) | GAP | `spatially_blocked_loss` NOT present in either `money_table.json` or `r3_money_table.json`. Required by EXPERIMENT_CONTRACT V4.5, V5.11. |
| B8 | Two-stage event aggregation (V4.10) | GAP | Not implemented — no evidence in money table of zcta-mean-then-county-mean aggregation |

---

## C. Spatial-Specific

| # | Item | Status | Evidence |
|---|------|--------|----------|
| C1 | Spatial autocorrelation in residuals tested (Moran's I) | DONE | LISA results for all scenarios x levels on S3 (`r0_*_lisa.json`, `r1_*_lisa.json`, `r2_*_lisa.json`); full LISA parquets in `sidecar/` |
| C2 | Spatial cross-validation used (not random CV) | DONE | County-blocked spatial CV as primary split; random split reported as diagnostic only. Fold files on S3 with spatial_fold column. |
| C3 | MAUP effects acknowledged | DONE | DOE_LOCKED.md Problem Statement explicitly names "B.1 MAUP"; DOE_R1_spatial.md adds HUC features to address it; diagnostics track `diag_leakage` across levels |
| C4 | Edge effects handled | PARTIAL | County crosswalk for spatial blocking handles boundary ZCTAs; adjacency matrix uses queen contiguity. But no explicit edge-trimming or buffer zone around metro boundaries documented. |
| C5 | Geary's C computed | DONE | `sidecar/geary_results.json` + `sidecar/geary_rollup.parquet` on S3 |
| C6 | GWR non-stationarity tested | DONE | `sidecar/gwr_nonstationarity.json` + per-scenario `gwr_local_r2_*.parquet` on S3 |
| C7 | Skater regionalization robustness | PARTIAL | `sidecar/robustness/` has region folds for 3/5 scenarios (houston, new_orleans, riverside_coachella). Missing: nyc, southwest_florida |

---

## D. Benchmark Completeness

| # | Item | Status | Evidence |
|---|------|--------|----------|
| D1 | R0 baseline complete (all 5 scenarios) | DONE | `r0_{scenario}.json` + `r0_{scenario}_predictions.parquet` for all 5 on S3 |
| D2 | R1 hydrology complete (all 5 scenarios) | DONE | `r1_hydrology_{scenario}.json` + predictions for all 5 on S3 |
| D3 | R2 temporal complete (all 5 scenarios) | DONE | `r2_{scenario}.json` + predictions for all 5 on S3 |
| D4 | R3 certified training complete (all 5 scenarios) | DONE | `r3_{scenario}.json` + predictions for all 5 on S3 |
| D5 | Diagnostics computed (R0/R1/R2) | DONE | `diagnostics_r0.json`, `diagnostics_r1.json`, `diagnostics_r2.json` on S3 |
| D6 | Certificates computed (R0/R1/R2) | DONE | `certificates_r0.json`, `certificates_r1.json`, `certificates_r2.json` on S3 |
| D7 | Geometry kappa (Phase 0.5) | DONE | `geometry_kappa.json` on S3 (2026-06-05, before training phases) |
| D8 | Gearbox warmup (Phase 0.75) | DONE | `gearbox_warmup.json` on S3 |
| D9 | Money table (R0-R2) | DONE | `money_table.json` on S3 (44KB, 8 cells, hypothesis evidence, bootstrap CIs) |
| D10 | R3 money table | DONE | `r3_money_table.json` on S3 (H5-H8, 3 variants, 8 rows) |
| D11 | DGM routing | DONE | `dgm_routing.json` on S3 |
| D12 | R3 block tests + admission | DONE | `r3_block_tests_{scenario}.json`, `r3_block_certificates_{scenario}.json`, `r3_block_admission_table.json`, `r3_dgm_admission_trace.json`, `r3_gear_summary.json` all on S3 |
| D13 | R3 order robustness | DONE | `r3_order_robustness_{scenario}.json` for all 5 on S3 |
| D14 | R4 VLM assessment (multi-model) | DONE | 6 VLMs x 5 scenarios: parquets + summaries + quality + calibration JSONs on S3 |
| D15 | FAST external validation (Phase 7) | DONE | `fast_validation.json` on S3 (2026-06-07) |
| D16 | R3 feature registry + candidate graph | DONE | `r3_feature_registry.json` + `r3_candidate_graph.json` on S3 |
| D17 | Kill rules evaluated | DONE | H1 pass implied by non-zero R2 across scenarios; H2a INCONCLUSIVE reported; kill rule "all uplift < 1%" not triggered (median 9.4%) |
| D18 | Verdicts file | DONE | `verdicts.json` on S3 (2026-06-07), covers n_geometries with summary |
| D19 | Leave-event-out (transfer) split | PARTIAL | DOE specifies leave-event-out as a split protocol; results contain `spatial_blocked` and `random` but leave-event-out status unclear from money table structure |

---

## E. Presentation

| # | Item | Status | Evidence |
|---|------|--------|----------|
| E1 | Publication-quality figures generated | PARTIAL | `sidecar/figures/fig5_lisa_triptych_houston.pdf` and `fig6_gwr_local_r2_houston.pdf` on S3. Only 2 figures, Houston only. No money table figure, no R0-R1-R2 comparison figure, no multi-scenario visualization. |
| E2 | Figures have proper captions with statistical details | GAP | No caption metadata found alongside figure files |
| E3 | Tables self-contained (readable without body text) | PARTIAL | `money_table.json` has full metadata (`methodology` block with test descriptions). But no formatted LaTeX/CSV table for paper insertion found locally or on S3. |
| E4 | Limitations section written | GAP | No dedicated limitations document found. Statistical-Considerations.md discusses pooling tradeoffs and R4 quarantine reasoning, but no consolidated limitations section for paper. |
| E5 | NeurIPS/SIGSpatial compute documentation | GAP | Instance types and wall-clock times not recorded in any results file. Required by venue (NeurIPS requires it per memory note). |

---

## F. Verification Gates (from EXPERIMENT_CONTRACT)

| Gate | Status | Notes |
|------|--------|-------|
| V0 (Pre-Launch) | DONE | Scripts have dry-run; CHECKLIST.md covers 9 dimensions |
| V1 (Job Completion) | DONE | All 5 scenarios have results for R0/R1/R2/R3 |
| V2 (Domain Invariants) | DONE | Simplex validity in certificates; geometry kappa pre-training |
| V3 (Cross-Fold Stability) | PARTIAL | Fold-level tests show variance but no explicit V3.1 (fold spread < 0.30) verification artifact |
| V4.5 (Spatially-blocked paired loss) | GAP | Not implemented in either money table |
| V4.10 (Two-stage event aggregation) | GAP | Not implemented |
| V5 (Paper-Readiness) | GAP | Multiple V5 sub-checks not met (V5.8 wall-clock, V5.9 instance, V5.11 spatially_blocked_loss, V5.15 two-stage) |

---

## Top 5 Priority Gaps (Ranked by Submission Impact)

### 1. Spatially-blocked paired loss test (B7, V4.5, V5.11)

**Impact:** CRITICAL -- The experiment contract requires this as a verification gate, and it provides the spatially-honest p-value that reviewers at SIGSpatial will demand. The current fold-level Wilcoxon pools observations that share spatial substrate within a fold; the spatially-blocked paired loss uses county-level aggregation to provide a conservative independence-respecting test.

**Action:** Run `compute_uplift_table.py` with spatially-blocked paired loss enabled (county crosswalk exists on S3). Must also implement two-stage aggregation (per-ZCTA mean across events, then per-county mean).

**Artifact needed:** `spatially_blocked_loss` block in `money_table.json` AND `r3_money_table.json`

---

### 2. Pinned dependency lockfile (A3)

**Impact:** HIGH -- Any reviewer attempting reproduction will hit version drift. `scikit-learn>=1.4` could resolve to 1.4 or 1.7 with different numerical results. The sklearn pickle versioning issue (per memory) makes this especially dangerous.

**Action:** Generate `requirements-lock.txt` (or `environment.yaml` with exact versions) from the SageMaker container that produced the canonical results. Record the exact scikit-learn, numpy, scipy versions used.

---

### 3. Publication figures for all scenarios (E1, E2)

**Impact:** HIGH -- Only 2 Houston-specific figures exist. A SIGSpatial submission needs: (a) money table as a formatted figure/table, (b) representation ladder comparison across all 5 metros, (c) LISA maps for at least 2-3 representative scenarios, (d) DGM routing visualization. All with proper captions including n, test statistics, and CIs.

**Action:** Run figure generation scripts for all scenarios; add caption metadata; generate LaTeX-ready formatted money table.

---

### 4. Wall-clock time and compute documentation (A7, E5)

**Impact:** MEDIUM-HIGH -- SIGSpatial (and NeurIPS-style venues) require documenting instance types and total compute hours for reproducibility and environmental impact assessment.

**Action:** Backfill from CloudWatch logs or SageMaker job history. Record in CHECKLIST.md execution records: instance type, duration per phase, total GPU/CPU hours.

---

### 5. Formal power analysis or post-hoc power statement (B5)

**Impact:** MEDIUM -- H2a came back INCONCLUSIVE (p=0.11, d=0.27). A post-hoc power analysis would tell reviewers whether n=20 regression folds was simply underpowered for the observed effect size, or whether the effect truly does not exist. This is especially important because the INCONCLUSIVE verdict is a major finding.

**Action:** Compute post-hoc power for Wilcoxon signed-rank with observed d=0.27 and n=20. State the sample size needed for 80% power at this effect size. Include in Statistical-Considerations.md and paper methodology.

---

## Summary Statistics

| Category | DONE | PARTIAL | GAP |
|----------|------|---------|-----|
| A. Reproducibility | 5 | 2 | 1 |
| B. Statistical Rigor | 4 | 1 | 3 |
| C. Spatial-Specific | 4 | 2 | 0 |
| D. Benchmark Completeness | 17 | 2 | 0 |
| E. Presentation | 0 | 2 | 3 |
| **Total** | **30** | **9** | **7** |

**Overall readiness:** ~65% paper-ready. The experimental execution is strong (D category is nearly complete). The primary gaps are in statistical presentation layer (spatially-blocked loss, figures, compute docs) and reproducibility infrastructure (lockfile, power analysis).
