# MISSING INVENTORY — GeoRSCT NeurIPS 2026

**Paper**: `V3/combine_georsct.tex` + 3 appendix inputs  
**Deadline**: Abstract May 4, Paper May 6  
**Last audit**: 2026-05-04

Legend: `[READY]` data exists, can fill now | `[NEEDS RUN]` requires SageMaker experiment | `[FIX]` structural issue in paper

---

## 1. COMPILATION BLOCKERS (paper won't build)

| # | Issue | File | Line(s) | Fix |
|---|-------|------|---------|-----|
| C1 | `figures/fig4_conus27_heatmap.pdf` missing from V3 | combine_georsct.tex | 259 | Copy from `../figures/fig4_trf_heatmap.pdf` (name mismatch) |
| C2 | `figures/fig5_spread_vs_trf.pdf` missing from V3 | combine_georsct.tex | 270 | Copy from `../figures/fig5_spread_vs_trf.pdf` (name mismatch) |
| C3 | `V3/figures/` directory does not exist | — | — | `mkdir V3/figures/` then copy or symlink |
| C4 | `table2_conus27.tex` missing (referenced by `\input` in appendix_h) | appendix_h_reproducibility.tex | 7 | Generate from `V2/figures/figures.py` CONUS_27 array |

---

## 2. TABLE 2 — Main Results (combine_georsct.tex lines 243-248)

**Status**: All XXX. Interpolation column data is `[READY]`, extrapolation data is `[NEEDS RUN]`.

### 2a. Interpolation columns (county holdout) — [READY]

Source: `V2/figures/figures.py` line 95, CONUS_27 array (canonical OOF, EVIDENCE_AUDIT 2026-04-29).

| Target family | PCA32 | GNN | Lag | N_ceil | Spread |
|---|---|---|---|---|---|
| Health (21) | 0.664 | 0.646 | 0.664 | 0.330 | 0.028 |
| Socioeconomic (3) | 0.680 | 0.653 | 0.693 | 0.303 | 0.044 |
| Environmental (4) | 0.594 | 0.675 | 0.661 | 0.320 | 0.085 |
| **All 27** | **0.655** | **0.651** | **0.666** | **0.326** | **0.037** |

**Category count mismatch**: Paper says Health(21), Socio(3), Environ(3) in combine but Socio=2, Environ=4 in data. Actual from CONUS_27: Health=21, Socio=2, Environ=4. Paper category assignments need reconciliation.

### 2b. Extrapolation columns (state holdout) — [NEEDS RUN: S018B]

No extrapolation R^2 data exists. S018B experiment ledger is all XXX.

**Options**:
- (a) Run S018B on SageMaker (~2hr on ml.m5.4xlarge) — fills all extrapolation columns
- (b) Drop extrapolation columns from Table 2, move to "future work" — simplifies table, avoids empty data
- (c) Keep structure, mark extrapolation as "---" with footnote — honest but looks incomplete

---

## 3. APPENDIX F — Experiment Ledger (54 XXX placeholders)

### S018A: Smoke Test — [READY to fill partially]

| Placeholder | Value | Source |
|---|---|---|
| row count | 31,789 | `zcta_features_labels.parquet` |
| Diabetes PCA32 test R^2 | 0.746 | CONUS_27 (interpolation) |
| Train R^2, RMSE, MAE | [NEEDS RUN] | Need train-fold metrics from S018A job |
| Test RMSE, MAE | [NEEDS RUN] | Need per-metric outputs |
| Wall-clock | [NEEDS RUN] | Need job metadata |
| Verdict | PASS (0.746 > 0.50) | Derivable from CONUS_27 |

### S018H: Canonical Certification — [READY to fill from evidence/predictions]

| Placeholder | Value | Source |
|---|---|---|
| Row count per family | 6,802 (test fold) | `certificate_rsn.parquet` (n_test column) |
| Certificate count | 81 (3 families x 27 tasks) | 324 total in parquet / 12 solvers; 81 for 3 benchmark families |
| Mean alpha (PCA/Lag/GNN) | 0.511 / 0.509 / 0.511 | `solver_metrics.parquet` |
| Mean kappa (PCA/Lag/GNN) | 0.223 / 0.222 / 0.222 | `solver_metrics.parquet` |
| Mean sigma (PCA/Lag/GNN) | 0.144 / 0.136 / 0.137 | `solver_metrics.parquet` |
| Gate pass rate | 100% / 100% / 100% | All EXECUTE (gate degeneracy confirmed) |
| Dominant decision | EXECUTE / EXECUTE / EXECUTE | All identical |
| H1 simplex integrity | [NEEDS COMPUTE] | Check max(abs(R+S+N - 1)) across certificate_rsn.parquet |
| H2 alpha range | [NEEDS COMPUTE] | max-min alpha across 3 families per task |
| H3 kappa range | [NEEDS COMPUTE] | max-min kappa across 3 families per task |
| H4 sigma stability | [NEEDS COMPUTE] | Count tasks with sigma < 0.30 |
| H5 cross-seed consensus | [NEEDS RUN] | Requires multi-seed outputs (seeds 42,123,456) |
| H6 gate discrimination | FAIL | Only 1 gate decision (all EXECUTE) — expected |
| H7 kappa-residual corr | [NEEDS COMPUTE] | Need per-sample kappa + residual vectors |
| Wall-clock | [NEEDS RUN] | Need SageMaker job metadata |
| Verdict | [NEEDS COMPUTE] | Depends on H1-H7; H6 FAIL is expected/known |

### S018B: State-Holdout Extrapolation — [NEEDS RUN]

All values require the S018B experiment to be executed.

| Placeholder | Can fill? | Note |
|---|---|---|
| Row count | 31,789 | Same dataset |
| Train/test split | 26,586 / 5,203 | From `geocert_splits_v24.parquet` |
| Per-family interp/extrap R^2 | [NEEDS RUN] | Full experiment required |
| Per-task extremes | [NEEDS RUN] | |
| Wall-clock | [NEEDS RUN] | |

### S018E: GNN with Spatial Features — [READY to fill partially]

| Placeholder | Value | Source |
|---|---|---|
| Graph nodes | 31,789 | CONUS ZCTAs |
| Graph edges | 184,260 | From Appendix B (queen contiguity) |
| Hidden dim | 64 | From Appendix C.4 (already documented) |
| Health (21) PCA/GNN/delta | 0.664 / 0.646 / -0.018 | CONUS_27 |
| Socio (2) PCA/GNN/delta | 0.680 / 0.653 / -0.027 | CONUS_27 |
| Environ (4) PCA/GNN/delta | 0.594 / 0.675 / +0.081 | CONUS_27 |
| All 27 PCA/GNN/delta | 0.655 / 0.651 / -0.004 | CONUS_27 |
| Elevation GNN-PCA | +0.167 | CONUS_27 |
| Tree cover GNN-PCA | +0.084 | CONUS_27 |
| Night lights GNN-PCA | +0.034 | CONUS_27 |
| Pop density GNN-PCA | +0.037 | CONUS_27 |
| Moran's I per task | [NEEDS COMPUTE] | Need spatial autocorrelation computation |
| Wall-clock | [NEEDS RUN] | Need job metadata |
| Verdict | PASS (3 environ tasks > +0.034) | Derivable |

### Appendix F duplicate Table 1 (lines 224-229) — [SAME AS Table 2]

Mirrors main Table 2; same data needed.

---

## 4. APPENDIX H — Reproducibility (1 XXX)

| Placeholder | Value | Source |
|---|---|---|
| S018H wall-clock minutes | [NEEDS RUN] | SageMaker job log |

---

## 5. FIGURES — Status

| Figure | Paper ref | File needed in V3 | Exists? | Source |
|---|---|---|---|---|
| Fig 4 (heatmap) | line 259 | `figures/fig4_conus27_heatmap.pdf` | NO (name mismatch) | `../figures/fig4_trf_heatmap.pdf` exists |
| Fig 5 (spread) | line 270 | `figures/fig5_spread_vs_trf.pdf` | NO (name mismatch) | `../figures/fig5_spread_vs_trf.pdf` exists |

**NOT referenced in combine but available:**
- `fig1_rank_inversion_scatter.pdf` — rank scatter (text retrieval domain, S6 of V2)
- `fig2_simplex_ternary.pdf` — simplex illustration
- `fig3_confusion_comparison.pdf` — confusion matrix comparison
- `fig8_synthetic_trf_validation.pdf` — TRF synthetic validation (App D)

---

## 6. STRUCTURAL ISSUES IN PAPER — [FIX]

| # | Issue | Location | Impact |
|---|---|---|---|
| F1 | Category counts wrong: paper says Socio(3)/Environ(3), data has Socio(2)/Environ(4) | Table 2 (line 244-245), S018B/S018E tables | Must reconcile — either reclassify pop_density as Socio or fix counts |
| F2 | `table2_conus27.tex` doesn't exist — `\input{table2_conus27}` in appendix_h will fail | appendix_h line 7 | Must generate this file (full 27-task table) |
| F3 | Appendix F category counts say Health(19) not 21 | ledger lines 131, 182, 224 | Must align with actual CONUS_27 data |
| F4 | `neurips_2026.sty` not verified in V3 | line 4 | Need style file for compilation |

---

## 7. BIBLIOGRAPHY — [OK]

All 11 citation keys present in `V3/references.bib`:
martin2026rsct, agarwal2025pdfm, klemmer2023satclip, geocert2026,
geman1992neural, kendall2017uncertainties, der2009aleatory,
wang2023decodingtrust, yeh2021sustainbench, efron1993bootstrap, ethayarajh2020utility

---

## 8. APPENDIX G (Failure Modes) — [OK]

Zero XXX placeholders. Complete.

---

## 9. CHECKLIST — [OK]

All checklist items have answers and justifications. No XXX.

---

## 10. PRIORITY ACTION PLAN

### Can do RIGHT NOW (no experiments needed)

1. **Fix figure paths** — copy/rename 2 PDFs into `V3/figures/` (5 min)
2. **Fill Table 2 interpolation columns** from CONUS_27 (15 min)
3. **Fill S018E table** (GNN vs PCA32 deltas) from CONUS_27 (10 min)
4. **Fill S018H certificate summary** (alpha/kappa/sigma/gate) from solver_metrics.parquet (10 min)
5. **Generate table2_conus27.tex** — full 27-task appendix table from CONUS_27 (15 min)
6. **Fill known constants**: row count (31,789), graph edges (184,260), hidden dim (64), test fold (6,802) (5 min)
7. **Fix category counts** to match actual data (Health=21, Socio=2, Environ=4) (10 min)
8. **Compute H1/H4/H6** from certificate_rsn.parquet (simplex integrity, sigma stability, gate discrimination) (15 min)

### Needs computation (local, no SageMaker)

9. **H2/H3** (alpha/kappa range across families per task) — derivable from certificate_rsn.parquet
10. **Moran's I** for spatial-advantage tasks — needs adjacency + target values

### Needs SageMaker run

11. **S018B** (state-holdout extrapolation) — fills extrapolation columns of Table 2 + full S018B ledger
12. **S018A train-fold metrics** (train R^2, RMSE, MAE) — or declare PASS from interpolation R^2
13. **H5** (cross-seed consensus) — needs 3-seed certificate outputs
14. **H7** (kappa-residual correlation) — needs per-sample kappa + residual vectors
15. **Wall-clock times** for S018A, S018B, S018E, S018H

### Decision needed (paper structure)

16. **Extrapolation columns**: Run S018B, drop columns, or mark "---"?
17. **Category assignment**: Is population_density Socio or Environ? (Currently Environ in data, paper says Socio(3))

---

## 11. XXX CENSUS

| File | Count | Fillable now | Needs run |
|---|---|---|---|
| combine_georsct.tex | 8 (5 table rows + 3 comments) | 5 (interp only) | 3 (extrap) |
| appendix_f_experiment_ledger.tex | 54 | ~25 | ~29 |
| appendix_h_reproducibility.tex | 1 | 0 | 1 |
| **Total** | **63** | **~30** | **~33** |
