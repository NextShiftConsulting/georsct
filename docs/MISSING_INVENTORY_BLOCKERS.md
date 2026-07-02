# SUBMISSION BLOCKERS — GeoRSCT NeurIPS 2026

**Paper**: `V3/combine_georsct.tex` + 3 appendix `\input` files  
**Deadline**: Abstract May 4, Paper May 6  
**Last audit**: 2026-05-04

This file lists everything that must be resolved before the paper compiles and is internally consistent. See `MISSING_INVENTORY.md` for the full XXX census and priority action plan.

---

## A. WON'T COMPILE (latex errors)

| # | Issue | File:Line | Fix | Effort |
|---|-------|-----------|-----|--------|
| A1 | `V3/figures/` directory missing | combine:259,270 | `mkdir V3/figures/` | 1 min |
| A2 | `figures/fig4_conus27_heatmap.pdf` not found | combine:259 | Copy `../figures/fig4_trf_heatmap.pdf` -> `V3/figures/fig4_conus27_heatmap.pdf` | 1 min |
| A3 | `figures/fig5_spread_vs_trf.pdf` not found | combine:270 | Copy `../figures/fig5_spread_vs_trf.pdf` -> `V3/figures/fig5_spread_vs_trf.pdf` | 1 min |
| A4 | `table2_conus27.tex` not found (`\input{table2_conus27}`) | appendix_h:7 | Generate 27-row LaTeX table from `V2/figures/figures.py` CONUS_27 | 15 min |
| A5 | `\ref{tab:conus27}` has no `\label{tab:conus27}` — label must be inside `table2_conus27.tex` which doesn't exist | combine:255 | Include `\label{tab:conus27}` in generated table2_conus27.tex | 0 min (part of A4) |

---

## B. DATA IN TABLE 2 MISSING (XXX placeholders in main body)

| # | Issue | File:Line | Data source | Status |
|---|-------|-----------|-------------|--------|
| B1 | Table 2 interpolation columns: Health/Socio/Environ/All R^2 + N_ceil | combine:243-248 | `V2/figures/figures.py` CONUS_27 (line 95) | **READY** — compute category means |
| B2 | Table 2 extrapolation columns (6 cells per row) | combine:243-248 | S018B experiment (state-holdout) | **NEEDS RUN** or restructure table |
| B3 | Table 2 cross-family spread row | combine:248 | Derivable from CONUS_27 | **READY** |

**Decision required for B2**: Run S018B (~2hr SageMaker), drop extrap columns, or mark "---"?

---

## C. CATEGORY COUNT CONTRADICTIONS (internally inconsistent)

The paper uses different category breakdowns in different places. The CONUS_27 canonical data has **Health=21, Socio=2, Environ=4**.

| # | Location | Claims | Actual | Discrepancy |
|---|----------|--------|--------|-------------|
| C1 | combine:225 (S7.1 setup) | "19 health, 5 socioeconomic, 3 environmental" | 21/2/4 | 19 vs 21, 5 vs 2, 3 vs 4 |
| C2 | combine:243-245 (Table 2) | "Health (21), Socioeconomic (3), Environmental (3)" | 21/2/4 | Socio 3 vs 2, Environ 3 vs 4 |
| C3 | combine:489 (App B source table) | "ACS: 33 input features + 5 targets" | 2 or 3 ACS targets | 5 vs 2-3 |
| C4 | combine:490 (App B source table) | "19 health prevalence targets" | 21 health in CONUS_27 | 19 vs 21 |
| C5 | combine:554 (App B target table) | "Health: 19, Socioeconomic: 5, Environmental: 3" | 21/2/4 | All three wrong |
| C6 | ledger:131,182,224 | "Health (19 tasks)" and "Socioeconomic (5)" | 21 and 2 | Both wrong |

**Root cause**: The category assignments in `V2/figures/figures.py` CONUS_27 are the ground truth. The paper text was written before final data was locked. Specific misclassifications:
- `population_density` is `Environ.` in data but the paper counts it as Socioeconomic
- 2 health tasks not in the "19" count: likely `binge_drinking` and `cholesterol_screening` (or `annual_checkup` and `sleep_less_7hr`) were added after the text was written

**Fix**: Either reclassify to match paper (move pop_density to Socio, etc.) or update all text to match data. The data is canonical — update the text.

---

## D. FACTUAL CLAIMS WITHOUT EVIDENCE

| # | Claim | File:Line | Issue |
|---|-------|-----------|-------|
| D1 | "dual-graph morphing on ZCTA adjacency" | combine:225 | GNN uses standard GraphSAGE with mean aggregation (App C.4). "Dual-graph morphing" is not described anywhere. Remove or document in App C. |
| D2 | "14 spatial-lag" features | combine:225 | App C.3 says 33 lag features (one per ACS column). The "14" appears to be a curated subset but is not documented. Clarify: is it 14 or 33? |
| D3 | 63 solver-usable features = 33 ACS + 14 lag + 16 enrichment | combine:225,342 | If lag is 33 (not 14), total would be 82 raw features before PCA. The "63" needs a precise decomposition. |
| D4 | "21 health outcomes (CDC PLACES)" | combine:225 | CDC PLACES 2023 has 36 measures. The 21 selected are not enumerated in the main body. App B lists 19. |

---

## E. VERSION / BUCKET INCONSISTENCIES

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| E1 | Version mismatch: `v23.001` vs `v23.002` | appendix_h:31 says v23.001; combine:342,943,983 say v23.002 | Pick one. If v23.002 is current, update appendix_h. |
| E2 | S018B script uses old bucket `swarm-yrsn-datasets` | experiments/s018b_freeze/run_s018b_freeze.py:33 | Should be `swarm-` prefix bucket in nsc-swarm account (865679935554) |
| E3 | Solver script uses old buckets `yrsn-datasets` / `yrsn-checkpoints` | evidence/solvers/train_and_export_v2.py:51-54 | Same bucket migration issue |

---

## F. APPENDIX F EXPERIMENT LEDGER (54 XXX placeholders)

### Fillable from existing evidence (~25 values)

| Experiment | Fillable items | Source |
|---|---|---|
| S018A | row count (31,789), test R^2 for diabetes (0.746), verdict (PASS) | CONUS_27 |
| S018H | alpha/kappa/sigma per family, gate pass rate (100%), dominant decision (EXECUTE), certificate count (81), test-fold rows (6,802) | `solver_metrics.parquet`, `certificate_rsn.parquet` |
| S018E | graph nodes (31,789), edges (184,260), hidden dim (64), per-family R^2 and deltas, spatial-advantage tasks, verdict | CONUS_27, Appendix C |

### Requires SageMaker or local computation (~29 values)

| Experiment | Missing items | Dependency |
|---|---|---|
| S018A | train R^2, RMSE, MAE, wall-clock | Need to run or infer |
| S018H | H1 (simplex integrity), H2-H3 (alpha/kappa range), H4 (sigma stability), H5 (cross-seed), H7 (kappa-residual corr), wall-clock, verdict | H1/H2/H3/H4 computable from `certificate_rsn.parquet`; H5/H7 need multi-seed or per-sample data |
| S018B | ALL values (interp/extrap R^2, per-task extremes, wall-clock, verdict) | Full experiment run needed |
| S018E | Moran's I per task, wall-clock | Local computation + job metadata |

---

## G. APPENDIX H — REPRODUCIBILITY (1 XXX)

| # | Issue | Fix |
|---|-------|-----|
| G1 | S018H wall-clock = XXX minutes | Needs SageMaker job log |

---

## H. ENRICHMENT SOURCE TABLE INCOMPLETE (App B)

| # | Issue | File:Line | Fix |
|---|-------|-----------|-----|
| H1 | Source data table (lines 486-496) lists only ACS, CDC, TIGER, VIIRS, USGS, Hansen | combine:486-496 | Add CDC SVI, FEMA NFHL, HIFLD, OSRM rows to match the 16 enrichment features documented in Appendix F (methodology) |

The enrichment sources are fully described in Appendix F (methodology, ~line 830+), but the "Source Data" summary table in App B omits them. A reviewer comparing App B's source table to the "63 features" claim in S7.1 would see 33 ACS + 27 targets + geometry = no enrichment.

---

## I. FIGURES NOT REFERENCED IN PAPER

Available but not `\includegraphics`'d in combine_georsct.tex:

| Figure | Content | Potential location |
|---|---|---|
| `fig1_rank_inversion_scatter.pdf` | Accuracy rank vs alpha rank (text retrieval, 16 models) | Was in V2 S6; no section in V3 uses text-retrieval results |
| `fig2_simplex_ternary.pdf` | Probability simplex illustration | Would strengthen S3 (RSN decomposition) |
| `fig3_confusion_comparison.pdf` | Confusion matrix comparison | Would support taxonomy exemplars |
| `fig8_synthetic_trf_validation.pdf` | TRF bootstrap synthetic validation | Referenced in App D text but no `\includegraphics` |

**Decision**: Which figures to include? Page budget is finite. Fig 8 is the most useful (validates App D claims).

---

## J. SUMMARY: BLOCKER COUNT BY SEVERITY

| Severity | Count | Description |
|---|---|---|
| **Won't compile** | 5 | A1-A5: missing figures dir, 2 PDFs, table2_conus27.tex, dangling label |
| **Empty main-body table** | 3 | B1-B3: Table 2 all XXX (interpolation fillable, extrapolation needs run) |
| **Internal contradictions** | 6 | C1-C6: category counts inconsistent across 6+ locations |
| **Unsupported claims** | 4 | D1-D4: dual-graph morphing, feature count arithmetic, health task count |
| **Version/infra** | 3 | E1-E3: version string mismatch, old S3 buckets |
| **Appendix placeholders** | 55 | F+G: experiment ledger + reproducibility |
| **Missing source rows** | 1 | H1: enrichment sources missing from App B table |
| **Unused figures** | 4 | I: available but not included |
| **Total** | **81** | |

### Critical path (must fix before submission)

1. **A1-A5**: Fix compilation (30 min)
2. **B1+B3**: Fill Table 2 interpolation columns + spread (15 min)
3. **C1-C6**: Reconcile category counts everywhere (30 min)
4. **D1**: Remove "dual-graph morphing" claim (2 min)
5. **D2-D3**: Clarify feature count (14 vs 33 lag, total 63 vs 82) (15 min)
6. **E1**: Fix version string (2 min)
7. **B2 decision**: Drop extrap columns or run S018B

Items 1-6 are ~90 minutes of editing. Item 7 is a structural decision.
