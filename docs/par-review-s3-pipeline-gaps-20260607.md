# PAR Review: FloodRSCT S3 Pipeline Gaps

**Date:** 2026-06-07
**Method:** Parallel Adversarial Review (2 independent reviewers, same model, aggregated)
**Target:** `s3://swarm-floodrsct-data/` artifact inventory vs paper verdict table requirements
**Reviewer agreement:** High convergence — both reviewers found the same critical/serious gaps independently

---

## Critical (blocks paper submission)

### C1. No Ranking artifacts exist

- **Evidence:** Zero matches for `ranking`, `tau_b`, `fidelity`, `kendall` across entire bucket
- **Paper impact:** Verdict table Ranking row is blank. Kendall's tau-b and fidelity-delta cannot be computed.
- **Resolution:** Post-processing script over existing R0/R1/R2 predictions. No new training required.

### C2. No Transfer/LEO artifacts exist

- **Evidence:** Zero matches for `leo`, `transfer`, `leave_event_out` across entire bucket
- **Paper impact:** Verdict table Transfer row is blank. Transfer retention ratio cannot be computed.
- **Resolution:** Requires leave-event-out CV runs — new SageMaker training jobs.

### C3. No Allocation artifacts exist

- **Evidence:** Zero matches for `allocation`, `alloc` across entire bucket
- **Paper impact:** Verdict table Allocation row is blank.
- **Resolution:** Derivable from existing money_table + vintage flags. Script, not SageMaker.

**Summary:** 3 of 6 decision geometries have zero S3 evidence.

---

## Serious (weakens paper, verdict table gaps)

### S1. H2a evidence lacks fold-level deltas and is INCONCLUSIVE (p=0.114)

- **File:** `results/s035/evidence/h2_evidence.json` (355 bytes, 2026-06-06 07:08:43)
- **Contents:** `n_paired_folds=20`, `mean_delta=0.061`, `wilcoxon_stat=125.0`, `wilcoxon_p_one_sided=0.114`, `verdict=INCONCLUSIVE`
- **Missing:** Per-fold delta array. A reviewer cannot reproduce the Wilcoxon signed-rank test without the 20 individual delta values.
- **Substantive issue:** p=0.114 means R0->R1 increment is not significant at any conventional alpha. H3 (R0->R2) does PASS (p=0.002), so the cumulative story holds, but the R1 increment alone is not statistically supported.
- **Resolution:** Paper must acknowledge R1 increment is inconclusive; lean on R0->R2 cumulative. Re-emit h2_evidence.json with per-fold delta array for reproducibility.

### S2. All 15 LISA files are stale vs predictions

- **LISA timestamps:** All dated 2026-06-03 23:37–23:38
- **Prediction timestamps:** Re-run 2026-06-05 to 2026-06-06
  - `r0_houston.json`: 2026-06-05 17:30:13
  - `r2_nyc.json`: 2026-06-06 06:00:07
  - `r1_hydrology_houston.json`: 2026-06-05 17:37:31
- **Impact:** Every Moran's I value, cluster map, and kappa_spatial in the paper may not reflect current model outputs.
- **Resolution:** Re-run LISA computation over Jun 5-6 predictions. Script, not SageMaker.

### S3. No `clustering_verdict.json` exists

- **Evidence:** Zero matches in bucket
- **Impact:** No formal PASS/FAIL for Clustering decision geometry row.
- **Resolution:** Run Moran's I permutation test (999 permutations) over LISA data, emit verdict JSON.

### S4. LISA JSONs lack global Moran's I p-values

- **Fields present:** `yrsn_morans_i`, `yrsn_kappa_spatial`, `n_significant`, `yrsn_expected_i`
- **Missing:** `morans_i_p_value` or equivalent global significance test
- **Example:** `r2_houston_lisa.json` has `yrsn_morans_i: 0.544` but no p-value
- **Note:** `n_significant` counts local LISA significance at ZCTA level, not the global Moran's I test against a permutation null.
- **Resolution:** Add global Moran's I p-value (999-permutation null) to LISA computation.

### S5. R1 ablation is Houston-only

- **Present:** `r1_no_target_lag_houston.json`, `r1_no_wlag_houston.json`, `r1_wlag_only_houston.json` (all 2026-06-03 22:34)
- **Missing:** No ablation runs for new_orleans, nyc, riverside_coachella, southwest_florida
- **Impact:** Relational ablation verdict cannot generalize beyond Houston.
- **Resolution:** Requires 4 variants x 4 scenarios = 16 SageMaker training jobs. Paper already flags Houston-only as partial result.

### S6. Figures are stale

- **`fig5_lisa_triptych_houston.pdf`:** 2026-06-04 00:29:27
- **`fig6_gwr_local_r2_houston.pdf`:** 2026-06-04 00:40:02
- **Latest predictions:** 2026-06-05/06
- **Resolution:** Re-render from current data. Script, not SageMaker.

---

## Minor

- **GWR skips New Orleans** — `INSUFFICIENT_N` (n_zcta=44). Legitimate but should be documented in paper's Relational geometry discussion.
- **MGWR convergence failure for Houston** — falls back to GWR (`'Sel_BW' object has no attribute 'bw_'`). GWR is primary method, but failure should be noted.
- **GWR sidecar missing New Orleans** — consistent with INSUFFICIENT_N skip.

---

## Verdict Table Readiness

| Decision Geometry | S3 Evidence | Verdict Issuable? | Resolution Type |
|---|---|---|---|
| **Prediction** | R0/R1/R2 for all 5 scenarios (Jun 5-6) | YES (H2a inconclusive, H3 passes) | — |
| **Ranking** | **Nothing** | **NO** | Script (post-processing) |
| **Clustering** | LISA exists but stale, no p-values, no verdict | **NO** | Script (re-run LISA + permutation test) |
| **Transfer** | **Nothing** | **NO** | SageMaker (LEO CV) |
| **Relational** | GWR for 4/5 scenarios (Jun 4, stale) | PARTIAL (Houston-only ablation) | SageMaker (ablation) + script (re-render) |
| **Allocation** | **Nothing** | **NO** | Script (cross-tabulate money_table) |

---

## Action Priority

### Blocked (need SageMaker compute)

1. **Transfer/LEO** — leave-event-out CV runs (new training)
2. **R1 ablation x 4 scenarios** — 16 new training jobs

### Derivable from existing artifacts (scripts only)

3. **Ranking** — compute tau-b over existing R0/R1/R2 predictions
4. **Allocation** — cross-tabulate money_table with vintage flags
5. **Clustering verdict** — Moran's I permutation test (999 perms) over LISA data
6. **Re-run LISA** on Jun 5-6 predictions (fixes S2 staleness)
7. **Re-render figures** (fixes S6)
8. **Re-emit h2_evidence.json** with per-fold delta array (fixes S1 reproducibility)

### Needs a decision

9. **H2a p=0.114** — R0->R1 is INCONCLUSIVE. Paper can lean on H3 (R0->R2, p=0.002) but must acknowledge the R1 increment alone is not significant.
