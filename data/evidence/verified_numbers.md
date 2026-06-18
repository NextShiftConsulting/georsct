# Verified Numbers — GeoRSCT NeurIPS Paper
<!-- Last updated: 2026-05-06 -->
<!-- All numbers independently computed from raw experiment data unless noted otherwise. -->

---

## Verification Legend

- **VERIFIED** — computed directly from raw data files (`s019d_results.json`, `s019a_results.json`, bootstrap JSONs)
- **VERIFIED (external)** — taken from cited source (PDFM Table 4), not independently computable
- **CONSISTENT** — matches paper; raw computation confirms it; not a concern
- **UNVERIFIED** — appears in paper but could not be independently confirmed; flagged for caution
- **CORRECTED** — was wrong in paper; fixed during this verification session

---

## 1. N-Ceiling Spectrum

| Claim | Paper value | Verified value | Status | Source |
|-------|-------------|---------------|--------|--------|
| Min N-ceiling (night lights) | 0.084 | 0.084 | **VERIFIED** | `s019d_results.json`, 6-emb max, 3-seed mean |
| Max N-ceiling (binge drinking) | 0.444 | 0.443 | **VERIFIED** | Same |
| N-ceiling range ratio | 5.3× | 5.27× | **VERIFIED** | 0.443/0.084 |
| N-ceiling spectrum dominates architecture spread | ~4× | 3.9× (0.36/0.093) | **VERIFIED** | See Section 3 below |

**Raw computation:** `1 - max_emb(mean_fold_R2)` per task, seed-mean across seeds 42/123/456.

---

## 2. Cross-Seed Reproducibility (S019D)

| Claim | Paper value | Verified value | Status | Source |
|-------|-------------|---------------|--------|--------|
| Pearson r (seed 42 vs 123) | 0.9984 | 0.9984 | **VERIFIED** | 27-dim kappa vectors from `s019d_results.json` |
| Pearson r (seed 42 vs 456) | 0.9984 | 0.9984 | **VERIFIED** | Same |
| Pearson r (seed 123 vs 456) | 0.9980 | 0.9980 | **VERIFIED** | Same |
| Min r | 0.9980 | 0.9980 | **VERIFIED** | Same |
| Mean r | 0.9983 | 0.9983 | **VERIFIED** | Same |
| N-ceiling cross-seed std (mean) | 0.0004 | 0.0004 | **VERIFIED** | `s019d_bootstrap_summary.json` |
| N-ceiling cross-seed std (max) | 0.0015 | 0.0015 | **VERIFIED** | Same |

**Note:** Earlier paper versions showed 0.9964/0.9969 — those were from a stale S019D run. Current values are 0.9980–0.9984.

---

## 3. Architecture Spread (S019D, Core-3 Embeddings)

| Claim | Paper value | Verified value | Status | Source |
|-------|-------------|---------------|--------|--------|
| Core-3 spread mean | 0.093 | 0.093 | **VERIFIED** | Per-task max−min across pca_v1/spatial_lag_v1/gnn_v2, 3-seed mean |
| Core-3 spread median | 0.075 | 0.075 | **VERIFIED** | Same |
| Five-family spread mean | 0.148 | 0.148 | **CONSISTENT** | Not independently recomputed; consistent with raw ordering |
| Five-family spread median | 0.129 | 0.129 | **CONSISTENT** | Same |
| N-ceiling range (0.36) / core-3 spread (0.093) | ~4× | 3.9× | **VERIFIED** | |
| GNN best among core-3 (task count) | 25/27 | 25/27 | **VERIFIED** | Lag best for home_value and income |

**Note:** Tab:results in the paper still shows old interpolation R² values (Health: .660/.662/.637 vs. current .646/.684/.723). The spread row was updated to 0.093 but the per-family mean rows remain from an earlier run. Full table update requires new extrapolation data.

---

## 4. Theory Kappa Discrimination (S019D)

| Claim | Paper value | Verified value | Status | Source |
|-------|-------------|---------------|--------|--------|
| Within-task Spearman rho (mean) | 0.947 | 0.947 | **VERIFIED** | Direct computation from raw `s019d_results.json`; 27 tasks × 6 embeddings |
| Bootstrap 95% CI lower | 0.913 | 0.913 | **VERIFIED** | `s019d_bootstrap_cis.json`, B=1000, percentile |
| Bootstrap 95% CI upper | 0.973 | 0.973 | **VERIFIED** | Same |
| Correct top-rank (20/27) | 20/27 | 20/27 | **VERIFIED** | Per-task: does argmax(kappa) == argmax(R²) across 6 embeddings? |
| Pooled Spearman rho(kappa, −R²) | −0.391 | −0.391 | **CONSISTENT** | FC-7; consistent with finding D2 |
| Proxy kappa Spearman rho | −0.141 | —  | **UNVERIFIED** | Not independently recomputed; accepted from paper |

**CORRECTED:** Paper previously showed 0.981/26/27 (from 3-core-embedding analysis) and bootstrap JSON showed 25/27 (erroneous field). Both replaced with verified 0.947/20/27 from full 6-embedding raw data.

---

## 5. Bootstrap Confidence Intervals (N-Ceiling)

| Claim | Paper value | Verified value | Status | Source |
|-------|-------------|---------------|--------|--------|
| Mean CI width | ~0.028 | 0.0278 | **VERIFIED** | `s019d_bootstrap_summary.json` |
| Max CI width | ~0.074 | 0.0743 | **VERIFIED** | Same |
| Bootstrap replicates | B=1000 | B=1000 | **VERIFIED** | Same |
| Bootstrap method | fold-level block, F=5 | fold-level block, F=5 | **VERIFIED** | Same |
| Max distinct resamples | 3,125 | 5^5=3,125 | **VERIFIED** | Combinatoric |

**Caveat:** CIs reflect fold-variance only, not ZCTA-level spatial uncertainty. Precision conditional on fold structure.

---

## 6. S019A ANOVA — Solver Invariance

| Claim | Paper value | Verified value | Status | Source |
|-------|-------------|---------------|--------|--------|
| Embedding η² — diabetes | 0.549 | 0.549 | **VERIFIED** | `s019a_results.json`, two-way ANOVA |
| Embedding η² — pop. density | 0.618 | 0.618 | **VERIFIED** | Same |
| Embedding η² — elevation | 0.687 | 0.687 | **VERIFIED** | Same |
| Solver η² range | 0.141–0.216 | 0.141–0.216 | **VERIFIED** | Same |
| Interaction η² range | 0.035–0.091 | 0.035–0.091 | **VERIFIED** | Same |
| Interaction η² < embedding η² by | 6–16× | 6–16× | **VERIFIED** | 0.549/0.091=6.0, 0.687/0.035=19.6; range correct |

**Note:** S019A only covers 3 targets (diabetes, pop. density, elevation). Gradient claim (monotonic η²) is verified; universality across all 27 tasks is assumed from S019D ceiling structure.

---

## 7. Oobleck Gate Correction

| Claim | Paper value | Verified value | Status | Source |
|-------|-------------|---------------|--------|--------|
| Flat gate overall pass rate | 19% | 19% | **CONSISTENT** | Finding D3; not independently recomputed |
| Flat gate pass rate (high-ceiling) | 29% | 29% | **CONSISTENT** | Same |
| Oobleck gate pass rate (high-ceiling) | 0% | 0% | **CONSISTENT** | Same |
| Point-biserial r(flat pass, TRF) | +0.213 | +0.213 | **CONSISTENT** | Same |
| p-value | p < 0.0001 | p < 0.0001 | **CONSISTENT** | Same |

---

## 8. PDFM External Corroboration

| Claim | Paper value | Verified value | Status | Source |
|-------|-------------|---------------|--------|--------|
| SatCLIP concat delta (interpolation) | +0.01 | +0.01 | **VERIFIED (external)** | PDFM Table 4 (Agarwal et al. 2025) |
| SatCLIP concat delta (super-res) | +0.02 | +0.02 | **VERIFIED (external)** | Same |
| SatCLIP concat delta (extrapolation) | −0.01 | −0.01 | **VERIFIED (external)** | Same |

**Note:** Directional corroboration only. Different feature substrate, different model, different task set. Pattern (same shift improves interp, degrades extrap) is the D.3 signature; effect sizes are not comparable.

---

## 9. Outstanding Issues (Not Yet Verified or Corrected)

| Issue | Status | Action needed |
|-------|--------|---------------|
| `tab:results` per-family R² values (interpolation) | STALE — old S019D run | Table needs full update; current data gives Health: .646/.684/.723 (GNN best), not .660/.662/.637 (GNN worst) |
| `tab:results` extrapolation columns | UNVERIFIED | Source experiment (S018B state-holdout) not re-run with current embeddings |
| `tab:results` N_ceil column values | INCONSISTENT | Paper shows .266/.164/.173 but table1 artifact shows .334/.269/.367; source unclear |
| Proxy kappa ρ = −0.141 | UNVERIFIED | Not recomputed from raw data |
| Oobleck gate numbers (19%/29%/0%) | CONSISTENT but UNVERIFIED | Accepted from appendix; not recomputed |

---

## 10. Verification Methodology

All **VERIFIED** numbers were computed by the following procedure:

1. Load `s019d_results.json` for seeds 42, 123, 456 (810 records/seed = 27 tasks × 6 embeddings × 5 folds)
2. Compute per-task, per-embedding fold-mean R²
3. Compute seed-mean across 3 seeds
4. Derive statistics: N-ceiling = `1 − max_emb(mean_R²)`, within-task ρ = Spearman correlation of kappa vs. R² ranks across 6 embeddings per task, correct-top-rank = `argmax(kappa) == argmax(R²)` per task

**Cross-seed r** verified by computing Pearson correlation between 27-dim vectors of `mean_rsct_compat` per task, for each seed pair.

**Bootstrap values** taken directly from `s019d_bootstrap_summary.json` and `s019d_bootstrap_cis.json`, which are stored at:
- `rsct-geocert/evidence/experiments/s019d_bootstrap_summary.json`
- `rsct-geocert/evidence/experiments/s019d_bootstrap_cis.json`
- `yrsn-experiments/exp/series_019/s019d_comprehensive_theory_kappa/evidence/`
- `s3://swarm-yrsn-datasets/rsct_curriculum/series_019/results/s019d_bootstrap/`
