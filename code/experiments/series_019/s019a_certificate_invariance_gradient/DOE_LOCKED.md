# DOE: S019A Certificate Invariance Gradient

**Experiment ID:** S019A
**Domain:** RSCT / GeoRSCT
**Status:** LOCKED
**Last Updated:** 2026-05-05
**Paper Reference:** GeoRSCT V3, Section 3.5 (illustration) + Appendix F (S019A ledger entry)

---

## Abstract

Three CONUS-27 targets spanning the architecture-invariance spectrum are evaluated
under a 3x3 factorial (3 embeddings x 3 solvers) with canonical RSCT certification
(Path A). The experiment tests whether the certificate decomposition (R, S_sup, N,
alpha, kappa, sigma) reveals regime-dependent structure that scalar R^2 alone cannot
distinguish, and whether gate decisions diverge across regimes as predicted.

---

## Hypotheses

### IG-1: Diabetes embedding main effect non-significant
**Statement:** For diabetes (health regime), the two-way ANOVA embedding main effect
on holdout R^2 is non-significant (p > 0.05), confirming architecture invariance.

| Variable Type | Description |
|---------------|-------------|
| Independent   | Embedding (PCA32, Spatial Lag, GNN) |
| Dependent     | Holdout R^2 |
| Control       | Target (diabetes), 5-fold county-holdout CV |

**Metrics:**
- `anova_embedding_p_diabetes`: F-test p-value > 0.05
- `eta_sq_embedding_diabetes`: eta-squared < 0.01

### IG-2: Elevation embedding main effect significant
**Statement:** For elevation (environmental regime), the two-way ANOVA embedding main
effect on holdout R^2 is significant (p < 0.05).

| Variable Type | Description |
|---------------|-------------|
| Independent   | Embedding (PCA32, Spatial Lag, GNN) |
| Dependent     | Holdout R^2 |
| Control       | Target (elevation), 5-fold county-holdout CV |

**Metrics:**
- `anova_embedding_p_elevation`: F-test p-value < 0.05
- `eta_sq_embedding_elevation`: eta-squared > 0.20

### IG-3: Eta-squared gradient across regimes
**Statement:** eta-squared(embedding) increases monotonically from diabetes to
population density to elevation, forming a gradient.

| Variable Type | Description |
|---------------|-------------|
| Independent   | Target regime (health, demographic, environmental) |
| Dependent     | eta-squared(embedding) on R^2 |
| Control       | Same factorial design across all three targets |

**Metrics:**
- `eta_sq_diabetes < eta_sq_pop_density < eta_sq_elevation`

### IG-4: Gate decisions diverge across targets
**Statement:** At least two of the three targets produce different gate decision
distributions (chi-square test on gate outcomes across 9 cells per target).

**Metrics:**
- `chi_sq_gate_p`: p < 0.05
- At least 2 distinct gate decisions observed across all 27 cells

### IG-5: Simplex integrity holds
**Statement:** |R + S_sup + N - 1| < 0.01 for all 27 cells (all folds).

**Metrics:**
- `simplex_max_deviation`: < 0.01

---

## Experimental Protocol

1. Load CONUS-27 data (zcta_features_labels.parquet, 31,789 ZCTAs)
2. For each target in [diabetes, population_density, elevation]:
   a. For each embedding in [PCA32, Spatial Lag, GNN]:
      b. For each solver in [HistGBDT, Ridge, MLP]:
         c. Run 5-fold county-holdout CV
         d. Per fold: train solver, collect OOF residuals on test fold
         e. Construct tercile labels from residuals
         f. Train MLP 3-class classifier on tercile labels
         g. Softmax probs -> aggregate_scores_from_probs -> (R, S_sup, N)
         h. sigma = compute_sigma_request(N)
         i. CPGatekeeperInput -> SequentialGatekeeper.evaluate()
         j. Record: R^2, certificate (R, S_sup, N, alpha, kappa, sigma),
            gate decision, gate_evidence (ADR-024), policy_id
3. Compute per-target two-way ANOVA (embedding x solver)
4. Compute Tukey HSD pairwise comparisons (embedding pairs, per target)
5. Compute certificate contrast table (paired t-tests on kappa)
6. Compute gate decision chi-square (target x gate outcome)
7. Report assumption checks (Shapiro-Wilk, Levene)

Total: 27 cells x 5 folds = 135 fits.

---

## Data Sources

| Source | Type | Size |
|--------|------|------|
| zcta_features_labels.parquet | Benchmark data | 31,789 ZCTAs |
| pca32_v1.npz | PCA transform (stored numerics) | 33 -> 32 dims |
| spatial_lag_v1.npz | Lag transform | 63 -> 32 dims |
| gnn_v2 latents | Frozen GraphSAGE | 32 dims |

---

## Success Criteria

| Hypothesis | Criterion | Status |
|------------|-----------|--------|
| IG-1 | ANOVA embedding p > 0.05 for diabetes | PENDING |
| IG-2 | ANOVA embedding p < 0.05 for elevation | PENDING |
| IG-3 | eta-sq gradient: diabetes < pop_density < elevation | PENDING |
| IG-4 | Chi-square p < 0.05, >= 2 distinct gate decisions | PENDING |
| IG-5 | Simplex max deviation < 0.01 | PENDING |

---

## Statistical Analysis Plan

### Primary: Two-Way ANOVA (per target)
- Factors: embedding (3 levels), solver (3 levels)
- Response: holdout R^2 (5 folds per cell)
- Report: F-statistic, p-value, eta-squared for each main effect + interaction

### Assumption checks (per target)
- Normality: Shapiro-Wilk on ANOVA residuals
- Homogeneity: Levene's test on cell variances
- If violated: use aligned-rank ANOVA (nonparametric)

### Post-hoc: Tukey HSD
- All pairwise embedding comparisons (3 choose 2 = 3 pairs)
- Report: mean difference, 95% CI, adjusted p, Cohen's d
- Only interpret for targets where ANOVA embedding effect is significant

### Certificate contrasts
- Paired t-test on fold-level kappa: diabetes vs elevation (same embedding x solver)
- Bootstrap 95% CI on gate margin (kappa - kappa_req), 10,000 resamples
- Chi-square on gate decision distribution (target x gate outcome)
- Point-biserial correlation: TRF vs gate pass (EXECUTE=1, other=0)

### Calibrated-policy sensitivity
- Re-run gatekeeper with S2 (moderate) calibrated policy
- Report which cells change gate decision

---

## DO NOT Constraints

- Do NOT use compute_sigma_from_kappa_array (batch sigma). Use compute_sigma_request(N).
- Do NOT mock the gatekeeper. Use production SequentialGatekeeper.
- Do NOT run locally. SageMaker only.
- Do NOT use random row splits. County-holdout only.
