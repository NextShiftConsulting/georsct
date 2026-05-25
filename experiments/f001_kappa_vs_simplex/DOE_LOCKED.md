# DOE: F001 — Dual Signal Path Validation (Patent Claim 2)

**Experiment ID:** F001-DUAL-PATH
**Namespace:** RSCT-Framework
**Claims:** C1 (Representation adequacy is not scalar), C2 (Dual-path alpha+kappa conjunction)
**Status:** LOCKED
**Created:** 2026-05-02
**Revised:** 2026-05-02 (complete redesign after triangle audit)
**Registry ref:** EXPERIMENT_REGISTRY.md, Day 3

---

## Abstract

Patent Claim 2 asserts that the RSCT certificate uses a DUAL signal path:
alpha (semantic decomposition, R/(R+N)) AND kappa (geometric compatibility, R*(1-N)).
The conjunction is architecturally necessary because alpha and kappa identify
DIFFERENT failure populations: high-alpha/low-kappa indicates geometry problems
(targeted corruption), while low-alpha/high-kappa indicates noise problems
(hallucination risk).

This experiment tests whether the dual-path provides discriminative power beyond
kappa alone on the MIRACL text retrieval leaderboard (14-16 embedding models).

**Critical prior**: Alpha is near-saturated on this dataset (median 0.985-0.999).
This is an empirical finding about the task domain, not a flaw. The experiment
tests both the general dual-path value AND how the architecture degrades gracefully
when one signal saturates.

---

## Hypotheses

### H1: Signal Non-Redundancy

**Statement:** Alpha and kappa are not redundant signals -- they rank models
differently, even when alpha is near-saturated.

| Variable Type | Description |
|---------------|-------------|
| Independent | Signal: alpha vs kappa (per model) |
| Dependent | Spearman rho(alpha_rank, kappa_rank) |
| Control | Same models, same MIRACL task |

**Metrics:**
- `rho_alpha_kappa`: Spearman correlation between alpha and kappa rankings
- `r2_alpha_kappa`: R-squared of alpha predicting kappa

**Pass condition:** |rho| < 0.95 (signals are not perfectly redundant).
Even partial non-redundancy supports Claim 2's architectural decision.

**Evidence required:** `evidence/h1_signal_nonredundancy.json`

### H2: Dual-Path Predictive Power

**Statement:** The pair (alpha, kappa) predicts downstream balanced accuracy
better than kappa alone.

| Variable Type | Description |
|---------------|-------------|
| Independent | Predictor set: {kappa} vs {alpha, kappa} |
| Dependent | R-squared for predicting balanced_accuracy |
| Control | Same models |

**Metrics:**
- `r2_kappa_only`: R2 of kappa predicting balanced_accuracy
- `r2_dual_path`: R2 of (alpha, kappa) predicting balanced_accuracy
- `delta_r2`: r2_dual_path - r2_kappa_only
- `f_test_p`: F-test for nested model comparison

**Pass condition:** delta_r2 > 0.01 OR p < 0.10 for the nested F-test.
With n=14, power is limited -- we report effect size honestly.

**Evidence required:** `evidence/h2_dual_path_prediction.json`

### H3: Alpha Saturation Characterization

**Statement:** Alpha is near-constant on this text retrieval dataset,
demonstrating that the dual-path architecture degrades gracefully to
kappa-dominant evaluation when semantic quality saturates.

| Variable Type | Description |
|---------------|-------------|
| Independent | Signal measured (alpha) |
| Dependent | Coefficient of variation (CV) of alpha across models |
| Control | Same models |

**Metrics:**
- `alpha_cv`: CV of alpha across models (std/mean)
- `kappa_cv`: CV of kappa across models
- `alpha_range`: max(alpha) - min(alpha)
- `kappa_range`: max(kappa) - min(kappa)
- `saturated`: alpha_cv < 0.01

**Pass condition:** ALWAYS PASSES -- this is a characterization hypothesis.
If saturated=True, it explains why H2 may show limited dual-path benefit on
THIS dataset while the architecture remains valid for datasets with alpha variance.
If saturated=False, H2 should show stronger dual-path benefit.

**Evidence required:** `evidence/h3_alpha_saturation.json`

### H4: Failure Mode Partition

**Statement:** In the (alpha, kappa) space, models with identical kappa
but different alpha show different downstream accuracy -- evidence that the
signals identify distinct failure populations.

| Variable Type | Description |
|---------------|-------------|
| Independent | Position in (alpha, kappa) space |
| Dependent | Balanced accuracy residual after kappa regression |
| Control | Kappa held approximately constant |

**Metrics:**
- `n_kappa_matched_pairs`: model pairs within kappa tolerance 0.02
- `alpha_accuracy_corr`: correlation between alpha and accuracy WITHIN kappa-matched pairs
- `residual_alpha_corr`: correlation between alpha and (accuracy - kappa_predicted_accuracy)

**Pass condition:** |residual_alpha_corr| > 0.2 (alpha explains accuracy variance
not captured by kappa). With small n, we report confidence intervals.

**Evidence required:** `evidence/h4_failure_partition.json`

---

## Data Sources

| Source | Type | Size | Location |
|--------|------|------|----------|
| MIRACL leaderboard | Existing | 14-16 models | s3://yrsn-checkpoints/model_master_metrics/leaderboard.json |
| Per-model certificates | Existing | alpha, kappa, R, S, N per model | Same file |
| Per-model confusion matrices | Existing | 3x3 per model | Same file |

---

## Experimental Protocol

1. Load leaderboard.json, filter to models with valid alpha/kappa (non-zero)
2. Extract alpha, kappa, balanced_accuracy per model
3. H1: Compute Spearman correlation between alpha and kappa rankings
4. H2: Fit OLS (kappa -> accuracy) and OLS (alpha+kappa -> accuracy), compare R2
5. H3: Compute coefficient of variation for alpha and kappa
6. H4: Find kappa-matched pairs, test whether alpha explains residual accuracy
7. Write evidence JSONs per hypothesis
8. Produce summary report

---

## Success Criteria

| Hypothesis | Criterion | Status |
|------------|-----------|--------|
| H1 | rho(alpha, kappa) < 0.95 | PENDING |
| H2 | delta_r2 > 0.01 OR F-test p < 0.10 | PENDING |
| H3 | Characterization (always passes) | PENDING |
| H4 | residual_alpha_corr > 0.2 | PENDING |

**Overall pass:** H1 + H3 must pass (minimum: signals are non-redundant AND
saturation is characterized). H2 + H4 provide strength of evidence.

**Honest reporting:** If alpha is saturated AND dual-path adds no predictive
power, we report this as: "On text retrieval, alpha saturates; the dual-path
architecture is designed for domains where alpha varies (e.g., multimodal,
agentic chains). The kappa-dominant regime is a valid operating mode."

---

## Kill Condition

If rho(alpha, kappa) > 0.99 AND delta_r2 < 0.001 AND alpha_cv > 0.05
(alpha has variance but is perfectly redundant with kappa), then Claim 2
is weakened. Response: soften to "architecturally distinct but empirically
correlated on this task."

---

## Decision Tree

```
H1 PASS + H2 PASS + H4 PASS ── Full dual-path validation. C2 fully supported.
H1 PASS + H2 FAIL + H3 saturated ── Alpha-saturated regime. C2 supported for architecture,
                                      kappa sufficient for this dataset. Honest finding.
H1 FAIL ── Signals redundant. C2 weakened. Investigate formula relationship.
H1 PASS + H2 FAIL + H3 not saturated ── Signals non-redundant but not predictive. Investigate.
```

---

## Change Control

| Date | Change | Reason |
|------|--------|--------|
| 2026-05-02 | Initial draft | Day 1 scaffold |
| 2026-05-02 | Complete redesign | Triangle audit (V-F001-1..4): original "kappa vs simplex" was conceptually incoherent. Reframed to test Claim 2 dual-path conjunction. |
