# DOE-C2a: Omega Bootstrap — Distributional Reliability per Construct

**Experiment:** s035-model-ladder / DOE-C2a
**Role:** Extend DOE-C1 certificates with omega (distributional reliability)
**Status:** COMPLETED (2026-06-17)
**Depends on:** DOE-C1 (COMPLETED)
**Blocks:** DOE-C2b (temporal prior needs omega for P16 blending)

---

## Hypothesis

**H-C2a:** Bootstrap variance of forward score differs across constructs
within the same geography. Constructs with higher omega (lower variance)
produce more stable certificates; constructs with low omega are
distribution-sensitive and require P16 blended quality for safe gating.

**Null:** Bootstrap CI width is uniform across constructs (omega is
uninformative).

---

## Design Matrix

| Factor | Levels | Type |
|--------|--------|------|
| Scenario | houston, southwest_florida, nyc, riverside_coachella, new_orleans | Fixed (5) |
| Construct | JRC, Deltares, FEMA, FAST, NFIP | Fixed (5, partial coverage) |
| Bootstrap iteration | 1..50 | Resampling |
| Fold protocol | spatial_blocked (5-fold) | Fixed (same as DOE-C1) |

**Total cells:** 5 scenarios x 5 constructs x 50 bootstrap = 1,250 certifications
(minus unavailable constructs per scenario; expect ~800-900 actual runs)

---

## Method

For each (scenario, construct) cell:

1. Load the DOE-C1 data: event_features, shared layers, folds, adjacency, coords.
2. For b in 1..B (B=50):
   a. Resample fold assignments with replacement (block bootstrap: resample
      which folds appear, not individual rows). This preserves spatial
      blocking structure while varying the train/test partition.
   b. Fit HistGBDT with the resampled folds (same frozen hyperparameters).
   c. Compute full certificate: forward_score, kappa_spatial, kappa_reconstruct.
   d. Record the certificate triple (f_b, s_b, r_b).
3. Compute omega per construct:
   - omega_forward = 1 - (std(f_1..f_B) / range_clip)
   - omega_spatial = 1 - (std(s_1..s_B) / range_clip)
   - omega_composite = min(omega_forward, omega_spatial)
   - Also compute: mean, std, 95% CI for each axis.
4. Compute alpha_omega (P16 blended quality):
   - alpha = forward_score from DOE-C1 (point estimate)
   - alpha_omega = omega * alpha + (1 - omega) * prior
   - prior = 0.5 (uninformative, per yrsn convention)
5. Recompute divergence matrix using alpha_omega instead of raw forward_score.
   Compare to DOE-C1 divergence matrix: does blending change the construct
   ordering?

---

## Independent Variables

| Variable | Range | Purpose |
|----------|-------|---------|
| B (bootstrap iterations) | 50 | Balance precision vs compute cost |
| Resample unit | Fold (block) | Preserve spatial blocking structure |

## Dependent Variables

| Variable | Metric | Interpretation |
|----------|--------|---------------|
| omega_forward | 1 - normalized_std(forward_score) | Higher = more stable predictions |
| omega_spatial | 1 - normalized_std(kappa_spatial) | Higher = stable spatial residuals |
| omega_composite | min(omega_forward, omega_spatial) | Conservative reliability bound |
| alpha_omega | omega * alpha + (1-omega) * 0.5 | P16 blended quality |
| CI_width_forward | 95% CI width of forward_score | Raw instability measure |
| divergence_shift | delta(d_blended, d_raw) | Does omega change construct ordering |

## Controlled Variables

| Variable | Value | Rationale |
|----------|-------|-----------|
| HistGBDT params | Frozen (ADR-014) | Same as DOE-C1 |
| Feature set | Same as DOE-C1 | Isolate bootstrap effect |
| Adjacency matrix | Same as DOE-C1 | No topology changes |
| Seed base | 42 + b | Reproducible bootstrap sequence |
| prior (P16) | 0.5 | Uninformative prior |

---

## Acceptance Criteria

| ID | Criterion | Test |
|----|-----------|------|
| AC-C2a-1 | omega varies across constructs within a scenario | Range(omega) > 0.1 for at least 3 scenarios |
| AC-C2a-2 | NFIP has lower omega than FEMA in Houston | omega_nfip < omega_fema (insurance is noisier than zones) |
| AC-C2a-3 | Alpha_omega changes at least one construct's relative position | rank(alpha_omega) != rank(alpha) for at least one pair |
| AC-C2a-4 | Bootstrap CIs are non-degenerate | CI_width > 0 for all available constructs |

---

## S3 Output Convention

```
results/s035/doe_c2a/
  omega_bootstrap_{scenario}.json         # Full result per scenario
  omega_bootstrap_summary.json            # Cross-scenario aggregate
  cache/
    bootstrap_samples_{scenario}.parquet  # B rows x 3 cert axes per construct
    omega_table_{scenario}.parquet        # 1 row per construct: omega + CI
  figures/
    fig_omega_by_construct.pdf            # Omega comparison across constructs
    fig_divergence_blended.pdf            # Blended vs raw divergence matrix
```

---

## Resource Estimate

- Instance: ml.m5.xlarge (4 vCPU, 16 GB)
- Per scenario: 50 bootstrap x 5 constructs x ~30s each = ~125 min
- Total (5 scenarios, parallel): ~2.5 hours wall clock per job
- Cost: ~$0.25/job x 5 = ~$1.25

---

## Connection to Theory

This experiment instantiates P16 (blended quality) for flood constructs.
The DoubleTake architecture (Sayed et al. ECCV 2024) validates this pattern:
their Hint MLP fuses prior geometry with learned cost volume via a confidence
weight, which is structurally identical to alpha_omega = omega * alpha + (1-omega) * prior.

If omega is uniformly high across constructs, P16 is unnecessary for flood
certification. If omega varies (expected), then raw forward_score overstates
confidence for distribution-sensitive constructs like NFIP.

---

## Results (2026-06-17)

### Omega by Scenario and Construct

| Scenario | JRC | Deltares | FEMA | FAST | NFIP |
|----------|-----|----------|------|------|------|
| Houston | 0.874 | -- | 0.952 | 0.931 | 0.961 |
| SW Florida | 0.935 | -- | 0.960 | 0.894 | 0.954 |
| NYC | 0.869 | 1.000 | null | -- | 0.871 |
| New Orleans | 0.970 | -- | 0.975 | -- | 0.909 |
| Riverside | 0.978 | -- | 0.927 | -- | 0.938 |

### Alpha vs Alpha_Omega (P16 Blended Quality)

| Scenario | Construct | alpha | alpha_omega | shift |
|----------|-----------|-------|-------------|-------|
| Houston | FEMA | 0.855 | 0.838 | -0.017 |
| Houston | NFIP | 0.769 | 0.758 | -0.011 |
| Houston | JRC | 0.170 | 0.212 | +0.042 |
| Houston | FAST | 0.064 | 0.094 | +0.030 |
| SW Florida | FEMA | 0.949 | 0.931 | -0.018 |
| SW Florida | NFIP | 0.555 | 0.552 | -0.003 |
| SW Florida | JRC | 0.503 | 0.503 | 0.000 |
| SW Florida | FAST | 0.324 | 0.342 | +0.018 |
| NYC | NFIP | 0.622 | 0.607 | -0.015 |
| NYC | JRC | 0.312 | 0.336 | +0.024 |
| New Orleans | NFIP | 0.849 | 0.817 | -0.032 |
| Riverside | FEMA | 0.347 | 0.358 | +0.011 |

### Acceptance Criteria Assessment

| ID | Criterion | Result | Notes |
|----|-----------|--------|-------|
| AC-C2a-1 | omega range > 0.1 in >=3 scenarios | **FAIL** (1/5) | Only NYC passes (range 0.131). Others 0.05-0.09. |
| AC-C2a-2 | NFIP omega < FEMA omega in Houston | **FAIL** | NFIP=0.961 > FEMA=0.952. NFIP is MORE stable. |
| AC-C2a-3 | alpha_omega changes construct ranking | **FAIL** | No rank changes in any scenario. |
| AC-C2a-4 | Non-degenerate CIs | **PARTIAL** | NYC FEMA: null (kappa_spatial computation failed). Several zero-width CIs. |

### Interpretation

**Key finding: omega is uniformly high (0.87-1.0) across all constructs
at ZCTA resolution.** The block bootstrap with 5-fold spatial blocking
does not produce enough certification variance to meaningfully differentiate
constructs. P16 blending shifts alpha_omega by at most 0.04 (Houston JRC)
and never changes construct ordering.

**This is a useful null result:** at ZCTA resolution (~33K regions
nationally), certificates are distribution-stable. The question of
whether P16 matters for flood certification may need to be tested at:
1. Finer spatial resolution (census tracts, DOE-C3)
2. More aggressive resampling (leave-one-event-out vs fold bootstrap)
3. Smaller sample sizes (sub-scenario geography)

**Data quality signals:**
- Deltares unavailable in 4/5 scenarios (no depth data)
- FAST unavailable in 3/5 scenarios (no FAST RP mapping)
- NYC FEMA: perfect forward_score=1.0 (flood zones perfectly predictable)
  producing null kappa_spatial and null omega
- New Orleans JRC: forward_score always 0.0 (JRC unpredictable from features)

**Implication for DOE-C2b:** Since omega is uniformly high, the temporal
prior experiment (P16 hint blending across sequential events) will produce
minimal blending effect. DOE-C2b should acknowledge this and test whether
temporal information provides value BEYOND what omega captures.
