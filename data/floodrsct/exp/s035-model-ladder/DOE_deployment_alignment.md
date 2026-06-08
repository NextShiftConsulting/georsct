# DOE: Deployment-Aligned Validation — Non-Locking Sidecar

**Experiment:** s035-model-ladder / Phase 4e (post-training deployment alignment)
**Role:** Deployment-alignment diagnostics and TWCV-reweighted metrics that sit ALONGSIDE the locked ladder. This is a reporting correction — it does not alter folds, solvers, features, hyperparameters, or primary inference.
**Status:** DESIGNED (DOE phase, not launched)
**Amends:** Nothing. This is additive. It does not modify DOE_LOCKED.md, the v1.2 amendment, DOE_spatial_diagnostics.md, or the kappa cascade.
**Depends on:** R0/R1/R2 predictions parquets (Phase 1-3); fold metadata (Phase 0.3); processed feature parquets; zcta_county_crosswalk.
**Method source:** Brenning & Suesse (2026), "Aligning validation with deployment: Target-weighted cross-validation for spatial prediction." arXiv:2603.29981.
**Code:** `georsct/rsct/validation/` (deployment_alignment.py, task_descriptors.py, deployment_domains.py, deployment_target.py).

---

## Why This Exists

The locked ladder answers: "Does representation upgrade improve held-out metrics?"
(H2a fold-level Wilcoxon, H3 spatial transfer, H5 target specificity.)

It does NOT answer: "Are those held-out metrics representative of deployment performance?"

Monitoring stations (NFIP claims, 311 reports, HWM observations) are preferentially sampled toward populated, flood-prone ZCTAs. The deployment domain — all ZCTAs in a metro where guidance will be issued — includes sparse, rural, and low-risk areas underrepresented in the validation folds. Standard CV estimates performance on the sample distribution, not the deployment distribution. TWCV reweights validation losses to match deployment task-descriptor distributions, producing deployment-aligned performance estimates.

This is NOT a new hypothesis test. It is a reporting correction that answers: "How much does the CV estimate shift when we account for deployment representativeness?"

---

## Governing Constraints

| Constraint | Why |
|-----------|-----|
| No new primary or secondary hypothesis test | This is a measurement correction, not inference |
| No change to locked folds, solvers, features, hyperparameters | The ladder result must remain exactly as registered |
| Separate S3 output namespace | `results/s035/sidecar/deployment_alignment/` |
| Pre-registration of descriptor set, binning, and gate thresholds | All frozen here, before launch |
| Deployment domain defined independently of sample | Prevents circular certificates (domain != observed sample) |

---

## Section 1 — Task Descriptors

### Per-Row Validation Descriptors

For each (scenario, fold), compute per validation row:

| Descriptor | Column | Source | Type |
|-----------|--------|--------|------|
| Nearest training distance | `nearest_train_km` | Haversine from held-out ZCTA centroid to nearest in-fold training ZCTA | Spatial difficulty |
| Flood zone exposure | `flood_pct_zone_a` | Processed parquet | Environmental covariate |
| Terrain wetness | `twi_twi` | Processed parquet | Environmental covariate |
| Impervious surface | `impervious_pct` | Processed parquet | Environmental covariate |
| Population density | `population` | Processed parquet | Demand proxy |

### Binning

All descriptors are binned into **5 quantile bins** (edges fit from the deployment target distribution, NOT the validation fold). Values outside the target range receive `SENTINEL_BIN = -1` (no-reference signal, preserved as explicit category).

### Pre-Registration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_bins | 5 | Matches fold count; avoids sparse cells |
| Descriptor set | nearest_train_km, flood_pct_zone_a, twi_twi, impervious_pct, population | Covers spatial difficulty + 4 orthogonal environmental axes |
| Distance metric | Haversine (km) | ZCTA centroids are lat/lon; no projection needed for distance-only |
| Shrinkage | 0.20 | Brenning & Suesse default; convex shrink toward uniform |
| Clip | 10.0 | Maximum ratio bound per marginal iteration |

---

## Section 2 — Deployment Domain

### Definition

The deployment domain for each metro is: **all ZCTAs in the metro's county set** (from `zcta_county_crosswalk.parquet`), not just those with observed NFIP claims. This is the universe for which the model will issue predictions.

### Regime Domains (Pre-Registered)

| Regime | ID | Rule | Note |
|--------|------|------|------|
| Pluvial (general flood) | `pluvial_full` | `full_universe=True` | All ZCTAs eligible — pluvial risk exists everywhere |

The pluvial regime uses `full_universe=True` because every ZCTA in a metro can experience rainfall-driven flooding. Coastal/fluvial sub-regimes (requiring hazard layers) are deferred to future work.

### Deployment Target Construction

```
deployment_target = all ZCTAs in metro (from crosswalk)
reference = observed/labeled ZCTAs (from processed parquet)
nearest_deploy_km = haversine(deployment_zcta, nearest reference_zcta)
```

This distance is computed ONCE for the deployment universe, NOT per-fold. It measures how far each deployment ZCTA is from ANY observed location — the fundamental support gap.

---

## Section 3 — Weighting Strategies

Three strategies, following Brenning & Suesse (2026):

| Strategy | Function | Primary? | Description |
|----------|----------|----------|-------------|
| **TWCV** (raking) | `rake_weights()` | **Yes — registered primary** | Full IPF matching validation marginals to deployment marginals. Converges to joint consistency. |
| TWCV-lite | `marginal_ratio_weights()` | Screening only | Product-of-marginals approximation. Correct only when descriptors are independent. |
| IWCV | `compute_iwcv_weights()` | Comparison | Logistic density-ratio estimation. Complementary to raking. |

### Certificate Gates (Pre-Registered)

| Gate | Threshold | Decision |
|------|-----------|----------|
| ESS fraction (hard floor) | < 0.15 | FAIL — estimate dominated by few tasks |
| ESS fraction (soft floor) | < 0.35 | WARN — seriously degraded |
| Missing deployment bins | > 0 | FAIL — uncloseable coverage gap |
| Dropped fraction | > 0.50 | FAIL — folds barely overlap deployment |
| No-reference fraction | > 0.10 | FAIL — too much pure extrapolation |
| Max weight × uniform | > 10.0 | WARN — single-task dominance |
| Mean JS divergence | > 0.20 | WARN — screening threshold |

Gate tag: `deploy_align_gates_v1`

---

## Section 4 — Outputs

### Per-Scenario Table

`results/s035/sidecar/deployment_alignment/{scenario}_alignment.parquet`:

| Column | Type | Description |
|--------|------|-------------|
| zcta_id | str | ZCTA identifier |
| event_id | str | Event identifier |
| fold_id | int | Fold assignment |
| nearest_train_km | float | Distance to nearest training ZCTA (km) |
| nearest_train_km_bin | int | Quantile bin (0-4 or SENTINEL) |
| flood_pct_zone_a_bin | int | Quantile bin |
| twi_twi_bin | int | Quantile bin |
| impervious_pct_bin | int | Quantile bin |
| population_bin | int | Quantile bin |
| w_twcv | float | Raking weight (normalized, sums to 1 within fold) |
| w_twcv_lite | float | Marginal-ratio weight |
| w_iwcv | float | Density-ratio weight |

### Per-Cell Summary

`results/s035/sidecar/deployment_alignment/alignment_summary.json`:

Per (scenario, target, level):

| Field | Description |
|-------|-------------|
| metric_unweighted | Standard CV metric (from locked results) |
| metric_twcv | TWCV-reweighted metric |
| metric_iwcv | IWCV-reweighted metric |
| delta_twcv | metric_twcv - metric_unweighted |
| delta_pct | delta_twcv / metric_unweighted × 100 |
| ess | Effective sample size |
| ess_fraction | ESS / n |
| missing_bins | Count of deployment bins with zero validation coverage |
| dropped_fraction | Fraction of validation mass outside deployment support |
| no_reference_fraction | Deployment ZCTAs with no nearby reference |
| mean_js_divergence | Mean JS across descriptors |
| alignment_decision | PASS / WARN / FAIL |
| raking_converged | bool |
| raking_iterations | int |

### Uplift Comparison Table

`results/s035/sidecar/deployment_alignment/uplift_comparison.json`:

Per (scenario, target):

| Field | Description |
|-------|-------------|
| uplift_R0_R1_unweighted | Standard R0→R1 metric delta |
| uplift_R0_R1_twcv | TWCV-reweighted R0→R1 delta |
| direction_agrees | sign(unweighted) == sign(twcv) |
| delta_magnitude | abs(twcv - unweighted) |

The critical question: **Does TWCV reweighting change the DIRECTION of R0→R1 uplift?** If not, the locked finding is robust to deployment representativeness.

---

## Section 5 — Phasing & Sequencing

| Phase | Name | Timing | Reads | Model-bearing? | Touches lock? |
|-------|------|--------|-------|----------------|---------------|
| 4e | Deployment alignment | After ALL levels' predictions exist | Predictions parquets, fold metadata, feature parquets, crosswalk | No (reweights existing metrics) | No |

Phase 4e runs AFTER Phase 4d (LISA/Geary) and AFTER all training is complete. It reads only existing artifacts. It writes only to `results/s035/sidecar/deployment_alignment/`.

---

## Section 6 — Interpretation Criteria (descriptive — no pass/fail on uplift)

| Signal | Supportive pattern | Honest null |
|--------|-------------------|-------------|
| TWCV uplift direction agrees with unweighted | Uplift is deployment-representative | Direction flips → uplift is an artifact of sample bias |
| TWCV magnitude shift < 20% | CV estimate is approximately representative | Large shift → CV systematically over/under-estimates |
| Alignment certificate PASS | Validation folds cover the deployment domain | FAIL → deployment coverage gap, metrics unreportable |
| ESS fraction > 0.35 | Reweighting is mild (DEFF < 2.86) | Low ESS → few tasks dominate, estimate fragile |

---

## Section 7 — DO NOT Constraints

- Do NOT let TWCV weights modify training or fold assignments.
- Do NOT add TWCV-adjusted metrics to the money table or primary inference.
- Do NOT use TWCV weights as a hypothesis test (no p-values on the weight shift).
- Do NOT define the deployment domain as the observed sample (circularity).
- Do NOT report TWCV-adjusted metrics WITHOUT the alignment certificate (ESS, coverage gaps).
- Do NOT tune descriptors, binning, or gates after seeing results.
- Do NOT use deployment distance in kappa_geom (model-free independence rule).

---

## Section 8 — Paper Integration

### Where This Goes

- **Section 7b** (Spatial Validity Diagnostics): Add a paragraph after the regionalization robustness paragraph. Frame as: "We additionally assess whether the spatially-blocked CV estimates are representative of deployment performance using target-weighted cross-validation (Brenning & Suesse 2026)."
- **Appendix**: Full descriptor distributions, weight diagnostics, ESS per fold.
- **Table**: Uplift comparison (unweighted vs TWCV) as a robustness table.

### Narrative Arc

The paper's spatial diagnostics thread becomes:
1. Errors are spatially structured (Moran, Geary) → Section 7b existing
2. LISA localizes WHERE → Figure 5 existing
3. GWR reveals WHY (non-stationarity) → Figure 6 existing
4. Skater regionalization is degenerate → existing paragraph
5. **NEW: TWCV confirms the CV estimate is deployment-representative** → new paragraph

---

## Dependencies

| Dependency | Used by |
|-----------|---------|
| `rsct.validation.task_descriptors` | Descriptor computation, binning |
| `rsct.validation.deployment_alignment` | Raking, IWCV, certificate summary |
| `rsct.validation.deployment_domains` | RegimeDomain resolution |
| `rsct.validation.deployment_target` | Deployment universe construction |
| `zcta_county_crosswalk.parquet` | Deployment universe definition |
| R0/R1/R2 predictions parquets | Per-row metrics to reweight |
| Fold metadata | Fold assignments for nearest-training distance |
| Processed feature parquets | Descriptor columns |
