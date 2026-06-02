# DOE: Kappa Diagnostic Cascade — Progressive Pre-Registration

**Experiment:** s035-model-ladder / Phase 4 (4a, 4b, 4c)
**Role:** Diagnostic + pre-registration — predicts which scenarios benefit from each fix BEFORE the fix is applied
**Status:** DESIGNED (DOE phase, not launched)
**Depends on:** Each phase depends on the preceding training phase

---

## Hypothesis

**H4 (cascade):** Kappa diagnostics computed at level L predict which scenarios
benefit from level L+1, and the diagnostics themselves improve (flags clear)
after the fix is applied.

**Primary test (v1.7):** Fold-level Wilcoxon signed-rank on paired
R0 vs R1 fold metrics. See DOE_R1_spatial.md for full specification.

**Cell-level predictor (v1.8, exploratory):** kappa_geom (pre-training
geometry, Phase 0.5) replaces diag_leakage as the cell-level predictor.
diag_leakage has a room-to-improve confound (shares R0_spatial with
uplift). kappa_geom is model-independent.

**Secondary tests (exploratory):**
- diag_transfer at R1 predicts R1→R2 uplift
- diag_residual_spatial at R0 predicts R0→R1 uplift
- diag_solver movement across levels

---

## Design: Sequential Pre-Registration Protocol

Each kappa computation is uploaded to S3 BEFORE the next training phase starts.
S3 timestamps prove temporal ordering — predictions precede observations.

```
Phase 1 (R0 train) ──> Phase 4a (kappa R0) ──────> Phase 2 (R1 train)
                        [uploaded to S3]              [S3 timestamp > 4a]
                        [predicts R1 uplift]

Phase 2 (R1 train) ──> Phase 4b (kappa R1) ──────> Phase 3 (R2 train)
                        [uploaded to S3]              [S3 timestamp > 4b]
                        [predicts R2 uplift]

Phase 3 (R2 train) ──> Phase 4c (kappa R2) ──────> Phase 5 (analysis)
                        [uploaded to S3]
                        [final diagnostics]
```

---

## Four Kappa Proxies

All computed per (scenario, target) cell. All use "higher = better" metric
(R2 for regression, ROC-AUC for classification). HistGBDT as reference solver
(except diag_solver which uses both).

### diag_leakage (maps to GeoRSCT A.1 — autocorrelation)

```
diag_leakage = 1 - (metric_random - metric_spatial) / max(|metric_random|, 0.01)
```

| Value | Interpretation | Prediction |
|-------|---------------|------------|
| LOW (< median) | Random split inflated by spatial autocorrelation | R1 spatial features SHOULD help |
| HIGH (>= median) | Performance robust to spatial blocking | R1 unlikely to help much |

### diag_transfer (maps to GeoRSCT D.3 — cross-event)

```
diag_transfer = max(0, metric_leave_event / max(|metric_spatial|, 0.01))
```

| Value | Interpretation | Prediction |
|-------|---------------|------------|
| LOW (< median) | Can't generalize across events | R2 temporal features SHOULD help |
| HIGH (>= median) | Static features transfer across events | R2 unlikely to help much |

### diag_solver (model uncertainty diagnostic)

```
diag_solver = 1 - |metric_hgbdt - metric_ridge| / max(|both|, 0.01)
```

| Value | Interpretation | Prediction |
|-------|---------------|------------|
| LOW | Solvers disagree — complex structure | More features or different solver may help |
| HIGH | Linear signal dominates | Representation may be sufficient |

### diag_residual_spatial (maps to GeoRSCT A.2/B.1 — via Moran's I)

```
diag_residual_spatial = 1 - kappa_spatial(yrsn)
```

Uses `yrsn.core.kappa.spatial.compute.compute_kappa_spatial` (NOT reinvented).
Moran's I on HistGBDT residuals aggregated per ZCTA.

| Value | Interpretation | Prediction |
|-------|---------------|------------|
| LOW | Prediction errors cluster geographically | R1 W-matrix features SHOULD reduce clustering |
| HIGH | Errors spatially random | Spatial structure adequately captured |

---

## Pre-Registration Specification

### Flag Threshold: Median Split (pre-committed, no tuning)

For each kappa proxy at each level:
- Cells with kappa BELOW median = "flagged" (predicted to benefit)
- Cells with kappa ABOVE median = "unflagged"
- Median guarantees 3-4 in each group (n=7 cells)
- NO degrees of freedom for researcher to exploit

### Pre-Registration JSON Schema

```json
{
  "level": "r0",
  "timestamp": "ISO-8601",
  "predictions": {
    "r1_should_help_most": ["houston", "nyc"],
    "r1_should_help_least": ["southwest_florida", "riverside_coachella"],
    "ordering_criterion": "diag_leakage ascending",
    "flag_threshold": "median_split",
    "median_value": 0.XXX
  },
  "cells": [
    {
      "scenario": "houston",
      "target": "obs_nfip_event_claims",
      "diag_leakage": 0.XXX,
      "diag_transfer": 0.XXX,
      "diag_solver": 0.XXX,
      "diag_residual_spatial": 0.XXX
    }
  ]
}
```

---

## Expected Cascade Patterns

| Diagnostic | R0→R1 expected | R1→R2 expected | If NOT observed |
|-----------|----------------|----------------|-----------------|
| diag_leakage | INCREASE | STABLE | R1 spatial features didn't fix autocorrelation |
| diag_transfer | STABLE | INCREASE | R2 temporal features didn't fix transfer |
| diag_residual_spatial | INCREASE | STABLE or INCREASE | W-matrix didn't decorrelate errors |
| diag_solver | INCREASE or STABLE | INCREASE or STABLE | Feature additions don't resolve model disagreement |

---

## Anti-Cherry-Picking Protocol

### Multiple Comparison Correction

**Primary test (H2a):** Single fold-level Wilcoxon signed-rank (R0 vs R1).
No multiple comparison correction needed — one pre-registered test.

**Exploratory cell-level tests (H2b, H4):**
8 associations: 4 diagnostic proxies x 2 transitions (R0→R1, R1→R2).
All at n=7 cells — too few for Holm-Bonferroni to have power (requires
|rho| >= 0.93 to reject). Instead:

1. Report ALL 8 Spearman correlations with bootstrap 95% CIs (10,000 resamples)
2. Report effect sizes and directional consistency (fraction positive)
3. Frame as observed associations, not hypothesis tests
4. Holm-Bonferroni corrected p-values reported for transparency but
   explicitly noted as decorative at n=7

### All-Cells Reporting

Every cell reported, not just confirming ones:

```
scenario | target | flagged_by | predicted_uplift | observed_uplift | correct?
```

Hit rate with exact binomial (Clopper-Pearson) CI.

### Negative Results Protocol

If primary hypothesis fails:
- Report null result with effect size and CI
- Report whether any exploratory test showed signal
- Report cascade movement table even if prediction-uplift link is weak
- Do NOT switch primary hypothesis post-hoc

---

## Outputs

| Artifact | S3 Key | Produced By |
|----------|--------|-------------|
| R0 diagnostics | `results/s035/diagnostics_r0.json` | Phase 4a |
| R1 diagnostics | `results/s035/diagnostics_r1.json` | Phase 4b |
| R2 diagnostics | `results/s035/diagnostics_r2.json` | Phase 4c |

---

## Dependencies

| Dependency | Source |
|-----------|--------|
| yrsn.core.kappa.spatial.compute | Moran's I (NOT reinvented) |
| _coverage_common.load_adjacency | ZCTA adjacency edge list |
| R0/R1/R2 predictions parquets | Per-row y_true, y_pred for Moran's I |
| R0/R1/R2 results JSONs | Aggregate metrics for kappa formulas |

---

## Compute

| Resource | Value |
|----------|-------|
| Instance | ml.m5.large or local |
| Est. duration | ~5 min per level |
| GPU | NOT NEEDED |

---

## DO NOT Constraints

- Do NOT compute kappa diagnostics for level L+1 before level L trains
- Do NOT change the median split threshold after seeing results
- Do NOT omit cells from the all-cells reporting table
- Do NOT switch the primary hypothesis
- Do NOT modify kappa formulas between levels
