# Cross-Fold Coherence Metric for RSCT Certificates

**Date:** 2026-06-08
**Context:** S035 model ladder, 5 flood scenarios, rsct-healthcheck Gate 2 validation
**Commit:** a1abe5f (inline), 1435376 (canonical)

## Problem Statement

RSCT Gate 2 (Consensus Gate) requires a `coherence` field in each
certificate to verify that cross-validation folds agree on the quality
assessment.  Prior to this change, the certifier (`compute_certificates.py`)
did not produce a coherence metric.  Gate 2 failed closed on all 33
cell-level evaluations (11 cells x 3 levels) with sub-signal
`COHERENCE_ABSENT_FAIL_CLOSED`, masking the actual gate outcomes for
cells with genuine signal.

Meanwhile, 13 of those 33 evaluations were already failing at Gate 1
(`N_FLOOR_BREACH_AND_ALPHA_LOW`) due to degenerate classification
targets with near-zero class variance.  The missing coherence field
made it impossible to distinguish "certificate is incomplete" from
"model has no signal."

## Design Constraints

1. **Coherence measures fold agreement, not performance quality.**
   A model can be coherent and bad (all folds agree the model is poor)
   or noisy but lucky on one fold.

2. **Degenerate targets must not receive a coherence value.**
   When alpha = 0 (no usable signal), computing coherence would produce
   a misleading number.  These cells should remain at Gate 1 REJECT.

3. **Every coherence value must have an audit trail.**
   The certificate must explain *why* coherence was or was not computed.

## Formula

```
coherence = valid_fold_rate * (0.6 * directional_agreement + 0.4 * magnitude_stability)
```

Where:

| Component | Definition | Rationale |
|-----------|-----------|-----------|
| `valid_fold_rate` | n_valid_folds / n_expected_folds | Penalizes missing folds |
| `directional_agreement` | Share of folds agreeing with median direction (metric > 0 vs <= 0) | Captures whether folds agree on the *direction* of signal |
| `magnitude_stability` | 1 - IQR / range, clipped to [0, 1] | Captures whether folds agree on *how much* signal, robust to single outlier fold via IQR |

For regression, the primary metric is R2 (spatial-blocked, HistGBDT).
For classification, AUC is mapped to 2*(AUC - 0.5), so the
direction threshold is 0 (above/below chance).

### Degenerate target handling

```python
if alpha < 0.01:
    coherence = None
    coherence_status = "NOT_COMPUTED_TARGET_DEGENERATE"
```

This preserves Gate 1 REJECT for targets where the model cannot
possibly learn (e.g., obs_has_hwm in NOLA where >99% of ZCTAs are
in the same class).

## Certificate Fields Added

| Field | Type | Description |
|-------|------|-------------|
| `coherence` | float or None | Composite coherence score, [0, 1] |
| `coherence_status` | string | `COMPUTED`, `NOT_COMPUTED_TARGET_DEGENERATE`, or `NOT_COMPUTED_INSUFFICIENT_FOLDS` |
| `coherence_detail.n_folds_expected` | int | Number of folds the split was designed to produce |
| `coherence_detail.n_folds_valid` | int | Number of folds with finite metric values |
| `coherence_detail.fold_directional_agreement` | float or None | Fraction of folds agreeing with median direction |
| `coherence_detail.fold_magnitude_stability` | float or None | 1 - IQR/range |
| `coherence_detail.failure_reason` | string or None | Why coherence was not computed (if applicable) |

## Observed Values (S035, R0 level)

| Scenario | Target | Coherence | Dir. Agreement | Mag. Stability | Status |
|----------|--------|-----------|----------------|----------------|--------|
| Houston | obs_has_311 | 0.852 | 1.00 | 0.63 | COMPUTED |
| Houston | obs_nfip_event_claims | 0.885 | 1.00 | 0.71 | COMPUTED |
| SWFL | obs_has_hwm | 0.808 | 1.00 | 0.52 | COMPUTED |
| SWFL | obs_nfip_event_claims | 0.810 | 0.80 | 0.83 | COMPUTED |
| NYC | obs_has_311 | 0.806 | 1.00 | 0.51 | COMPUTED |
| NYC | obs_has_hwm | 0.688 | 1.00 | 0.22 | COMPUTED |
| NYC | obs_nfip_event_claims | None | -- | -- | DEGENERATE (alpha=0) |
| Riverside | obs_has_hwm | None | -- | -- | DEGENERATE (alpha=0) |
| Riverside | obs_nfip_event_claims | 0.747 | 0.80 | 0.67 | COMPUTED |
| NOLA | obs_has_hwm | None | -- | -- | DEGENERATE (alpha=0) |
| NOLA | obs_nfip_event_claims | 0.865 | 1.00 | 0.66 | COMPUTED |

## Healthcheck Impact

| Metric | Before | After |
|--------|--------|-------|
| Healthy cells | 0 | 2 |
| Warning cells | 0 | 4 |
| Failing cells | 0 | 0 |
| Critical cells | 11 | 5 |
| COHERENCE_ABSENT_FAIL_CLOSED | 20 | 1 |
| N_FLOOR_BREACH (real failures) | 13 | 13 (unchanged) |
| Routing conflicts | 33 | 14 |

The 5 remaining critical cells:
- 3 degenerate targets (obs_has_hwm with <1% positive class in NOLA, Riverside, SWFL-R0)
- 2 cells with alpha collapse at R1/R2 (NOLA and Riverside obs_nfip_event_claims)
- 1 correctly degenerate (NYC obs_nfip_event_claims R0, alpha=0)

All of these are real failures that should not be rescued.

## Relationship to Gearbox Warmup

The gearbox warmup (Phase 0.75) already computes a coherence metric
using a simpler formula: `1 - IQR / |median|`.  The certificate
coherence is intentionally different:

- Gearbox coherence uses Ridge regression on assembled features
  (pre-training warmup).
- Certificate coherence uses the primary solver (HistGBDT) on
  spatial-blocked folds (post-training evaluation).
- Certificate coherence adds directional agreement and degenerate
  target handling.

The two metrics serve different purposes: gearbox coherence informs
gear assignment (exploration vs exploitation), while certificate
coherence gates whether the result is trustworthy enough to act on.

## Paper Relevance

This metric is relevant to Section 4 (Experimental Setup) and
Section 5 (Results) of the NeurIPS/SIGSpatial submission:

- **Section 4:** Coherence as a quality gate in the RSCT certification
  pipeline.  The formula should be stated in the methods section
  alongside the other gate definitions.

- **Section 5:** The before/after healthcheck table demonstrates that
  the gating framework correctly separates "incomplete certificate"
  from "genuinely failed model."  The degenerate target handling
  shows the framework's ability to distinguish data quality issues
  from model quality issues.

- **Appendix:** Full coherence values per scenario x target x level
  provide transparency into the certification process.

## Implementation Notes

Two implementations exist:
1. **Canonical:** `rsct/experiment_cert.py` -- `compute_coherence()`
   returns `CoherenceResult` dataclass with full audit trail.
2. **Inline fallback:** `wsp/floodrsct/jobs/compute_certificates.py` --
   `_compute_coherence_inline()` computes the same metric as a dict,
   used when the rsct wheel on SageMaker predates the canonical
   implementation.

The inline version will be removed once the rsct wheel is rebuilt.
