# A5: Prithvi-TJEPA Fusion Probe — Status: CANCELLED

**Date:** 2026-06-26
**MMAR reviewed:** Yes (claude, deepseek-r1, kimi, nemotron — unanimous CANCEL)

## What Happened

A5 was designed to test whether Prithvi-EO-2.0 satellite embeddings (1024-dim)
and TJEPA self-supervised embeddings are redundant or complementary
representations for flood damage prediction.

### Two independent blockers prevented execution:

**Blocker 1 (technical):** s037 TJEPA job exports only `y_true`/`y_pred` —
no latent embeddings serialized. The TJEPA encoder produces 128-dim internal
representations but `run_vertical_slice.py` does not save them. Unblocking
would require modifying the job script and re-running all 5 scenarios (~1hr
SageMaker compute).

**Blocker 2 (scientific):** The DOE's own gate condition is met: "If Prithvi
adds no predictive value (A1 FAIL), fusion is moot." A1 FAILed — 0/5 scenarios
show R2 improvement, 3/5 show negative delta.

### Why cancellation is correct (not just deferred)

The original question is logically dead by transitivity:

1. Prithvi captures satellite land-cover information (1024-dim)
2. TJEPA captures learned structure in R0 tabular features (128-dim)
3. A1 proved Prithvi is redundant WITH R0 tabular features
4. TJEPA is a function OF R0 tabular features
5. Therefore Prithvi is redundant with TJEPA — fusion adds nothing

All 4 MMAR reviewers confirmed this reasoning unanimously. No reviewer
proposed keeping the original hypothesis.

### Design flaw discovered (MMAR finding)

The original hypotheses had a **dimensionality bug**:
- AC-A5-1 specifies cosine similarity between Prithvi (1024-dim) and TJEPA
  embeddings — but TJEPA is 128-dim, not 1024-dim. Cosine similarity requires
  equal dimensions.
- AC-A5-2 specifies PCA on "concatenated 2048-dim space" — but actual
  concatenation would be 1024+128 = 1152-dim, not 2048-dim.

Even if both blockers were lifted, the hypotheses could not execute as designed
without dimension alignment (projection or padding).

## What Could Replace A5 (Research Directions)

One reviewer (deepseek-r1) proposed a redesigned question, though all reviewers
rated it as minor priority:

**Redesigned question:** Does TJEPA self-supervised pretraining improve
predictions over raw R0 tabular features within each scenario?

This tests whether the learned 128-dim TJEPA representation captures non-linear
feature interactions that HistGBDT on raw features misses. The vertical slice
job (s037) already ran this comparison — the results exist in the s037 JSONs
but have not been analyzed in the cross-analysis context.

**ADR connection:** ADR-054 (memristor measurement probe) and ADR-058
(per-domain oobleck coupling sign) establish that RSCT measurements should be
domain-calibrated, not universal. The cross-analysis battery confirmed this
empirically: flood prediction is regime-specific. A redesigned A5 could test
whether per-scenario TJEPA pretraining captures regime-specific structure
that a universal feature set cannot.

**Recommended action:** Analyze existing s037 results (TJEPA vs R0 vs MAE
per scenario) rather than launching new compute. The data already exists.

## Paper Framing

A5 should be mentioned in the paper as:

> "A fifth analysis (Prithvi-TJEPA fusion) was pre-registered but cancelled
> after A1 demonstrated that satellite embeddings are redundant with the
> tabular features from which TJEPA representations are derived. The fusion
> question is moot by transitivity."

This is honest, clean, and shows the DOE's gate conditions working as designed.

## MMAR Process Notes

- 4 reviewers, unanimous CANCEL recommendation
- No substantive disagreements in cross-critique
- Key novel finding: dimensionality bug in original hypothesis design
- Highest-value next action identified: NFIP ablation (from cross-analysis
  agenda item 2), not A5 redesign
