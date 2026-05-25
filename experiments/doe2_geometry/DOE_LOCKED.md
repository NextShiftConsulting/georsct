# DOE-2: Threshold Derivation

**Experiment ID:** DOE-2
**Status:** DRAFT
**Last Updated:** 2026-04-03

---

## Question

Can τ_edge (decision threshold) be derived without grid search?

---

## Hypotheses

| ID | Statement | Metric | Threshold | Status |
|----|-----------|--------|-----------|--------|
| H1 | Breakpoint in P(S_sup) distribution maps to stable τ_edge | `threshold_stability` | variance < 5pp | PENDING |
| H2 | Frozen threshold transfers across datasets with < 3pp accuracy drop | `transfer_gap` | < 0.03 | PENDING |

---

## AWS Infrastructure

**S3 Paths:**
```
Input:  s3://yrsn-datasets/s017/s016c_foundation/
Output: s3://yrsn-datasets/s017/doe2_geometry/results/
Code:   s3://yrsn-datasets/rsct_code/doe2_geometry/
```

**Instance:** `ml.m5.xlarge` (CPU sufficient for threshold analysis)

---

## Experiment Matrix

| Run | Dataset | Task | Est. Duration |
|-----|---------|------|---------------|
| 2.0 | MIRACL dev | Extract P(S_sup) distribution, find breakpoint | ~5 min |
| 2.1 | MIRACL test | Freeze threshold, evaluate | ~5 min |
| 2.2 | FEVER/HotpotQA/SciFact | Cross-domain transfer eval | ~10 min |
| 2.3 | Bootstrap (100x) | Threshold stability | ~15 min |

---

## Kill Rules

- Threshold variance > 5pp across bootstraps → present as local calibration only
- Transfer gap > 10pp → threshold is dataset-specific

---

## Change Control

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-04-03 | Initial design |
