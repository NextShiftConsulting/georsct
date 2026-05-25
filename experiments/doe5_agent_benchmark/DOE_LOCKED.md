# DOE-5: Feature Analysis

**Experiment ID:** DOE-5
**Status:** PARTIAL (S016C feature importance done)
**Last Updated:** 2026-04-03

---

## Question

Which geometric features matter and why?

---

## Hypotheses

| ID | Statement | Metric | Threshold | Status |
|----|-----------|--------|-----------|--------|
| H1 | directional_asymmetry in top-5 features across all seeds | `feature_rank` | ≤ 5 | **PASS** (rank 3) |
| H2 | Top-5 features stable across datasets (Jaccard ≥ 0.6) | `jaccard_similarity` | ≥ 0.6 | PENDING |
| H3 | SHAP values for directional features have consistent sign | `shap_sign_consistency` | > 0.9 | PENDING |

---

## AWS Infrastructure

**S3 Paths:**
```
Input:  s3://yrsn-datasets/s017/s016c_foundation/
Output: s3://yrsn-datasets/s017/doe5_agent_benchmark/results/
Code:   s3://yrsn-datasets/rsct_code/doe5_agent_benchmark/
```

**Instance:** `ml.m5.2xlarge` (SHAP is memory-intensive)

---

## S016C Feature Importance (Baseline)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | v_proj_on_diff | 0.054 |
| 2 | cosine_uv | 0.037 |
| 3 | directional_asymmetry | 0.035 |
| 4 | delta_norm | 0.024 |
| 5 | u_proj_on_diff | 0.023 |

---

## Experiment Matrix

| Run | Task | Est. Duration |
|-----|------|---------------|
| 5.0 | SHAP on frozen S016C tree (MIRACL) | ~30 min |
| 5.1 | Permutation importance on FEVER | ~15 min |
| 5.2 | Permutation importance on HotpotQA | ~15 min |
| 5.3 | Permutation importance on SciFact | ~15 min |
| 5.4 | Cross-dataset top-k stability analysis | ~5 min |

---

## Kill Rules

- If top features differ completely across datasets → geometry is substrate-specific
- If SHAP signs inconsistent → feature interpretation unreliable

---

## Change Control

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-04-03 | Initial design with S016C baseline |
