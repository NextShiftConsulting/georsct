# DOE-1: Substrate Independence

**Experiment ID:** DOE-1
**Status:** DRAFT
**Last Updated:** 2026-04-03

---

## Question

Does the geometry tree hold across embedding substrates?

---

## Hypotheses

| ID | Statement | Metric | Threshold | Status |
|----|-----------|--------|-----------|--------|
| H1 | Geometry tree achieves AUC ≥ 0.60 on MiniLM and Nemotron embeddings | `auc_roc` | ≥ 0.60 | PENDING |
| H2 | Tree outperforms cosine by ≥ 3pp on 2/3 substrates | `delta_auc` | ≥ 0.03 | PENDING |
| H3 | AUC std across substrates < 5pp | `auc_std` | < 0.05 | PENDING |

---

## AWS Infrastructure

**S3 Paths:**
```
Input:  s3://yrsn-datasets/s017/doe1_substrate/embeddings/
Output: s3://yrsn-datasets/s017/doe1_substrate/results/
Code:   s3://yrsn-datasets/rsct_code/doe1_substrate/
```

**Instance:** `ml.g4dn.xlarge` (embedding generation), `ml.m5.xlarge` (tree eval)

**Image:** `pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker`

---

## Experiment Matrix

| Run | Model | Job Type | Instance | Est. Duration |
|-----|-------|----------|----------|---------------|
| 1.0 | S016C baseline | Processing | ml.m5.xlarge | ~10 min |
| 1.1 | MiniLM-L6-v2 | Processing | ml.g4dn.xlarge | ~30 min |
| 1.2 | Nemotron 1B | Processing | ml.g4dn.xlarge | ~45 min |

---

## Kill Rules

- Cosine beats tree by >3pp on all substrates → narrow to characterization
- AUC < 0.55 on any substrate → geometry doesn't transfer

---

## Change Control

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-04-03 | Initial design |
