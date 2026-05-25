# DOE-6: Geospatial Transfer

**Experiment ID:** DOE-6
**Status:** NOT STARTED
**Last Updated:** 2026-04-03

---

## Question

Do geometric features transfer to geospatial embeddings (non-text modality)?

---

## Hypotheses

| ID | Statement | Metric | Threshold | Status |
|----|-----------|--------|-----------|--------|
| H1 | AUC ≥ 0.55 on PDFM geospatial pairs | `auc_roc` | ≥ 0.55 | PENDING |
| H2 | Top-5 features overlap ≥ 40% with text domain | `feature_overlap` | ≥ 0.4 | PENDING |
| H3 | directional_asymmetry remains predictive | `feature_rank` | ≤ 10 | PENDING |

---

## AWS Infrastructure

**S3 Paths:**
```
Input:  s3://yrsn-datasets/s017/doe6_geospatial/pdfm_conus27.csv
Output: s3://yrsn-datasets/s017/doe6_geospatial/results/
Code:   s3://yrsn-datasets/rsct_code/doe6_geospatial/
```

**Instance:** `ml.m5.xlarge`

---

## Data

**PDFM CONUS-27:** 35,000 US zip codes with 27 ground truth variables

**Label Construction (quintile-based):**
- Same quintile → R (Relevant)
- Adjacent quintile → S_sup (Superfluous)
- Distant quintile (≥2 apart) → N (Noise)

---

## Experiment Matrix

| Run | Task | Dependency | Est. Duration |
|-----|------|------------|---------------|
| 6.0 | Load PDFM conus27, construct pairs | Data on S3 | ~5 min |
| 6.1 | Compute geometric features (native PDFM dim) | Run 6.0 | ~10 min |
| 6.2 | Train tree on PDFM pairs | Run 6.1 | ~10 min |
| 6.3 | Evaluate AUC, feature importance | Run 6.2 | ~5 min |
| 6.4 | Compare features with text domain | DOE-5 complete | ~5 min |

---

## PDFM Embedding Access

**Status:** Benchmarks available (conus27.csv uploaded to S3)

**Full embeddings:** Require approval from Google Research

**Fallback:** If embeddings not approved, use conus27 ground truth variables directly as "embedding" proxy

---

## Kill Rules

- AUC < 0.52 → Drop from paper, save for domain-specific follow-up
- Feature overlap < 20% → Geometry is modality-specific

---

## Change Control

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-04-03 | Initial design |
