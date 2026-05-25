# DOE-3: Confusion Zone Characterization

**Experiment ID:** DOE-3
**Status:** PARTIAL (S016C baseline complete)
**Last Updated:** 2026-04-03

---

## Question

Is the R/S_sup confusion zone real and consistent across datasets?

---

## Hypotheses

| ID | Statement | Metric | Threshold | Status |
|----|-----------|--------|-----------|--------|
| H1 | Gap(S-R) = P(S_sup\|S_sup) - P(S_sup\|R) > 0 for all datasets | `gap_s_minus_r` | > 0 | **PASS** (S016C) |
| H2 | Cohen's d between R and S_sup score distributions ≥ 0.5 | `cohens_d` | ≥ 0.5 | PENDING |
| H3 | Error concentration inside confusion zone > 60% | `error_concentration` | > 0.60 | PENDING |

---

## AWS Infrastructure

**S3 Paths:**
```
Input:  s3://yrsn-datasets/s017/s016c_foundation/
Output: s3://yrsn-datasets/s017/doe3_confusion_zone/results/
Code:   s3://yrsn-datasets/rsct_code/doe3_confusion_zone/
```

**Instance:** `ml.m5.xlarge`

---

## Baseline Results (S016C)

| Dataset | P(S\|R) | P(S\|S) | P(S\|N) | Gap(S-R) | AUC |
|---------|---------|---------|---------|----------|-----|
| MIRACL | 0.349 | 0.362 | 0.120 | +0.012 | 0.684 |
| FEVER | 0.325 | 0.349 | 0.141 | +0.024 | 0.674 |
| HotpotQA | 0.334 | 0.348 | 0.140 | +0.015 | 0.666 |
| SciFact | 0.358 | 0.392 | 0.209 | +0.034 | 0.651 |

**All gaps positive → H1 PASS**

---

## Remaining Work

| Run | Task | Est. Duration |
|-----|------|---------------|
| 3.0 | Compute Cohen's d per dataset | ~5 min |
| 3.1 | Compute overlap coefficient | ~5 min |
| 3.2 | Measure error concentration in zone | ~10 min |
| 3.3 | Generate density overlap figures | ~5 min |

---

## Kill Rules

- If confusion zone disappears cross-dataset → corpus-specific, not general

---

## Change Control

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-04-03 | Initial design with S016C baseline |
