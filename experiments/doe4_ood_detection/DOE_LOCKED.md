# DOE-4: Governance Workflow

**Experiment ID:** DOE-4
**Status:** DESIGN ONLY
**Last Updated:** 2026-04-03

---

## Question

Can the confusion zone become a governance workflow (auto-resolve vs escalate)?

---

## Hypotheses

| ID | Statement | Metric | Threshold | Status |
|----|-----------|--------|-----------|--------|
| H1 | ≥ 70% of samples can be auto-resolved (outside confusion zone) | `pct_auto_resolved` | ≥ 0.70 | PENDING |
| H2 | Error capture rate in escalation bucket ≥ 80% | `error_capture` | ≥ 0.80 | PENDING |
| H3 | Escalation rate < 30% | `pct_escalated` | < 0.30 | PENDING |

---

## AWS Infrastructure

**S3 Paths:**
```
Input:  s3://yrsn-datasets/s017/doe3_confusion_zone/results/
Output: s3://yrsn-datasets/s017/doe4_ood_detection/results/
Code:   s3://yrsn-datasets/rsct_code/doe4_ood_detection/
```

**Instance:** `ml.m5.large` (lightweight analysis)

---

## Three-Region Certificate Schema

```
Region 1 (PROCEED):      P(S_sup) < τ_low   → Auto-approve
Region 2 (ESCALATE):     τ_low ≤ P(S_sup) ≤ τ_high → Human review
Region 3 (RE_ENCODE):    P(S_sup) > τ_high  → Reject/retry
```

**Threshold derivation:** From DOE-3 confusion zone bounds

---

## Experiment Matrix

| Run | Task | Dependency | Est. Duration |
|-----|------|------------|---------------|
| 4.0 | Define τ_low, τ_high from DOE-3 | DOE-3 complete | ~5 min |
| 4.1 | Compute region assignments | Run 4.0 | ~5 min |
| 4.2 | Measure auto-resolve/escalate/reject rates | Run 4.1 | ~5 min |
| 4.3 | Compute error capture per region | Run 4.1 | ~5 min |

---

## SR 11-7 Mapping (Regulatory)

| RSCT Region | SR 11-7 Requirement | Action |
|-------------|---------------------|--------|
| PROCEED | Low-risk, automated | Audit log only |
| ESCALATE | Material uncertainty | Manager review |
| RE_ENCODE | High-risk | Block + remediation |

---

## Kill Rules

- If escalation rate > 50% → workflow too conservative
- If error capture < 60% → zone boundaries wrong

---

## Change Control

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-04-03 | Initial design |
