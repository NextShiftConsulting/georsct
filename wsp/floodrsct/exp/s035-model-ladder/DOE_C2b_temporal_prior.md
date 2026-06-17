# DOE-C2b: Temporal Prior — Sequential Certification with P16 Hint

**Experiment:** s035-model-ladder / DOE-C2b
**Role:** Test whether prior certificates improve sequential flood certification
**Status:** DESIGNED
**Depends on:** DOE-C2a (needs omega for blending weight)
**Blocks:** None (informational for floodcaster engine design)

---

## Hypothesis

**H-C2b:** Using a prior event's certificate as a P16 hint (via alpha_omega
blending) reduces certificate variance for subsequent events in the same
geography, without inflating forward score.

**Null:** Sequential blending produces identical certificates to independent
certification (prior adds no information).

**Anti-hypothesis:** If the prior INFLATES forward score, we have construct
leakage through time (the FEMA->Hazus->NFIP chain carries forward).

---

## Design Matrix

| Factor | Levels | Type |
|--------|--------|------|
| Scenario | houston (3+ events), nyc (2 events) | Fixed |
| Construct | NFIP, FEMA, JRC | Fixed (3 — available across events) |
| Event ordering | Chronological | Fixed |
| Blending | independent, sequential_p16 | Treatment (2 arms) |

Houston events (chronological): harvey2017, imelda2019, beryl2024
NYC events: sandy2012, nyc2023

**Total cells:** 2 scenarios x 3 constructs x 2 arms x (2-3 events) = ~30 certifications

---

## Method

### Arm A: Independent (control)
Certify each event independently using DOE-C1 protocol. No information
flows between events.

### Arm B: Sequential P16
For event sequence E_1, E_2, ..., E_k:
1. Certify E_1 independently -> certificate C_1.
2. For E_t (t >= 2):
   a. Take prior certificate C_{t-1} as hint.
   b. Compute raw certificate C_t^raw via DOE-C1 protocol on E_t data.
   c. Use omega from DOE-C2a as blending weight.
   d. Compute blended: alpha_omega_t = omega * C_t^raw.forward + (1-omega) * C_{t-1}.forward
   e. Record both C_t^raw and C_t^blended.

### Diagnostic
Compare Arm A vs Arm B certificates:
- If blended forward_score > raw forward_score: prior is informative (good).
- If blended forward_score > actual next-event forward_score: leakage (bad).
- If blended == raw within CI: prior adds nothing (omega too high or prior uninformative).

---

## Acceptance Criteria

| ID | Criterion | Test |
|----|-----------|------|
| AC-C2b-1 | Sequential blending reduces certificate variance | std(C_blended) < std(C_independent) for NFIP |
| AC-C2b-2 | No construct leakage through time | Blended R2 does not exceed holdout R2 |
| AC-C2b-3 | FEMA prior is less informative than NFIP prior | delta(FEMA_blended, FEMA_raw) < delta(NFIP_blended, NFIP_raw) |

---

## S3 Output Convention

```
results/s035/doe_c2b/
  temporal_prior_{scenario}.json
  cache/
    sequential_certificates_{scenario}.parquet
```

---

## Resource Estimate

- Instance: ml.m5.large (2 vCPU, 8 GB)
- Small experiment: ~30 certifications total
- Wall clock: ~30 min per scenario
- Cost: ~$0.15

---

## Connection to Theory

This is the DoubleTake "second pass" applied to flood certification.
In DoubleTake, the prior geometry (TSDF render) informs the second
cost-volume pass. Here, the prior certificate from event t-1 informs
the certification of event t via P16 blending. The key constraint
is P5 (no runtime learning): the prior is a cached measurement,
not a gradient signal.
